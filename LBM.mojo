from std.gpu.host import DeviceContext
from std.sys import has_accelerator
from std.gpu import block_idx,thread_idx
from layout import TileTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout

from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from std.collections import InlineArray
from std.memory import Pointer
from std.collections import Set,Dict

from flags import *
from contextTensor import ContextTileTensor
from vector import Vector

struct LatticeModel[D:Int,Q:Int,float_dtype:DType,int_dtype:DType](ImplicitlyCopyable):
    comptime int_vector = Vector[Self.int_dtype,Self.D]
    comptime float_vector = Vector[Self.float_dtype,Self.D]
    comptime dimension = Self.Q
    comptime int_scalar = Scalar[Self.int_dtype]
    comptime float_scalar = Scalar[Self.float_dtype]

    var directions:InlineArray[Self.int_vector,Self.Q]
    var float_directions:InlineArray[Self.float_vector,Self.Q]

    var weights:Vector[Self.float_dtype,Self.Q]
    var opposite_indices:InlineArray[Self.int_scalar,Self.Q]

    def __init__(out self,directions:InlineArray[Self.int_vector,Self.Q],float_directions:InlineArray[Self.float_vector,Self.Q],weights:Vector[Self.float_dtype,Self.Q]):
        self.directions = directions
        self.weights = weights
        self.opposite_indices = InlineArray[self.int_scalar,Self.Q](fill = 0)
        self.float_directions = float_directions
        self._get_opposite_indices()
        
    def _get_opposite_indices(mut self):
        for i in range(Self.Q): # Cant be bothered making an effecient algorithim to search opposite
            opp_direction = self.directions[i].copy()
            for j in range(Self.D):
                opp_direction[j] = opp_direction[j]*(-1)
            for k in range(Self.Q):
                if opp_direction == self.directions[k]:
                    self.opposite_indices[i] = self.int_scalar(k)
                    break


def get_D2Q9[float_dtype:DType = DType.float32,int_dtype:DType = DType.int32]() -> LatticeModel[2,9,float_dtype,int_dtype]:  
    comptime D = 2
    comptime Q = 9
    comptime int_vector = Vector[int_dtype,D]
    comptime float_vector = Vector[float_dtype,D]
    
    float_directions_list:List[List[Scalar[float_dtype]]]  =  
                                        [
                                        [ 0,  0], # 0: Center (rest)
                                        [ 1,  0], # 1: East
                                        [ 0,  1], # 2: North
                                        [-1,  0], # 3: West
                                        [ 0, -1], # 4: South
                                        [ 1,  1], # 5: North-East
                                        [-1,  1], # 6: North-West
                                        [-1, -1], # 7: South-West
                                        [ 1, -1]  # 8: South-East
                                        ]
    float_directions = InlineArray[float_vector,Q](uninitialized = True)
    for i in range(Q):
        _ = float_directions[i].__init__(float_directions_list[i])
    
    directions_list:List[List[Scalar[int_dtype]]] =
                                    [
                                        [ 0,  0], # 0: Center (rest)
                                        [ 1,  0], # 1: East
                                        [ 0,  1], # 2: North
                                        [-1,  0], # 3: West
                                        [ 0, -1], # 4: South
                                        [ 1,  1], # 5: North-East
                                        [-1,  1], # 6: North-West
                                        [-1, -1], # 7: South-West
                                        [ 1, -1]  # 8: South-East
                                    ]

    directions = InlineArray[int_vector,Q](uninitialized = True)
    for i in range(Q):
        _ = directions[i].__init__(directions_list[i])

    weights =  Vector[float_dtype,Q](
                                    4./9.,                          # 0: Center
                                    1./9., 1./9., 1./9., 1./9.,           # 1-4: Axis
                                    1./36., 1/36., 1./36., 1./36.        # 5-8: Diagonal
                                    )

    return LatticeModel[D,Q,float_dtype,int_dtype](directions,float_directions,weights)
    

struct LBM_Grid[float_dtype:DType,int_dtype:DType,D:Int,Q:Int,//,latticeModel:LatticeModel[D,Q,float_dtype,int_dtype],nx:Int,ny:Int,nz:Int]():
    comptime float_scalar = Scalar[Self.float_dtype]
    var dx:Self.float_scalar
    var area:Self.float_scalar
    var volume:Self.float_scalar
    var shape:InlineArray[Int,3]
    var num_points:Int
    var f_field_size: Int
    var vel_field_size: Int
    var bc_field_size:Int
    def __init__(out self,dx:Self.float_scalar):
        # (self.nx,self.ny,self.nz) = (nx,ny,nz)
        self.dx = dx
        self.area = dx**2
        self.volume = dx**3
        self.shape = [Self.nx,Self.ny,Self.nz]
        self.num_points = Self.nx*Self.ny*Self.nz
        self.f_field_size = Self.Q*self.num_points
        self.vel_field_size = Self.D*self.num_points
        self.bc_field_size = (Self.D+1)*self.num_points


def set_outer_walls[float_dtype:DType where float_dtype.is_floating_point(),BCLayout:TensorLayout,FlagLayout:TensorLayout,flag_origin:Origin[mut=True],bc_origin:Origin[mut=True],
                    //,
                    D:Int,
                    nx:Int,
                    ny:Int,
                    nz:Int,
                    ]
                    (flags:TileTensor[DType.uint8,FlagLayout,flag_origin],
                            bc:TileTensor[float_dtype,BCLayout,bc_origin],
                            side:String,
                            boundary_type:Int,
                            velocity:List[Scalar[float_dtype]],
                            density:Scalar[float_dtype]) raises:

    
    axes:Dict[String,Int] = {'X':0,
                    'Y':1,
                    'Z':2,}
    valid_strings:Set[String] = {'-X','+X','-Y','+Y','-Z','+Z'}
    # (side) in valid_strings
    assert side in valid_strings, 'Must be valid string'
    axis = axes[String(side[byte = 1])]
    # range_values = [[0,nx-1 if nx > 1 else 1],[0,ny-1 if ny > 1 else 1],[0,nz-1 if nz > 1 else 1]]
    range_values = [[0,nx],[0,ny],[0,nz]]
    
    if side[byte = 0] == '-':
        range_values[axis] = [0,1]
    else:
        end_index = range_values[axis][1] - 1
        range_values[axis] = [end_index,end_index+1]
    
    x_slice = Tuple(range_values[0][0],range_values[0][1])
    y_slice = Tuple(range_values[1][0],range_values[1][1])
    z_slice = Tuple(range_values[2][0],range_values[2][1])
    
    #Set Flags
    boundary = flags.slice(x_slice,y_slice,z_slice)
    _ = boundary.fill(flags.ElementType(boundary_type))

    # Velocity
    for i in range(D):
        bc_vel = bc.slice(x_slice,y_slice,z_slice,(i,i+1))
        _ = bc_vel.fill(velocity[i])
    
    # Density
    bc_rho = bc.slice(x_slice,y_slice,z_slice,(D,D+1))
    _ = bc_rho.fill(density)


def LBM_kernel[ float_dtype:DType,D:Int,Q:Int,Flayout:TensorLayout,BCLayout:TensorLayout,FlagLayout:TensorLayout,
                //,
                lattice_model:LatticeModel[D,Q,float_dtype,DType.int32],
                ]
                ( 
                f_out:TileTensor[float_dtype,Flayout,MutAnyOrigin],
                f_in:TileTensor[float_dtype,Flayout,MutAnyOrigin],
                bc:TileTensor[float_dtype,BCLayout,MutAnyOrigin],
                flags:TileTensor[DType.uint8,FlagLayout,MutAnyOrigin],
                grid_shape:Vector[DType.int32,3],
                inv_tau:Scalar[float_dtype]
                ):
    
    comptime assert f_in.flat_rank == 4 and f_in.flat_rank == f_out.flat_rank
    comptime assert bc.flat_rank == 4
    comptime assert flags.flat_rank == 3
    
    x = block_dim.x * block_idx.x + thread_idx.x
    y = block_dim.y * block_idx.y + thread_idx.y
    z = block_dim.z * block_idx.z + thread_idx.z
    index = Vector[DType.int32,3](Int32(x),Int32(y),Int32(z))

    if index[0] < grid_shape[0] and index[1] < grid_shape[1] and index[2] < grid_shape[3]: # Basic Guard
        var f_new = Vector[float_dtype,D](fill = 0.)
        var velocity = Vector[float_dtype,D]()
        for q in range(Q):
            f_opp = f_in[lattice_model.opposite_indices[q],x,y,z]
            direction = lattice_model.directions[q]
            
            pull_index = get_adjacent_idx[D,-1](index,grid_shape,direction) # Pulling Scheme
            pulled_f = f_in[q,pull_index[0],pull_index[1],pull_index[2]]
            
            comptime for ii in range(D):
                velocity[ii] = bc[pull_index[0],pull_index[1],pull_index[2],ii]
            rho = bc[pull_index[0],pull_index[1],pull_index[2],D]

            if flags[pull_index[0],pull_index[1],pull_index[2]] == 0: # Stream
                f_new[q] = pulled_f
            elif flags[pull_index[0],pull_index[1],pull_index[2]] == 1: # BounceBack with moving wall BC put together (2nd term is 0 if stationary wall)
                f_new[q] = f_opp + 2.*3.*lattice_model.weights[q]*rho*(lattice_model.float_directions[q].dot(velocity))

        # Get Velocity and Density
        velocity.fill(0)
        rho = 0
        for q in range(Q):
            rho += f_new[q]
            velocity += f_new[q]*lattice_model.float_directions[q]
        velocity /= rho

        # Collision Term
        for q in range(Q):
            f_eq = BGK_Collision(lattice_model.weights[q],rho,velocity,lattice_model.float_directions[q])
            f_new[q] += inv_tau*(f_eq - f_new[q])

        for q in range(Q):
            f_out[q,x,y,z] = f_new[q]

def get_adjacent_idx[D:Int,shift:Int32 = 1](index:Vector[DType.int32,3],grid_shape:Vector[DType.int32,3],direction:Vector[DType.int32,D],) -> Vector[DType.int32,3]:
    comptime assert D <= 3 
    adj_index = Vector[DType.int32,3]()
    comptime for d in range(D):
        adj_index[d] = (index[d] + shift*direction[d]) % grid_shape[d]
    return adj_index


def BGK_Collision[dtype:DType,D:Int,//](weight:Scalar[dtype],density:Scalar[dtype],velocity:Vector[dtype,D],direction:Vector[dtype,D]) -> Scalar[dtype]:
    comptime assert dtype.is_floating_point(), 'DType to BGK_collision term should be Float point like' # Weied using where statement cause compile error?
    ei_dot_u = velocity.dot(direction)
    return weight*density*(1 + 3.*ei_dot_u + 4.5*ei_dot_u*ei_dot_u +- 1.5*velocity.dot(velocity))


    

    





    


