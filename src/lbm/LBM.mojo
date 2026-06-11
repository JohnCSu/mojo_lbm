from std.gpu.host import DeviceContext
from layout import TileTensor,LayoutTensor
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from std.math import ceildiv
from std.collections import InlineArray
from std.memory import Pointer
from std.collections import Set,Dict

# from  import ContextTileTensor,Vector
# from ..utils import ContextTileTensor,Vector
from src.utils import Vector,ContextTileTensor

# struct LatticeModel[D:Int,Q:Int,float_dtype:DType,int_dtype:DType](ImplicitlyCopyable):
#     comptime int_vector = Vector[Self.int_dtype,Self.D]
#     comptime float_vector = Vector[Self.float_dtype,Self.D]
#     comptime dimension = Self.Q
#     comptime int_scalar = Scalar[Self.int_dtype]
#     comptime float_scalar = Scalar[Self.float_dtype]

#     var directions:InlineArray[Self.int_vector,Self.Q]
#     var float_directions:InlineArray[Self.float_vector,Self.Q]

#     var weights:Vector[Self.float_dtype,Self.Q]
#     var opposite_indices:InlineArray[Self.int_scalar,Self.Q]

#     def __init__(out self,directions:InlineArray[Self.int_vector,Self.Q],float_directions:InlineArray[Self.float_vector,Self.Q],weights:Vector[Self.float_dtype,Self.Q]):
#         self.directions = directions
#         self.weights = weights
#         self.opposite_indices = InlineArray[self.int_scalar,Self.Q](fill = 0)
#         self.float_directions = float_directions
#         self._get_opposite_indices()
        
#     def _get_opposite_indices(mut self):
#         for i in range(Self.Q): # Cant be bothered making an effecient algorithim to search opposite
#             opp_direction = self.directions[i].copy()
#             for j in range(Self.D):
#                 opp_direction[j] = opp_direction[j]*(-1)
#             for k in range(Self.Q):
#                 if opp_direction == self.directions[k]:
#                     self.opposite_indices[i] = self.int_scalar(k)
#                     break

# def get_D2Q9[float_dtype:DType = DType.float32,int_dtype:DType = DType.int32]() -> LatticeModel[2,9,float_dtype,int_dtype]:  
#     comptime D = 2
#     comptime Q = 9
#     comptime int_vector = Vector[int_dtype,D]
#     comptime float_vector = Vector[float_dtype,D]
    
#     float_directions_list:List[List[Scalar[float_dtype]]]  =  
#                                         [
#                                         [ 0,  0], # 0: Center (rest)
#                                         [ 1,  0], # 1: East
#                                         [ 0,  1], # 2: North
#                                         [-1,  0], # 3: West
#                                         [ 0, -1], # 4: South
#                                         [ 1,  1], # 5: North-East
#                                         [-1,  1], # 6: North-West
#                                         [-1, -1], # 7: South-West
#                                         [ 1, -1]  # 8: South-East
#                                         ]
#     float_directions = InlineArray[float_vector,Q](uninitialized = True)
#     for i in range(Q):
#         float_directions[i].fill_from_list(float_directions_list[i])
    
#     directions_list:List[List[Scalar[int_dtype]]] =
#                                     [
#                                         [ 0,  0], # 0: Center (rest)
#                                         [ 1,  0], # 1: East
#                                         [ 0,  1], # 2: North
#                                         [-1,  0], # 3: West
#                                         [ 0, -1], # 4: South
#                                         [ 1,  1], # 5: North-East
#                                         [-1,  1], # 6: North-West
#                                         [-1, -1], # 7: South-West
#                                         [ 1, -1]  # 8: South-East
#                                     ]

#     directions = InlineArray[int_vector,Q](uninitialized = True)
#     for i in range(Q):
#         directions[i].fill_from_list(directions_list[i])

#     weights =  Vector[float_dtype,Q](
#                                     4./9.,                          # 0: Center
#                                     1./9., 1./9., 1./9., 1./9.,           # 1-4: Axis
#                                     1./36., 1/36., 1./36., 1./36.        # 5-8: Diagonal
#                                     )

#     return LatticeModel[D,Q,float_dtype,int_dtype](directions,float_directions,weights)
    

struct LBM_Grid[float_dtype:DType,int_dtype:DType,D:Int,Q:Int,//,
                latticeModel:LatticeModel[D,Q,float_dtype,int_dtype],
                nx:Int,ny:Int,nz:Int,
                
                ](): 
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



def set_outer_walls[float_dtype:DType,
                    flag_origin:Origin[mut=True],
                    bc_origin:Origin[mut=True],
                    nx:Int,ny:Int,nz:Int,
                    D:Int,Q:Int,
                    latticeModel:LatticeModel[D,Q,float_dtype,DType.int32],
                    //,
                    grid:LBM_Grid[latticeModel,nx,ny,nz],
                    FlagLayout:Layout[_,_],
                    BCLayout:Layout[_,_],
                    ]
                    (flags:TileTensor[DType.uint8,type_of(FlagLayout),flag_origin],
                            bc:TileTensor[float_dtype,type_of(BCLayout),bc_origin],
                            side:String,
                            boundary_type:Scalar[DType.uint8],
                            velocity:List[Scalar[float_dtype]],
                            density:Scalar[float_dtype]) raises:
    '''
    Apply Boundary conditions to the outer walls. Uses LayoutTensor conversion to allow for layout independent indexing based on
    Tensor rank.

    '''
    comptime assert float_dtype.is_floating_point()
    comptime assert flags.rank == 3 and bc.rank == 4
    
    comptime flag_as_lt = LayoutTensor[DType.uint8,FlagLayout.to_layout(),flag_origin]
    comptime bc_as_lt = LayoutTensor[float_dtype,BCLayout.to_layout(),bc_origin]

    flags_lt = flag_as_lt(flags.ptr)
    bc_lt = bc_as_lt(bc.ptr)

    axes:Dict[String,Int] = {'X':0,
                    'Y':1,
                    'Z':2,}
    valid_strings:Set[String] = {'-X','+X','-Y','+Y','-Z','+Z'}
    # (side) in valid_strings
    assert side in valid_strings, 'Must be valid string'
    axis = axes[String(side[byte = 1])]
    
    # Layout independent
    end_values = [nx,ny,nz]
    if side[byte = 0] == '-':
        fixed = 0
    else:
        fixed = end_values[axis] - 1

    if axis == 0: # X-axis, fix x and loop
        for y in range(ny):
            for z in range(nz):
                flags_lt[fixed,y,z] = flags.ElementType(boundary_type)
                comptime for i in range(D):
                    bc_lt[fixed,y,z,i] = velocity[i]
                bc_lt[fixed,y,z,D] = density
    elif axis == 1:
        for x in range(nx):
            for z in range(nz):
                flags_lt[x,fixed,z] = flags.ElementType(boundary_type)
                comptime for i in range(D):
                    bc_lt[x,fixed,z,i] = velocity[i]
                bc_lt[x,fixed,z,D] = density
    else: # Loop Z-face
        for x in range(nx):
            for y in range(ny):
                flags_lt[x,y,fixed] = flags.ElementType(boundary_type)
                comptime for i in range(D):
                    bc_lt[x,y,fixed,i] = velocity[i]
                bc_lt[x,y,fixed,D] = density






    


