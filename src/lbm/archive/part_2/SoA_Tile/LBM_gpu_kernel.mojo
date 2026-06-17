from std.gpu import block_dim,block_idx,thread_idx,barrier
from layout import TileTensor,LayoutTensor,coord
from layout.tile_tensor import stack_allocation
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from std.gpu.memory import AddressSpace
from src.lbm.lattice_models import LatticeModel
from src.lbm import LBM_Grid
from src.lbm.flags import SOLID_NODE,FLUID_NODE
from src.utils import Vector,ContextTileTensor

from std.algorithm.functional import vectorize
from std.sys import simd_width_of

def LBM_kernel[ float_dtype:DType,D:Int,Q:Int,
                lattice_model:LatticeModel[D,Q,float_dtype,DType.int32],
                nx:Int,ny:Int,nz:Int,tile_size:Int,
                //,
                grid: LBM_Grid[lattice_model,nx,ny,nz,tile_size], 
                Flayout:Layout[...] where Flayout.rank == 4,  
                BClayout:Layout[...] where BClayout.rank == 4,
                Flaglayout:Layout[...] where Flaglayout.rank == 3,
                simd_width:Int,
                *,
                reorder_threads:Bool = True,
                ]
                ( 
                f_out:TileTensor[float_dtype,type_of(Flayout),MutAnyOrigin],
                f_in:TileTensor[float_dtype,type_of(Flayout),ImmutAnyOrigin],
                bc:TileTensor[float_dtype,type_of(BClayout),ImmutAnyOrigin],
                flags:TileTensor[DType.uint8,type_of(Flaglayout),ImmutAnyOrigin],
                inv_tau:Scalar[float_dtype]
                ):
    '''
    From part_1/tiled. Default layout should be Col_major Tile and ORw major tiler. Uses Compile time unrolling for last 2 for loops. Main Stream/BC comptime looping does not work.
    ''' 
    # Convience Variable Names and constants
    comptime assert Flaglayout.flat_rank == 6
    comptime assert Flayout.rank == 4 and BClayout.rank == 4 and Flaglayout.rank == 3
    comptime assert Flayout.static_shape[6] == Q
    comptime weights = lattice_model.weights
    comptime float_directions = lattice_model.float_directions
    comptime directions = lattice_model.directions
    comptime opposite_index = lattice_model.opposite_indices
    comptime grid_shape:InlineArray[Int,3] = [nx,ny,nz]
    
    comptime if reorder_threads:
        comptime if D==1: # Indexing based on Dimension of Grid
            block_x,block_dim_x = block_idx.x,block_dim.x
            block_dim_y,block_dim_z = 0,0
            block_y,block_z = 0,0

            local_x = thread_idx.y
            local_y = 0
            local_z = 0
        elif D == 2:
            
            block_x,block_dim_x = block_idx.y,block_dim.y
            block_y,block_dim_y = block_idx.x,block_dim.x
            block_z,block_dim_z = 0,0

            local_x = thread_idx.y
            local_y = thread_idx.x
            local_z = 0

        else:
            block_x,block_dim_x = block_idx.z,block_dim.z
            block_y,block_dim_y = block_idx.y,block_dim.y
            block_z,block_dim_z = block_idx.x,block_dim.x

            local_x = thread_idx.z
            local_y = thread_idx.y
            local_z = thread_idx.x
    else:
        block_x,block_dim_x = block_idx.x,block_dim.x
        block_y,block_dim_y = block_idx.y,block_dim.y
        block_z,block_dim_z = block_idx.z,block_dim.z

        local_x = thread_idx.x
        local_y = thread_idx.y
        local_z = thread_idx.z
    
    x = block_x*block_dim_x + local_x
    y = block_y*block_dim_y + local_y
    z = block_z*block_dim_z + local_z
    
    index:InlineArray[Int,3] = [x,y,z]    

    # Main Compute
    if (index[0] < grid_shape[0]) and (index[1] < grid_shape[1]) and (index[2] < grid_shape[2]): # Basic Guard
        var f_new = Vector[float_dtype,Q](fill = 0.)
        var velocity = Vector[float_dtype,D]()
        var rho:Scalar[float_dtype] = 0
        coord_x = coord[DType.uint32]( (local_x,block_x) )
        coord_y = coord[DType.uint32]((local_y,block_y))
        coord_z = coord[DType.uint32]((local_z,block_z))
        
        flags.prefetch(Coord(coord_x,coord_y,coord_z))

        comptime for q in range(Q):
            direction = directions[q]
            pull_index = get_adjacent_idx[D,-1](index,grid_shape,direction) # Pulling Scheme
            pulled_f = f_in.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2],q)))[0]
            pulled_flag = flags.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2])))[0]
            
            f_new[q] = pulled_f if pulled_flag == FLUID_NODE else f_new[q]

            if pulled_flag == SOLID_NODE:
                f_opp = f_in.load(coord[DType.uint32]((x,y,z,Int(opposite_index[q]))))[0] # Need this as  Element Type is a Simd Vec of size 1
                comptime for ii in range(D):
                    velocity[ii] = bc.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2],ii)))[0]
                rho = bc.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2],D)))[0]
                f_new[q] = f_opp + 2.*3.*weights[q]*rho*(float_directions[q].dot(velocity))
                
        # Get Velocity and Density
        velocity.fill(0)
        rho = 0
        comptime for q in range(Q):    
            rho += f_new[q]
            velocity += f_new[q]*float_directions[q]

        velocity /= rho
        # Collision Term
        u_dot_u = velocity.dot(velocity)

        comptime for q in range(Q):
            f_eq = SRT(weights[q],rho,velocity,u_dot_u,float_directions[q])            
            f_out.store(coord = coord[DType.uint32]((x,y,z,q)),value = f_new[q] -  inv_tau*(f_new[q]- f_eq))

@always_inline
def get_adjacent_idx[D:Int,shift:Int = 1](index:InlineArray[Int,3],grid_shape:InlineArray[Int,3],direction:Vector[DType.int32,D],) -> InlineArray[Int,3]:
    comptime assert D <= 3 
    adj_index = InlineArray[Int,3](fill = 0 )
    comptime for d in range(D):
        adj_index[d] = (index[d] + shift*Int(direction[d])) % grid_shape[d]
    return adj_index

@always_inline
def SRT[dtype:DType,D:Int,//](weight:Scalar[dtype],density:Scalar[dtype],velocity:Vector[dtype,D],u_dot_u:Scalar[dtype],direction:Vector[dtype,D]) -> Scalar[dtype]:
    comptime assert dtype.is_floating_point(), 'DType to BGK_collision term should be Float point like' # Weied using where statement cause compile error?
    ei_dot_u = velocity.dot(direction)
    return weight*density*(1 + 3.*ei_dot_u + 4.5*ei_dot_u*ei_dot_u - 1.5*u_dot_u)


    
