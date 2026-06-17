from std.gpu import block_dim,block_idx,thread_idx,barrier
from layout import TileTensor,LayoutTensor
from layout.tile_tensor import stack_allocation
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from std.gpu.memory import AddressSpace
from src.lbm.lattice_models import LatticeModel
from src.lbm.LBM import LBM_Grid
from src.lbm.flags import SOLID_NODE,FLUID_NODE
from src.utils import Vector,ContextTileTensor


def LBM_kernel[ float_dtype:DType,D:Int,Q:Int,
                lattice_model:LatticeModel[D,Q,float_dtype,DType.int32],
                nx:Int,ny:Int,nz:Int,
                tile_size:Int,
                //,
                grid: LBM_Grid[lattice_model,nx,ny,nz,tile_size], 
                Flayout:Layout[...] where Flayout.rank == 4,  
                BClayout:Layout[...] where BClayout.rank == 4,
                Flaglayout:Layout[...] where Flaglayout.rank == 3,
                ]
                ( 
                f_out:TileTensor[float_dtype,type_of(Flayout),MutAnyOrigin],
                f_in:TileTensor[float_dtype,type_of(Flayout),MutAnyOrigin],
                bc:TileTensor[float_dtype,type_of(BClayout),MutAnyOrigin],
                flags:TileTensor[DType.uint8,type_of(Flaglayout),MutAnyOrigin],
                inv_tau:Scalar[float_dtype]
                ):
    '''
    From reorderThreads. This uses tiletensor Indexing. This example is used to compare speed to converting to layout tensor (which should be zero cost)

    ''' 

    comptime assert Flayout.flat_rank == 8 and BClayout.flat_rank == 8 and Flaglayout.flat_rank == 6
    comptime tile_size = Flaglayout.static_shape[0] # For now lets assime tile size is the same

    # Convience Variable Names and constants
    comptime weights = lattice_model.weights
    comptime float_directions = lattice_model.float_directions
    comptime directions = lattice_model.directions
    comptime opposite_index = lattice_model.opposite_indices
    comptime grid_shape = Vector[DType.int32,3](Int32(nx),Int32(ny),Int32(nz))

    
    # We are Row Major for tiler
    block_x,block_dim_x = block_idx.y,block_dim.y
    block_y,block_dim_y = block_idx.x,block_dim.x
    block_z = 0

    # We are Col Major for tiles
    local_x = thread_idx.y
    local_y = thread_idx.x
    local_z = 0
    
    x = block_x*block_dim_x + local_x
    y = block_y*block_dim_y + local_y
    z = 0
    #Right Now we index by block y the fastest
    index = Vector[DType.int32,3](Int32(x),Int32(y),Int32(z))
    local_index:InlineArray[Int,3] = [local_x,local_y,local_z]
    block_index:InlineArray[Int,3] = [block_x,block_y,block_z]
    # Main Compute
    if (index[0] < grid_shape[0]) and (index[1] < grid_shape[1]) and (index[2] < grid_shape[2]): # Basic Guard
        var f_new = Vector[float_dtype,Q](fill = 0.)
        var velocity = Vector[float_dtype,D]()
        comptime for q in range(Q):
            direction = directions[q]
            pull_local,pull_block = get_adjacent_idx[D,Flaglayout,tile_size,-1](local_index,block_index,direction) # Pulling Scheme
            pulled_flag = flags[pull_local[0],pull_block[0],     pull_local[1],pull_block[1],   pull_local[2],pull_block[2]]
            
            pulled_f = f_in[0,q,pull_local[0],pull_block[0],     pull_local[1],pull_block[1],   pull_local[2],pull_block[2]]
            f_new[q] = pulled_f if pulled_flag == FLUID_NODE else f_new[q]
            if pulled_flag == SOLID_NODE: # BounceBack with moving wall BC put together (2nd term is 0 if stationary wall)
                f_opp = f_in[0,opposite_index[q],   local_x,block_x,    local_y,block_y,    local_z,block_z] # Need (local_idx,block_idx)
                comptime for ii in range(D):
                    velocity[ii] = bc[pull_local[0],pull_block[0],     pull_local[1],pull_block[1],   pull_local[2],pull_block[2],   ii,0]
                rho = bc[pull_local[0],pull_block[0],     pull_local[1],pull_block[1],   pull_local[2],pull_block[2],   D,0]

                f_new[q] = f_opp + 2.*3.*weights[q]*rho*(float_directions[q].dot(velocity))
        # Get Velocity and Density
        velocity.fill(0)
        rho = 0
        comptime for q in range(Q):
            rho += f_new[q]
            velocity += f_new[q]*float_directions[q]
        velocity /= rho
        # Collision Term
        comptime for q in range(Q):
            f_eq = SRT(weights[q],rho,velocity,float_directions[q])            
            f_out[0,q,   local_x,block_x,    local_y,block_y,    local_z,block_z] = f_new[q] -  inv_tau*(f_new[q]- f_eq)

@always_inline
def get_adjacent_idx[D:Int,flag_layout:Layout[...],tile_size:Int,shift:Int = 1]
                    (local_index:InlineArray[Int,3],block_index:InlineArray[Int,3],direction:Vector[DType.int32,D]) -> Tuple[InlineArray[Int,3],InlineArray[Int,3]]:
    comptime assert flag_layout.flat_rank == 6 and flag_layout.rank == 3
    adj_local_index = InlineArray[Int,3](fill =0)
    adj_block_index = InlineArray[Int,3](fill =0)
    
    comptime for d in range(D):
        direction_d = shift*Int(direction[d])
        current_pull_index = local_index[d] + direction_d
        adj_local_index[d] = current_pull_index % tile_size # Modulo as we flip back
        next_block = current_pull_index < 0 or current_pull_index >= tile_size
        adj_block_index[d] = (block_index[d] + direction_d if next_block else 0) % flag_layout.static_shape[1+2*d]
        # if current_pull_index < 0 or current_pull_index >= tile_size: # If spill outside of block we shift the blockindex by direction
        #     adj_block_index[d] = (block_index[d] + direction_d) % flag_layout.static_shape[1+2*d] # Modulo to flip back around
        # else:
        #     adj_block_index[d] = block_index[d]
    
    return adj_local_index,adj_block_index



def get_adjacent_idx[D:Int,shift:Int32 = 1](index:Vector[DType.int32,3],grid_shape:Vector[DType.int32,3],direction:Vector[DType.int32,D],) -> Vector[DType.int32,3]:
    comptime assert D <= 3 
    adj_index = Vector[DType.int32,3]()
    comptime for d in range(D):
        adj_index[d] = (index[d] + shift*direction[d]) % grid_shape[d]
    return adj_index

@always_inline
def SRT[dtype:DType,D:Int,//](weight:Scalar[dtype],density:Scalar[dtype],velocity:Vector[dtype,D],direction:Vector[dtype,D]) -> Scalar[dtype]:
    comptime assert dtype.is_floating_point(), 'DType to BGK_collision term should be Float point like' # Weied using where statement cause compile error?
    ei_dot_u = velocity.dot(direction)
    return weight*density*(1 + 3.*ei_dot_u + 4.5*ei_dot_u*ei_dot_u - 1.5*velocity.dot(velocity))

