from std.gpu import block_dim,block_idx,thread_idx
from max.gpu.sync import barrier
from layout import TileTensor,LayoutTensor
from layout.tile_tensor import stack_allocation
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from max.gpu.memory import AddressSpace
from src.lbm import Lattice
from src.lbm import LBM_Grid
from src.lbm import SOLID_NODE,FLUID_NODE
from src.utils import Vector,ContextTileTensor


def LBM_kernel[
                Flayout:Layout,
                BClayout:Layout,
                Flaglayout:Layout,
                grid: LBM_Grid,
                ]
                (
                f_out:TileTensor[grid.float_dtype,type_of(Flayout),MutAnyOrigin],
                f_in:TileTensor[grid.float_dtype,type_of(Flayout),ImmutAnyOrigin],
                bc:TileTensor[grid.float_dtype,type_of(BClayout),ImmutAnyOrigin],
                flags:TileTensor[DType.uint8,type_of(Flaglayout),ImmutAnyOrigin],
                inv_tau:Scalar[grid.float_dtype]
                )
                where Flayout.rank == 4 and BClayout.rank == 4 and Flaglayout.rank == 3:
    '''
    From reorderThreads. This uses tiletensor Indexing. This example is used to compare speed to converting to layout tensor (which should be zero cost).

    '''
    comptime D = grid.D
    comptime Q = grid.Q
    comptime float_dtype = grid.float_dtype
    comptime lattice = grid.lattice
    comptime nx = grid.nx
    comptime ny = grid.ny
    comptime nz = grid.nz

    comptime assert Flayout.flat_rank == 8 and BClayout.flat_rank == 8 and Flaglayout.flat_rank == 6
    comptime tile_size = Flaglayout.static_shape[0] # For now lets assime tile size is the same

    # Convience Variable Names and constants
    var weights = materialize[lattice.weights]()
    var directions = materialize[lattice.directions]()
    var opposite_index = materialize[lattice.opposite_indices]()
    var grid_shape = Vector[DType.int32,3](Int32(nx),Int32(ny),Int32(nz))

    
    # We are Row Major for tiler
    var block_x,block_dim_x = block_idx.y,block_dim.y
    var block_y,block_dim_y = block_idx.x,block_dim.x
    var block_z = 0

    # We are Col Major for tiles
    var local_x = thread_idx.y
    var local_y = thread_idx.x
    var local_z = 0
    
    var x = block_x*block_dim_x + local_x
    var y = block_y*block_dim_y + local_y
    var z = 0
    #Right Now we index by block y the fastest
    var index = Vector[DType.int32,3](Int32(x),Int32(y),Int32(z))
    var local_index:InlineArray[Int,3] = [local_x,local_y,local_z]
    var block_index:InlineArray[Int,3] = [block_x,block_y,block_z]
    # Main Compute
    var rho: Scalar[float_dtype]
    if (index[0] < grid_shape[0]) and (index[1] < grid_shape[1]) and (index[2] < grid_shape[2]): # Basic Guard
        var f_new = Vector[float_dtype,Q](fill = 0.)
        var velocity = Vector[float_dtype,D](uninitialized = True)
        comptime for q in range(Q):
            var f_opp = f_in[0,opposite_index[q],   local_x,block_x,    local_y,block_y,    local_z,block_z] # Need (local_idx,block_idx)
            var direction = directions[q]
            ref pull_local,pull_block = get_adjacent_idx[_,D,Flaglayout,tile_size,-1](local_index,block_index,direction) # Pulling Scheme

            var pulled_f = f_in[0,q,pull_local[0],pull_block[0],     pull_local[1],pull_block[1],   pull_local[2],pull_block[2]]
            var pulled_flag = flags[pull_local[0],pull_block[0],     pull_local[1],pull_block[1],   pull_local[2],pull_block[2]]

            if pulled_flag == FLUID_NODE:
                f_new[q] = pulled_f
            elif pulled_flag == SOLID_NODE:
                comptime for ii in range(D):
                    velocity[ii] = bc[pull_local[0],pull_block[0],     pull_local[1],pull_block[1],   pull_local[2],pull_block[2],   ii,0]
                rho = bc[pull_local[0],pull_block[0],     pull_local[1],pull_block[1],   pull_local[2],pull_block[2],   D,0]
                f_new[q] = f_opp + 2.*3.*weights[q]*rho*(directions[q].cast_to[float_dtype]().dot(velocity))
            # f_new[q] = pulled_f if pulled_flag == FLUID_NODE else f_new[q]
            #  # BounceBack with moving wall BC put together (2nd term is 0 if stationary wall)
            # comptime for ii in range(D):
            #     velocity[ii] = bc[pull_local[0],pull_block[0],     pull_local[1],pull_block[1],   pull_local[2],pull_block[2],   ii,0]
            # rho = bc[pull_local[0],pull_block[0],     pull_local[1],pull_block[1],   pull_local[2],pull_block[2],   D,0]
            # f_new[q] = f_opp + 2.*3.*weights[q]*rho*(directions[q].cast_to[float_dtype]().dot(velocity)) if pulled_flag == SOLID_NODE else f_new[q]

        velocity.fill(0)
        rho = 0
        comptime for q in range(Q):
            rho += f_new[q]
            velocity += f_new[q]*directions[q].cast_to[float_dtype]()
        velocity /= rho
        # Collision Term
        comptime for q in range(Q):
            var f_eq = SRT(weights[q],rho,velocity,directions[q].cast_to[float_dtype]())            
            f_out[0,q,   local_x,block_x,    local_y,block_y,    local_z,block_z] = f_new[q] -  inv_tau*(f_new[q]- f_eq)

@always_inline
def get_adjacent_idx[int_dtype:DType,D:Int,flag_layout:Layout[...],tile_size:Int,shift:Int = 1]
                    (local_index:InlineArray[Int,3],block_index:InlineArray[Int,3],direction:Vector[int_dtype,D]) -> Tuple[InlineArray[Int,3],InlineArray[Int,3]]:
    comptime assert flag_layout.flat_rank == 6 and flag_layout.rank == 3
    var adj_local_index = InlineArray[Int,3](fill =0)
    var adj_block_index = InlineArray[Int,3](fill =0)
    
    comptime for d in range(D):
        var direction_d = shift*Int(direction[d])
        var current_pull_index = local_index[d] + direction_d
        adj_local_index[d] = current_pull_index % tile_size # Modulo as we flip back
        var next_block = current_pull_index < 0 or current_pull_index >= tile_size
        adj_block_index[d] = (block_index[d] + direction_d if next_block else 0) % flag_layout.static_shape[1+2*d]
    return adj_local_index^, adj_block_index^



def get_adjacent_idx[int_dtype:DType,D:Int,shift:Int32 = 1](index:Vector[DType.int32,3],grid_shape:Vector[DType.int32,3],direction:Vector[int_dtype,D],) -> Vector[DType.int32,3]:
    comptime assert D <= 3 
    var adj_index = Vector[DType.int32,3](uninitialized = True)
    comptime for d in range(D):
        adj_index[d] = (index[d] + shift*Int32(direction[d])) % grid_shape[d]
    return adj_index^

@always_inline
def SRT[dtype:DType,D:Int,//](weight:Scalar[dtype],density:Scalar[dtype],velocity:Vector[dtype,D],direction:Vector[dtype,D]) -> Scalar[dtype]:
    comptime assert dtype.is_floating_point(), 'DType to BGK_collision term should be Float point like' # Weied using where statement cause compile error?
    var ei_dot_u = velocity.dot(direction)
    return weight*density*(1 + 3.*ei_dot_u + 4.5*ei_dot_u*ei_dot_u - 1.5*velocity.dot(velocity))

# last modified by: muse-spark-1.2 on 2026/09/01
