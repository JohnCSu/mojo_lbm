from std.gpu import block_dim,block_idx,thread_idx,barrier
from layout import TileTensor,LayoutTensor,coord
from layout.tile_tensor import stack_allocation
from layout.tile_layout import Layout,col_major,Coord,TensorLayout
from std.gpu.memory import AddressSpace
from std.gpu import barrier
from std.gpu.memory import async_copy, async_copy_wait_all

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
                Flayout:Layout[...],
                BClayout:Layout[...],
                Flaglayout:Layout[...],
                simd_width:Int,
                ]
                (
                f_out:TileTensor[float_dtype,type_of(Flayout),MutAnyOrigin],
                f_in:TileTensor[float_dtype,type_of(Flayout),ImmutAnyOrigin],
                bc:TileTensor[float_dtype,type_of(BClayout),ImmutAnyOrigin],
                flags:TileTensor[DType.uint8,type_of(Flaglayout),ImmutAnyOrigin],
                inv_tau:Scalar[float_dtype]
                )
                where tile_size >= 1 and Flayout.rank == 4 and BClayout.rank == 4 and Flaglayout.rank == 3:
    '''
    Shared Memory Load all flags in local threads to shared memory. Boundaries we just pull from global
    ''' 
    # Convience Variable Names and constants
    comptime assert Flaglayout.flat_rank == 3 or Flaglayout.flat_rank == 6

    comptime assert Flayout.rank == 4 and BClayout.rank == 4 and Flaglayout.rank == 3
    comptime assert Flayout.static_shape[6] == Q
    comptime weights = lattice_model.weights
    comptime float_directions = lattice_model.float_directions
    comptime directions = lattice_model.directions
    comptime opposite_index = lattice_model.opposite_indices
    comptime grid_shape:InlineArray[Int,3] = [nx,ny,nz]
    
    # comptime assert tile_size >= 5 if D == 2 else tile_size >= 8

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
    tid = thread_idx.z * block_dim.x * block_dim.y 
        + thread_idx.y * block_dim.x 
        + thread_idx.x

    # Load Flags into shared. For 3D we have 10x10x10 so we halve the x dim so we do 5x10x10 
    comptime shared_x_dim = grid.tile_size if nx > 1 else 1
    comptime shared_y_dim = grid.tile_size if ny > 1 else 1
    comptime shared_z_dim = grid.tile_size if nz > 1 else 1
    local_index:InlineArray[Int,3] = [local_x,local_y,local_z]
    
    pull_local:InlineArray[Int,3] = [local_x,local_y,local_z]
    shared_flags = stack_allocation[DType.uint8,address_space=AddressSpace.SHARED](col_major[shared_x_dim,shared_y_dim,shared_z_dim]())
    shared_flags[local_x,local_y,local_z] = flags.load(coord[DType.uint32]((x,y,z))) # Assume tile_size == block dims
    barrier()
    # Main Compute
    if (index[0] < grid_shape[0]) and (index[1] < grid_shape[1]) and (index[2] < grid_shape[2]): # Basic Guard
        var f_new = Vector[float_dtype,Q](fill = 0.)
        var velocity = Vector[float_dtype,D](uninitialized = True)
        var rho:Scalar[float_dtype] = 0

        comptime for q in range(Q):
            # Streaming Step (Pull f and flags from direction)
            direction = directions[q]
            pull_index = get_adjacent_idx[D,-1](index,grid_shape,direction) # Pulling Scheme
            pulled_f = f_in.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2],q)))[0]
            # Pull Flags from shared mem or global
            if at_block_boundary[D,tile_size,-1](local_index,direction): # if at Boundary of block so we just pull from global
                pulled_flag = flags.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2])))[0]
            else: # Else pull from shared mem
                comptime for k in range(D): 
                    pull_local[k] = local_index[k] -Int(direction[k])
                pulled_flag = shared_flags[pull_local[0],pull_local[1],pull_local[2]]
                # pulled_flag = shared_flags[local_x-Int(direction[0]),local_y-Int(direction[1]),local_z-Int(direction[2])]

            # Apply Boundary Conditions
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
def at_block_boundary[D:Int,tile_size:Int,shift:Int = 1](
                        local_index:InlineArray[Int,3],
                        direction:Vector[DType.int32,D]
                    ) -> Bool:
    _ = False
    comptime for d in range(D):
        direction_d = shift*Int(direction[d])
        current_pull_index = local_index[d] + direction_d
        if (current_pull_index < 0 or current_pull_index >= tile_size):
            return True
        # is_adj =  if not is_adj else is_adj
    return False

@always_inline
def get_global_xyz_from_block_and_local_idx[D:Int,flag_layout:Layout[...],tile_size:Int,shift:Int = 1]
                    (
                        local_index:InlineArray[Int,3],
                        block_index:InlineArray[Int,3]
                    ) -> Tuple[InlineArray[Int,3],InlineArray[Int,3]]:
    comptime assert flag_layout.rank == 3
    comptime assert flag_layout.flat_rank == 6 or flag_layout.flat_rank == 3
    comptime is_nested = flag_layout.flat_rank == 6
    
    adj_local_index = InlineArray[Int,3](fill =0)
    adj_block_index = InlineArray[Int,3](fill =0)
    
    comptime for d in range(D):
        adj_local_index[d] = local_index[d] % tile_size # Modulo as we flip back
        sign = -1 if local_index[d] < 0 else 1
        next_block =  local_index[d] < 0 or local_index[d] >= tile_size
        adj_block_index[d] = (block_index[d] + sign if next_block else 0) % flag_layout.static_shape[1+2*d if is_nested else d]
    return adj_local_index,adj_block_index




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


    
