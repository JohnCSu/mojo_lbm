from std.gpu import block_dim,block_idx,thread_idx,barrier,grid_dim
from layout import TileTensor,LayoutTensor,coord
from layout.tile_tensor import stack_allocation
from layout.tile_layout import Layout,col_major,Coord,TensorLayout
from std.gpu.memory import AddressSpace
from std.gpu import barrier
from src.lbm.lattice_models import Lattice
from src.lbm import LBM_Grid
from src.lbm.constants import SOLID_NODE,FLUID_NODE
from src.utils import Vector,ContextTileTensor


from std.algorithm.functional import vectorize
from std.sys import simd_width_of




def LBM_kernel[
                Flayout:Layout,
                BClayout:Layout,
                Flaglayout:Layout,
                grid: LBM_Grid,
                simd_width:Int,
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
    Base LBM to also handle 3D and non_square Grids. Key assumption is that block dim == tile-size 
    i.e. grid can be non-square but block is squre (same block dim in each x,y,z).
    ''' 
    comptime D = grid.D
    comptime Q = grid.Q
    comptime float_dtype = grid.float_dtype
    comptime lattice_model = grid.lattice_model
    comptime nx = grid.nx
    comptime ny = grid.ny
    comptime nz = grid.nz
    comptime tile_size = grid.tile_shape[0]

    # Convience Variable Names and constants
    comptime assert Flaglayout.flat_rank == 3 or Flaglayout.flat_rank == 6
    comptime assert Flayout.rank == 4 and BClayout.rank == 4 and Flaglayout.rank == 3
    comptime assert Flayout.static_shape[6] == Q
    comptime assert tile_size % 2 == 0 or tile_size == 1,'Tile size must be even or 1'

    comptime weights = lattice_model.weights
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
    comptime shared_x_dim = grid.tile_size + 2 if nx > 1 else 1
    comptime shared_y_dim = grid.tile_size + 2 if ny > 1 else 1
    comptime shared_z_dim = grid.tile_size + 2 if nz > 1 else 1
    
    shared_flags = stack_allocation[DType.uint8,AddressSpace.SHARED](col_major[shared_x_dim,shared_y_dim,shared_z_dim]())

    block_index:InlineArray[Int,3] = [block_x,block_y,block_z]
    NUM_BLOCKS:InlineArray[Int,3] = [grid_dim.x,grid_dim.y,grid_dim.z]

    comptime tid_limit = shared_x_dim//2 * shared_y_dim * shared_z_dim

    if tid < tid_limit:
        sync_set_shared_flags[D,nx,ny,nz,tile_size](tid,block_index,shared_flags,flags)
    
    comptime shift_x = 1 if nx > 1 else 0
    comptime shift_y = 1 if ny > 1 else 0
    comptime shift_z = 1 if nz > 1 else 0
    
    var f_new = Vector[float_dtype,Q](fill = 0.)
    var velocity = Vector[float_dtype,D](uninitialized = True)
    var rho:Scalar[float_dtype] = 0
    barrier()
    
    if (index[0] < grid_shape[0]) and (index[1] < grid_shape[1]) and (index[2] < grid_shape[2]): # Basic Guard
        comptime for q in range(Q):
            direction = directions[q]
            pull_index = get_adjacent_idx[_,D,-1](index,grid_shape,direction) # Pulling Scheme
            pulled_f = f_in.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2],q)))[0]
            
            local_flag_x = local_x + shift_x - Int(direction[0])
            local_flag_y = local_y + shift_y - (Int(direction[1]) if D >= 2 else 0)
            local_flag_z = local_z + shift_z - (Int(direction[2]) if D ==3 else 0)
            pulled_flag = shared_flags[local_flag_x,local_flag_y,local_flag_z]
            f_new[q] = pulled_f #if pulled_flag == FLUID_NODE else f_new[q]

            if pulled_flag == SOLID_NODE:
                f_opp = f_in.load(coord[DType.uint32]((x,y,z,Int(opposite_index[q]))))[0] # Need this as  Element Type is a Simd Vec of size 1
                comptime for ii in range(D):
                    velocity[ii] = bc.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2],ii)))[0]
                rho = bc.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2],D)))[0]
                f_new[q] = f_opp + 2.*3.*weights[q]*rho*(directions[q].cast_to[float_dtype]().dot(velocity))

    # Get Velocity and Density
    velocity.fill(0)
    rho = 0
    comptime for q in range(Q):    
        rho += f_new[q]
        velocity += f_new[q]*directions[q].cast_to[float_dtype]()

    velocity /= rho
    # Collision Term
    u_dot_u = velocity.dot(velocity)

    comptime for q in range(Q):
        f_eq = SRT(weights[q],rho,velocity,u_dot_u,directions[q].cast_to[float_dtype]())            
        f_out.store(coord = coord[DType.uint32]((x,y,z,q)),value = f_new[q] -  inv_tau*(f_new[q]- f_eq))




@always_inline
def sync_set_shared_flags[dtype:DType,
                        flagLayoutType:TensorLayout,
                        //,
                        D:Int,
                        nx:Int,
                        ny:Int,
                        nz:Int,
                        tile_size:Int]
                        (tid:Int,
                        block_index:InlineArray[Int,3],
                        shared_flags:TileTensor[dtype,_,MutAnyOrigin,address_space = AddressSpace.SHARED,...],
                        flags:TileTensor[dtype,flagLayoutType,_],
                        ):
    comptime assert shared_flags.flat_rank == 3
    comptime shared_x_dim = shared_flags.static_shape[0]
    comptime shared_y_dim = shared_flags.static_shape[1]
    comptime shared_z_dim = shared_flags.static_shape[2]

    comptime max_shared_tid = shared_x_dim//2 + shared_y_dim + shared_z_dim
    comptime shift_x = 1 if nx > 1 else 0
    comptime shift_y = 1 if ny > 1 else 0
    comptime shift_z = 1 if nz > 1 else 0
    
    # Halo: 2 passes of 5x10x10 = 500 cells each
    comptime half_x = shared_x_dim // 2
    comptime tid_limit = half_x * shared_y_dim * shared_z_dim

    shared_local_index = InlineArray[Int,3](uninitialized = True)

    comptime for pass_id in range(2): # Each thread is responsible for 2 elements along x axis (1st dim)
        var sx = 2*(tid % half_x)+ pass_id
        var sy = (tid // half_x) % shared_y_dim 
        var sz = tid // (half_x * shared_y_dim)
        
        shared_local_index[0] = sx - shift_x
        shared_local_index[1] = sy - shift_y
        shared_local_index[2] = sz - shift_z
    
        s_local_index,s_block_index = get_global_xyz_from_block_and_local_idx_old[D,flagLayoutType,tile_size](shared_local_index,block_index)
        gx = s_local_index[0] + s_block_index[0]*tile_size
        gy = s_local_index[1] + s_block_index[1]*tile_size
        gz = s_local_index[2] + s_block_index[2]*tile_size
        
        shared_flags[sx,sy,sz] = flags.load(coord[DType.int32]((gx,gy,gz)))
    # barrier()


@always_inline
def get_global_xyz_from_block_and_local_idx_old[D:Int,FlagLayoutType:TensorLayout,tile_size:Int]
                    (
                        local_index:InlineArray[Int,3],
                        block_index:InlineArray[Int,3]
                    ) -> Tuple[InlineArray[Int,3],InlineArray[Int,3]]:
    comptime assert FlagLayoutType.rank == 3
    comptime assert FlagLayoutType.flat_rank == 6 or FlagLayoutType.flat_rank == 3
    comptime is_nested = FlagLayoutType.flat_rank == 6
    
    adj_local_index = InlineArray[Int,3](fill =0)
    adj_block_index = InlineArray[Int,3](fill =0)
    
    comptime for d in range(D):
        adj_local_index[d] = local_index[d] % tile_size # Modulo as we flip back
        sign = -1 if local_index[d] < 0 else 1
        next_block =  local_index[d] < 0 or local_index[d] >= tile_size
        adj_block_index[d] = (block_index[d] + (sign if next_block else 0)) % FlagLayoutType.static_shape[1+2*d if is_nested else d]
    return adj_local_index,adj_block_index



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
        adj_block_index[d] = (block_index[d] + (sign if next_block else 0)) % flag_layout.static_shape[1+2*d if is_nested else d]
    return adj_local_index,adj_block_index




@always_inline
def get_adjacent_idx[int_dtype:DType,D:Int,shift:Int = 1](index:InlineArray[Int,3],grid_shape:InlineArray[Int,3],direction:Vector[int_dtype,D],) -> InlineArray[Int,3]:
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


    
