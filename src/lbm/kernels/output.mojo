from std.gpu import block_dim,block_idx,thread_idx,grid_dim,barrier
from layout import TileTensor,LayoutTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout,col_major
from layout.tile_tensor import stack_allocation
from std.gpu.memory import AddressSpace

from src.lbm import LBM_Grid,LBM_Config,LatticeModel
from src.utils import Vector,ContextTileTensor
from src.lbm.kernels.utils.index import get_adjacent_idx,is_index_valid
from src.lbm.kernels.utils.load_and_store import load_f,store_f
from src.lbm.kernels.utils.moment import get_density,get_velocity
from src.lbm.kernels.utils.finite_difference import get_velocity_gradient
from src.lbm.kernels.utils.shared_tile import get_global_index_for_shared_memory


def calculate_rho_and_velocity[ 
    float_dtype:DType,D:Int,Q:Int,
    lattice_model:LatticeModel[D,Q,float_dtype,DType.int32],
    nx:Int,ny:Int,nz:Int,tile_size:Int,
    //,
    grid: LBM_Grid[lattice_model,nx,ny,nz,tile_size], 
    Flayout:Layout[...],
    BClayout:Layout[...],
    Flaglayout:Layout[...],
    RhoLayout:Layout[...],
    VelocityLayout:Layout[...],
    config:LBM_Config = LBM_Config(),
    *,
    f_dtype:DType = float_dtype if config.f_dtype is None else config.f_dtype.value()
    ]
    (
        density:TileTensor[float_dtype,type_of(RhoLayout),MutAnyOrigin],
        velocity:TileTensor[float_dtype,type_of(VelocityLayout),MutAnyOrigin],

        f:TileTensor[f_dtype,type_of(Flayout),ImmutAnyOrigin],
        bc:TileTensor[float_dtype,type_of(BClayout),ImmutAnyOrigin],
        flags:TileTensor[DType.uint8,type_of(Flaglayout),ImmutAnyOrigin],
    )
    where velocity.rank == 4 and density.rank == 3 and Flayout.rank == 4
        and bc.rank == 4:
                                
    # Run on GPU
    '''
    Compute the Velocity and Density from f dist. Converts to layout tensor to allow layout independent assignment. This should be run on the gpu
    '''
    comptime grid_shape:InlineArray[Int,3] = [nx,ny,nz]
    comptime assert velocity.rank == velocity.flat_rank and density.rank == density.flat_rank, 'Velocity and Density Tensors should be non-nested and row-major or col-major'
    
    comptime D_is_last_dim = (VelocityLayout.static_shape[0] == nx and VelocityLayout.static_shape[1] == ny and VelocityLayout.static_shape[2] == nz and VelocityLayout.static_shape[3] == (D+1)) 

    var x = block_dim.x * block_idx.x + thread_idx.x
    var y = block_dim.y * block_idx.y + thread_idx.y
    var z = block_dim.z * block_idx.z + thread_idx.z
    var index:InlineArray[Int,3] = [x,y,z]
    var f_vec = Vector[float_dtype,Q](fill = 0)
    coord_index = coord[DType.int32]((index[0],index[1],index[2]))

    var flag = flags.load(coord_index)[0]
    
    var u = Vector[float_dtype,D](fill=0)
    if index[0] < grid_shape[0] and index[1] < grid_shape[1] and index[2] < grid_shape[2]: # Basic Guard
        if flag != SOLID_NODE:
            comptime for q in range(Q):
                f_vec[q] = load_f[float_dtype,config.use_float16c](f,index,q)

            rho = get_density[config.DDF_shift](f_vec)
            u = get_velocity[lattice_model.float_directions](f_vec,rho)
        else:# Get the BC For that node
            comptime for ii in range(D):
                u[ii] = bc.load(coord[DType.int32]((index[0],index[1],index[2],ii)))[0]
            rho = bc.load(coord[DType.int32]((index[0],index[1],index[2],D)))[0]
            
        density.store(coord_index,rho)
        comptime for d in range(D):
            comptime if D_is_last_dim:
                velocity.store(coord[DType.int32]((index[0],index[1],index[2],d)), value = u[d])
            else:
                velocity.store(coord[DType.int32]((d,index[0],index[1],index[2])), value = u[d])

def sync_load_rank4_tensor_to_shared_with_halo[
    float_dtype:DType,
    sharedLayoutType:TensorLayout,
    srcLayoutType:TensorLayout,
    srcOrigin:Origin,
    //,
    tile_size:Int,
    D:Int,
    ]
    (
    shared_tile:TileTensor[float_dtype,sharedLayoutType, MutExternalOrigin, address_space=AddressSpace.SHARED],
    src_tensor:TileTensor[float_dtype,srcLayoutType,srcOrigin],
    tid:Int,
    mut shared_local_index:InlineArray[Int,3],
    block_index:InlineArray[Int,3],
    tiler_shape:InlineArray[Int,3],
    ):
    comptime assert shared_tile.rank == 4 and shared_tile.flat_rank == 4
    comptime assert src_tensor.rank == 4 and (src_tensor.flat_rank == 4 or src_tensor.flat_rank == 8)

    comptime is_nested = src_tensor.rank != src_tensor.flat_rank
    comptime src_N = src_tensor.static_shape[6]*src_tensor.static_shape[7] if is_nested else src_tensor.static_shape[3]
    comptime N = shared_tile.static_shape[3] # Guranteed to be non-nested layout from assertion above
    comptime assert N == src_N, 'The last dimension of the shared tile and soruce tensor must match'
    
    comptime SHARED_x = tile_size + 2
    comptime SHARED_y = tile_size + 2 if D >= 2 else 1 
    comptime SHARED_z = tile_size + 2 if D == 3 else 1
    comptime NUM_THREADS = tile_size**D
    comptime NUM_SHARED_XYZ_POINTS = SHARED_x*SHARED_y*SHARED_z

    for i in range(tid,NUM_SHARED_XYZ_POINTS,NUM_THREADS): # loop only iterates 1-2 per thread
        # Indexes for shared array
        shared_local_index[0] = i % SHARED_x
        shared_local_index[1] = (i % (SHARED_x*SHARED_y))//SHARED_x 
        shared_local_index[2] = i // (SHARED_x * SHARED_y)
        # Index for the current i threadindex
        shared_global_index = get_global_index_for_shared_memory[D,tile_size](shared_local_index,block_index,tiler_shape)
        comptime for n in range(N):
            val = src_tensor.load(coord[DType.int32]((shared_global_index[0],shared_global_index[1],shared_global_index[2],n)))[0]
            shared_tile[shared_local_index[0],shared_local_index[1],shared_local_index[2],n] = val
    
    



def calculate_Q_criterion[
    float_dtype:DType,D:Int,Q:Int,
    lattice_model:LatticeModel[D,Q,float_dtype,DType.int32],
    nx:Int,ny:Int,nz:Int,tile_size:Int,
    //,
    grid: LBM_Grid[lattice_model,nx,ny,nz,tile_size], 
    Flayout:Layout[...] ,
    FlagLayout:Layout[...] ,
    VelocityLayout:Layout[...],
    QLayout:Layout[...],
    config:LBM_Config = LBM_Config(),
    *,
    f_dtype:DType = float_dtype if config.f_dtype is None else config.f_dtype.value()
    ]
    (
        Q_tensor:TileTensor[float_dtype,type_of(QLayout),MutAnyOrigin],

        f:TileTensor[f_dtype,type_of(Flayout),ImmutAnyOrigin],
        flags:TileTensor[DType.uint8,type_of(FlagLayout),ImmutAnyOrigin],
        velocity:TileTensor[float_dtype,type_of(VelocityLayout),ImmutAnyOrigin],
    )
    where velocity.rank == 4 and f.rank == 4 and Q_tensor.rank == 3 and flags.rank == 3:

    comptime SHARED_x = tile_size + 2
    comptime SHARED_y = tile_size + 2 if D >= 2 else 1 
    comptime SHARED_z = tile_size + 2 if D == 3 else 1
    comptime grid_shape:InlineArray[Int,3] = [nx,ny,nz]
    comptime NUM_THREADS = tile_size**D
    comptime NUM_SHARED_XYZ_POINTS = SHARED_x*SHARED_y*SHARED_z
    comptime load_v = load_f[float_dtype]
     
    # comptime ijk_layout = col_major[]
    var shared_u = stack_allocation[float_dtype,AddressSpace.SHARED](col_major[SHARED_x,SHARED_y,SHARED_z,D]())

    # 0 to 511
    var tid = thread_idx.x + thread_idx.y * block_dim.x + thread_idx.z * block_dim.x * block_dim.y
    # var shared_global_index = InlineArray[Int,3](uninitialized = True)
    var shared_local_index = InlineArray[Int,3](uninitialized = True)
    var block_index:InlineArray[Int,3] = [block_idx.x,block_idx.y,block_idx.z]
    var tiler_shape:InlineArray[Int,3] = [grid_dim.x,grid_dim.y,grid_dim.z]

    # To Pull
    
    for i in range(tid,NUM_SHARED_XYZ_POINTS,NUM_THREADS): # loop only iterates
        # Indexes for shared array
        shared_local_index[0] = i % SHARED_x
        shared_local_index[1] = (i % (SHARED_x*SHARED_y))//SHARED_x 
        shared_local_index[2] = i // (SHARED_x * SHARED_y)
        # Index for the current i threadindex
        shared_global_index = get_global_index_for_shared_memory[D,tile_size](shared_local_index,block_index,tiler_shape)
        comptime for d in range(D):
            vel_i = velocity.load(coord[DType.int32]((d,shared_global_index[0],shared_global_index[1],shared_global_index[2])))[0]
            shared_u[shared_local_index[0],shared_local_index[1],shared_local_index[2],d] = vel_i
    barrier()
    
    var x = block_dim.x * block_idx.x + thread_idx.x
    var y = block_dim.y * block_idx.y + thread_idx.y
    var z = block_dim.z * block_idx.z + thread_idx.z

    comptime shift_x = 1
    comptime shift_y = 1 if D >= 2 else 0
    comptime shift_z = 1 if D == 3 else 0
    index:InlineArray[Int,3] = [x,y,z]
    coord_index = coord[DType.int32]((index[0],index[1],index[2]))
    flag = flags.load(coord_index)
    if index[0] < grid_shape[0] and index[1] < grid_shape[1] and index[2] < grid_shape[2] and flag != SOLID_NODE: # Basic Guard
        
        shared_local_index[0] = thread_idx.x + shift_x
        shared_local_index[1] = thread_idx.y + shift_y
        shared_local_index[2] = thread_idx.z + shift_z
        # Calculate Voricity lets do 2D for now
        comptime if D == 2: # We are mixing lattice unit so actually dx is = 1
            dv_dx = get_velocity_gradient[1](shared_u,flags,shared_local_index,index,grid_shape,velocity_direction = 1,axis = 0)
            du_dy = get_velocity_gradient[1](shared_u,flags,shared_local_index,index,grid_shape,velocity_direction = 0,axis = 1)
            Q_tensor.store(coord_index,value= dv_dx- du_dy)
        else:
            comptime assert False, 'D = 3 not implemented yet'
        






def rowMajor1D[int_dtype:DType]() -> type_of( row_major(coord[int_dtype]((1,))) ):
    return row_major(coord[int_dtype]((1,)))

def rowMajor2D[int_dtype:DType]() -> type_of(row_major(coord[int_dtype]((1,2))) ):
    return row_major(coord[int_dtype]((1,2)))


def calculate_drag_around_object[
    float_dtype:DType,
    int_dtype:DType,
    D:Int,Q:Int,
    lattice_model:LatticeModel[D,Q,float_dtype,int_dtype],
    nx:Int,ny:Int,nz:Int,tile_size:Int,
    //,
    grid: LBM_Grid[lattice_model,nx,ny,nz,tile_size], 
    FLayout:Layout[...] ,
    FlagLayout:Layout[...],
    config:LBM_Config = LBM_Config(),
    *,
    f_dtype:DType = float_dtype if config.f_dtype is None else config.f_dtype.value()
    ](
        f:TileTensor[f_dtype,type_of(FLayout),ImmutAnyOrigin],
        flags:TileTensor[DType.uint8,type_of(FlagLayout),ImmutAnyOrigin],
        fluid_boundary:TileTensor[int_dtype,type_of(rowMajor1D[int_dtype]()),MutAnyOrigin],
        force_tensor:TileTensor[float_dtype,type_of(rowMajor2D[int_dtype]()),MutAnyOrigin],
    ):

    # Should be a 1D based kernel loop
    tid = block_dim.x * block_idx.x + thread_idx.x
    if tid < fluid_boundary.layout.size():
        crd = FlagLayout.idx2crd[out_dtype = int_dtype](Int(fluid_boundary[tid])).flatten()
        grid_index = InlineArray[Int,3](uninitialized = True)
        
        comptime if FlagLayout.rank*2 == FlagLayout.flat_rank:
            comptime for i in range(3):
                loc_x = Int(crd[2*i].value()) # local
                til_x = Int(crd[(2*i)+1].value())
                grid_index[i] = tile_size*til_x + loc_x
        else:
            comptime assert FlagLayout.rank == FlagLayout.flat_rank
            comptime for i in range(3):
                grid_index[i] = Int(crd[i].value())
        
        comptime grid_shape:InlineArray[Int,3] = [nx,ny,nz]
        comptime opposite_index = lattice_model.opposite_indices

        if grid_index[0] < grid_shape[0] and grid_index[1] < grid_shape[1] and grid_index[2] < grid_shape[2]:
            var push_flags = InlineArray[UInt8,Q](uninitialized = True)
            var push_indices = InlineArray[InlineArray[Int,3],Q](uninitialized = True)
            
            # Gather Neighboring Flags in PUSH direction not Pull
            comptime for q in range(Q):
                comptime direction = lattice_model.directions[q]
                push_indices[q] = get_adjacent_idx[D,1](grid_index,grid_shape,direction) # push Scheme as
                push_flags[q] = flags.load(coord[DType.uint32]((push_indices[q][0],push_indices[q][1],push_indices[q][2])))[0]
                
            # Compute Forces
            var force_vec = Vector[float_dtype,D](fill = 0.)
            comptime for q in range(Q):
                comptime direction = lattice_model.directions[q]
                comptime float_dir = lattice_model.float_directions[q]
                if push_flags[q] == SOLID_NODE:
                    var f_local = load_f[float_dtype,config.use_float16c](f,grid_index,q)
                    comptime if config.DDF_shift:
                        comptime weight = lattice_model.weights[q]
                        f_link = f_local + weight
                    else:
                        f_link = f_local
                    force_vec += (2*f_link)*lattice_model.float_directions[q] # Only stationary wall for now

            # push to global
            comptime for i in range(D): # Overwrite
                force_tensor[tid,i] = force_vec[i]


    

