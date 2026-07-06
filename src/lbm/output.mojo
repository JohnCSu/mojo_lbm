from std.gpu import block_dim,block_idx,thread_idx
from layout import TileTensor,LayoutTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from .LBM import LBM_Grid
from .config import LBM_Config
from .lattice_models import LatticeModel
from src.utils import Vector,ContextTileTensor
from src.lbm.utils.index import get_adjacent_idx
from src.lbm.utils.load_and_store import load_f,store_f

def calculate_rho_and_velocity[ float_dtype:DType,D:Int,Q:Int,
                                lattice_model:LatticeModel[D,Q,float_dtype,DType.int32],
                                nx:Int,ny:Int,nz:Int,tile_size:Int,
                                //,
                                grid: LBM_Grid[lattice_model,nx,ny,nz,tile_size], 
                                Flayout:Layout[...] ,
                                RhoLayout:Layout[...],
                                VelocityLayout:Layout[...] ,
                                config:LBM_Config = LBM_Config(),
                                *,
                                f_dtype:DType = float_dtype if config.f_dtype is None else config.f_dtype.value()
                                ]
                                (
                                    f:TileTensor[f_dtype,type_of(Flayout),MutAnyOrigin],
                                    density:TileTensor[float_dtype,type_of(RhoLayout),MutAnyOrigin],
                                    velocity:TileTensor[float_dtype,type_of(VelocityLayout),MutAnyOrigin],
                                
                                )
                                where VelocityLayout.rank == 4 and RhoLayout.rank == 3 and Flayout.rank == 4:
                                
    # Run on GPU
    '''
    Compute the Velocity and Density from f dist. Converts to layout tensor to allow layout independent assignment. This should be run on the gpu
    '''
    
    comptime grid_shape = Vector[DType.int32,3](Int32(nx),Int32(ny),Int32(nz))
    comptime f_as_lt = LayoutTensor[f_dtype,Flayout.to_layout(),MutAnyOrigin]
    comptime vel_as_lt = LayoutTensor[float_dtype,VelocityLayout.to_layout(),MutAnyOrigin]
    comptime rho_as_lt = LayoutTensor[float_dtype,RhoLayout.to_layout(),MutAnyOrigin]

    comptime f_is_first_index = (f_as_lt.shape[0]() == Q and f_as_lt.shape[1]() == nx and f_as_lt.shape[2]() == ny and f_as_lt.shape[3]() == nz)
    f_lt = f_as_lt(f.ptr)
    velocity_lt = vel_as_lt(velocity.ptr)
    density_lt = rho_as_lt(density.ptr)

    x = block_dim.x * block_idx.x + thread_idx.x
    y = block_dim.y * block_idx.y + thread_idx.y
    z = block_dim.z * block_idx.z + thread_idx.z
    index = Vector[DType.int32,3](Int32(x),Int32(y),Int32(z))
    f_vec = Vector[float_dtype,Q](fill = 0)

    
    if index[0] < grid_shape[0] and index[1] < grid_shape[1] and index[2] < grid_shape[2]: # Basic Guard
        comptime if config.use_float16c:
            comptime assert f_dtype == DType.uint16
            comptime for q in range(Q):
                f_vec[q] = Scalar[float_dtype](config.fp16c_to_fp32(f_lt[x,y,z,q][0]))
        else:
            comptime assert f_dtype == float_dtype
            comptime for q in range(Q):
                f_vec[q] = Scalar[float_dtype](f_lt[x,y,z,q][0])

        var u = Vector[float_dtype,D](fill = 0.)
        var rho = Scalar[float_dtype](0)
    
        for q in range(Q):
            rho += f_vec[q]
            u += f_vec[q]*lattice_model.float_directions[q]
        
        comptime if config.DDF_shift:
            rho += 1
        u /= rho

        density_lt[x,y,z] = rho
        comptime for i in range(D):
            velocity_lt[i,x,y,z] = u[i]




comptime Runtime_rowMajor_1D_Type = type_of(row_major(coord[DType.int32]((3,))))
comptime Runtime_rowMajor_2D_Type = type_of(row_major(coord[DType.int32]((3,3))))

def calculate_drag_around_object[
    float_dtype:DType,D:Int,Q:Int,
    lattice_model:LatticeModel[D,Q,float_dtype,DType.int32],
    nx:Int,ny:Int,nz:Int,tile_size:Int,
    //,
    grid: LBM_Grid[lattice_model,nx,ny,nz,tile_size], 
    FLayout:Layout[...] ,
    FlagLayout:Layout[...],
    VelocityLayout:Layout[...],
    config:LBM_Config = LBM_Config(),
    *,
    f_dtype:DType = float_dtype if config.f_dtype is None else config.f_dtype.value()
    ](
        f:TileTensor[f_dtype,type_of(FLayout),ImmutAnyOrigin],
        flags:TileTensor[DType.uint8,type_of(FlagLayout),ImmutAnyOrigin],
        fluid_boundary:TileTensor[DType.int32,Runtime_rowMajor_2D_Type,MutAnyOrigin],
        force_tensor:TileTensor[float_dtype,Runtime_rowMajor_2D_Type,MutAnyOrigin],
    ):

    # Should be a 1D based kernel loop
    tid = block_dim.x * block_idx.x + thread_idx.x
    grid_index:InlineArray[Int,3] = [Int(fluid_boundary[tid,0]),Int(fluid_boundary[tid,1]),Int(fluid_boundary[tid,2])]
    
    comptime grid_shape:InlineArray[Int,3] = [nx,ny,nz]
    comptime opposite_index = lattice_model.opposite_indices

    var pull_flags = InlineArray[UInt8,Q](uninitialized = True)
    var pull_indices = InlineArray[InlineArray[Int,3],Q](uninitialized = True)
    # Get flags of surrounding fluid
    if grid_index[0] < grid_shape[0] and grid_index[1] < grid_shape[1] and grid_index[2] < grid_shape[2]:
        var f_pulled = Vector[float_dtype,Q](fill = 0.)
        

        # Pull F from neighbors
        comptime for q in range(Q):
            comptime direction = lattice_model.directions[q]
            pull_indices[q] = get_adjacent_idx[D,-1](grid_index,grid_shape,direction) # Pulling Scheme
            pull_flags[q] = flags.load(coord[DType.uint32]((pull_indices[q][0],pull_indices[q][1],pull_indices[q][2])))[0]
            f_pulled[q] =  load_f[float_dtype,config.use_float16c](f,pull_indices[q],q)
        

        # Compute Forces
        var force_vec = Vector[float_dtype,D](fill = 0.)
        comptime zero_vec = Vector[float_dtype,D](fill = 0)
        comptime for q in range(Q):
            comptime float_dir = lattice_model.float_directions[q]
            comptime opp_index = Int(opposite_index[q])
            force_vec += (f_pulled[q] + f_pulled[opp_index])*float_dir if pull_flags[q] == SOLID_NODE else zero_vec
        
        # Push to global
        comptime for i in range(D):
            force_tensor[tid,i] = force_vec[i]



    

