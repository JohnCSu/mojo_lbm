from std.gpu import block_dim,block_idx,thread_idx,grid_dim,barrier
from layout import TileTensor,LayoutTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout,col_major
from layout.tile_tensor import stack_allocation
from std.gpu.memory import AddressSpace

from src.lbm import LBM_Grid,LBM_Config,Lattice
from src.utils import Vector,ContextTileTensor

from src.lbm.kernels.utils.index import get_adjacent_idx,is_index_valid
from src.lbm.kernels.ops import wall_bc
from src.lbm.kernels.utils.load_and_store import load_f,store_f,esoteric_pull_load_f_vec,double_buffer_pull_load_f
from src.lbm.kernels.utils.moment import get_density,get_velocity,get_strain_rate_tensor,get_strain_rate_tensor_norm_squared,get_non_eq_second_order_moment
from src.lbm.kernels.utils.finite_difference import get_velocity_gradient
from src.lbm.kernels.utils.shared_tile import get_global_index_for_shared_memory,sync_load_rank4_tensor_to_shared_with_halo
from src.lbm.kernels.utils.equilibrium import get_f_eq_vec,get_f_noneq_vec


def calculate_rho_and_velocity[
    Flayout:Layout[...],
    BClayout:Layout[...],
    Flaglayout:Layout[...],
    RhoLayout:Layout[...],
    VelocityLayout:Layout[...],
    grid: LBM_Grid,
    config:LBM_Config,
    *,
    current_step_is_odd:Optional[Bool] = None,
    ]
    (
        density:TileTensor[grid.float_dtype,type_of(RhoLayout),MutAnyOrigin],
        velocity:TileTensor[grid.float_dtype,type_of(VelocityLayout),MutAnyOrigin],
        
        f:TileTensor[config.set_f_dtype(grid.float_dtype),type_of(Flayout),ImmutAnyOrigin],
        bc:TileTensor[grid.float_dtype,type_of(BClayout),ImmutAnyOrigin],
        flags:TileTensor[DType.uint8,type_of(Flaglayout),ImmutAnyOrigin],
    )
    where velocity.rank == 4 and density.rank == 3 and Flayout.rank == 4
        and bc.rank == 4:

    # Run on GPU
    """Computes the density and velocity fields from the distribution function.

    For each non-solid node, loads `f`, computes the density and velocity
    from the moments, and stores them into the `density` and `velocity`
    tensors. For solid nodes, copies the boundary-condition values into the
    output tensors instead.

    Parameters:
        Flayout: The compile-time `Layout` of the distribution function `f`.
        BClayout: The compile-time `Layout` of the boundary-condition tensor.
        Flaglayout: The compile-time `Layout` of the `uint8` flag tensor.
        RhoLayout: The compile-time `Layout` of the density output tensor.
        VelocityLayout: The compile-time `Layout` of the velocity output
            tensor.
        grid: The compile-time `LBM_Grid` describing the domain.
        config: The `LBM_Config` used to select storage options.
        f_dtype: The storage `DType` for `f` (defaults to the config's
            `f_dtype` or `float_dtype`).

    Args:
        density: The output density tile tensor (rank 3).
        velocity: The output velocity tile tensor (rank 4).
        f: The input distribution function tile tensor (rank 4).
        bc: The boundary-condition tile tensor (rank 4).
        flags: The `uint8` tile tensor labeling each node (rank 3).
    """
    comptime D = grid.D
    comptime Q = grid.Q
    comptime float_dtype = grid.float_dtype
    comptime lattice = grid.lattice
    comptime directions = lattice.directions
    comptime opposite_indices = lattice.opposite_indices
    comptime weights = lattice.weights
    comptime grid_shape:InlineArray[Int,3] = grid.shape
    comptime assert velocity.rank == velocity.flat_rank and density.rank == density.flat_rank, 'Velocity and Density Tensors should be non-nested and row-major or col-major'

    comptime D_is_last_dim = (VelocityLayout.static_shape[0] == grid.nx and VelocityLayout.static_shape[1] == grid.ny and VelocityLayout.static_shape[2] == grid.nz and VelocityLayout.static_shape[3] == (D))

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
            var pull_flags = InlineArray[UInt8,Q](uninitialized=True)
            pull_flags[0] = flag
            comptime if config.lbm_method == ESOTERIC_PULL:
                comptime assert current_step_is_odd, 'If lbm_method is set to esoteric_pull, is_even_time_step must be defined'
                f_vec = esoteric_pull_load_f_vec[float_dtype,lattice.directions,current_step_is_odd.value(),config.use_float16c](f,index,grid_shape)
            
            elif config.lbm_method == DOUBLE_BUFFER:
                f_vec = double_buffer_pull_load_f[float_dtype,directions,config.use_float16c](f,index,grid_shape)
            
            else:
                comptime assert False, 'lbm_method not valid'
                            
            comptime include_bounceback = False if config.lbm_method == ESOTERIC_PULL else True
            wall_bc[include_bounceback,directions,opposite_indices,weights,config.use_float16c](f_vec,pull_flags,f,flags,bc,index,grid_shape)

            rho = get_density[config.DDF_shift](f_vec)
            u = get_velocity[lattice.directions](f_vec,rho)
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




# def calculate_esoteric_rho_and_velocity[
#     is_even_time_step:Bool,
#     Flayout:Layout[...],
#     BClayout:Layout[...],
#     Flaglayout:Layout[...],
#     RhoLayout:Layout[...],
#     VelocityLayout:Layout[...],
#     grid: LBM_Grid,
#     config:LBM_Config = LBM_Config(),
#     *,
#     f_dtype:DType = grid.float_dtype if config.f_dtype is None else config.f_dtype.value()
#     ]
#     (
#         density:TileTensor[grid.float_dtype,type_of(RhoLayout),MutAnyOrigin],
#         velocity:TileTensor[grid.float_dtype,type_of(VelocityLayout),MutAnyOrigin],

#         f:TileTensor[f_dtype,type_of(Flayout),ImmutAnyOrigin],
#         bc:TileTensor[grid.float_dtype,type_of(BClayout),ImmutAnyOrigin],
#         flags:TileTensor[DType.uint8,type_of(Flaglayout),ImmutAnyOrigin],
#     )
#     where velocity.rank == 4 and density.rank == 3 and Flayout.rank == 4
#         and bc.rank == 4:

#     # Run on GPU
#     """Computes the density and velocity fields from the distribution function.

#     For each non-solid node, loads `f`, computes the density and velocity
#     from the moments, and stores them into the `density` and `velocity`
#     tensors. For solid nodes, copies the boundary-condition values into the
#     output tensors instead.

#     Parameters:
#         Flayout: The compile-time `Layout` of the distribution function `f`.
#         BClayout: The compile-time `Layout` of the boundary-condition tensor.
#         Flaglayout: The compile-time `Layout` of the `uint8` flag tensor.
#         RhoLayout: The compile-time `Layout` of the density output tensor.
#         VelocityLayout: The compile-time `Layout` of the velocity output
#             tensor.
#         grid: The compile-time `LBM_Grid` describing the domain.
#         config: The `LBM_Config` used to select storage options.
#         f_dtype: The storage `DType` for `f` (defaults to the config's
#             `f_dtype` or `float_dtype`).

#     Args:
#         density: The output density tile tensor (rank 3).
#         velocity: The output velocity tile tensor (rank 4).
#         f: The input distribution function tile tensor (rank 4).
#         bc: The boundary-condition tile tensor (rank 4).
#         flags: The `uint8` tile tensor labeling each node (rank 3).
#     """
#     comptime D = grid.D
#     comptime Q = grid.Q
#     comptime float_dtype = grid.float_dtype
#     comptime lattice = grid.lattice
#     comptime grid_shape:InlineArray[Int,3] = grid.shape
#     comptime assert velocity.rank == velocity.flat_rank and density.rank == density.flat_rank, 'Velocity and Density Tensors should be non-nested and row-major or col-major'

#     comptime D_is_last_dim = (VelocityLayout.static_shape[0] == grid.nx and VelocityLayout.static_shape[1] == grid.ny and VelocityLayout.static_shape[2] == grid.nz and VelocityLayout.static_shape[3] == (D))

#     var x = block_dim.x * block_idx.x + thread_idx.x
#     var y = block_dim.y * block_idx.y + thread_idx.y
#     var z = block_dim.z * block_idx.z + thread_idx.z
#     var index:InlineArray[Int,3] = [x,y,z]
#     var f_vec = Vector[float_dtype,Q](fill = 0)
#     coord_index = coord[DType.int32]((index[0],index[1],index[2]))

#     var flag = flags.load(coord_index)[0]

#     var u = Vector[float_dtype,D](fill=0)
#     if index[0] < grid_shape[0] and index[1] < grid_shape[1] and index[2] < grid_shape[2]: # Basic Guard
#         if flag != SOLID_NODE:
#             f_vec = esoteric_pull_load_f_vec[float_dtype,lattice.directions,is_even_time_step,config.use_float16c](f,index,grid_shape)
#             # comptime for q in range(Q):
#             #     f_vec[q] = load_f[float_dtype,config.use_float16c](f,index,q)

#             rho = get_density[config.DDF_shift](f_vec)
#             u = get_velocity[lattice.directions](f_vec,rho)


#         else:# Get the BC For that node
#             comptime for ii in range(D):
#                 u[ii] = bc.load(coord[DType.int32]((index[0],index[1],index[2],ii)))[0]
#             rho = bc.load(coord[DType.int32]((index[0],index[1],index[2],D)))[0]

#         density.store(coord_index,rho)
#         comptime for d in range(D):
#             comptime if D_is_last_dim:
#                 velocity.store(coord[DType.int32]((index[0],index[1],index[2],d)), value = u[d])
#             else:
#                 velocity.store(coord[DType.int32]((d,index[0],index[1],index[2])), value = u[d])

