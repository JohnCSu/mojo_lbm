"""Computes density and velocity output fields from the distribution
function.

Reads the distribution function, applies boundary conditions, and
computes the macroscopic density and velocity moments for each lattice
node.
"""
from std.gpu import block_dim,block_idx,thread_idx,grid_dim,barrier
from layout import TileTensor,LayoutTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout,col_major
from layout.tile_tensor import stack_allocation
from std.gpu.memory import AddressSpace

from src.lbm import LBM_Grid,LBM_Config,Lattice,GridLike,LBM_method
from src.utils import Vector,ContextTileTensor

from src.lbm.kernels.utils.index import get_adjacent_idx,is_index_valid
from src.lbm.kernels.ops import wall_bc,set_f_vector_and_flags
from src.lbm.kernels.utils.load_and_store import load_f,store_f,esoteric_pull_load_f_vec,double_buffer_pull_load_f,set_adjacent_flags
from src.lbm.kernels.utils.moment import get_density,get_velocity,get_strain_rate_tensor,get_strain_rate_tensor_norm_squared,get_non_eq_second_order_moment
from src.lbm.kernels.utils.finite_difference import get_velocity_gradient
from src.lbm.kernels.utils.shared_tile import get_global_index_for_shared_memory,sync_load_rank4_tensor_to_shared_with_halo
from src.lbm.kernels.utils.equilibrium import get_f_eq_vec,get_f_noneq_vec

def calculate_rho_and_velocity[
    FlayoutType:TensorLayout,
    BClayoutType:TensorLayout,
    FlaglayoutType:TensorLayout,
    RhoLayoutType:TensorLayout,
    VelocityLayoutType:TensorLayout,
    grid: Some[GridLike],
    config:LBM_Config,
    *,
    after_odd_step:Optional[Bool] = None,
    ]
    (
        density:TileTensor[grid.float_dtype,RhoLayoutType,MutAnyOrigin],
        velocity:TileTensor[grid.float_dtype,VelocityLayoutType,MutAnyOrigin],
        
        f:TileTensor[config.set_f_dtype(grid.float_dtype),FlayoutType,ImmutAnyOrigin],
        bc:TileTensor[grid.float_dtype,BClayoutType,ImmutAnyOrigin],
        flags:TileTensor[DType.uint8,FlaglayoutType,ImmutAnyOrigin],
    )
    where VelocityLayoutType.rank == 4 and RhoLayoutType.rank == 3 and FlayoutType.rank == 4
        and BClayoutType.rank == 4:

    # Run on GPU
    """Computes the density and velocity fields from the distribution function.

    For each non-solid node, loads `f`, computes the density and velocity
    from the moments, and stores them into the `density` and `velocity`
    tensors. For solid nodes, copies the boundary-condition values into the
    output tensors instead.

    Parameters:
        FlayoutType: The compile-time `Layout` of the distribution function
            `f`.
        BClayoutType: The compile-time `Layout` of the boundary-condition
            tensor.
        FlaglayoutType: The compile-time `Layout` of the `uint8` flag tensor.
        RhoLayoutType: The compile-time `Layout` of the density output
            tensor.
        VelocityLayoutType: The compile-time `Layout` of the velocity output
            tensor.
        grid: The compile-time `LBM_Grid` describing the domain.
        config: The `LBM_Config` used to select storage options.
        after_odd_step: When `True`, load from the positive half in the
            esoteric-pull scheme (defaults to `None`). Must not be None if used for esoteric step

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

    comptime D_is_last_dim = (VelocityLayoutType.static_shape[0] == grid.nx and
                             VelocityLayoutType.static_shape[1] == grid.ny and 
                             VelocityLayoutType.static_shape[2] == grid.nz and 
                             VelocityLayoutType.static_shape[3] == (D)
                             )

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
            comptime is_even_time_step = after_odd_step # after_odd_step implies is_even_time_step
            set_f_vector_and_flags[grid,config,is_even_time_step = is_even_time_step](f_vec,pull_flags,f,flags,index,grid_shape)
            
            # comptime include_bounceback = False if config.lbm_method == ESOTERIC_PULL else True
            comptime if config.include_moving_boundary:
                wall_bc[directions,opposite_indices,weights,config.use_float16c](f_vec,pull_flags,bc,index,grid_shape)

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
