"""Provides post-processing GPU kernels for the LBM solver.

Contains the density/velocity extraction kernel, the Q-criterion kernel, and
the drag-force kernel around an immersed object. Each kernel is parameterized
by the compile-time `LBM_Grid` and `LBM_Config` and runs on the GPU using
natural `(x, y, z, q)` indexing.
"""
from std.gpu import block_dim,block_idx,thread_idx,grid_dim,barrier
from layout import TileTensor,LayoutTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout,col_major
from layout.tile_tensor import stack_allocation
from std.gpu.memory import AddressSpace

from src.lbm import LBM_Grid,LBM_Config,Lattice
from src.utils import Vector,ContextTileTensor
from src.lbm.kernels.utils.index import get_adjacent_idx,is_index_valid
from src.lbm.kernels.utils.load_and_store import load_f,store_f,esoteric_pull_load_f_vec,esoteric_pull_store_f_vec
from src.lbm.kernels.utils.moment import get_density,get_velocity,get_strain_rate_tensor,get_strain_rate_tensor_norm_squared,get_non_eq_second_order_moment
from src.lbm.kernels.utils.finite_difference import get_velocity_gradient
from src.lbm.kernels.utils.shared_tile import get_global_index_for_shared_memory,sync_load_rank4_tensor_to_shared_with_halo
from src.lbm.kernels.utils.equilibrium import get_f_eq_vec,get_f_noneq_vec




def calculate_Q_criterion[
    Flayout:Layout[...] ,
    FlagLayout:Layout[...] ,
    VelocityLayout:Layout[...],
    QLayout:Layout[...],
    grid: LBM_Grid,
    config:LBM_Config,
    *,
    f_dtype:DType = grid.float_dtype if config.f_dtype is None else config.f_dtype.value()
    ]
    (
        Q_tensor:TileTensor[grid.float_dtype,type_of(QLayout),MutAnyOrigin],

        f:TileTensor[f_dtype,type_of(Flayout),ImmutAnyOrigin],
        flags:TileTensor[DType.uint8,type_of(FlagLayout),ImmutAnyOrigin],
        velocity:TileTensor[grid.float_dtype,type_of(VelocityLayout),ImmutAnyOrigin],
        tau:Scalar[grid.float_dtype],
    )
    where velocity.rank == 4 and f.rank == 4 and Q_tensor.rank == 3 and flags.rank == 3:

    """Computes the Q-criterion field for the current state.

    Loads the velocity field into shared memory with a halo, computes the
    vorticity magnitude squared via finite differences, computes the
    strain-rate tensor from the non-equilibrium populations, and stores
    $$Q = 0.25 \\|\\omega\\|^2 - 0.5 \\|S\\|_F^2$$.

    Parameters:
        Flayout: The compile-time `Layout` of the distribution function `f`.
        FlagLayout: The compile-time `Layout` of the `uint8` flag tensor.
        VelocityLayout: The compile-time `Layout` of the velocity input
            tensor, indexed as `[x, y, z, D]`.
        QLayout: The compile-time `Layout` of the Q-criterion output tensor.
        grid: The compile-time `LBM_Grid` describing the domain.
        config: The `LBM_Config` used to select storage options.
        f_dtype: The storage `DType` for `f` (defaults to the config's
            `f_dtype` or `float_dtype`).

    Args:
        Q_tensor: The output Q-criterion tile tensor (rank 3).
        f: The input distribution function tile tensor (rank 4).
        flags: The `uint8` tile tensor labeling each node (rank 3).
        velocity: The input velocity tile tensor (rank 4), indexed as
            `[x, y, z, D]`.
        tau: The relaxation time used for the strain-rate reconstruction.
    """
    comptime D = grid.D
    comptime Q = grid.Q
    comptime float_dtype = grid.float_dtype
    comptime lattice = grid.lattice
    comptime tile_shape = grid.tile_shape
    comptime SHARED_x = tile_shape[0] + 2
    comptime SHARED_y = tile_shape[1] + 2 if D >= 2 else 1
    comptime SHARED_z = tile_shape[2] + 2 if D == 3 else 1
    comptime grid_shape:InlineArray[Int,3] = grid.shape
    comptime D_is_last_dim = (VelocityLayout.static_shape[0] == grid.nx and VelocityLayout.static_shape[1] == grid.ny and VelocityLayout.static_shape[2] == grid.nz and VelocityLayout.static_shape[3] == (D))

    comptime assert D_is_last_dim, 'Velocity Tensor must be indexed as [x,y,z,D]'
    comptime assert D == 2 or D == 3,'Calculating Q criterion can only be 2D or 3D'

    # 0 to 511
    var tid = thread_idx.x + thread_idx.y * block_dim.x + thread_idx.z * block_dim.x * block_dim.y
    var block_index:InlineArray[Int,3] = [block_idx.x,block_idx.y,block_idx.z]
    var tiler_shape:InlineArray[Int,3] = [grid_dim.x,grid_dim.y,grid_dim.z]
    var shared_u = stack_allocation[float_dtype,AddressSpace.SHARED](col_major[SHARED_x,SHARED_y,SHARED_z,D]())
    sync_load_rank4_tensor_to_shared_with_halo[tile_shape,D](shared_u,velocity,tid,block_index,tiler_shape)
    barrier()

    var x = block_dim.x * block_idx.x + thread_idx.x
    var y = block_dim.y * block_idx.y + thread_idx.y
    var z = block_dim.z * block_idx.z + thread_idx.z

    comptime shift_x = 1
    comptime shift_y = 1 if D >= 2 else 0
    comptime shift_z = 1 if D == 3 else 0
    comptime vorticity_size = 1 if D == 2 else 3

    var shared_local_index = InlineArray[Int,3](uninitialized = True)
    var index:InlineArray[Int,3] = [x,y,z]
    var coord_index = coord[DType.int32]((index[0],index[1],index[2]))
    var flag = flags.load(coord_index)

    comptime stress_indices = lattice.stress_indices
    comptime directions = lattice.directions
    comptime weights = lattice.weights
    comptime assert not config.LES, 'Q criterion currently assumes post-collision so doesnt work for LES'
    if index[0] < grid_shape[0] and index[1] < grid_shape[1] and index[2] < grid_shape[2] and flag != SOLID_NODE: # Basic Guard
        shared_local_index[0] = thread_idx.x + shift_x
        shared_local_index[1] = thread_idx.y + shift_y
        shared_local_index[2] = thread_idx.z + shift_z
        # Calculate Voricity
        comptime if D == 2: # Lattice Units so dx is = 1
            dv_dx = get_velocity_gradient[1](shared_u,flags,shared_local_index,index,grid_shape,velocity_direction = 1,axis = 0)
            du_dy = get_velocity_gradient[1](shared_u,flags,shared_local_index,index,grid_shape,velocity_direction = 0,axis = 1)
            vorticity = dv_dx- du_dy
            vort_norm_sq = vorticity*vorticity
        else:
            # ex
            dw_dy = get_velocity_gradient[1](shared_u,flags,shared_local_index,index,grid_shape,velocity_direction = 2,axis = 1)
            dv_dz = get_velocity_gradient[1](shared_u,flags,shared_local_index,index,grid_shape,velocity_direction = 1,axis = 2)
            # ey
            du_dz = get_velocity_gradient[1](shared_u,flags,shared_local_index,index,grid_shape,velocity_direction = 0,axis = 2)
            dw_dx = get_velocity_gradient[1](shared_u,flags,shared_local_index,index,grid_shape,velocity_direction = 2,axis = 0)
            # ez
            dv_dx = get_velocity_gradient[1](shared_u,flags,shared_local_index,index,grid_shape,velocity_direction = 1,axis = 0)
            du_dy = get_velocity_gradient[1](shared_u,flags,shared_local_index,index,grid_shape,velocity_direction = 0,axis = 1)

            vorticity_vector = Vector[float_dtype,3](dw_dy-dv_dz,du_dz-dw_dx,dv_dx-du_dy)
            vort_norm_sq = vorticity_vector.norm_squared() # This is 2xRotation Tensor magnitude

        # Get Strain Rate Tensor
        var f_vec = Vector[float_dtype,Q](uninitialized = True)
        comptime for q in range(Q):
            f_vec[q] = load_f[float_dtype,config.use_float16c](f,index,q)

        var rho = get_density[config.DDF_shift](f_vec)
        var u = get_velocity[lattice.directions](f_vec,rho)

        var f_neq = get_f_noneq_vec[True,directions,weights,config.DDF_shift](f_vec,rho,u,tau)
        var second_moment_neq = get_non_eq_second_order_moment[directions,stress_indices](f_neq)
        var strain_rate = get_strain_rate_tensor(second_moment_neq,rho,tau)

        ss_norm_sq = get_strain_rate_tensor_norm_squared[stress_indices](strain_rate)

        Q_crit = 0.25*vort_norm_sq - 0.5*ss_norm_sq
        Q_tensor.store(coord_index,value= Q_crit)







