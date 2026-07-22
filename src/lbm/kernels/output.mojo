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

from src.lbm import LBM_Grid,LBM_Config,LatticeModel
from src.utils import Vector,ContextTileTensor
from src.lbm.kernels.utils.index import get_adjacent_idx,is_index_valid
from src.lbm.kernels.utils.load_and_store import load_f,store_f,esoteric_pull_load_f_vec,esoteric_pull_store_f_vec
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
    config:LBM_Config = LBM_Config(),
    *,
    f_dtype:DType = grid.float_dtype if config.f_dtype is None else config.f_dtype.value()
    ]
    (
        density:TileTensor[grid.float_dtype,type_of(RhoLayout),MutAnyOrigin],
        velocity:TileTensor[grid.float_dtype,type_of(VelocityLayout),MutAnyOrigin],
        f:TileTensor[f_dtype,type_of(Flayout),ImmutAnyOrigin],
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
    comptime lattice_model = grid.lattice_model
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
            comptime for q in range(Q):
                f_vec[q] = load_f[float_dtype,config.use_float16c](f,index,q)

            rho = get_density[config.DDF_shift](f_vec)
            u = get_velocity[lattice_model.directions](f_vec,rho)
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

def calculate_esoteric_rho_and_velocity[
    is_even_time_step:Bool,
    Flayout:Layout[...],
    BClayout:Layout[...],
    Flaglayout:Layout[...],
    RhoLayout:Layout[...],
    VelocityLayout:Layout[...],
    grid: LBM_Grid,
    config:LBM_Config = LBM_Config(),
    *,
    f_dtype:DType = grid.float_dtype if config.f_dtype is None else config.f_dtype.value()
    ]
    (
        density:TileTensor[grid.float_dtype,type_of(RhoLayout),MutAnyOrigin],
        velocity:TileTensor[grid.float_dtype,type_of(VelocityLayout),MutAnyOrigin],

        f:TileTensor[f_dtype,type_of(Flayout),ImmutAnyOrigin],
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
    comptime lattice_model = grid.lattice_model
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
            f_vec = esoteric_pull_load_f_vec[float_dtype,lattice_model.directions,is_even_time_step,config.use_float16c](f,index,grid_shape)
            comptime for q in range(Q):
                f_vec[q] = load_f[float_dtype,config.use_float16c](f,index,q)

            rho = get_density[config.DDF_shift](f_vec)
            u = get_velocity[lattice_model.directions](f_vec,rho)


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








def calculate_Q_criterion[
    Flayout:Layout[...] ,
    FlagLayout:Layout[...] ,
    VelocityLayout:Layout[...],
    QLayout:Layout[...],
    grid: LBM_Grid,
    config:LBM_Config = LBM_Config(),
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
    comptime lattice_model = grid.lattice_model
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

    comptime stress_indices = lattice_model.stress_indices
    comptime directions = lattice_model.directions
    comptime weights = lattice_model.weights
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
        var u = get_velocity[lattice_model.directions](f_vec,rho)

        var f_neq = get_f_noneq_vec[True,directions,weights,config.DDF_shift](f_vec,rho,u,tau)
        var second_moment_neq = get_non_eq_second_order_moment[directions,stress_indices](f_neq)
        var strain_rate = get_strain_rate_tensor(second_moment_neq,rho,tau)

        ss_norm_sq = get_strain_rate_tensor_norm_squared[stress_indices](strain_rate)

        Q_crit = 0.25*vort_norm_sq - 0.5*ss_norm_sq
        Q_tensor.store(coord_index,value= Q_crit)


def rowMajor1D[int_dtype:DType]() -> type_of( row_major(coord[int_dtype]((1,))) ):
    """Returns a 1D row-major `Int`-dtype layout instance.

    Parameters:
        int_dtype: The `DType` used for the layout coordinates.

    Returns:
        A 1D row-major layout instance.
    """
    return row_major(coord[int_dtype]((1,)))

def rowMajor2D[int_dtype:DType]() -> type_of(row_major(coord[int_dtype]((1,2))) ):
    """Returns a 2D row-major `Int`-dtype layout instance.

    Parameters:
        int_dtype: The `DType` used for the layout coordinates.

    Returns:
        A 2D row-major layout instance.
    """
    return row_major(coord[int_dtype]((1,2)))


def calculate_drag_around_object[
    FLayout:Layout[...] ,
    FlagLayout:Layout[...],
    grid: LBM_Grid,
    config:LBM_Config = LBM_Config(),
    *,
    f_dtype:DType = grid.float_dtype if config.f_dtype is None else config.f_dtype.value()
    ](
        f:TileTensor[f_dtype,type_of(FLayout),ImmutAnyOrigin],
        flags:TileTensor[DType.uint8,type_of(FlagLayout),ImmutAnyOrigin],
        fluid_boundary:TileTensor[grid.int_dtype,type_of(rowMajor1D[grid.int_dtype]()),MutAnyOrigin],
        force_tensor:TileTensor[grid.float_dtype,type_of(rowMajor2D[grid.int_dtype]()),MutAnyOrigin],
    ):

    """Computes the drag force on the fluid nodes adjacent to an immersed object.

    Iterates over the linear fluid boundary indices, gathers the push-scheme
    neighbor flags, and accumulates the momentum-exchange contribution

    $$F = \\sum_q 2 f_{link} e_q$$

    for every direction `q` whose push neighbor is solid. The result is
    written into `force_tensor[tid, i]` for each dimension `i`.

    Parameters:
        FLayout: The compile-time `Layout` of the distribution function `f`.
        FlagLayout: The compile-time `Layout` of the `uint8` flag tensor.
        grid: The compile-time `LBM_Grid` describing the domain.
        config: The `LBM_Config` used to select storage options.
        f_dtype: The storage `DType` for `f` (defaults to the config's
            `f_dtype` or `float_dtype`).

    Args:
        f: The input distribution function tile tensor (rank 4).
        flags: The `uint8` tile tensor labeling each node (rank 3).
        fluid_boundary: The 1D tile tensor of linear fluid boundary indices.
        force_tensor: The 2D output tile tensor of per-node force vectors.
    """
    comptime D = grid.D
    comptime Q = grid.Q
    comptime float_dtype = grid.float_dtype
    comptime int_dtype = grid.int_dtype
    comptime lattice_model = grid.lattice_model
    comptime tile_shape = grid.tile_shape
    # Should be a 1D based kernel loop
    tid = block_dim.x * block_idx.x + thread_idx.x
    if tid < fluid_boundary.layout.size():
        crd = FlagLayout.idx2crd[out_dtype = int_dtype](Int(fluid_boundary[tid])).flatten()
        grid_index = InlineArray[Int,3](uninitialized = True)

        comptime if FlagLayout.rank*2 == FlagLayout.flat_rank:
            comptime for i in range(3):
                loc_x = Int(crd[2*i].value()) # local
                til_x = Int(crd[(2*i)+1].value())
                grid_index[i] = tile_shape[i]*til_x + loc_x
        else:
            comptime assert FlagLayout.rank == FlagLayout.flat_rank
            comptime for i in range(3):
                grid_index[i] = Int(crd[i].value())

        comptime grid_shape:InlineArray[Int,3] = grid.shape
        comptime opposite_index = lattice_model.opposite_indices

        if grid_index[0] < grid_shape[0] and grid_index[1] < grid_shape[1] and grid_index[2] < grid_shape[2]:
            var push_flags = InlineArray[UInt8,Q](uninitialized = True)
            var push_indices = InlineArray[InlineArray[Int,3],Q](uninitialized = True)

            # Gather Neighboring Flags in PUSH direction not Pull
            comptime for q in range(Q):
                comptime direction = lattice_model.directions[q]
                push_indices[q] = get_adjacent_idx[1](grid_index,grid_shape,direction) # push Scheme as
                push_flags[q] = flags.load(coord[DType.uint32]((push_indices[q][0],push_indices[q][1],push_indices[q][2])))[0]

            # Compute Forces
            var force_vec = Vector[float_dtype,D](fill = 0.)
            comptime for q in range(Q):
                comptime direction = lattice_model.directions[q]
                comptime float_direction = lattice_model.directions[q].cast_to[float_dtype]()
                if push_flags[q] == SOLID_NODE:
                    var f_local = load_f[float_dtype,config.use_float16c](f,grid_index,q)
                    comptime if config.DDF_shift:
                        comptime weight = lattice_model.weights[q]
                        f_link = f_local + weight
                    else:
                        f_link = f_local
                    force_vec += (2*f_link)*float_direction # Only stationary wall for now

            # push to global
            comptime for i in range(D): # Overwrite
                force_tensor[tid,i] = force_vec[i]
