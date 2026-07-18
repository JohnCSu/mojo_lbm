"""Defines the deprecated single-buffer SRT LBM GPU kernel.

`LBM_kernel` is the original single-buffer SRT kernel kept for reference and
benchmarking. New code should use `double_buffer_kernel` from
`src/lbm/kernels/double_buffer/`, which exposes the same behavior with a
safer double-buffer interface.
"""
from layout import TileTensor,LayoutTensor,coord
from layout.tile_tensor import stack_allocation
from layout.tile_layout import Layout,row_major,Coord,TensorLayout

from std.gpu import block_dim,block_idx,thread_idx,barrier
from std.gpu.memory import AddressSpace
from std.utils.numerics import nan,isnan
from std.math import sqrt

from src.lbm import LBM_Grid,LBM_Config,LatticeModel
from src.lbm.constants import SOLID_NODE,FLUID_NODE,Flags,cs_squared
from src.lbm.kernels.utils.index import get_adjacent_idx
from src.lbm.kernels.utils.load_and_store import load_f,store_f

from src.utils import Vector,ContextTileTensor
from src.lbm.kernels.utils.moment import (
                                            get_density,
                                            get_velocity,
                                            get_strain_rate_tensor,
                                            get_non_eq_second_order_moment,
                                            get_density_and_velocity_for_eq_BC)
from src.lbm.kernels.utils.turbulence import get_Smagorinsky_LES_tau
from src.lbm.kernels.utils.equilibrium import get_f_eq_vec, get_f_noneq_vec
from src.lbm.kernels.double_buffer import double_buffer_kernel

@deprecated(use=double_buffer_kernel)
def LBM_kernel[ float_dtype:DType,D:Int,Q:Int,
                lattice_model:LatticeModel[D,Q,float_dtype,DType.int32],
                nx:Int,ny:Int,nz:Int,tile_size:Int,
                //,
                Flayout:Layout[...],
                BClayout:Layout[...],
                Flaglayout:Layout[...],
                grid: LBM_Grid[lattice_model,nx,ny,nz,tile_size],
                config:LBM_Config = LBM_Config(),
                *,
                f_dtype:DType = config.f_dtype.value() if config.f_dtype is not None else float_dtype
                ]
                (
                f_out:TileTensor[f_dtype,type_of(Flayout),MutAnyOrigin],

                f_in:TileTensor[f_dtype,type_of(Flayout),ImmutAnyOrigin],
                bc:TileTensor[float_dtype,type_of(BClayout),ImmutAnyOrigin],
                flags:TileTensor[DType.uint8,type_of(Flaglayout),ImmutAnyOrigin],
                tau:Scalar[float_dtype],
                )
                where tile_size >= 1 and Flayout.rank == 4 and BClayout.rank == 4 and Flaglayout.rank == 3:
    """Runs one SRT LBM time step from `f_in` into `f_out` (deprecated).

    Performs pull-scheme streaming with mid-grid bounce-back at solid nodes,
    optional equilibrium boundary-condition handling, density and velocity
    extraction, optional Smagorinsky LES relaxation-time correction, and the
    BGK collision. Deprecated in favor of `double_buffer_kernel`, which
    exposes the same behavior with a safer double-buffer interface.

    Parameters:
        Flayout: The compile-time `Layout` of the distribution function.
        BClayout: The compile-time `Layout` of the boundary-condition tensor.
        Flaglayout: The compile-time `Layout` of the `uint8` flag tensor.
        grid: The compile-time `LBM_Grid` describing the domain.
        config: The `LBM_Config` selecting DDF shift, Float16C, LES, and the
            valid boundary-condition flags.
        f_dtype: The storage `DType` for `f` (defaults to the config's
            `f_dtype` or `float_dtype`).

    Args:
        f_out: The output distribution function tile tensor (rank 4).
        f_in: The input distribution function tile tensor (rank 4).
        bc: The boundary-condition tile tensor (rank 4).
        flags: The `uint8` tile tensor labeling each node (rank 3).
        tau: The base SRT relaxation time.
    """
    '''
    Base LBM to also handle 3D and non_square Grids. Key assumption is that block dim == tile-size
    i.e. grid can be non-square but block is squre (same block dim in each x,y,z).
    '''
    # Convience Variable Names and constants
    comptime assert f_out.flat_rank == 8
    comptime weights = lattice_model.weights
    comptime float_directions = lattice_model.float_directions
    comptime directions = lattice_model.directions
    comptime opposite_index = lattice_model.opposite_indices
    comptime grid_shape:InlineArray[Int,3] = [nx,ny,nz]
    comptime non_temporal = True
    comptime load_f_from_xyzq = load_f[float_dtype,config.use_float16c,non_temporal]
    comptime stress_indices = lattice_model.stress_indices
    # Comptime asserts
    comptime assert not directions[0].all_true(), 'The first direction for the lattice model should be all 0s i.e directions[0]=[0,0,0]'

    x = block_idx.x*block_dim.x + thread_idx.x
    y = block_idx.y*block_dim.y + thread_idx.y
    z = block_idx.z*block_dim.z + thread_idx.z

    var index:InlineArray[Int,3] = [x,y,z]
    var pull_flags = InlineArray[UInt8,Q](uninitialized = True)
    # var pull_indices = InlineArray[InlineArray[Int,3],Q](uninitialized = True)
    # Main Compute
    pull_flags[0] = flags.load(coord[DType.uint32]((x,y,z)))[0]

    if (index[0] < grid_shape[0]) and (index[1] < grid_shape[1]) and (index[2] < grid_shape[2]) and pull_flags[0] != SOLID_NODE: # Basic Guard
        var f_new = Vector[float_dtype,Q](fill = 0.)
        var velocity = Vector[float_dtype,D](uninitialized = True)
        var rho:Scalar[float_dtype] = 0
        # Pull Stream Step # This is different for methods
        comptime for q in range(Q):
            comptime direction = directions[q]
            pull_index = get_adjacent_idx[-1](index,grid_shape,direction) # Pulling Scheme
            f_new[q] =  load_f_from_xyzq(f_in,pull_index,q)


        # Function this
        # Bounce Back AND PULL FLAGS
        comptime for q in range(Q):
            comptime direction = directions[q]
            pull_index = get_adjacent_idx[-1](index,grid_shape,direction) # Pulling Scheme
            comptime if q > 0: # we pulled the flag[0] earlier
                pull_flags[q] = flags.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2])))[0]

            if pull_flags[q] == SOLID_NODE:
                opp_q = Int(opposite_index[q])
                comptime for ii in range(D):
                    velocity[ii] = bc.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2],ii)))[0]
                rho = bc.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2],D)))[0]
                f_new[q] = load_f_from_xyzq(f_in,index,opp_q) + 2.*3.*weights[q]*rho*(float_directions[q].dot(velocity))


        # Function This
        # Equilibrium BC
        comptime if Flags.EQUILIBRIUM in config.INCLUDED_BCs:
            current_flag = pull_flags[0] # comptime assert gurantees this is the flag for the current node
            if current_flag  == Flags.EQUILIBRIUM:
                comptime for ii in range(D):
                    velocity[ii] = bc.load(coord[DType.uint32]((x,y,z,ii)))[0]
                rho = bc.load(coord[DType.uint32]((x,y,z,D)))[0]
                rho_local,u_l = get_density_and_velocity_for_eq_BC[float_directions,directions,config.DDF_shift](f_new,weights,index,grid_shape)
                u_local = u_l if isnan(velocity[0]) else velocity # nan means the vel is free
                rho_local = rho_local if isnan(rho) else rho # Nan means density is free
                f_new = get_f_eq_vec[float_directions,weights,config.DDF_shift](f_new,rho_local,u_local)


        # Get Velocity and Density
        rho = get_density[config.DDF_shift](f_new)
        velocity = get_velocity[float_directions](f_new,rho)
        tau_local = tau # Create a local variable if we need to modify tau with LES,KBC EELBM etc
        f_eq = get_f_eq_vec[float_directions,weights,config.DDF_shift](f_new,rho,velocity)

        # LES
        comptime if config.implies_f_noneq():
            f_neq = get_f_noneq_vec[post_collision = False](f_new,f_eq,tau_local)
            second_moment_neq = get_non_eq_second_order_moment[float_directions,stress_indices](f_neq)
            strain_rate = get_strain_rate_tensor(second_moment_neq,rho,tau_local)
            comptime if config.LES:
                comptime Cs = 0.1
                tau_eddy = get_Smagorinsky_LES_tau[stress_indices](strain_rate,Cs)
                tau_local += tau_eddy

        # Collision Term
        u_dot_u = velocity.dot(velocity)
        inv_tau = 1./tau_local # This is faster by 0.4 ms on the 256^3 benchmark

        # Store f back to Global
        comptime for q in range(Q):
            store_f[config.use_float16c,non_temporal](f_out,(f_new[q] -  inv_tau*(f_new[q]- f_eq[q]) ),index,q)
