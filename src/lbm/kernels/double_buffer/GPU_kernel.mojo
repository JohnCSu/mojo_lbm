"""Defines the double-buffer SRT LBM GPU kernel.

The kernel reads pre-collision populations from `f_in` and writes
post-collision populations to `f_out`, so the caller swaps the two buffers
between time steps. A single kernel serves the D2Q9, D3Q19, and D3Q27
lattice models through compile-time parameterization.
"""
from layout import TileTensor,LayoutTensor,coord
from layout.tile_tensor import stack_allocation
from layout.tile_layout import Layout,row_major,Coord,TensorLayout

from std.gpu import block_dim,block_idx,thread_idx,barrier
from std.gpu.memory import AddressSpace
from std.math import sqrt

from src.lbm import LBM_Config,LatticeModel,GridLike,LBM_Grid
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
from src.lbm.kernels.ops.turbulence import get_Smagorinsky_LES_tau
from src.lbm.kernels.utils.equilibrium import get_f_eq_vec, get_f_noneq_vec
from src.lbm.kernels.ops import wall_bc,equilibrium_bc,SRT


def double_buffer_kernel[
    Flayout:Layout,
    BClayout:Layout,
    Flaglayout:Layout,
    grid: LBM_Grid,
    config:LBM_Config,
    ]
    (
    f_out:TileTensor[config.set_f_dtype(grid.float_dtype),type_of(Flayout),MutAnyOrigin],
    f_in:TileTensor[config.set_f_dtype(grid.float_dtype),type_of(Flayout),ImmutAnyOrigin],
    bc:TileTensor[grid.float_dtype,type_of(BClayout),ImmutAnyOrigin],
    flags:TileTensor[DType.uint8,type_of(Flaglayout),ImmutAnyOrigin],
    tau:Scalar[grid.float_dtype],
    )
    where Flayout.rank == 4 and BClayout.rank == 4 and Flaglayout.rank == 3:
    """Runs one SRT LBM time step from `f_in` into `f_out`.

    Performs pull-scheme streaming with mid-grid bounce-back at solid nodes,
    optional equilibrium boundary-condition handling, density and velocity
    extraction, optional Smagorinsky LES relaxation-time correction, and the
    BGK collision. The result is written to `f_out`; the caller swaps
    `f_in` and `f_out` between calls.

    Parameters:
        Flayout: The compile-time `Layout` of the distribution function.
        BClayout: The compile-time `Layout` of the boundary-condition tensor.
        Flaglayout: The compile-time `Layout` of the `uint8` flag tensor.
        grid: The compile-time `LBM_Grid` describing the domain.
        config: The `LBM_Config` selecting DDF shift, Float16C, LES, and the
            valid boundary-condition flags.

    Args:
        f_out: The output distribution function tile tensor (rank 4).
        f_in: The input distribution function tile tensor (rank 4).
        bc: The boundary-condition tile tensor (rank 4).
        flags: The `uint8` tile tensor labeling each node (rank 3).
        tau: The base SRT relaxation time.
    """
    # Convience Variable Names and constants
    comptime D = grid.D
    comptime Q = grid.Q
    comptime float_dtype = grid.float_dtype
    comptime int_dtype = grid.int_dtype
    comptime lattice_model = grid.lattice_model
    comptime weights = lattice_model.weights
    comptime directions = lattice_model.directions
    comptime opposite_indices = lattice_model.opposite_indices
    comptime grid_shape:InlineArray[Int,3] = grid.shape
    comptime non_temporal = True
    comptime load_f_from_xyzq = load_f[float_dtype,config.use_float16c,non_temporal]
    comptime stress_indices = lattice_model.stress_indices
    # comptime assert f_out.flat_rank == 8
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

        # Pull Stream Step # This is different for methods
        comptime for q in range(Q):
            comptime direction = directions[q]
            pull_index = get_adjacent_idx[shift = -1](index,grid_shape,direction) # Pulling Scheme
            f_new[q] =  load_f_from_xyzq(f_in,pull_index,q)
        
        # Bounce Back AND PULL FLAGS
        comptime include_bounceback = True
        wall_bc[include_bounceback,directions,opposite_indices,weights,config.use_float16c](f_new,pull_flags,f_in,flags,bc,index,grid_shape)
        
        # Equilibrium BC
        comptime if Flags.EQUILIBRIUM in config.INCLUDED_BCs:
            equilibrium_bc[directions,weights,config.DDF_shift](f_new,pull_flags,bc,index,grid_shape)

        # Get Velocity and Density
        rho = get_density[config.DDF_shift](f_new)
        velocity = get_velocity[directions](f_new,rho)
        tau_local = tau # Create a local variable if we need to modify tau with LES,KBC EELBM etc
        
        # Non eq ops
        comptime if config.implies_f_noneq():
            f_neq = get_f_noneq_vec[False,directions,weights,config.use_float16c](f_new,rho,velocity,tau_local)
            second_moment_neq = get_non_eq_second_order_moment[directions,stress_indices](f_neq)
            strain_rate = get_strain_rate_tensor(second_moment_neq,rho,tau_local)
            comptime if config.LES:
                comptime Cs = 0.1
                tau_eddy = get_Smagorinsky_LES_tau[stress_indices](strain_rate,Cs)
                tau_local += tau_eddy

        # Collision Term
        u_dot_u = velocity.dot(velocity)
        inv_tau = 1./tau_local # This is faster by 0.4 ms on the 256^3 benchmark

        SRT[directions,weights,config.DDF_shift](f_new,velocity,rho,tau_local)

        # Store f back to Global
        comptime for q in range(Q):
            store_f[config.use_float16c,non_temporal](f_out,f_new[q],index,q)




