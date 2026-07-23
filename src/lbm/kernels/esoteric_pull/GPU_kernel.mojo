"""Defines the esoteric-pull LBM GPU kernel (incomplete).

The esoteric-pull streaming scheme reads and writes populations in-place to
halve memory traffic compared with the double-buffer variant. The
implementation here is a work in progress and only contains the start of the
streaming step.
"""
from layout import TileTensor,LayoutTensor,coord
from layout.tile_tensor import stack_allocation
from layout.tile_layout import Layout,row_major,Coord,TensorLayout

from std.gpu import block_dim,block_idx,thread_idx,barrier
from std.gpu.memory import AddressSpace
from std.utils.numerics import nan,isnan
from std.math import sqrt

from src.lbm import LBM_Grid,LBM_Config,Lattice
from src.lbm.constants import SOLID_NODE,FLUID_NODE,Flags,cs_squared
from src.lbm.kernels.utils.index import get_adjacent_idx
from src.lbm.kernels.utils.load_and_store import load_f,store_f,esoteric_pull_load_f_vec,esoteric_pull_store_f_vec

from src.utils import Vector,ContextTileTensor
from src.lbm.kernels.utils.moment import (
                                            get_density,
                                            get_velocity,
                                            get_strain_rate_tensor,
                                            get_non_eq_second_order_moment,
                                            get_density_and_velocity_for_eq_BC)
# from src.lbm.kernels.utils.turbulence import get_Smagorinsky_LES_tau

from src.lbm.kernels.utils.equilibrium import get_f_eq_vec, get_f_noneq_vec,f_eq

from src.lbm.kernels.ops import SRT,wall_bc

def esoteric_pull_kernel[ 
                is_even_time_step:Bool,
                F_layout:Layout,
                BC_layout:Layout,
                Flag_layout:Layout,
                grid: LBM_Grid,
                config:LBM_Config,
                ]
                (
                f:TileTensor[config.set_f_dtype(grid.float_dtype),type_of(F_layout),MutAnyOrigin],
                bc:TileTensor[grid.float_dtype,type_of(BC_layout),ImmutAnyOrigin],
                flags:TileTensor[DType.uint8,type_of(Flag_layout),ImmutAnyOrigin],
                tau:Scalar[grid.float_dtype],
                )
                where F_layout.rank == 4 and BC_layout.rank == 4 and Flag_layout.rank == 3:
    """Runs one esoteric-pull SRT LBM time step in place (incomplete).

    Intended to perform the pull-scheme streaming and collision in place by
    pulling from the positive half of the lattice on even time steps and the
    negative half on odd time steps, halving memory traffic versus the
    double-buffer kernel. The implementation is a work in progress and only
    contains the start of the streaming step.

    Parameters:
        is_even_time_step: When `True`, pull from the positive half of the
            lattice; otherwise pull from the negative half.
        grid: The compile-time `LBM_Grid` describing the domain.
        Flayout: The compile-time `Layout` of the distribution function.
        BClayout: The compile-time `Layout` of the boundary-condition tensor.
        Flaglayout: The compile-time `Layout` of the `uint8` flag tensor.
        config: The `LBM_Config` selecting DDF shift, Float16C, LES, and the
            valid boundary-condition flags.
        f_dtype: The storage `DType` for `f` (defaults to the config's
            `f_dtype` or `float_dtype`).

    Args:
        f: The distribution function tile tensor (rank 4), updated in place.
        bc: The boundary-condition tile tensor (rank 4).
        flags: The `uint8` tile tensor labeling each node (rank 3).
        tau: The base SRT relaxation time.
    """
    # Convience Variable Names and constants
    comptime D = grid.D
    comptime Q = grid.Q
    comptime float_dtype = grid.float_dtype
    comptime int_dtype = grid.int_dtype
    comptime lattice = grid.lattice
    comptime weights = lattice.weights
    comptime directions = lattice.directions
    comptime opposite_indices = lattice.opposite_indices
    comptime stress_indices = lattice.stress_indices
    comptime grid_shape:InlineArray[Int,3] = grid.shape
    
    # Comptime asserts
    comptime assert not directions[0].all_true(), 'The first direction for the lattice model should be all 0s i.e directions[0]=[0,0,0]'
    comptime assert lattice.is_valid_for_esoteric_pull(),'Except the first direction, velocitys direction should be followed by their opposite direction'

    x = block_idx.x*block_dim.x + thread_idx.x
    y = block_idx.y*block_dim.y + thread_idx.y
    z = block_idx.z*block_dim.z + thread_idx.z

    var index:InlineArray[Int,3] = [x,y,z]
    var pull_flags = InlineArray[UInt8,Q](uninitialized = True)
    pull_flags[0] = flags.load(coord[DType.uint32]((x,y,z)))[0]
    

    if (index[0] < grid_shape[0]) and (index[1] < grid_shape[1]) and (index[2] < grid_shape[2]) and pull_flags[0] != SOLID_NODE: # Basic Guard
        # Load and Stream From Global
        f_new: Vector[float_dtype,Q] = esoteric_pull_load_f_vec[float_dtype,directions,is_even_time_step,config.use_float16c](f,index,grid_shape)

        # Apply BC
        comptime if config.include_moving_boundary: # We have moving walls
            comptime include_bounceback = False
            wall_bc[include_bounceback,directions,opposite_indices,weights,config.use_float16c](f_new,pull_flags,f,flags,bc,index,grid_shape)

        # Get Local Variables moments
        rho = get_density[config.DDF_shift](f_new) 
        velocity = get_velocity[directions](f_new,rho)
        tau_local = tau # Create a local variable if we need to modify tau with LES,KBC EELBM etc

        # Collision Term
        SRT[directions,weights,config.DDF_shift](f_new,velocity,rho,tau_local)
        
        # Store To Global
        esoteric_pull_store_f_vec[directions,is_even_time_step,config.use_float16c](f,f_new,index,grid_shape)