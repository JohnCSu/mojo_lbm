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

from src.lbm import LBM_Grid,LBM_Config,LatticeModel
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
from src.lbm.kernels.utils.turbulence import get_Smagorinsky_LES_tau
from src.lbm.kernels.utils.equilibrium import get_f_eq_vec, get_f_noneq_vec,f_eq

def esoteric_pull_kernel[ float_dtype:DType,D:Int,Q:Int,
                lattice_model:LatticeModel[D,Q,float_dtype,DType.int32],
                nx:Int,ny:Int,nz:Int,tile_size:Int,
                //,
                is_even_time_step:Bool,
                implicitBounceBackOnly:Bool,
                Flayout:Layout[...],
                BClayout:Layout[...],
                Flaglayout:Layout[...],
                grid: LBM_Grid[lattice_model,nx,ny,nz,tile_size],
                config:LBM_Config = LBM_Config(),
                *,
                f_dtype:DType = config.f_dtype.value() if config.f_dtype is not None else float_dtype
                ]
                (
                f:TileTensor[f_dtype,type_of(Flayout),MutAnyOrigin],
                bc:TileTensor[float_dtype,type_of(BClayout),ImmutAnyOrigin],
                flags:TileTensor[DType.uint8,type_of(Flaglayout),ImmutAnyOrigin],
                tau:Scalar[float_dtype],
                )
                where tile_size >= 1 and Flayout.rank == 4 and BClayout.rank == 4 and Flaglayout.rank == 3:
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
    comptime assert lattice_model.is_valid_for_esoteric_pull(),'Except the first direction, velocitys direction should be followed by their opposite direction'

    x = block_idx.x*block_dim.x + thread_idx.x
    y = block_idx.y*block_dim.y + thread_idx.y
    z = block_idx.z*block_dim.z + thread_idx.z

    var index:InlineArray[Int,3] = [x,y,z]
    var pull_flags = InlineArray[UInt8,Q](uninitialized = True)
    # var pull_indices = InlineArray[InlineArray[Int,3],Q](uninitialized = True)
    pull_flags[0] = flags.load(coord[DType.uint32]((x,y,z)))[0]
    comptime load_f_from_xyzq = load_f[float_dtype,config.use_float16c]
    if (index[0] < grid_shape[0]) and (index[1] < grid_shape[1]) and (index[2] < grid_shape[2]) and pull_flags[0] != SOLID_NODE: # Basic Guard
        
        var velocity = Vector[float_dtype,D](uninitialized = True)
        var rho:Scalar[float_dtype] = 0

        f_new: Vector[float_dtype,Q] = esoteric_pull_load_f_vec[float_dtype,directions,is_even_time_step,config.use_float16c](f,index,grid_shape)
        
        # # Regular LBM
        comptime if not implicitBounceBackOnly: # We have moving walls
            comptime for q in range(1,Q):
                comptime direction = directions[q]
                pull_index = get_adjacent_idx[shift = -1](index,grid_shape,direction) # Pulling Scheme
                # comptime if q > 0: # we pulled the flag[0] earlier
                pull_flags[q] = flags.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2])))[0]
                # var correction:Scalar[float_dtype] = 0.
                if pull_flags[q] == Flags.SOLID:
                    float_direction = (float_directions[q])
                    weight = weights[q]
                    comptime for ii in range(D):
                        velocity[ii] = bc.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2],ii)))[0]
                    rho = bc.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2],D)))[0]
                    f_new[q] += 2.*3.*weight*rho*(float_direction.dot(velocity))

        # var velocity = Vector[float_dtype,D](uninitialized = True)
        # var rho:Scalar[float_dtype] = 0
        # Get Velocity and Density
        rho = get_density[config.DDF_shift](f_new) 
        velocity = get_velocity[float_directions](f_new,rho)
        tau_local = tau # Create a local variable if we need to modify tau with LES,KBC EELBM etc
        # f_eq = get_f_eq_vec[float_directions,weights,config.DDF_shift](f_new,rho,velocity)

        # Collision Term
        u_dot_u = velocity.dot(velocity)
        inv_tau = 1./tau_local # This is faster by 0.4 ms on the 256^3 benchmark

        # Collide
        comptime for q in range(Q):
            comptime direction = float_directions[q]
            comptime weight = weights[q]
            f_new[q] -= inv_tau*(f_new[q]- f_eq[config.DDF_shift](weight,rho,velocity,u_dot_u,direction))

        esoteric_pull_store_f_vec[directions,is_even_time_step,config.use_float16c](f,f_new,index,grid_shape)
        
