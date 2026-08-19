from layout import TileTensor,LayoutTensor,coord
from layout.tile_tensor import stack_allocation
from layout.tile_layout import Layout,row_major,Coord,TensorLayout

from src.lbm import LBM_Config,Lattice,GridLike,LBM_Grid,RuntimeParams
from src.lbm.constants import SOLID_NODE,FLUID_NODE,Flags,cs_squared,Collisions
from src.lbm.kernels.utils.index import get_adjacent_idx,get_rank4_coord
from src.lbm.kernels.utils.load_and_store import load_f,store_f

from src.utils import Vector,ContextTileTensor
from src.lbm.kernels.utils.moment import (
                                            get_density,
                                            get_velocity,
                                            get_strain_rate_tensor,
                                            get_non_eq_second_order_moment,
                                        )

from src.lbm.kernels.utils.equilibrium import get_f_eq_vec, get_f_noneq_vec
from src.lbm.kernels.ops.turbulence import get_Smagorinsky_LES_tau
from src.lbm.kernels.ops import SRT,TRT,KBC,RLBM

@always_inline
def collide[
    FlayoutType:TensorLayout,
    BClayoutType:TensorLayout,
    FlaglayoutType:TensorLayout,
    //,
    grid: LBM_Grid,
    config:LBM_Config,
    ](
    mut f_vec:Vector[grid.float_dtype,grid.Q],
    f:TileTensor[config.set_f_dtype(grid.float_dtype),FlayoutType,_],
    bc:TileTensor[grid.float_dtype,BClayoutType,_],
    flags:TileTensor[DType.uint8,FlaglayoutType,_],
    pull_flags:InlineArray[UInt8,grid.Q],
    index:InlineArray[Int,3],
    tau:Scalar[grid.float_dtype],
    ):
    comptime D = grid.D
    comptime float_dtype = grid.float_dtype
    comptime int_dtype = grid.int_dtype
    comptime lattice = grid.lattice
    comptime weights = lattice.weights
    comptime directions = lattice.directions
    comptime opposite_indices = lattice.opposite_indices
    comptime grid_shape:InlineArray[Int,3] = grid.shape
    comptime stress_indices = lattice.stress_indices

    # Get Velocity and Density
    rho = get_density[config.DDF_shift](f_vec)
    velocity = get_velocity[directions](f_vec,rho)
    tau_local = tau # Create a local variable if we need to modify tau with LES,KBC EELBM etc

    comptime if config.capture_density:
        comptime assert bc.mut,'BC tiletensor must be mutable if config.capture_velocity or capture_density is set'
        if Flags.has[Flags.CAPTURE_DENSITY](pull_flags[0]):
            bc.store(get_rank4_coord(index,D),rho)
    
    comptime if config.capture_velocity:
        comptime assert bc.mut,'BC tiletensor must be mutable if config.capture_velocity or capture_density is set'
        if Flags.has[Flags.CAPTURE_VELOCITY](pull_flags[0]):
            comptime for d in range(D):
                bc.store(get_rank4_coord(index,d),velocity[d])
    
    # Non eq ops
    comptime if config.implies_f_noneq():
        f_neq = get_f_noneq_vec[False,directions,weights,config.DDF_shift](f_vec,rho,velocity,tau_local)
        second_moment_neq = get_non_eq_second_order_moment[directions,stress_indices](f_neq)
        strain_rate = get_strain_rate_tensor(second_moment_neq,rho,tau_local)
        comptime if config.LES:
            comptime Cs = 0.1
            tau_eddy = get_Smagorinsky_LES_tau[stress_indices](strain_rate,Cs)
            tau_local += tau_eddy

        comptime if config.collision_op == Collisions.RLBM:
            RLBM[directions,weights,stress_indices,config.DDF_shift](f_vec,f_neq,second_moment_neq,rho,velocity,tau_local)
        elif config.collision_op == Collisions.KBC:
            KBC[directions,weights,config.DDF_shift](f_vec,f_neq,second_moment_neq,rho,velocity,tau_local)
    
    # Collision Term
    # comptime assert config.collision_op_is_valid(), 'Collision operator must be either SRT or TRT'
    comptime if config.collision_op == Collisions.SRT:
        SRT[directions,weights,config.DDF_shift](f_vec,velocity,rho,tau_local)
    elif config.collision_op == Collisions.TRT:
        comptime TRT_magic_param = 3./16.
        tau_asymm = 0.5 + TRT_magic_param/(tau_local-0.5)
        TRT[directions,weights,config.DDF_shift](f_vec,velocity,rho,tau_local,tau_asymm)
    else:
        comptime assert config.collision_op_is_valid() ,'Invalid Collision Operator specified'

