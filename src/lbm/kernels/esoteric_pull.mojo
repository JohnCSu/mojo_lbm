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
from max.gpu.memory import AddressSpace
from std.utils.numerics import nan,isnan
from std.math import sqrt

from src.lbm import LBM_Grid,LBM_Config,Lattice
from src.lbm.constants import SOLID_NODE,FLUID_NODE,Flags,cs_squared,LBM_method

from src.lbm.kernels.utils.checks import is_valid_thread
from src.utils import Vector
from src.lbm.kernels.steps import stream,collide,apply_boundary_conditions,store_f_vec_to_global

def esoteric_pull_kernel[ 
    is_even_time_step:Bool,
    FlayoutType:TensorLayout,
    BClayoutType:TensorLayout,
    FlaglayoutType:TensorLayout,
    grid: LBM_Grid,
    config:LBM_Config,
    ]
    (
    f:TileTensor[config.set_f_dtype(grid.float_dtype),FlayoutType,MutAnyOrigin],
    bc:TileTensor[grid.float_dtype,BClayoutType,MutAnyOrigin if config.implies_bc_is_mutable() else ImmutAnyOrigin],
    flags:TileTensor[DType.uint8,FlaglayoutType,ImmutAnyOrigin],
    tau:Scalar[grid.float_dtype],
    )
    where FlayoutType.rank == 4 and BClayoutType.rank == 4 and FlaglayoutType.rank == 3:
    """Runs one esoteric-pull SRT LBM time step in place (incomplete).

    Intended to perform the pull-scheme streaming and collision in place by
    pulling from the positive half of the lattice on even time steps and the
    negative half on odd time steps, halving memory traffic versus the
    double-buffer kernel. The implementation is a work in progress and only
    contains the start of the streaming step.

    Parameters:
        is_even_time_step: When `True`, pull from the positive half of the
            lattice; otherwise pull from the negative half.
        FlayoutType: The compile-time `Layout` of the distribution function.
        BClayoutType: The compile-time `Layout` of the boundary-condition tensor.
        FlaglayoutType: The compile-time `Layout` of the `uint8` flag tensor.
        grid: The compile-time `LBM_Grid` describing the domain.
        config: The `LBM_Config` selecting DDF shift, Float16C, LES, and the
            valid boundary-condition flags.

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
    comptime assert config.lbm_method == LBM_method.ESOTERIC_PULL

    comptime D = grid.D
    comptime Q = grid.Q
    comptime float_dtype = grid.float_dtype
    comptime lattice = grid.lattice
    comptime directions = lattice.directions
    comptime grid_shape:InlineArray[Int,3] = grid.shape
    comptime non_temporal = True
    # comptime assert f_out.flat_rank == 8
    comptime assert not directions[0].all_true(), 'The first direction for the lattice model should be all 0s i.e directions[0]=[0,0,0]'

    x = block_idx.x*block_dim.x + thread_idx.x
    y = block_idx.y*block_dim.y + thread_idx.y
    z = block_idx.z*block_dim.z + thread_idx.z

    var index:InlineArray[Int,3] = [x,y,z]

    # Main Compute
    coord_index = coord[DType.int32]((index[0],index[1],index[2]))
    var flag = flags.load(coord_index)[0]

    if is_valid_thread(index,grid_shape,flag):
        # Streaming And Load Step
        var f_vec = Vector[float_dtype,Q](fill = 0)
        var pull_flags = InlineArray[UInt8,Q](uninitialized=True)
        stream[grid,config,is_even_time_step = is_even_time_step](f_vec,pull_flags,f,flags,flag,index,grid_shape)
        #BC
        apply_boundary_conditions[grid,config](f_vec,f,bc,flags,pull_flags,index,tau)
        # Collision Step + Any Turbulence modelling etc.
        collide[grid,config](f_vec,f,bc,flags,pull_flags,index,tau)

        # Store To Global
        store_f_vec_to_global[grid,config,is_even_time_step = is_even_time_step](f,f_vec,index)
        