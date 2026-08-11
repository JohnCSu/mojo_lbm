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

from src.lbm import LBM_Config,Lattice,GridLike,LBM_Grid,RuntimeParams
from src.lbm.constants import Flags,cs_squared,Collisions

from src.lbm.kernels.utils.checks import is_valid_thread
from src.lbm.kernels.utils.index import get_adjacent_idx
from src.lbm.kernels.utils.load_and_store import store_f

from src.utils import Vector
from src.lbm.kernels.steps import stream,collide,apply_boundary_conditions


def double_buffer_kernel[
    FlayoutType:TensorLayout,
    BClayoutType:TensorLayout,
    FlaglayoutType:TensorLayout,
    grid: LBM_Grid,
    config:LBM_Config,
    ]
    (
    f_out:TileTensor[config.set_f_dtype(grid.float_dtype),FlayoutType,MutAnyOrigin],
    f_in:TileTensor[config.set_f_dtype(grid.float_dtype),FlayoutType,ImmutAnyOrigin],
    bc:TileTensor[grid.float_dtype,BClayoutType,ImmutAnyOrigin],
    flags:TileTensor[DType.uint8,FlaglayoutType,ImmutAnyOrigin],
    tau:Scalar[grid.float_dtype],
    # params:RuntimeParams[grid.float_dtype]
    )
    where FlayoutType.rank == 4 and BClayoutType.rank == 4 and FlaglayoutType.rank == 3:
    """Runs one SRT LBM time step from `f_in` into `f_out`.

    Performs pull-scheme streaming with mid-grid bounce-back at solid nodes,
    optional equilibrium boundary-condition handling, density and velocity
    extraction, optional Smagorinsky LES relaxation-time correction, and the
    BGK collision. The result is written to `f_out`; the caller swaps
    `f_in` and `f_out` between calls.

    Parameters:
        FlayoutType: The compile-time `Layout` of the distribution function.
        BClayoutType: The compile-time `Layout` of the boundary-condition tensor.
        FlaglayoutType: The compile-time `Layout` of the `uint8` flag tensor.
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
        stream[grid,config](f_vec,pull_flags,f_in,flags,flag,index,grid_shape)
        #BC
        apply_boundary_conditions[grid,config](f_vec,f_in,bc,flags,pull_flags,index,tau)
        # Collision Step + Any Turbulence modelling etc.
        collide[grid,config](f_vec,f_in,bc,flags,pull_flags,index,tau)

        # Store f back to Global
        comptime for q in range(Q):
            store_f[config.use_float16c,non_temporal](f_out,f_vec[q],index,q)

