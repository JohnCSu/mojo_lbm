"""Provides the tiled-layout SRT LBM benchmark for 3D grids.

`benchmark_func_3D` builds column-major tile and tiler layouts for the flag,
`f`, and `bc` fields and delegates to `run_benchmark` with row-major density
and velocity outputs.
"""
from .benchmark import run_benchmark
from layout import TileTensor
from layout.tile_layout import (row_major,col_major,TensorLayout,blocked_product,Layout)
from std.python import Python, PythonObject
from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from std.collections import InlineArray
from src.lbm import SOLID_NODE,FLUID_NODE,LBM_Grid,get_D2Q9,LatticeModel,set_exterior_walls,LBM_Config
from .LBM_gpu_kernel import LBM_kernel
from src.utils import Vector,ContextTileTensor
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep,run
from std.utils import Variant

@always_inline
def benchmark_func_3D[
    float_dtype:DType,D:Int,Q:Int,
    lattice_model:LatticeModel[D,Q,float_dtype,DType.int32],
    nx:Int,ny:Int,nz:Int,
    tile_size:Int,
    //,
    grid: LBM_Grid[lattice_model,nx,ny,nz,tile_size],
    U:Scalar[float_dtype],
    tau:Scalar[float_dtype],
    config:LBM_Config = LBM_Config(),
    *,
    reorder_threads:Bool = True
    ]
    (mut b:Bencher) capturing raises where tile_size >= 1:
    """Benchmarks one SRT LBM time step on a 3D tiled layout.

    Builds column-major `(tile_size, tile_size, tile_size[, Q|D+1])` tiles
    composed with column-major tilers via `blocked_product`, then delegates
    to `run_benchmark` with row-major density and velocity outputs.

    Parameters:
        grid: The compile-time `LBM_Grid` describing the domain.
        U: The lid velocity in lattice units.
        tau: The SRT relaxation time.
        config: The `LBM_Config` selecting runtime toggles (defaults to a
            fresh `LBM_Config()`).
        reorder_threads: Reserved; currently unused (defaults to `True`).

    Args:
        b: The `Bencher` used to time the kernel.
    """
    comptime assert tile_size > 1 and D == 3
    # This can be stored in LBM Grid
    comptime GRID_DIM:Tuple[Int,Int,Int] = grid.GRID_DIM
    comptime BLOCK_SHAPE:Tuple[Int,Int,Int] = grid.BLOCK_SHAPE
    comptime all_slice = slice(None,None,None)
    comptime simd_width = 4
    comptime flag_tile = col_major[tile_size,tile_size,tile_size]()
    comptime f_tile = col_major[tile_size,tile_size,tile_size,Q]()
    comptime bc_tile = col_major[tile_size,tile_size,tile_size,D+1]()

    comptime flag_tiler = col_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z]()
    comptime f_tiler = col_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z,1]()
    comptime bc_tiler = col_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z,1]()

    comptime flag_layout = blocked_product(flag_tile,flag_tiler)
    comptime f_layout = blocked_product(f_tile,f_tiler)
    comptime bc_layout = blocked_product(bc_tile,bc_tiler)

    comptime density_layout = row_major[nx,ny,nz]()
    comptime velocity_layout = row_major[D,nx,ny,nz]()

    run_benchmark[grid,U,tau,simd_width,f_layout,flag_layout,bc_layout,velocity_layout,density_layout,config](b)
