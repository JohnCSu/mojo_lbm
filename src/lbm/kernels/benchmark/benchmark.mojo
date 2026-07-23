"""Provides the SRT LBM benchmark harness.

`run_benchmark` allocates flag, `bc`, `f`, and `f_out` buffers for a given
grid and layout, applies a lid-driven-cavity-style wall setup, compiles
`double_buffer_kernel`, and times one in-place time step using the Mojo
benchmarking framework.

Provides the tiled-layout SRT LBM benchmark for 3D grids.

`benchmark_func_3D` builds column-major tile and tiler layouts for the flag,
`f`, and `bc` fields and delegates to `run_benchmark` with row-major density
and velocity outputs.
"""
from std.gpu.host import DeviceContext
from layout import TileTensor
from layout.tile_layout import (
    row_major,
    col_major,
    TensorLayout,
    blocked_product,
    Layout,
)
from std.gpu import block_dim, block_idx, thread_idx
from std.collections import InlineArray
from src.lbm import (
    SOLID_NODE,
    FLUID_NODE,
    LBM_Grid,
    get_D2Q9,
    Lattice,
    set_exterior_walls,
    LBM_Config,
)
from src.lbm.preprocess import initialize_fluid_at_rest
from src.utils import Vector, ContextTileTensor
# from .GPU_kernel import esoteric_pull_kernel
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep,run

from . import esoteric
from . import double_buffer


@always_inline
def benchmark_func_tiled_3D[
    float_dtype:DType,D:Int,Q:Int,
    lattice:Lattice[D,Q,float_dtype,DType.int32],
    nx:Int,ny:Int,nz:Int,
    //,
    lbm_method:StaticString,
    U:Scalar[float_dtype],
    tau:Scalar[float_dtype],
    grid: LBM_Grid[lattice,nx,ny,nz,...],
    config:LBM_Config = LBM_Config(),
    ]
    (mut b:Bencher) capturing raises:
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
    comptime assert D == 3
    comptime assert lbm_method in {'esoteric pull','double buffer'}
    # This can be stored in LBM Grid
    #(32,32,64), (8,8,4)
    
    comptime all_slice = slice(None,None,None)
    comptime simd_width = 4
    comptime flag_tile = col_major[grid.tile_shape[0],grid.tile_shape[1],grid.tile_shape[2]]()
    comptime f_tile = col_major[grid.tile_shape[0],grid.tile_shape[1],grid.tile_shape[2],Q]()
    comptime bc_tile = col_major[grid.tile_shape[0],grid.tile_shape[1],grid.tile_shape[2],D+1]()

    comptime flag_tiler = col_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z]()
    comptime f_tiler = col_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z,1]()
    comptime bc_tiler = col_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z,1]()

    comptime flag_layout = blocked_product(flag_tile,flag_tiler)
    comptime f_layout = blocked_product(f_tile,f_tiler)
    comptime bc_layout = blocked_product(bc_tile,bc_tiler)

    comptime density_layout = row_major[nx,ny,nz]()
    comptime velocity_layout = row_major[D,nx,ny,nz]()

    comptime if lbm_method == 'esoteric pull':
        esoteric.run_benchmark[grid,U,tau,simd_width,f_layout,flag_layout,bc_layout,velocity_layout,density_layout,config](b)
    comptime if lbm_method == 'double buffer':
        double_buffer.run_benchmark[grid,U,tau,simd_width,f_layout,flag_layout,bc_layout,velocity_layout,density_layout,config](b)


@always_inline
def benchmark_func_3D_non_tiled[
    float_dtype:DType,D:Int,Q:Int,
    lattice:Lattice[D,Q,float_dtype,DType.int32],
    nx:Int,ny:Int,nz:Int,
    //,
    lbm_method:StaticString,
    U:Scalar[float_dtype],
    tau:Scalar[float_dtype],
    grid: LBM_Grid[lattice,nx,ny,nz,_],
    config:LBM_Config = LBM_Config(),
    ]
    (mut b:Bencher) capturing raises:
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

    comptime assert D == 3
    comptime assert lbm_method in {'esoteric pull','double buffer'}
    # This can be stored in LBM Grid
    comptime all_slice = slice(None,None,None)
    comptime simd_width = 4

    comptime flag_layout = col_major[nx,ny,nz]()
    comptime f_layout = col_major[nx,ny,nz,Q]()
    comptime bc_layout = col_major[nx,ny,nz,D+1]()

    comptime density_layout = row_major[nx,ny,nz]()
    comptime velocity_layout = row_major[D,nx,ny,nz]()

    comptime if lbm_method == 'esoteric pull':
        esoteric.run_benchmark[grid,U,tau,simd_width,f_layout,flag_layout,bc_layout,velocity_layout,density_layout,config](b)
    comptime if lbm_method == 'double buffer':
        double_buffer.run_benchmark[grid,U,tau,simd_width,f_layout,flag_layout,bc_layout,velocity_layout,density_layout,config](b)



