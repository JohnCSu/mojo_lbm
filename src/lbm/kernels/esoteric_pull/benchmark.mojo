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
    LatticeModel,
    set_exterior_walls,
    LBM_Config,
)
from src.lbm.preprocess import initialize_fluid_at_rest
from src.utils import Vector, ContextTileTensor
from .GPU_kernel import esoteric_pull_kernel
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep,run

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


@always_inline
def run_benchmark[
    float_dtype: DType,
    D: Int,
    Q: Int,
    lattice_model: LatticeModel[D, Q, float_dtype, DType.int32],
    nx: Int,
    ny: Int,
    nz: Int,
    tile_size: Int,
    //,
    grid: LBM_Grid[lattice_model, nx, ny, nz, tile_size],
    U: Scalar[float_dtype],
    tau: Scalar[float_dtype],
    simd_width: Int,
    f_layout: Layout[...],
    flag_layout: Layout[...],
    bc_layout: Layout[...],
    velocity_layout: Layout[...],
    density_layout: Layout[...],
    config: LBM_Config,
](mut b: Bencher) raises where (
    tile_size >= 1
    and f_layout.rank == 4
    and flag_layout.rank == 3
    and bc_layout.rank == 4
    and velocity_layout.rank == 4
    and density_layout.rank == 3
):
    """Benchmarks one SRT LBM time step for a given grid and layout.

    Allocates the flag, `bc`, `f`, and `f_out` buffers, fills `f` with a
    uniform rest distribution, applies four solid exterior walls (with a
    moving `+Y` lid at velocity `U`), copies the buffers to the GPU, compiles
    `double_buffer_kernel`, and times one in-place step.

    Parameters:
        grid: The compile-time `LBM_Grid` describing the domain.
        U: The lid velocity in lattice units.
        tau: The SRT relaxation time.
        simd_width: The SIMD width hint (currently unused).
        f_layout: The compile-time layout of the distribution function.
        flag_layout: The compile-time layout of the flag field.
        bc_layout: The compile-time layout of the boundary-condition field.
        velocity_layout: The compile-time layout of the velocity output.
        density_layout: The compile-time layout of the density output.
        config: The `LBM_Config` selecting runtime toggles.

    Args:
        b: The `Bencher` used to time the kernel.
    """
    comptime GRID_DIM: Tuple[Int, Int, Int] = grid.GRID_DIM
    comptime BLOCK_SHAPE: Tuple[Int, Int, Int] = grid.BLOCK_SHAPE
    comptime Float = Scalar[float_dtype]
    comptime f_dtype = config.f_dtype.value() if config.f_dtype else float_dtype
    ctx = DeviceContext()
    flags = ContextTileTensor[DType.uint8](ctx, flag_layout)
    bc = ContextTileTensor[float_dtype](ctx, bc_layout)
    f = ContextTileTensor[f_dtype](ctx, f_layout)

    initialize_fluid_at_rest[grid,config](f.cpu())

    set_exterior_walls[grid,config](flags.cpu(),bc.cpu(),'+Y',SOLID_NODE,[U,0,0],1.)
    set_exterior_walls[grid,config](flags.cpu(),bc.cpu(),'-Y',SOLID_NODE,[0,0,0],1.)
    set_exterior_walls[grid,config](flags.cpu(),bc.cpu(),'+X',SOLID_NODE,[0,0,0],1.)
    set_exterior_walls[grid,config](flags.cpu(),bc.cpu(),'-X',SOLID_NODE,[0,0,0],1.)
    ctx.synchronize()
    # Copy To GPU()
    _ = flags.gpu()
    _ = bc.gpu()
    _ = f.gpu()

    #Compile Functions
    comptime LBM_Even = esoteric_pull_kernel[True,f_layout,bc_layout,flag_layout,grid,config]
    comptime LBM_Odd = esoteric_pull_kernel[False,f_layout,bc_layout,flag_layout,grid,config]
    LBM_even_step = ctx.compile_function[LBM_Even,LBM_Even]()
    LBM_odd_step = ctx.compile_function[LBM_Odd,LBM_Odd]()

    ctx.synchronize()
    # Compile Functions
    comptime LBM_kernel_ = double_buffer_kernel[
        f_layout, bc_layout, flag_layout, grid, config
    ]
    LBM_func = ctx.compile_function[LBM_kernel_, LBM_kernel_]()
    ctx.synchronize()

    @always_inline
    def run_kernel(ctx: DeviceContext) capturing raises:
        ctx.enqueue_function(
            LBM_even_step,
            f.gpu(),
            bc.gpu().as_immut(),
            flags.gpu().as_immut(),
            Float(tau),
            grid_dim=GRID_DIM,
            block_dim=BLOCK_SHAPE,
        )
        ctx.enqueue_function(
            LBM_odd_step,
            f.gpu(),
            bc.gpu().as_immut(),
            flags.gpu().as_immut(),
            Float(tau),
            grid_dim=GRID_DIM,
            block_dim=BLOCK_SHAPE,
        )
        ctx.synchronize()

    b.iter_custom[run_kernel](ctx)
    keep(f.gpu_buffer().unsafe_ptr())
    keep(flags.gpu_buffer().unsafe_ptr())
    keep(bc.gpu_buffer().unsafe_ptr())
    ctx.synchronize()
