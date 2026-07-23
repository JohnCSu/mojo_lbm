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
from src.lbm.kernels import esoteric_pull_kernel
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep,run


@always_inline
def run_benchmark[
    float_dtype: DType,
    D: Int,
    Q: Int,
    lattice_model: Lattice[D, Q, float_dtype, DType.int32],
    nx: Int,
    ny: Int,
    nz: Int,
    //,
    grid: LBM_Grid[lattice_model, nx, ny, nz,_],
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
    f_layout.rank == 4
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
    # """
    # comptime GRID_DIM:Tuple[Int,Int,Int] = (grid.GRID_DIM[0],grid.GRID_DIM[1],grid.GRID_DIM[2]*2)
    # comptime BLOCK_SHAPE:Tuple[Int,Int,Int] = (grid.BLOCK_SHAPE[0],grid.BLOCK_SHAPE[1],grid.BLOCK_SHAPE[2]//2)

    comptime GRID_DIM: Tuple[Int, Int, Int] = grid.GRID_DIM
    comptime BLOCK_SHAPE: Tuple[Int, Int, Int] = grid.BLOCK_SHAPE
    # print('Kernel dims: ',GRID_DIM,BLOCK_SHAPE)
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
