from std.gpu.host import DeviceContext
from layout import TileTensor
from layout.tile_layout import (row_major,col_major,TensorLayout,blocked_product)
from std.python import Python, PythonObject
from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from std.collections import InlineArray
from src.lbm import SOLID_NODE,FLUID_NODE,set_outer_walls,LBM_Grid,get_D2Q9,LBM_kernel
from src.lbm.variations import reorderThreads

from src.utils import Vector,ContextTileTensor
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep,run

comptime float_dtype = DType.float32
comptime int_dtype = DType.int32
comptime float_scalar = Scalar[float_dtype]
comptime D2Q9 = get_D2Q9[DType.float32,DType.int32]()
comptime D,Q = (2,9)
comptime N = 4048
comptime L = 1.
comptime dx = L/float_scalar(N-1)
comptime (nx,ny,nz) = (N,N,1)
comptime num_points = nx*ny*nz

comptime THREADS_PER_BLOCK = 256
comptime BLOCK_SHAPE = (16,16,1)
comptime GRID_DIM = ((nx) // BLOCK_SHAPE[0]+1,(ny) // BLOCK_SHAPE[1]+1, 1 )# Plus one

comptime grid = LBM_Grid[D2Q9,nx,ny,nz](dx)

comptime U_phs:float_scalar = 1.
comptime U:float_scalar = 0.05
comptime viscosity:float_scalar = 1/100.
comptime dt = dx*U/U_phs 
comptime Re = 1/viscosity
comptime L_lat:float_scalar = N
comptime v_lat = U*L_lat/Re
comptime tau = v_lat/(1/3.) +0.5

@always_inline
def benchmark_row_major_LBM_FusedKernel(mut b:Bencher) capturing raises:
    # This can be stored in LBM Grid
    comptime flag_layout = row_major[nx,ny,nz]()
    comptime f_layout = row_major[Q,nx,ny,nz]()
    comptime bc_layout = row_major[nx,ny,nz,D+1]()
    comptime density_layout = row_major[nx,ny,nz]()
    comptime velocity_layout = row_major[D,nx,ny,nz]()

    ctx = DeviceContext()
    flags = ContextTileTensor[DType.uint8](ctx,flag_layout)
    bc = ContextTileTensor[float_dtype](ctx,bc_layout)
    f = ContextTileTensor[float_dtype](ctx,f_layout)
    f_out = ContextTileTensor[float_dtype](ctx,f_layout)

    u = ContextTileTensor[float_dtype](ctx,velocity_layout)
    rho = ContextTileTensor[float_dtype](ctx,density_layout)
    # Set up
    f.fill(1./Float32(Q))
    f_out.fill(1./Float32(Q))

    set_outer_walls[D,nx,ny,nz](flags.cpu(),bc.cpu(),'+Y',SOLID_NODE,[U,0],1.)
    set_outer_walls[D,nx,ny,nz](flags.cpu(),bc.cpu(),'-Y',SOLID_NODE,[0,0],1.)
    set_outer_walls[D,nx,ny,nz](flags.cpu(),bc.cpu(),'-X',SOLID_NODE,[0,0],1.)
    set_outer_walls[D,nx,ny,nz](flags.cpu(),bc.cpu(),'+X',SOLID_NODE,[0,0],1.)

    ctx.synchronize()
    # Copy To GPU()
    _ = flags.gpu()
    _ = bc.gpu()
    _ = f.gpu()
    _ = f_out.gpu()

    ctx.synchronize()
    #Compile Functions
    LBM_func = ctx.compile_function[LBM_kernel[grid,f_layout,bc_layout,flag_layout],LBM_kernel[grid,f_layout,bc_layout,flag_layout]]()
    # calc_rho_and_u_gpu = ctx.compile_function[calculate_rho_and_velocity[grid,f_layout,density_layout,velocity_layout],calculate_rho_and_velocity[grid,f_layout,density_layout,velocity_layout]]()
    ctx.synchronize()
    

    @always_inline
    def run_kernel(ctx:DeviceContext) capturing raises:
        ctx.enqueue_function(LBM_func,f_out.gpu(),f.gpu(),bc.gpu(),flags.gpu(),Float32(1/tau),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)

    b.iter_custom[run_kernel](ctx)
    keep(f_out.gpu_buffer().unsafe_ptr())
    keep(f.gpu_buffer().unsafe_ptr())
    keep(flags.gpu_buffer().unsafe_ptr())
    keep(bc.gpu_buffer().unsafe_ptr())
    ctx.synchronize()


@always_inline
def benchmark_row_major_ThreadReorder_LBM_FusedKernel(mut b:Bencher) capturing raises:
    # This can be stored in LBM Grid
    comptime flag_layout = row_major[nx,ny,nz]()
    comptime f_layout = row_major[Q,nx,ny,nz]()
    comptime bc_layout = row_major[nx,ny,nz,D+1]()
    comptime density_layout = row_major[nx,ny,nz]()
    comptime velocity_layout = row_major[D,nx,ny,nz]()

    ctx = DeviceContext()
    flags = ContextTileTensor[DType.uint8](ctx,flag_layout)
    bc = ContextTileTensor[float_dtype](ctx,bc_layout)
    f = ContextTileTensor[float_dtype](ctx,f_layout)
    f_out = ContextTileTensor[float_dtype](ctx,f_layout)

    u = ContextTileTensor[float_dtype](ctx,velocity_layout)
    rho = ContextTileTensor[float_dtype](ctx,density_layout)
    # Set up
    f.fill(1./Float32(Q))
    f_out.fill(1./Float32(Q))

    set_outer_walls[D,nx,ny,nz](flags.cpu(),bc.cpu(),'+Y',SOLID_NODE,[U,0],1.)
    set_outer_walls[D,nx,ny,nz](flags.cpu(),bc.cpu(),'-Y',SOLID_NODE,[0,0],1.)
    set_outer_walls[D,nx,ny,nz](flags.cpu(),bc.cpu(),'-X',SOLID_NODE,[0,0],1.)
    set_outer_walls[D,nx,ny,nz](flags.cpu(),bc.cpu(),'+X',SOLID_NODE,[0,0],1.)

    ctx.synchronize()
    # Copy To GPU()
    _ = flags.gpu()
    _ = bc.gpu()
    _ = f.gpu()
    _ = f_out.gpu()

    ctx.synchronize()
    #Compile Functions
    LBM_func = ctx.compile_function[reorderThreads.LBM_kernel[grid,f_layout,bc_layout,flag_layout],reorderThreads.LBM_kernel[grid,f_layout,bc_layout,flag_layout]]()
    # calc_rho_and_u_gpu = ctx.compile_function[calculate_rho_and_velocity[grid,f_layout,density_layout,velocity_layout],calculate_rho_and_velocity[grid,f_layout,density_layout,velocity_layout]]()
    ctx.synchronize()
    
    @always_inline
    def run_kernel(ctx:DeviceContext) capturing raises:
        ctx.enqueue_function(LBM_func,f_out.gpu(),f.gpu(),bc.gpu(),flags.gpu(),Float32(1/tau),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)

    b.iter_custom[run_kernel](ctx)
    keep(f_out.gpu_buffer().unsafe_ptr())
    keep(f.gpu_buffer().unsafe_ptr())
    keep(flags.gpu_buffer().unsafe_ptr())
    keep(bc.gpu_buffer().unsafe_ptr())
    ctx.synchronize()



@always_inline
def benchmark_16x16_Tiled_LBM_FusedKernel(mut b:Bencher) capturing raises:
    comptime tile_shape = 16
    comptime n_tiles = N//tile_shape
    comptime assert N % tile_shape == 0 and N > 0

    # blocked product ss bugged rn
    comptime flag_tile = row_major[tile_shape,tile_shape,1]()
    comptime f_tile = row_major[1,tile_shape,tile_shape,1]()
    comptime bc_tile = row_major[tile_shape,tile_shape,1,1]()

    comptime flag_layout = blocked_product(flag_tile,col_major[n_tiles,n_tiles,1]())
    comptime f_layout = blocked_product(f_tile,col_major[Q,n_tiles,n_tiles,1]())
    comptime bc_layout = blocked_product(bc_tile,col_major[n_tiles,n_tiles,1,D+1]())

    comptime density_layout = row_major[nx,ny,nz]()
    comptime velocity_layout = row_major[D,nx,ny,nz]()

    ctx = DeviceContext()
    flags = ContextTileTensor[DType.uint8](ctx,flag_layout)
    bc = ContextTileTensor[float_dtype](ctx,bc_layout)
    f = ContextTileTensor[float_dtype](ctx,f_layout)
    f_out = ContextTileTensor[float_dtype](ctx,f_layout)

    u = ContextTileTensor[float_dtype](ctx,velocity_layout)
    rho = ContextTileTensor[float_dtype](ctx,density_layout)
    # Set up
    f.fill(1./Float32(Q))
    f_out.fill(1./Float32(Q))

    set_outer_walls[D,nx,ny,nz](flags.cpu(),bc.cpu(),'+Y',SOLID_NODE,[U,0],1.)
    set_outer_walls[D,nx,ny,nz](flags.cpu(),bc.cpu(),'-Y',SOLID_NODE,[0,0],1.)
    set_outer_walls[D,nx,ny,nz](flags.cpu(),bc.cpu(),'-X',SOLID_NODE,[0,0],1.)
    set_outer_walls[D,nx,ny,nz](flags.cpu(),bc.cpu(),'+X',SOLID_NODE,[0,0],1.)

    ctx.synchronize()
    # Copy To GPU()
    _ = flags.gpu()
    _ = bc.gpu()
    _ = f.gpu()
    _ = f_out.gpu()

    ctx.synchronize()
    #Compile Functions
    LBM_func = ctx.compile_function[LBM_kernel[grid,f_layout,bc_layout,flag_layout],LBM_kernel[grid,f_layout,bc_layout,flag_layout]]()
    # calc_rho_and_u_gpu = ctx.compile_function[calculate_rho_and_velocity[grid,f_layout,density_layout,velocity_layout],calculate_rho_and_velocity[grid,f_layout,density_layout,velocity_layout]]()
    ctx.synchronize()
    

    @always_inline
    def run_kernel(ctx:DeviceContext) capturing raises:
        ctx.enqueue_function(LBM_func,f_out.gpu(),f.gpu(),bc.gpu(),flags.gpu(),Float32(1/tau),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)

    b.iter_custom[run_kernel](ctx)
    keep(f_out.gpu_buffer().unsafe_ptr())
    keep(f.gpu_buffer().unsafe_ptr())
    keep(flags.gpu_buffer().unsafe_ptr())
    keep(bc.gpu_buffer().unsafe_ptr())
    ctx.synchronize()


def main() raises:
    total_bytes =  Q*num_points*2*4 + num_points*(D+1)*4 + num_points # 4btes per Q (fp32) , 4 byters per bc (fp32) , 1 byte per flag (fp) 
    print('Benchmark for fp32/fp32 LBM')
    print('Grid Shape: {},{},{}'.format(nx,ny,nz))
    print('Num Points On grid: {}'.format(num_points))
    print('Approximate Total Bytes {} or MB {}'.format(total_bytes,Float64(total_bytes)/1e6))
    print(GRID_DIM)
    print(BLOCK_SHAPE)
    print('Tau {}'.format(tau))

    var bench_config = BenchConfig(max_iters=20, num_warmup_iters=1)
    var bench = Bench(bench_config.copy())

    bench.bench_function[benchmark_row_major_LBM_FusedKernel](BenchId('Base Row Major LBM Kernel '))
    bench.bench_function[benchmark_row_major_ThreadReorder_LBM_FusedKernel](BenchId('Base with Thread Reordering'))
    print(bench)