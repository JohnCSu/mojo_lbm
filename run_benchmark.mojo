from std.gpu.host import DeviceContext
from layout import TileTensor
from layout.tile_layout import (row_major,col_major,TensorLayout,blocked_product)
from std.python import Python, PythonObject
from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from std.collections import InlineArray
from src.lbm import SOLID_NODE,FLUID_NODE,set_outer_walls,LBM_Grid,get_D2Q9,LBM_kernel
from src.lbm.variations import reorderThreads,tiled,base

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
comptime tile_size = 16
comptime BLOCK_SHAPE = (16,16,1)
comptime GRID_DIM = ((nx) // BLOCK_SHAPE[0],(ny) // BLOCK_SHAPE[1], 1 )# Plus one

comptime grid = LBM_Grid[D2Q9,nx,ny,nz](dx)

comptime U_phs:float_scalar = 1.
comptime U:float_scalar = 0.05
comptime viscosity:float_scalar = 1/100.
comptime dt = dx*U/U_phs 
comptime Re = 1/viscosity
comptime L_lat:float_scalar = N
comptime v_lat = U*L_lat/Re
comptime tau = v_lat/(1/3.) +0.5


comptime benchmark_1 = base.benchmark_func[grid,GRID_DIM,BLOCK_SHAPE,U,tau]
comptime benchmark_2 = reorderThreads.benchmark_func[grid,GRID_DIM,BLOCK_SHAPE,U,tau]
comptime benchmark_3a = tiled.benchmark_func_row_tile[grid,GRID_DIM,BLOCK_SHAPE,U,tau,tile_size]
comptime benchmark_3b = tiled.benchmark_func_col_tile[grid,GRID_DIM,BLOCK_SHAPE,U,tau,tile_size]
comptime benchmark_3c = tiled.benchmark_func_row_tile[grid,GRID_DIM,BLOCK_SHAPE,U,tau,tile_size,reorder_threads = False]
comptime benchmark_3d = tiled.benchmark_func_col_tile[grid,GRID_DIM,BLOCK_SHAPE,U,tau,tile_size,reorder_threads = False]

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

    bench.bench_function[benchmark_1](BenchId('Base Row Major LBM Kernel '))
    bench.bench_function[benchmark_2](BenchId('Base with Thread Reordering'))
    bench.bench_function[benchmark_3a](BenchId('Tiled 16x16 Layout Tile:Row major Tiler: Row major'))
    bench.bench_function[benchmark_3b](BenchId('Tiled 16x16 Layout Tile:Col major Tiler: Row major'))
    bench.bench_function[benchmark_3c](BenchId('Tiled 16x16 Layout Tile:Row major Tiler: Row major No Thread Reorder'))
    bench.bench_function[benchmark_3d](BenchId('Tiled 16x16 Layout Tile:Col major Tiler: Row major No Thread Reorder'))


    print(bench)