from std.gpu.host import DeviceContext
from layout import TileTensor
from layout.tile_layout import (row_major,col_major,TensorLayout,blocked_product)
from std.python import Python, PythonObject
from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from std.collections import InlineArray
from src.lbm import SOLID_NODE,FLUID_NODE,set_exterior_walls,LBM_Grid,get_D2Q9
from src.lbm.archive.part_1 import reorderThreads,tiled,branchless,immutable_inputs,loop_unroll,tiled_no_layout
from src.lbm.archive.part_2 import tensor_loads,prefetch,tiled_no_layout_immut,SoA_Tile,AoS_Tile
from src.lbm.archive import base

from src.utils import Vector,ContextTileTensor
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep,run

comptime float_dtype = DType.float32
comptime int_dtype = DType.int32
comptime float_scalar = Scalar[float_dtype]
comptime D2Q9 = get_D2Q9[DType.float32,DType.int32]()
comptime D,Q = (2,9)
comptime N = 4096
comptime L = 1.
comptime dx = L/float_scalar(N-1)
comptime (nx,ny,nz) = (N,N,1)
comptime num_points = nx*ny*nz

comptime tile_size = 16
comptime grid = LBM_Grid[D2Q9,nx,ny,nz,tile_size](dx)

comptime BLOCK_SHAPE = grid.BLOCK_SHAPE
comptime GRID_DIM = grid.GRID_DIM

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
comptime benchmark_3a = tiled.benchmark_func_row_tile[grid,GRID_DIM,BLOCK_SHAPE,U,tau]
comptime benchmark_3b = tiled.benchmark_func_col_tile[grid,GRID_DIM,BLOCK_SHAPE,U,tau]
comptime benchmark_3c = tiled.benchmark_func_row_tile[grid,GRID_DIM,BLOCK_SHAPE,U,tau,reorder_threads = False]
comptime benchmark_3d = tiled.benchmark_func_col_tile[grid,GRID_DIM,BLOCK_SHAPE,U,tau,reorder_threads = False]
comptime benchmark_4 = loop_unroll.benchmark_func[grid,GRID_DIM,BLOCK_SHAPE,U,tau]
comptime benchmark_5 = branchless.benchmark_func[grid,GRID_DIM,BLOCK_SHAPE,U,tau]
comptime benchmark_6 = immutable_inputs.benchmark_func[grid,GRID_DIM,BLOCK_SHAPE,U,tau]

comptime benchmark_7 = tensor_loads.benchmark_func[grid,GRID_DIM,BLOCK_SHAPE,U,tau]
comptime benchmark_8 = tiled_no_layout.benchmark_func_col_tile[grid,GRID_DIM,BLOCK_SHAPE,U,tau]
comptime benchmark_9 = prefetch.benchmark_func[grid,GRID_DIM,BLOCK_SHAPE,U,tau]
comptime benchmark_8b = tiled_no_layout_immut.benchmark_func_col_tile[grid,GRID_DIM,BLOCK_SHAPE,U,tau]
comptime benchmark_10 = SoA_Tile.benchmark_func[grid,GRID_DIM,BLOCK_SHAPE,U,tau]
comptime benchmark_10b = SoA_Tile.benchmark_func[grid,GRID_DIM,BLOCK_SHAPE,U,tau,reorder_threads = False]
comptime benchmark_11 = AoS_Tile.benchmark_func[grid,GRID_DIM,BLOCK_SHAPE,U,tau]
comptime benchmark_11b = AoS_Tile.benchmark_func[grid,GRID_DIM,BLOCK_SHAPE,U,tau,reorder_threads = False]


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

    print('Base Unoptimized Layout Changes ')
    bench.bench_function[benchmark_1](BenchId('1. Base Row Major LBM Kernel '))
    
    print('Part 1 Layout Changes ')
    bench.bench_function[benchmark_2](BenchId('2. Base with Thread Reordering'))
    bench.bench_function[benchmark_3a](BenchId('3a Tiled 16x16 Layout Tile:Row major Tiler: Row major'))
    bench.bench_function[benchmark_3b](BenchId('3b. Tiled 16x16 Layout Tile:Col major Tiler: Row major'))
    bench.bench_function[benchmark_3c](BenchId('3c. Tiled 16x16 Layout Tile:Row major Tiler: Row major No Thread Reorder'))
    bench.bench_function[benchmark_3d](BenchId('3d. Tiled 16x16 Layout Tile:Col major Tiler: Row major No Thread Reorder'))

    print('Part 2 Algorithim Changes')
    bench.bench_function[benchmark_4](BenchId('4. Tiled 16x16 Layout Col/Row Loop Unroll'))
    bench.bench_function[benchmark_5](BenchId('5. Tiled 16x16 Layout Col/Row Loop Unroll branchless'))
    bench.bench_function[benchmark_6](BenchId('6. Tiled 16x16 Layout Col/Row Loop Unroll branchless Immutable'))


    print('Part 3 Additional Analysis')
    bench.bench_function[benchmark_7](BenchId('7. Benchmark_6 but with TileTensor.load and .store'))
    bench.bench_function[benchmark_8](BenchId('8. Tiled 16x16 Layout Tile:Col major Tiler: Row major Nested Indexing'))
    bench.bench_function[benchmark_8b](BenchId('8b. Tiled 16x16 Layout Tile:Col major Tiler: Row major Nested Indexing With Immut Inputs'))
    bench.bench_function[benchmark_9](BenchId('9. Benchmark 8 with Prefetching flags and BC'))
    bench.bench_function[benchmark_10](BenchId('10. Col Major SoA Tile Thread Reordering'))
    bench.bench_function[benchmark_10b](BenchId('10b. Col Major SoA Tile No Reordering'))
    bench.bench_function[benchmark_11](BenchId('11. Col Major AoS Tile Thread Reordering'))
    bench.bench_function[benchmark_11b](BenchId('11b. Col Major AoS Tile No Reordering'))
    
    print(bench)