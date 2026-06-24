from std.gpu.host import DeviceContext
from layout import TileTensor
from layout.tile_layout import (row_major,col_major,TensorLayout,blocked_product)
from std.python import Python, PythonObject
from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from std.collections import InlineArray
from src.lbm import SOLID_NODE,FLUID_NODE,set_outer_walls,LBM_Grid,get_D3Q27
from src.lbm.archive.part_3 import base
from src.lbm.archive.part_3 import sharedmemory_p1, sharedmemory_all
# from src.lbm.variations import base

from src.utils import Vector,ContextTileTensor
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep,run

comptime float_dtype = DType.float32
comptime int_dtype = DType.int32
comptime float_scalar = Scalar[float_dtype]
comptime D3Q27 = get_D3Q27[DType.float32,DType.int32]()
comptime D,Q = (D3Q27.D,D3Q27.Q)
comptime N = 256
comptime L = 1.
comptime dx = L/float_scalar(N-1)
comptime (nx,ny,nz) = (N,N,N)
comptime num_points = nx*ny*nz

comptime tile_size = 8
comptime tiled_grid = LBM_Grid[D3Q27,nx,ny,nz,tile_size](dx)
comptime non_tiled_grid = LBM_Grid[D3Q27,nx,ny,nz,1](dx)

comptime U_phs:float_scalar = 1.
comptime U:float_scalar = 0.05
comptime viscosity:float_scalar = 1/100.
comptime dt = dx*U/U_phs 
comptime Re = 1/viscosity
comptime L_lat:float_scalar = N
comptime v_lat = U*L_lat/Re
comptime tau = v_lat/(1/3.) +0.5

comptime benchmark_1 = base.benchmark_func_row_major_AoS[non_tiled_grid,U,tau]
comptime benchmark_2 = base.benchmark_func_col_major_SoA[non_tiled_grid,U,tau]

comptime benchmark_3 = base.benchmark_func_col_tile_row_tiler[tiled_grid,U,tau]
comptime benchmark_4 = base.benchmark_func_row_tile_col_tiler[tiled_grid,U,tau]

comptime benchmark_5 = base.benchmark_func_col_tile_col_tiler[tiled_grid,U,tau]
comptime benchmark_6 = base.benchmark_func_row_tile_row_tiler[tiled_grid,U,tau]

comptime benchmark_7 = sharedmemory_p1.benchmark_func_col_tile_col_tiler[tiled_grid,U,tau]
comptime benchmark_8 = sharedmemory_all.benchmark_func_col_tile_col_tiler[tiled_grid,U,tau]

def main() raises:
    total_bytes =  Q*num_points*2*4 + num_points*(D+1)*4 + num_points # 4btes per Q (fp32) , 4 byters per bc (fp32) , 1 byte per flag (fp) 
    print('Benchmark for fp32/fp32 LBM')
    print('Grid Shape: {},{},{}'.format(nx,ny,nz))
    print('Num Points On grid: {}'.format(num_points))
    print('Approximate Total Bytes {} or MB {}'.format(total_bytes,Float64(total_bytes)/1e6))
    print('Non Tiled Grid Dim: {} Block_Shape {} '.format(non_tiled_grid.GRID_DIM,non_tiled_grid.BLOCK_SHAPE))
    print('Tiled Grid Dim: {} Block_Shape {} '.format(tiled_grid.GRID_DIM,tiled_grid.BLOCK_SHAPE))
    print('Tau {}'.format(tau))

    var bench_config = BenchConfig(max_iters=10, num_warmup_iters=1)
    var bench = Bench(bench_config.copy())

    print('All Indexing assumes of the form: (x,y,z,q)')
    # bench.bench_function[benchmark_1](BenchId('1. Base Row Major AoS'))
    bench.bench_function[benchmark_2](BenchId('2. Base Col Major SoA'))
    bench.bench_function[benchmark_3](BenchId('3. Tile Col, Tiler Row'))
    bench.bench_function[benchmark_4](BenchId('4. Tile Row, Tile Col'))
    bench.bench_function[benchmark_5](BenchId('5. Tile Col, Tiler Col'))
    bench.bench_function[benchmark_6](BenchId('6. Tile Row, Tiler Row'))
    bench.bench_function[benchmark_7](BenchId('7. Shared Memory For Flags Global Pull if on boundary'))
    bench.bench_function[benchmark_8](BenchId('8. Shared Memory For Flags No Global Pull'))
    

    
    print(bench)