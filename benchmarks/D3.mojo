from std.gpu.host import DeviceContext
from layout import TileTensor
from layout.tile_layout import (row_major,col_major,TensorLayout,blocked_product)
from std.python import Python, PythonObject
from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from std.collections import InlineArray
from src.lbm import SOLID_NODE,FLUID_NODE,set_exterior_walls,LBM_Grid,get_D3Q19,get_D3Q27,LBM_Config
from src.lbm.archive.part_3 import base
from src.lbm.kernels import double_buffer,esoteric_pull

from src.utils import Vector,ContextTileTensor
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep,run
import std.sys as sys

comptime float_dtype = DType.float32
comptime int_dtype = DType.int32
comptime float_scalar = Scalar[float_dtype]
comptime N = 256
comptime L = 1.
comptime dx = L/float_scalar(N-1)
comptime (nx,ny,nz) = (N,N,N)
comptime tile_size = 4

# For D3Q19
comptime D3Q19 = get_D3Q19[float_dtype,DType.int32]()
comptime D,Q = (D3Q19.D,D3Q19.Q)
comptime tiled_grid = LBM_Grid[D3Q19,nx,ny,nz,tile_size](dx)
comptime non_tiled_grid = LBM_Grid[D3Q19,nx,ny,nz,1](dx)

# # -------- For D3Q27 Becnhmarks Uncomment below and comment above -----------
# comptime D3Q27 = get_D3Q27[DType.float32,DType.int32]()
# comptime D,Q = (D3Q27.D,D3Q27.Q)
# comptime tiled_grid = LBM_Grid[D3Q27,nx,ny,nz,tile_size](dx)
# comptime non_tiled_grid = LBM_Grid[D3Q27,nx,ny,nz,1](dx)

comptime num_points = nx*ny*nz
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


comptime benchmark_9 = double_buffer.benchmark_func_3D[tiled_grid,U,tau,LBM_Config()]
comptime benchmark_10 = double_buffer.benchmark_func_3D[tiled_grid,U,tau,LBM_Config(use_float16c = True,DDF_shift = True)]
comptime benchmark_11 = double_buffer.benchmark_func_3D[tiled_grid,U,tau,LBM_Config(LES = True,DDF_shift = True)]


comptime benchmark_12 = esoteric_pull.benchmark_func_3D[tiled_grid,U,tau,LBM_Config()]
comptime benchmark_13 = esoteric_pull.benchmark_func_3D[tiled_grid,U,tau,LBM_Config(use_float16c = True,DDF_shift = True)]
# comptime benchmark_14 = esoteric_pull.benchmark_func_3D[tiled_grid,U,tau,LBM_Config(LES = True,DDF_shift = True)]
comptime benchmark_15 = esoteric_pull.benchmark_func_3D_non_tiled[non_tiled_grid,U,tau,LBM_Config(LES = True,DDF_shift = True)]

def main() raises:
    ctx = DeviceContext()
    total_bytes =  Q*num_points*2*4 + num_points*(D+1)*4 + num_points # 4btes per Q (fp32) , 4 byters per bc (fp32) , 1 byte per flag (fp) 
    print('{}^3 LDC Cube at Re=100 Benchmark for fp32/fp32 D{}Q{} LBM'.format(N,D,Q))
    print('Running On GPU Device: {}'.format(ctx.name()))
    print("Mojo Version: {}.{}.{}".format(sys.defines.MojoVersion().major, sys.defines.MojoVersion().minor,sys.defines.MojoVersion().patch))
    print('Grid Shape: {},{},{}'.format(nx,ny,nz))
    print('Total Number of Points On grid: {}'.format(num_points))
    print('Approximate Total Bytes {} or {} MB'.format(total_bytes,Float64(total_bytes)/1e6))
    print('Non Tiled GPU Launch: Grid Dim: {} Block_Shape {} '.format(non_tiled_grid.GRID_DIM,non_tiled_grid.BLOCK_SHAPE))
    print('Tiled GPU Launch: Grid Dim: {} Block_Shape {} '.format(tiled_grid.GRID_DIM,tiled_grid.BLOCK_SHAPE))
    print('All Indexing assumes of the form: (x,y,z,q)')

    var bench_config = BenchConfig(max_iters=2, num_warmup_iters=1)
    var bench = Bench(bench_config.copy())
    # bench.bench_function[benchmark_1](BenchId('1. Base Row Major AoS'))
    # bench.bench_function[benchmark_2](BenchId('2. Base Col Major SoA'))
    # bench.bench_function[benchmark_3](BenchId('3. Tile Col, Tiler Row'))
    # bench.bench_function[benchmark_4](BenchId('4. Tile Row, Tile Col'))
    # bench.bench_function[benchmark_5](BenchId('5. Tile Col, Tiler Col'))
    # bench.bench_function[benchmark_6](BenchId('6. Tile Row, Tiler Row'))
    # bench.bench_function[benchmark_7](BenchId('7. Shared Memory For Flags tile, Global Pull For boundary'))
    # bench.bench_function[benchmark_8](BenchId('8. Map Flags + Halo region to Shared'))
    # bench.bench_function[benchmark_9](BenchId('9. LBM with Default LBM_Config'))
    # bench.bench_function[benchmark_10](BenchId('10. LBM float16c + DDF_shift'))
    # bench.bench_function[benchmark_11](BenchId('11. LES + DDF_Shift'))

    # bench.bench_function[benchmark_12](BenchId('12. Esoteric Pull LBM with Default LBM_Config'))
    bench.bench_function[benchmark_13](BenchId('13. Esoteric Pull LBM float16c + DDF_shift'))
    # bench.bench_function[benchmark_14](BenchId('14. Esoteric Pull LES + DDF_Shift'))
    # bench.bench_function[benchmark_15](BenchId('15. float16c + DDF_Shift Col Major'))
    print(bench)