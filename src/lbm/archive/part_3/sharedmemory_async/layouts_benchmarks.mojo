from .benchmark import run_benchmark
from layout import TileTensor
from layout.tile_layout import (row_major,col_major,TensorLayout,blocked_product,Layout)
from std.python import Python, PythonObject
from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from std.collections import InlineArray
from src.lbm import SOLID_NODE,FLUID_NODE,LBM_Grid,get_D2Q9,Lattice,set_exterior_walls
from .LBM_gpu_kernel import LBM_kernel
from src.utils import Vector,ContextTileTensor
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep,run
from std.utils import Variant


@always_inline
def benchmark_func_row_major_AoS[
    float_dtype:DType,D:Int,Q:Int,
    lattice_model:Lattice[D,Q,float_dtype,DType.int32],
    nx:Int,ny:Int,nz:Int,
    //,
    grid: LBM_Grid[lattice_model,nx,ny,nz,_],
    U:Scalar[float_dtype],
    tau:Scalar[float_dtype],
    *,
    ]
    (mut b:Bencher) capturing raises:
    # This can be stored in LBM Grid
    comptime assert grid.x_tile == 1
    comptime GRID_DIM:Tuple[Int,Int,Int] = grid.GRID_DIM
    comptime BLOCK_SHAPE:Tuple[Int,Int,Int] = grid.BLOCK_SHAPE
    comptime all_slice = slice(None,None,None)
    comptime simd_width = 4

    comptime flag_layout = row_major[nx,ny,nz]()
    comptime f_layout = row_major[nx,ny,nz,Q]()
    comptime bc_layout = row_major[nx,ny,nz,D+1]()

    comptime density_layout = row_major[nx,ny,nz]()
    comptime velocity_layout = row_major[D,nx,ny,nz]()

    run_benchmark[grid,U,tau,simd_width,f_layout,flag_layout,bc_layout,velocity_layout,density_layout](b)


@always_inline
def benchmark_func_col_major_SoA[
    float_dtype:DType,D:Int,Q:Int,
    lattice_model:Lattice[D,Q,float_dtype,DType.int32],
    nx:Int,ny:Int,nz:Int,
    //,
    grid: LBM_Grid[lattice_model,nx,ny,nz,_],
    U:Scalar[float_dtype],
    tau:Scalar[float_dtype],
    *,
    reorder_threads:Bool = True
    ]
    (mut b:Bencher) capturing raises:
    # This can be stored in LBM Grid
    comptime assert grid.x_tile == 1
    comptime GRID_DIM:Tuple[Int,Int,Int] = grid.GRID_DIM
    comptime BLOCK_SHAPE:Tuple[Int,Int,Int] = grid.BLOCK_SHAPE
    comptime all_slice = slice(None,None,None)
    comptime simd_width = 4

    comptime flag_layout = col_major[nx,ny,nz]()
    comptime f_layout = col_major[nx,ny,nz,Q]()
    comptime bc_layout = col_major[nx,ny,nz,D+1]()

    comptime density_layout = row_major[nx,ny,nz]()
    comptime velocity_layout = row_major[D,nx,ny,nz]()

    run_benchmark[grid,U,tau,simd_width,f_layout,flag_layout,bc_layout,velocity_layout,density_layout](b)



@always_inline
def benchmark_func_col_tile_row_tiler[
    float_dtype:DType,D:Int,Q:Int,
    lattice_model:Lattice[D,Q,float_dtype,DType.int32],
    nx:Int,ny:Int,nz:Int,
    //,
    grid: LBM_Grid[lattice_model,nx,ny,nz,_],
    U:Scalar[float_dtype],
    tau:Scalar[float_dtype],
    *,
    reorder_threads:Bool = True
    ]
    (mut b:Bencher) capturing raises:
    comptime assert grid.x_tile > 1 
    # This can be stored in LBM Grid
    comptime GRID_DIM:Tuple[Int,Int,Int] = grid.GRID_DIM
    comptime BLOCK_SHAPE:Tuple[Int,Int,Int] = grid.BLOCK_SHAPE
    comptime all_slice = slice(None,None,None)
    comptime simd_width = 4
    comptime flag_tile = col_major[grid.x_tile,grid.y_tile,1]()
    comptime f_tile = col_major[grid.x_tile,grid.y_tile,1,Q]()
    comptime bc_tile = col_major[grid.x_tile,grid.y_tile,1,D+1]()
        
    comptime flag_tiler = row_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z]()
    comptime f_tiler = row_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z,1]()
    comptime bc_tiler = row_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z,1]()

    comptime flag_layout = blocked_product(flag_tile,flag_tiler)
    comptime f_layout = blocked_product(f_tile,f_tiler)
    comptime bc_layout = blocked_product(bc_tile,bc_tiler)
    comptime density_layout = row_major[nx,ny,nz]()
    comptime velocity_layout = row_major[D,nx,ny,nz]()

    run_benchmark[grid,U,tau,simd_width,f_layout,flag_layout,bc_layout,velocity_layout,density_layout](b)



@always_inline
def benchmark_func_row_tile_col_tiler[
    float_dtype:DType,D:Int,Q:Int,
    lattice_model:Lattice[D,Q,float_dtype,DType.int32],
    nx:Int,ny:Int,nz:Int,
    //,
    grid: LBM_Grid[lattice_model,nx,ny,nz,_],
    U:Scalar[float_dtype],
    tau:Scalar[float_dtype],
    *,
    reorder_threads:Bool = True
    ]
    (mut b:Bencher) capturing raises:
    comptime assert grid.x_tile > 1
    # This can be stored in LBM Grid
    comptime GRID_DIM:Tuple[Int,Int,Int] = grid.GRID_DIM
    comptime BLOCK_SHAPE:Tuple[Int,Int,Int] = grid.BLOCK_SHAPE
    comptime all_slice = slice(None,None,None)
    comptime simd_width = 4
    comptime flag_tile = row_major[grid.x_tile,grid.y_tile,1]()
    comptime f_tile = row_major[grid.x_tile,grid.y_tile,1,Q]()
    comptime bc_tile = row_major[grid.x_tile,grid.y_tile,1,D+1]()
        
    comptime flag_tiler = col_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z]()
    comptime f_tiler = col_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z,1]()
    comptime bc_tiler = col_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z,1]()

    comptime flag_layout = blocked_product(flag_tile,flag_tiler)
    comptime f_layout = blocked_product(f_tile,f_tiler)
    comptime bc_layout = blocked_product(bc_tile,bc_tiler)

    comptime density_layout = row_major[nx,ny,nz]()
    comptime velocity_layout = row_major[D,nx,ny,nz]()

    run_benchmark[grid,U,tau,simd_width,f_layout,flag_layout,bc_layout,velocity_layout,density_layout](b)


@always_inline
def benchmark_func_row_tile_row_tiler[
    float_dtype:DType,D:Int,Q:Int,
    lattice_model:Lattice[D,Q,float_dtype,DType.int32],
    nx:Int,ny:Int,nz:Int,
    //,
    grid: LBM_Grid[lattice_model,nx,ny,nz,_],
    U:Scalar[float_dtype],
    tau:Scalar[float_dtype],
    *,
    reorder_threads:Bool = True
    ]
    (mut b:Bencher) capturing raises:
    comptime assert grid.x_tile > 1
    # This can be stored in LBM Grid
    comptime GRID_DIM:Tuple[Int,Int,Int] = grid.GRID_DIM
    comptime BLOCK_SHAPE:Tuple[Int,Int,Int] = grid.BLOCK_SHAPE
    comptime all_slice = slice(None,None,None)
    comptime simd_width = 4
    comptime flag_tile = row_major[grid.x_tile,grid.y_tile,1]()
    comptime f_tile = row_major[grid.x_tile,grid.y_tile,1,Q]()
    comptime bc_tile = row_major[grid.x_tile,grid.y_tile,1,D+1]()
        
    comptime flag_tiler = row_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z]()
    comptime f_tiler = row_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z,1]()
    comptime bc_tiler = row_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z,1]()

    comptime flag_layout = blocked_product(flag_tile,flag_tiler)
    comptime f_layout = blocked_product(f_tile,f_tiler)
    comptime bc_layout = blocked_product(bc_tile,bc_tiler)

    comptime density_layout = row_major[nx,ny,nz]()
    comptime velocity_layout = row_major[D,nx,ny,nz]()

    run_benchmark[grid,U,tau,simd_width,f_layout,flag_layout,bc_layout,velocity_layout,density_layout](b)

@always_inline
def benchmark_func_col_tile_col_tiler[
    float_dtype:DType,D:Int,Q:Int,
    lattice_model:Lattice[D,Q,float_dtype,DType.int32],
    nx:Int,ny:Int,nz:Int,
    //,
    grid: LBM_Grid[lattice_model,nx,ny,nz,_],
    U:Scalar[float_dtype],
    tau:Scalar[float_dtype],
    *,
    reorder_threads:Bool = True
    ]
    (mut b:Bencher) capturing raises:
    comptime assert grid.x_tile > 1
    # This can be stored in LBM Grid
    comptime GRID_DIM:Tuple[Int,Int,Int] = grid.GRID_DIM
    comptime BLOCK_SHAPE:Tuple[Int,Int,Int] = grid.BLOCK_SHAPE
    comptime all_slice = slice(None,None,None)
    comptime simd_width = 4
    comptime flag_tile = col_major[grid.x_tile,grid.y_tile,1]()
    comptime f_tile = col_major[grid.x_tile,grid.y_tile,1,Q]()
    comptime bc_tile = col_major[grid.x_tile,grid.y_tile,1,D+1]()
        
    comptime flag_tiler = col_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z]()
    comptime f_tiler = col_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z,1]()
    comptime bc_tiler = col_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z,1]()

    comptime flag_layout = blocked_product(flag_tile,flag_tiler)
    comptime f_layout = blocked_product(f_tile,f_tiler)
    comptime bc_layout = blocked_product(bc_tile,bc_tiler)

    comptime density_layout = row_major[nx,ny,nz]()
    comptime velocity_layout = row_major[D,nx,ny,nz]()

    run_benchmark[grid,U,tau,simd_width,f_layout,flag_layout,bc_layout,velocity_layout,density_layout](b)