from std.gpu.host import DeviceContext
from layout import TileTensor
from layout.tile_layout import (row_major,col_major,TensorLayout,blocked_product)
from std.python import Python, PythonObject
from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from std.collections import InlineArray
from src.lbm import SOLID_NODE,FLUID_NODE,set_outer_walls,LBM_Grid,get_D2Q9,LatticeModel

from .LBM_gpu_kernel import LBM_kernel
from src.utils import Vector,ContextTileTensor
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep,run

@always_inline
def benchmark_func[
    float_dtype:DType,D:Int,Q:Int,
    lattice_model:LatticeModel[D,Q,float_dtype,DType.int32],
    nx:Int,ny:Int,nz:Int,
    tile_size:Int,
    //,
    grid: LBM_Grid[lattice_model,nx,ny,nz,tile_size],
    GRID_DIM:Tuple[Int,Int,Int],
    BLOCK_SHAPE:Tuple[Int,Int,Int],
    U:Scalar[float_dtype],
    tau:Scalar[float_dtype], 
    ]
    (mut b:Bencher) capturing raises:
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
    f.fill(1./Scalar[float_dtype](Q))
    f_out.fill(1./Scalar[float_dtype](Q))

    set_outer_walls[grid,flag_layout,bc_layout](flags.cpu(),bc.cpu(),'+Y',SOLID_NODE,[U,0],1.)
    set_outer_walls[grid,flag_layout,bc_layout](flags.cpu(),bc.cpu(),'-Y',SOLID_NODE,[0,0],1.)
    set_outer_walls[grid,flag_layout,bc_layout](flags.cpu(),bc.cpu(),'-X',SOLID_NODE,[0,0],1.)
    set_outer_walls[grid,flag_layout,bc_layout](flags.cpu(),bc.cpu(),'+X',SOLID_NODE,[0,0],1.)

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

