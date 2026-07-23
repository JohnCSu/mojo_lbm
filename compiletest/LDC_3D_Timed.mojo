from std.gpu.host import DeviceContext
from layout import TileTensor,coord
from layout.tile_layout import Layout,row_major,TensorLayout,blocked_product,col_major
from std.python import Python, PythonObject
from std.collections import InlineArray
from src.lbm import (
                    Flags,SOLID_NODE,FLUID_NODE,
                    LBM_Grid,LBM_Config,
                    get_D3Q19,set_exterior_walls,calculate_rho_and_velocity,
                    UnitSystem,DoubleBufferConfig,EsotericPullConfig
                    )

from src.lbm.kernels.double_buffer import double_buffer_kernel
from src.utils import Vector,ContextTileTensor
from src.lbm.geometry.primatives import add_sphere,add_box
from src.lbm.preprocess.initial_condition import initialize_fluid_at_rest
from std.collections import Set
from std.time import perf_counter
from std import sys
comptime float_dtype = DType.float32
comptime int_dtype = DType.int32
comptime float_scalar = Scalar[float_dtype]
comptime D3Q19 = get_D3Q19[DType.float32,DType.int32]()
comptime D,Q = (D3Q19.D,D3Q19.Q)
comptime N = 32
comptime L = 1.
comptime dx = L/float_scalar(N-1)
comptime (nx,ny,nz) = (N,N,N)
comptime num_points = nx*ny*nz
comptime tile_size = 8
comptime grid = LBM_Grid[D3Q19,nx,ny,nz,tile_size](dx)
comptime config = DoubleBufferConfig(DDF_shift = True,LES = True)

comptime all_slice = slice(None,None,None)

comptime BLOCK_SHAPE = grid.BLOCK_SHAPE
comptime GRID_DIM = grid.GRID_DIM
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

def main() raises:
    comptime assert N % tile_size == 0 , 'tile_size must divide N'
    
    assert N % tile_size == 0, 'Tile Size must Divide N' 

    U_phs:float_scalar = 1.
    U:float_scalar = 0.1
    L_phys:float_scalar = 1.
    Re:float_scalar = 100

    valid_Re:Set[Int] = {100,400,1000,3200,5000,7500,10000}

    Re_=Int(Re)
    if Re_ not in valid_Re:
        raise Error('Re for LDC must be the following {}. Got Re = {} instead'.format(valid_Re,Re_))

    
    units = grid.get_UnitSystem_with_Re(U_phs,U,L_phys,Re=Re)
    tau = units.tau
    dt = units.dt

    ctx = DeviceContext()
    
    flags = ContextTileTensor[DType.uint8](ctx,flag_layout)
    bc = ContextTileTensor[float_dtype](ctx,bc_layout)
    f = ContextTileTensor[float_dtype](ctx,f_layout)
    f_out = ContextTileTensor[float_dtype](ctx,f_layout)

    # Set up
    print('Setting Initial Conditions')
    initialize_fluid_at_rest[grid,config](f.cpu())

    print('Setting Boundary Conditions')
    set_exterior_walls[grid,config](flags.cpu(),bc.cpu(),'+Y',SOLID_NODE,[U,0,0],1.)
    set_exterior_walls[grid,config](flags.cpu(),bc.cpu(),'-Y',SOLID_NODE,[0,0,0],1.)
    set_exterior_walls[grid,config](flags.cpu(),bc.cpu(),'+X',SOLID_NODE,[0,0,0],1.)
    set_exterior_walls[grid,config](flags.cpu(),bc.cpu(),'-X',SOLID_NODE,[0,0,0],1.)
    set_exterior_walls[grid,config](flags.cpu(),bc.cpu(),'-Z',SOLID_NODE,[0,0,0],1.)
    set_exterior_walls[grid,config](flags.cpu(),bc.cpu(),'+Z',SOLID_NODE,[0,0,0],1.)
    
    ctx.synchronize()
    # Copy To GPU()
    _ = flags.gpu()
    _ = bc.gpu()
    _ = f.gpu()
    _ = f_out.gpu()

    #Compile Functions
    comptime LBM_ = double_buffer_kernel[f_layout,bc_layout,flag_layout,grid,config]
    LBM_func = ctx.compile_function[LBM_,LBM_]()

    ctx.synchronize()
    
    comptime MAX_ITERS = 5
    total_iters = MAX_ITERS*2
    # Run Simulation
    print('{}^3 LDC Cube at Re=100 Benchmark for fp32/fp32 D{}Q{} LBM'.format(N,D,Q))
    print('Running On GPU Device: {}'.format(ctx.name()))
    print("Mojo Version: {}.{}.{}".format(sys.defines.MojoVersion().major, sys.defines.MojoVersion().minor,sys.defines.MojoVersion().patch))
    print('Number of Lattice Points: ',num_points)
    print('Number of Iterations: ',total_iters)
    print('LES:= ',config.LES)
    print('Valid BC: {}',materialize[config.INCLUDED_BCs]())
    print('Grid Dim: ',GRID_DIM)
    print('BLOCK_SHAPE: ', BLOCK_SHAPE)

    time_start = perf_counter()
    for t in range(MAX_ITERS):
        ctx.enqueue_function(LBM_func,f_out.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),tau,grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        ctx.enqueue_function(LBM_func,f.gpu(),f_out.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),tau,grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
    ctx.synchronize()
    time_end = perf_counter()
    

    Wall_clock = (time_end-time_start) # Convert from ns to s
    num_points = grid.num_points

    MLUPs = Float64(num_points*total_iters)/(Wall_clock*1e6) # MLUPs = Millions of Lattice Point Updates per second
    
    print('Wallclock Time: ',Wall_clock)
    print('MLUPs averaged over {} iterations: {}'.format(total_iters,MLUPs))
