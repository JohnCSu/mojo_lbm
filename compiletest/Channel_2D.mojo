from std.gpu.host import DeviceContext
from layout import TileTensor,coord
from layout.tile_layout import Layout,row_major,TensorLayout,blocked_product,col_major
from std.python import Python, PythonObject
from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv

from std.collections import InlineArray
from src.lbm import (
                    Flags,SOLID_NODE,FLUID_NODE,
                    LBM_Grid,LBM_Config,
                    get_D2Q9,set_exterior_walls,calculate_rho_and_velocity,DoubleBufferConfig,EsotericPullConfig)

from src.lbm.kernels.double_buffer import double_buffer_kernel


from src.utils import Vector,ContextTileTensor

comptime float_dtype = DType.float32
comptime int_dtype = DType.int32
comptime float_scalar = Scalar[float_dtype]
comptime D2Q9 = get_D2Q9()
comptime D,Q = (2,9)
comptime N = 32
comptime L = 1.
comptime dx = L/float_scalar(N-1)
comptime (nx,ny,nz) = (2*N,N,1)
comptime tile_size = 16
comptime grid = LBM_Grid[D2Q9,nx,ny,nz,tile_size](dx)
comptime valid_bcs = {Flags.EQUILIBRIUM}
comptime config = DoubleBufferConfig(BCs = valid_bcs,DDF_shift = False)

comptime BLOCK_SHAPE = grid.BLOCK_SHAPE
comptime GRID_DIM = grid.GRID_DIM

# comptime BLOCK_SHAPE = (16,16,1)
# comptime GRID_DIM = (2,2,1)

comptime simd_width = 4
comptime flag_tile = col_major[tile_size,tile_size,1]()
comptime f_tile = col_major[tile_size,tile_size,1,Q]()
comptime bc_tile = col_major[tile_size,tile_size,1,D+1]()

comptime flag_tiler = col_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z]()
comptime f_tiler = col_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z,1]()
comptime bc_tiler = col_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z,1]()

comptime flag_layout = blocked_product(flag_tile,flag_tiler)
comptime f_layout = blocked_product(f_tile,f_tiler)
comptime bc_layout = blocked_product(bc_tile,bc_tiler)

comptime density_layout = row_major[nx,ny,nz]()
comptime velocity_layout = row_major[D,nx,ny,nz]()


comptime all_slice = slice(None,None,None)

def main() raises:
    comptime assert N % tile_size == 0 , 'tile_size must divide N'
    print(grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z)
    print('Grid Dim: ',GRID_DIM)
    print('BLOCK_SHAPE: ', BLOCK_SHAPE)
    assert N % tile_size == 0, 'Tile Size must Divide N' 
    print(grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z)

    U_phs:float_scalar = 1.
    U:float_scalar = 0.1
    viscosity:float_scalar = 1/10.
    dt = dx*U/U_phs 
    Re = 1/viscosity
    L_lat:float_scalar = N
    v_lat = U*L_lat/Re
    tau = v_lat/(1/3.) +0.5
    print('Tau {}'.format(tau))

    ctx = DeviceContext()
    
    flags = ContextTileTensor[DType.uint8](ctx,flag_layout)
    bc = ContextTileTensor[float_dtype](ctx,bc_layout)
    f = ContextTileTensor[float_dtype](ctx,f_layout)
    f_out = ContextTileTensor[float_dtype](ctx,f_layout)

    u = ContextTileTensor[float_dtype](ctx,velocity_layout)
    rho = ContextTileTensor[float_dtype](ctx,density_layout)

    # Set up
    comptime if not config.DDF_shift:
        f.fill(1./Float32(Q))
        f_out.fill(1./Float32(Q))
    else:
        f.fill(0.)
        f_out.fill(0.)
    

    set_exterior_walls[grid,config](flags.cpu(),bc.cpu(),'+X',Flags.EQUILIBRIUM,[],1.)
    set_exterior_walls[grid,config](flags.cpu(),bc.cpu(),'-X',Flags.EQUILIBRIUM,[U,0],1.)
    set_exterior_walls[grid,config](flags.cpu(),bc.cpu(),'+Y',SOLID_NODE,[0,0],1.)
    set_exterior_walls[grid,config](flags.cpu(),bc.cpu(),'-Y',SOLID_NODE,[0,0],1.)

    ctx.synchronize()
    # Copy To GPU()
    _ = flags.gpu()
    _ = bc.gpu()
    _ = f.gpu()
    _ = f_out.gpu()

    ctx.synchronize()
    #Compile Functions
    comptime LBM_ = double_buffer_kernel[f_layout,bc_layout,flag_layout,grid,config]
    LBM_func = ctx.compile_function[LBM_,LBM_]()

    comptime get_u_and_rho = calculate_rho_and_velocity[f_layout,bc_layout,flag_layout,density_layout,velocity_layout,grid,config]
    calc_rho_and_u_gpu = ctx.compile_function[get_u_and_rho,get_u_and_rho]()
 
    ctx.synchronize()
    comptime MAX_ITERS = 5
    # Run Simulation
    for t in range(MAX_ITERS):
        ctx.enqueue_function(LBM_func,f_out.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),1/tau,grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        ctx.enqueue_function(LBM_func,f.gpu(),f_out.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),1/tau,grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        ctx.synchronize()
        ctx.enqueue_function(calc_rho_and_u_gpu,rho.gpu(),u.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        ctx.synchronize()
        u_np = u.buffer_to_numpy()/U
        print('step = {} max ={} avg = {}'.format(t,u_np.max(),u_np.mean()))
    ctx.synchronize()
    # Get Final U and rho
    ctx.enqueue_function(calc_rho_and_u_gpu,rho.gpu(),u.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
    ctx.synchronize()
    u_np = (u.buffer_to_numpy()/U).reshape(D,nx,ny,nz)
    print('Final: step = {} max ={} avg = {}'.format(MAX_ITERS,u_np.max(),u_np.mean()))
