from std.gpu.host import DeviceContext
from layout import TileTensor
from layout.tile_layout import Layout,row_major,TensorLayout,blocked_product,col_major
from std.python import Python, PythonObject
from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv

from std.collections import InlineArray
from src.lbm import SOLID_NODE,FLUID_NODE,LBM_Grid,get_D2Q9,set_outer_walls,calculate_rho_and_velocity
from src.lbm.variations.part_2.SoA_Tile import LBM_kernel
from src.utils import Vector,ContextTileTensor

comptime float_dtype = DType.float32
comptime int_dtype = DType.int32
comptime float_scalar = Scalar[float_dtype]
comptime D2Q9 = get_D2Q9[DType.float32,DType.int32]()
comptime D,Q = (2,9)
comptime N = 128
comptime L = 1.
comptime dx = L/float_scalar(N-1)
comptime (nx,ny,nz) = (N,N,1)
comptime num_points = nx*ny*nz

comptime THREADS_PER_BLOCK = 256
comptime BLOCK_SHAPE = (16,16,1)
comptime GRID_DIM = ((nx) // BLOCK_SHAPE[0]+1,(ny) // BLOCK_SHAPE[1]+1, 1 )# Plus one
comptime tile_size = 16
comptime n_tiles = N//tile_size
comptime simd_width = 4
comptime flag_tile = col_major[tile_size,tile_size,1]()
comptime f_tile = col_major[tile_size,tile_size,1,Q]()
comptime bc_tile = col_major[tile_size,tile_size,1,D+1]()
    
comptime flag_tiler = row_major[n_tiles,n_tiles,1]()
comptime f_tiler = row_major[n_tiles,n_tiles,1,1]()
comptime bc_tiler = row_major[n_tiles,n_tiles,1,1]()

comptime flag_layout = blocked_product(flag_tile,flag_tiler)
comptime f_layout = blocked_product(f_tile,f_tiler)
comptime bc_layout = blocked_product(bc_tile,bc_tiler)

comptime density_layout = row_major[nx,ny,nz]()
comptime velocity_layout = row_major[D,nx,ny,nz]()

comptime grid = LBM_Grid[D2Q9,nx,ny,nz](dx)
comptime all_slice = slice(None,None,None)

def main() raises:
    comptime assert N % tile_size == 0 , 'tile_size must divide N'
    print(GRID_DIM)
    print(BLOCK_SHAPE)
    assert N % tile_size == 0, 'Tile Size must Divide N' 
    
    U_phs:float_scalar = 1.
    U:float_scalar = 0.05
    viscosity:float_scalar = 1/100.
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

    f.fill(1./Float32(Q))
    f_out.fill(1./Float32(Q))

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
    comptime LBM_ = LBM_kernel[grid,f_layout,bc_layout,flag_layout,simd_width]
    LBM_func = ctx.compile_function[LBM_,LBM_]()

    comptime get_u_and_rho = calculate_rho_and_velocity[grid,f_layout,density_layout,velocity_layout,f_is_AoS = True]
    calc_rho_and_u_gpu = ctx.compile_function[get_u_and_rho,get_u_and_rho]()
 
    ctx.synchronize()

    # Run Simulation
    for t in range(10_000):
        ctx.enqueue_function(LBM_func,f_out.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),1/tau,grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        ctx.enqueue_function(LBM_func,f.gpu(),f_out.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),1/tau,grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        if t % 1000 == 0 :
            pass
            ctx.synchronize()
            ctx.enqueue_function(calc_rho_and_u_gpu,f.gpu(),rho.gpu(),u.gpu(),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
            ctx.synchronize()
            u_np = u.buffer_to_numpy()/U
            print('step = {} max ={} avg = {}'.format(t,u_np.max(),u_np.mean()))
    ctx.synchronize()
    # Get Final U and rho
    ctx.enqueue_function(calc_rho_and_u_gpu,f.gpu(),rho.gpu(),u.gpu(),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)

    ctx.synchronize()
    
    np = Python.import_module('numpy')
    pd = Python.import_module('pandas')
    pv = Python.import_module('pyvista')
    plt = Python.import_module('matplotlib.pyplot')
    ctypes = Python.import_module("ctypes")

    u_np = (u.buffer_to_numpy()/U).reshape(D,nx,ny,nz)
    print('step = {} max ={} avg = {}'.format(0,u_np.max(),u_np.mean()) )

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    m = np.meshgrid(x, y)
    xx,yy = m[0],m[1]
    pv_mesh = pv.StructuredGrid(xx, yy, np.zeros_like(xx))
    
    

    u_plot = u_np[0,all_slice,all_slice,all_slice]
    v_plot = u_np[1,all_slice,all_slice,all_slice]

    u_mag = np.sqrt(u_plot**2 + v_plot**2)
    pv_mesh.point_data['U_mag'] = u_mag.ravel()
    pv_mesh.point_data['U velocity'] = u_plot.ravel()
    pv_mesh.point_data['V velocity'] = v_plot.ravel()
    
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh,scalars ='U_mag',show_edges = False, cmap= 'jet',clim = [0,1],nan_color='white',)
    plotter.view_xy()
    plotter.show() # screenshot = 'LDC_Re100.png'
   
    
    # import pandas as pd
    v_benchmark = pd.read_csv('v_velocity_results.csv',sep = ',')
    u_benchmark = pd.read_csv('u_velocity_results.txt',sep= '\t')
    
    
    horizontal_line = pv_mesh.sample_over_line( Python.tuple(0,L/2,0),Python.tuple(L,L/2,0),resolution= nx)
    v_05 = horizontal_line.point_data['V velocity']
    
    plt.plot(v_benchmark['%x'],v_benchmark['100'],'o',label = 'Ghia et al')
    plt.plot(horizontal_line.points[all_slice,0],v_05,label ='lbm Results')
    plt.xlabel('x')
    plt.ylabel('v velocity')
    plt.legend()
    plt.show()
    
    
    vertical_line = pv_mesh.sample_over_line(Python.tuple(L/2,0,0),Python.tuple(L/2,L,0),resolution= ny)
    u_05 = vertical_line.point_data['U velocity']
    
    plt.plot(u_benchmark['%y'],u_benchmark['100'],'o',label = 'Ghia et al')
    plt.plot(vertical_line.points[all_slice,1],u_05,label ='lbm Results')
    plt.xlabel('y')
    plt.ylabel('u velocity')
    plt.legend()
    plt.show()
        
    
    # print(b_np.reshape(nx,ny,D+1)[slice(None,None,None),slice(None,None,None),0])
    # print(f_np.reshape(Q,nx,ny)[0,slice(None,None,None),slice(None,None,None)])
    # print(flag_np.reshape(nx,ny))


    # print(u_np.reshape(N,N,D)[slice(None,None,None),slice(None,None,None),0])
    
    
    # flag_ptr = flags.cpu_buffer().unsafe_ptr()
    # f_add = Int(flag_ptr) # Need to get the pointer address as Int type
    # p_int = ctypes.POINTER(ctypes.c_uint8) # Set Dtype
    # np_ptr = ctypes.cast(f_add, p_int)
    # np_flag = np.ctypeslib.as_array(np_ptr, shape=Python.tuple(flags.size))

    # print(np_flag.reshape(Python.tuple(nx,ny)))
    

    # f_ptr = f.cpu_buffer().unsafe_ptr()
    # f_add = Int(f_ptr) # Need to get the pointer address as Int type
    # p_float = ctypes.POINTER(ctypes.c_float) # Set Dtype
    # np_ptr = ctypes.cast(f_add, p_float)
    # np_f = np.ctypeslib.as_array(np_ptr, shape=Python.tuple(f.size))

    # print(np_f)



    # f_buffer =  ctx.enqueue_create_host_buffer[grid.float_dtype](grid.f_field_size)
    # f_out_buffer =  ctx.enqueue_create_host_buffer[grid.float_dtype](grid.f_field_size)

    # flag_buffer = ctx.enqueue_create_host_buffer[DType.uint8](grid.num_points)
    # bc_buffer = ctx.enqueue_create_host_buffer[grid.float_dtype](grid.bc_field_size)

    # ctx.synchronize()
    # # flags = TileTensor[DType.uint8,RowMajorType,MutAnyOrigin](flag_buffer,layout)
    
    # # Do BC on this
    # flags = TileTensor[DType.uint8](flag_buffer,flag_layout)
    # bc = TileTensor[float_dtype](bc_buffer,bc_layout)
    # f = TileTensor[float_dtype](f_buffer,f_layout)
    # Make buffers to GPU

    # f_buffer_gpu =  ctx.enqueue_create_buffer[grid.float_dtype](grid.f_field_size)
    # f_out_buffer_gpu =  ctx.enqueue_create_buffer[grid.float_dtype](grid.f_field_size)

    # flag_buffer_gpu = ctx.enqueue_create_buffer[DType.uint8](grid.num_points)
    # bc_buffer_gpu = ctx.enqueue_create_buffer[grid.float_dtype](grid.bc_field_size)

    # ctx.synchronize()

    # # Copy Buffers from cpu to GPU

    # ctx.enqueue_copy(dst_buf = f_buffer_gpu,src_buf = f_buffer)
    # ctx.enqueue_copy(dst_buf = f_out_buffer_gpu,src_buf = f_out_buffer)
    # ctx.enqueue_copy(dst_buf = flag_buffer_gpu,src_buf = flag_buffer)
    # ctx.enqueue_copy(dst_buf = bc_buffer_gpu,src_buf = bc_buffer)

    # ctx.synchronize()

