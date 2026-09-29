from max.gpu.host import DeviceContext
from layout import TileTensor,coord
from layout.tile_layout import Layout,row_major,TensorLayout,blocked_product,col_major
from std.python import Python, PythonObject
from std.sys import argv
from std.sys.defines import get_defined_int
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
comptime N = get_defined_int["N", 128]()
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

comptime flag_tiler = col_major[grid.layouts.n_tiles_x,grid.layouts.n_tiles_y,grid.layouts.n_tiles_z]()
comptime f_tiler = col_major[grid.layouts.n_tiles_x,grid.layouts.n_tiles_y,grid.layouts.n_tiles_z,1]()
comptime bc_tiler = col_major[grid.layouts.n_tiles_x,grid.layouts.n_tiles_y,grid.layouts.n_tiles_z,1]()

comptime flag_layout = blocked_product(flag_tile,flag_tiler)
comptime f_layout = blocked_product(f_tile,f_tiler)
comptime bc_layout = blocked_product(bc_tile,bc_tiler)

comptime density_layout = row_major[nx,ny,nz]()
comptime velocity_layout = row_major[D,nx,ny,nz]()


comptime all_slice = slice(None,None,None)

def get_iters(default: Int) raises -> Int:
    var iters = default
    var args = argv()
    var i = 1
    while i < len(args):
        var parts = String(args[i]).split('--iters=')
        if len(parts) == 2:
            iters = atol(parts[1])
        elif String(args[i]) == '--iters':
            if i + 1 < len(args):
                iters = atol(args[i + 1])
                i += 1
            else:
                raise Error('--iters requires a value')
        i += 1
    return iters


def get_plot(default: Bool) raises -> Bool:
    var plot = default
    var args = argv()
    var i = 1
    while i < len(args):
        var parts = String(args[i]).split('--plot=')
        if len(parts) == 2:
            var v = String(parts[1])
            plot = v == '1' or v == 'true' or v == 'True' or v == 'on' or v == 'ON'
        elif String(args[i]) == '--plot':
            plot = True
        elif String(args[i]) == '--no-plot':
            plot = False
        i += 1
    return plot


def main() raises:
    comptime assert N % tile_size == 0 , 'tile_size must divide N'
    print(grid.layouts.n_tiles_x,grid.layouts.n_tiles_y,grid.layouts.n_tiles_z)
    print('Grid Dim: ',GRID_DIM)
    print('BLOCK_SHAPE: ', BLOCK_SHAPE)
    assert N % tile_size == 0, 'Tile Size must Divide N' 
    print(grid.layouts.n_tiles_x,grid.layouts.n_tiles_y,grid.layouts.n_tiles_z)

    var U_phs:float_scalar = 1.
    var U:float_scalar = 0.1
    var viscosity:float_scalar = 1/10.
    _ = dx*U/U_phs
    var Re = 1/viscosity
    var L_lat:float_scalar = float_scalar(N)
    var v_lat = U*L_lat/Re
    var tau = v_lat/(1/3.) +0.5
    print('Tau {}'.format(tau))

    var ctx = DeviceContext()
    
    var flags = ContextTileTensor[DType.uint8](ctx,flag_layout)
    var bc = ContextTileTensor[float_dtype](ctx,bc_layout)
    var f = ContextTileTensor[float_dtype](ctx,f_layout)
    var f_out = ContextTileTensor[float_dtype](ctx,f_layout)

    var u = ContextTileTensor[float_dtype](ctx,velocity_layout)
    var rho = ContextTileTensor[float_dtype](ctx,density_layout)

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
    comptime LBM_ = double_buffer_kernel[type_of(f_layout),type_of(bc_layout),type_of(flag_layout),grid,config]
    var LBM_func = ctx.compile_function[LBM_]()

    comptime get_u_and_rho = calculate_rho_and_velocity[type_of(f_layout),type_of(bc_layout),type_of(flag_layout),type_of(density_layout),type_of(velocity_layout),grid,config]
    var calc_rho_and_u_gpu = ctx.compile_function[get_u_and_rho]()
 
    ctx.synchronize()
    var MAX_ITERS = get_iters(10_000)
    # Run Simulation
    for t in range(MAX_ITERS):
        ctx.enqueue_function[LBM_](f_out.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),1/tau,grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        ctx.enqueue_function[LBM_](f.gpu(),f_out.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),1/tau,grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        if (t % max((MAX_ITERS//10),1)) == 0 :
            # pass
            ctx.synchronize()
            ctx.enqueue_function[get_u_and_rho](rho.gpu(),u.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
            ctx.synchronize()
            var u_np = u.buffer_to_numpy()/U
            print('step = {} max ={} avg = {}'.format(t,u_np.max(),u_np.mean()))
    ctx.synchronize()
    # Get Final U and rho
    ctx.enqueue_function[get_u_and_rho](rho.gpu(),u.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
    ctx.synchronize()

    var u_np = (u.buffer_to_numpy()/U).reshape(D,nx,ny,nz)
    print('step = {} max ={} avg = {}'.format(0,u_np.max(),u_np.mean()) )

    var plot = get_plot(True)
    if plot:
        var np = Python.import_module('numpy')
        var pv = Python.import_module('pyvista')

        var x = np.linspace(0, grid.domain_size[0], nx)
        var y = np.linspace(0, grid.domain_size[1], ny)
        var m = np.meshgrid(x, y,indexing = 'ij')
        var xx,yy = m[0],m[1]
        var pv_mesh = pv.StructuredGrid(xx, yy, np.zeros_like(xx))
        print(pv_mesh)


        var u_plot = u_np[0,all_slice,all_slice,all_slice].T
        var v_plot = u_np[1,all_slice,all_slice,all_slice].T

        var u_mag = np.sqrt(u_plot**2 + v_plot**2)
        pv_mesh.point_data['U_mag'] = u_mag.ravel()
        pv_mesh.point_data['U velocity'] = u_plot.ravel()
        pv_mesh.point_data['V velocity'] = v_plot.ravel()

        var plotter = pv.Plotter()
        plotter.add_mesh(pv_mesh,scalars ='U_mag',show_edges = False, cmap= 'jet',clim = [0,1],nan_color='white',)
        plotter.view_xy()
        plotter.show_axes()
        plotter.show() # screenshot = 'LDC_Re100.png'
        
    
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

