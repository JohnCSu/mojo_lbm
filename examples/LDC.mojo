from std.gpu.host import DeviceContext
from layout import TileTensor,coord
from layout.tile_layout import Layout,row_major,TensorLayout,blocked_product,col_major
from std.python import Python, PythonObject
from std.collections import InlineArray
from src.lbm import (
                    Flags,SOLID_NODE,FLUID_NODE,
                    LBM_Grid,LBM_Config,
                    get_D2Q9,set_exterior_walls,calculate_rho_and_velocity,
                    UnitSystem,DoubleBufferConfig,EsotericPullConfig
                    )
from src.lbm.preprocess.initial_condition import initialize_fluid_at_rest
from src.lbm.kernels.double_buffer import double_buffer_kernel
from src.utils import Vector,ContextTileTensor
from src.lbm.geometry.primatives import add_sphere,add_box
from src.visualization import pyvista_viewer_import,grid_viewer

from std.collections import Set
comptime float_dtype = DType.float32
comptime int_dtype = DType.int32
comptime float_scalar = Scalar[float_dtype]
comptime D2Q9 = get_D2Q9()
comptime D,Q = (2,9)
comptime N = 512
comptime L = 1.
comptime dx = L/float_scalar(N-1)
comptime (nx,ny,nz) = (N,N,1)
comptime tile_size = 16
comptime grid = LBM_Grid[D2Q9,nx,ny,nz,tile_size](dx,[0.,0.,0.])
comptime config = DoubleBufferConfig(collision_op ='RLBM',DDF_shift = True,LES = False)

comptime BLOCK_SHAPE = grid.BLOCK_SHAPE
comptime GRID_DIM = grid.GRID_DIM

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
    '''
    LDC Benchmark For Fluid Dynamice. This Script does not compare to benchmark data 
    and so can be set to any Reynolds number.
    '''
    comptime assert N % tile_size == 0 , 'tile_size must divide N'
    print(grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z)
    print('Grid Dim: ',GRID_DIM)
    print('BLOCK_SHAPE: ', BLOCK_SHAPE)
    assert N % tile_size == 0, 'Tile Size must Divide N' 
    print(grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z)

    U_phs:float_scalar = 1.
    U:float_scalar = 0.1
    L_phys:float_scalar = 1.
    Re:float_scalar = 10000
    Re_=Int(Re)
    
    units = grid.get_UnitSystem_with_Re(U_phs,U,L_phys,Re=Re)
    tau = units.tau
    dt = units.dt
    # Cs:float_scalar = 0.17

    print(units.tau,units.Re, units.kinematic_viscosity)

    ctx = DeviceContext()
    
    flags = ContextTileTensor[DType.uint8](ctx,flag_layout)
    bc = ContextTileTensor[float_dtype](ctx,bc_layout)
    f = ContextTileTensor[float_dtype](ctx,f_layout)
    f_out = ContextTileTensor[float_dtype](ctx,f_layout)

    u = ContextTileTensor[float_dtype](ctx,velocity_layout)
    rho = ContextTileTensor[float_dtype](ctx,density_layout)
    pv_view = pyvista_viewer_import()

    # Set up

    initialize_fluid_at_rest[grid,config](f.cpu())

    set_exterior_walls[grid,config](flags.cpu(),bc.cpu(),'+Y',SOLID_NODE,[U,0],1.)
    set_exterior_walls[grid,config](flags.cpu(),bc.cpu(),'-Y',SOLID_NODE,[0,0],1.)
    set_exterior_walls[grid,config](flags.cpu(),bc.cpu(),'+X',SOLID_NODE,[0,0],1.)
    set_exterior_walls[grid,config](flags.cpu(),bc.cpu(),'-X',SOLID_NODE,[0,0],1.)
    ctx.synchronize()
    # Copy To GPU()
    _ = flags.gpu()
    _ = bc.gpu()
    _ = f.gpu()
    _ = f_out.gpu()

    #Compile Functions
    comptime LBM_ = double_buffer_kernel[f_layout,bc_layout,flag_layout,grid,config]
    LBM_func = ctx.compile_function[LBM_,LBM_]()

    comptime get_u_and_rho = calculate_rho_and_velocity[f_layout,bc_layout,flag_layout,density_layout,velocity_layout,grid,config]
    calc_rho_and_u_gpu = ctx.compile_function[get_u_and_rho,get_u_and_rho]()

    ctx.synchronize()

    # Animation Code
    np = Python.import_module('numpy')
    pd = Python.import_module('pandas')
    u_np = (u.buffer_to_numpy()/U).reshape(D,nx,ny,nz)
    viewer = grid_viewer[grid](subplot_shape= (1,1))
    
    u_plot = u_np[0,all_slice,all_slice,all_slice].T
    v_plot = u_np[1,all_slice,all_slice,all_slice].T
    u_mag = np.sqrt(u_plot**2 + v_plot**2)
    viewer.point_data['U_mag'] = u_mag.ravel()
    viewer.point_data['U velocity'] = u_plot.ravel()
    viewer.point_data['V velocity'] = v_plot.ravel()
    
    viewer.set_mesh_display('U_mag',clim = [0,1],cmap ='jet')

    viewer.set_animation('LDC_Re{}.gif'.format(Int(Re)),framerate = 16)
    viewer.plotter.add_title('Lid Driven Cavity for Re = {}'.format(Re_))
    viewer.plotter.add_text("step: {} time: {}".format(0,0.), position="lower_edge", name="dynamic_text",font_size = 14)
   
    comptime MAX_ITERS = 400_000
    # Run Simulation
    for step in range(MAX_ITERS):
        ctx.enqueue_function(LBM_func,f_out.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),tau,grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        ctx.enqueue_function(LBM_func,f.gpu(),f_out.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),tau,grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        if (step % (MAX_ITERS//100)) == 0:
            ctx.synchronize()
            ctx.enqueue_function(calc_rho_and_u_gpu,rho.gpu(),u.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
            ctx.synchronize()
            u_np = (u.buffer_to_numpy()/U).reshape(D,nx,ny,nz)
            t = 2.*Scalar[float_dtype](step)*dt
            print('step = {}, time = {} max ={} avg = {}'.format(step,t,u_np.max(),u_np.mean()))
            u_plot = u_np[0,all_slice,all_slice,all_slice].T
            v_plot = u_np[1,all_slice,all_slice,all_slice].T
            u_mag = np.sqrt(u_plot**2 + v_plot**2)
            viewer.point_data['U_mag'] = u_mag.ravel()
            viewer.point_data['U velocity'] = u_plot.ravel()
            viewer.point_data['V velocity'] = v_plot.ravel()
            viewer.plotter.add_text("step: {} time: {}".format(step,t), position="lower_edge", name="dynamic_text",font_size = 14)
            viewer.update_frame()
            ctx.synchronize()

    ctx.synchronize()
    viewer.close()
