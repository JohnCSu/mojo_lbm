from std.gpu.host import DeviceContext
from layout import TileTensor,coord
from layout.tile_layout import Layout,row_major,TensorLayout,blocked_product,col_major
from std.python import Python, PythonObject
from std.collections import InlineArray
from src.lbm import (
                    Flags,SOLID_NODE,FLUID_NODE,
                    LBM_Grid,LBM_Config,
                    get_D2Q9,set_exterior_walls,calculate_rho_and_velocity,
                    UnitSystem
                    )

from src.lbm.kernels.SRT import LBM_kernel
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
comptime config = LBM_Config(DDF_shift = True,LES = True)

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

    valid_Re:Set[Int] = {100,400,1000,3200,5000,7500,10000}

    Re_=Int(Re)
    if Re_ not in valid_Re:
        raise Error('Re for LDC must be the following {}. Got Re = {} instead'.format(valid_Re,Re_))

    # units = UnitSystem(U_phs,U,radius,radius/dx,1.,Re = 100.)
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
    comptime if not config.DDF_shift:
        f.fill(1./Float32(Q)) # Should be initialising with respective weight for each dist but should be ok as IC is fluid at rest
        f_out.fill(1./Float32(Q))
    else:
        f.fill(0.)
        f_out.fill(0.)

    
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
    comptime LBM_ = LBM_kernel[grid,f_layout,bc_layout,flag_layout,config]
    LBM_func = ctx.compile_function[LBM_,LBM_]()

    comptime get_u_and_rho = calculate_rho_and_velocity[grid,f_layout,bc_layout,flag_layout,density_layout,velocity_layout,config]
    calc_rho_and_u_gpu = ctx.compile_function[get_u_and_rho,get_u_and_rho]()

    ctx.synchronize()
    
    # Animation Code
    np = Python.import_module('numpy')
    pd = Python.import_module('pandas')
    u_np = (u.buffer_to_numpy()/U).reshape(D,nx,ny,nz)
    pv_mesh = grid_viewer[grid](subplot_shape= (3,1))
    
    u_plot = u_np[0,all_slice,all_slice,all_slice].T
    v_plot = u_np[1,all_slice,all_slice,all_slice].T
    u_mag = np.sqrt(u_plot**2 + v_plot**2)
    pv_mesh.point_data['U_mag'] = u_mag.ravel()
    pv_mesh.point_data['U velocity'] = u_plot.ravel()
    pv_mesh.point_data['V velocity'] = v_plot.ravel()
    
    pv_mesh.set_mesh_display('U_mag',clim = [0,1],cmap ='jet')


    # Chart Data
    v_benchmark = pd.read_csv('v_velocity_results.csv',sep = ',')
    u_benchmark = pd.read_csv('u_velocity_results.txt',sep= '\t')

    pv_mesh.add_chart(Python.tuple(1,0),'V velocity',Python.tuple(0,L/2,0),Python.tuple(L,L/2,0),0,resolution= N,label = 'LBM')
    pv_mesh.add_data_to_chart(Python.tuple(1,0),v_benchmark['%x'],v_benchmark['{}'.format(Int(Re))],color = 'r',label = 'Ghia et al')
    
    pv_mesh.add_chart(Python.tuple(2,0),'U velocity',Python.tuple(L/2,0,0),Python.tuple(L/2,L,0),1,resolution= N,label = 'LBM')
    pv_mesh.add_data_to_chart(Python.tuple(2,0),u_benchmark['%y'],u_benchmark['{}'.format(Int(Re))],color = 'r',label = 'Ghia et al')


    pv_mesh.set_animation('LDC_Re{}.gif'.format(Int(Re)))
    # pv_mesh.show()
   
    comptime MAX_ITERS = 400_000
    # Run Simulation
    for t in range(MAX_ITERS):
        ctx.enqueue_function(LBM_func,f_out.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),tau,grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        ctx.enqueue_function(LBM_func,f.gpu(),f_out.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),tau,grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        if (t % (MAX_ITERS//100)) == 0:
            ctx.synchronize()
            ctx.enqueue_function(calc_rho_and_u_gpu,rho.gpu(),u.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
            ctx.synchronize()
            u_np = (u.buffer_to_numpy()/U).reshape(D,nx,ny,nz)
            print('step = {}, time = {} max ={} avg = {}'.format(t,2.*Scalar[float_dtype](t)*dt,u_np.max(),u_np.mean()))
            u_plot = u_np[0,all_slice,all_slice,all_slice].T
            v_plot = u_np[1,all_slice,all_slice,all_slice].T
            u_mag = np.sqrt(u_plot**2 + v_plot**2)
            pv_mesh.point_data['U_mag'] = u_mag.ravel()
            pv_mesh.point_data['U velocity'] = u_plot.ravel()
            pv_mesh.point_data['V velocity'] = v_plot.ravel()
            pv_mesh.update_frame()
            ctx.synchronize()

    ctx.synchronize()
    # Get Final U and rho
    ctx.enqueue_function(calc_rho_and_u_gpu,rho.gpu(),u.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
    ctx.synchronize()
    

    pv_mesh.close()
    # u_np = (u.buffer_to_numpy()/U).reshape(D,nx,ny,nz)
    # # pv_mesh = grid_viewer[grid](subplot_shape= (1,1))
    
    # u_plot = u_np[0,all_slice,all_slice,all_slice].T
    # v_plot = u_np[1,all_slice,all_slice,all_slice].T

    # u_mag = np.sqrt(u_plot**2 + v_plot**2)
    # pv_mesh.point_data['U_mag'] = u_mag.ravel()
    # pv_mesh.point_data['U velocity'] = u_plot.ravel()
    # pv_mesh.point_data['V velocity'] = v_plot.ravel()
    
    # pv_mesh.set_mesh_display('U_mag',clim = [0,1.5],cmap ='jet')
    # pv_mesh.show()
   