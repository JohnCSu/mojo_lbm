from std.gpu.host import DeviceContext
from layout import TileTensor,coord
from layout.tile_layout import Layout,row_major,TensorLayout,blocked_product,col_major
from std.python import Python, PythonObject
from std.collections import InlineArray
from src.lbm import (
                    Flags,SOLID_NODE,FLUID_NODE,
                    LBM_Grid,LBM_Config,
                    get_D2Q9,set_exterior_walls,calculate_rho_and_velocity,
                    UnitSystem,DoubleBufferConfig,DoubleBufferSolver,Assembly
                    )
from src.lbm.constants import Collisions
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
comptime config = DoubleBufferConfig(collision_op = Collisions.KBC,DDF_shift = False,LES = False)

comptime all_slice = slice(None,None,None)

def main() raises:
    print(grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z)
    print('Grid Dim: ',grid.GRID_DIM)
    print('BLOCK_SHAPE: ', grid.BLOCK_SHAPE)
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

    units = grid.get_UnitSystem_with_Re(U_phs,U,L_phys,Re=Re)
    tau = units.tau
    dt = units.dt
    print(units.tau,units.Re, units.kinematic_viscosity)

    ctx = DeviceContext()
    
    assembly = Assembly[grid,config](ctx,units)
    solver = DoubleBufferSolver[grid,config](ctx)

    assembly.initialize_f_at_rest()

    assembly.set_exterior_walls('+Y',SOLID_NODE,[U,0],1.,in_lattice_units = True)
    assembly.set_exterior_walls('-Y',SOLID_NODE,[0,0],1.,in_lattice_units = True)
    assembly.set_exterior_walls('+X',SOLID_NODE,[0,0],1.,in_lattice_units = True)
    assembly.set_exterior_walls('-X',SOLID_NODE,[0,0],1.,in_lattice_units = True)

    u = ContextTileTensor[float_dtype](ctx,grid.layouts.velocity_layout)
    rho = ContextTileTensor[float_dtype](ctx,grid.layouts.density_layout)
    
    pv_view = pyvista_viewer_import()
    ctx.synchronize()

    #Compile Functions
    comptime get_u_and_rho = calculate_rho_and_velocity[grid.layouts.f_layout,grid.layouts.bc_layout,grid.layouts.flag_layout,grid.layouts.density_layout,grid.layouts.velocity_layout,grid,config]
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

   
    comptime MAX_ITERS = 400_000
    # Run Simulation
    for t in range(MAX_ITERS):
        solver.even_step(assembly,tau)
        solver.odd_step(assembly,tau)
        # ctx.enqueue_function(LBM_func,f_out.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),tau,grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        # ctx.enqueue_function(LBM_func,f.gpu(),f_out.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),tau,grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        if (t % (MAX_ITERS//100)) == 0:
            ctx.synchronize()
            ctx.enqueue_function(calc_rho_and_u_gpu,rho.gpu(),u.gpu(),assembly.f.gpu().as_immut(),assembly.bc.gpu().as_immut(),assembly.flags.gpu().as_immut(),grid_dim = grid.GRID_DIM,block_dim = grid.BLOCK_SHAPE)
            ctx.synchronize()
            u_np = (u.buffer_to_numpy()/U).reshape(nx,ny,nz,D)
            print('step = {}, time = {} max ={} avg = {}'.format(t,2.*Scalar[float_dtype](t)*dt,u_np.max(),u_np.mean()))
            u_plot = u_np[all_slice,all_slice,all_slice,0].T
            v_plot = u_np[all_slice,all_slice,all_slice,1].T
            u_mag = np.sqrt(u_plot**2 + v_plot**2)
            pv_mesh.point_data['U_mag'] = u_mag.ravel()
            pv_mesh.point_data['U velocity'] = u_plot.ravel()
            pv_mesh.point_data['V velocity'] = v_plot.ravel()
            pv_mesh.update_frame()
            ctx.synchronize()

    ctx.synchronize()
    # Get Final U and rho
    # ctx.enqueue_function(calc_rho_and_u_gpu,rho.gpu(),u.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
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
   