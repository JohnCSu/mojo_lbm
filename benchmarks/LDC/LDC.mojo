from max.gpu.host import DeviceContext
from layout import TileTensor,coord
from layout.tile_layout import Layout,row_major,TensorLayout,blocked_product,col_major
from std.python import Python, PythonObject
from std.sys import argv
from std.sys.defines import get_defined_int
from std.collections import InlineArray
from src.lbm import (
                    Flags,SOLID_NODE,FLUID_NODE,
                    LBM_Grid,LBM_Config,
                    get_D2Q9,set_exterior_walls,calculate_rho_and_velocity,
                    UnitSystem,DoubleBufferConfig
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
comptime N = get_defined_int["N", 512]()
comptime L = 1.
comptime dx = L/float_scalar(N-1)
comptime (nx,ny,nz) = (N,N,1)
comptime tile_size = 16
comptime grid = LBM_Grid[D2Q9,nx,ny,nz,tile_size](dx,[0.,0.,0.])
comptime config = DoubleBufferConfig(collision_op = Collisions.KBC,DDF_shift = False,LES = False)

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
    print(grid.layouts.n_tiles_x,grid.layouts.n_tiles_y,grid.layouts.n_tiles_z)
    print('Grid Dim: ',grid.GRID_DIM)
    print('BLOCK_SHAPE: ', grid.BLOCK_SHAPE)
    assert N % tile_size == 0, 'Tile Size must Divide N' 
    print(grid.layouts.n_tiles_x,grid.layouts.n_tiles_y,grid.layouts.n_tiles_z)

    var U_phs:float_scalar = 1.
    var U:float_scalar = 0.1
    var L_phys:float_scalar = 1.
    var Re:float_scalar = 10000

    var valid_Re:Set[Int] = {100,400,1000,3200,5000,7500,10000}

    var Re_=Int(Re)
    if Re_ not in valid_Re:
        raise Error('Re for LDC must be the following {}. Got Re = {} instead'.format(valid_Re,Re_))

    # units = UnitSystem(U_phs,U,radius,radius/dx,1.,Re = 100.)
    var units = grid.get_UnitSystem_with_Re(U_phs,U,L_phys,Re=Re)
    var tau = units.tau
    var dt = units.dt
    # Cs:float_scalar = 0.17

    print(units.tau,units.Re, units.kinematic_viscosity)

    var ctx = DeviceContext()

    var flags = ContextTileTensor[DType.uint8](ctx,grid.layouts.flag_layout)
    var bc = ContextTileTensor[float_dtype](ctx,grid.layouts.bc_layout)
    var f = ContextTileTensor[float_dtype](ctx,grid.layouts.f_layout)
    var f_out = ContextTileTensor[float_dtype](ctx,grid.layouts.f_layout)

    var u = ContextTileTensor[float_dtype](ctx,grid.layouts.velocity_layout)
    var rho = ContextTileTensor[float_dtype](ctx,grid.layouts.density_layout)
    var plot = get_plot(True)
    if plot:
        _ = pyvista_viewer_import()

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
    comptime LBM_ = double_buffer_kernel[type_of(grid.layouts.f_layout),type_of(grid.layouts.bc_layout),type_of(grid.layouts.flag_layout),grid,config]

    comptime get_u_and_rho = calculate_rho_and_velocity[type_of(grid.layouts.f_layout),type_of(grid.layouts.bc_layout),type_of(grid.layouts.flag_layout),type_of(grid.layouts.density_layout),type_of(grid.layouts.velocity_layout),grid,config]
    var calc_rho_and_u_gpu = ctx.compile_function[get_u_and_rho]()

    ctx.synchronize()
    
    # Animation Code
    var np = Python.import_module('numpy')
    var pd = Python.import_module('pandas')
    var u_np = (u.buffer_to_numpy()/U).reshape(D,nx,ny,nz)
    var pv_mesh = PythonObject()
    if plot:
        pv_mesh = grid_viewer[grid](subplot_shape= (3,1))
        var u_plot = u_np[0,all_slice,all_slice,all_slice].T
        var v_plot = u_np[1,all_slice,all_slice,all_slice].T
        var u_mag = np.sqrt(u_plot**2 + v_plot**2)
        pv_mesh.point_data['U_mag'] = u_mag.ravel()
        pv_mesh.point_data['U velocity'] = u_plot.ravel()
        pv_mesh.point_data['V velocity'] = v_plot.ravel()

        pv_mesh.set_mesh_display('U_mag',clim = [0,1],cmap ='jet')


        # Chart Data
        var v_benchmark = pd.read_csv('v_velocity_results.csv',sep = ',')
        var u_benchmark = pd.read_csv('u_velocity_results.txt',sep= '\t')

        pv_mesh.add_chart(Python.tuple(1,0),'V velocity',Python.tuple(0,L/2,0),Python.tuple(L,L/2,0),0,resolution= N,label = 'LBM')
        pv_mesh.add_data_to_chart(Python.tuple(1,0),v_benchmark['%x'],v_benchmark['{}'.format(Int(Re))],color = 'r',label = 'Ghia et al')

        pv_mesh.add_chart(Python.tuple(2,0),'U velocity',Python.tuple(L/2,0,0),Python.tuple(L/2,L,0),1,resolution= N,label = 'LBM')
        pv_mesh.add_data_to_chart(Python.tuple(2,0),u_benchmark['%y'],u_benchmark['{}'.format(Int(Re))],color = 'r',label = 'Ghia et al')


        pv_mesh.set_animation('LDC_Re{}.gif'.format(Int(Re)))
        # pv_mesh.show()
   
    var MAX_ITERS = get_iters(400_000)
    # Run Simulation
    for t in range(MAX_ITERS):
        ctx.enqueue_function[LBM_](f_out.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),tau,grid_dim = grid.GRID_DIM,block_dim = grid.BLOCK_SHAPE)
        ctx.enqueue_function[LBM_](f.gpu(),f_out.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),tau,grid_dim = grid.GRID_DIM,block_dim = grid.BLOCK_SHAPE)
        # ctx.enqueue_function[LBM_](f_out.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),tau,grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        # ctx.enqueue_function[LBM_](f.gpu(),f_out.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),tau,grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        if (t % max((MAX_ITERS//100),1)) == 0:
            ctx.synchronize()
            ctx.enqueue_function[get_u_and_rho](rho.gpu(),u.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),grid_dim = grid.GRID_DIM,block_dim = grid.BLOCK_SHAPE)
            ctx.synchronize()
            u_np = (u.buffer_to_numpy()/U).reshape(D,nx,ny,nz)
            print('step = {}, time = {} max ={} avg = {}'.format(t,2.*Scalar[float_dtype](t)*dt,u_np.max(),u_np.mean()))
            if plot:
                var u_plot = u_np[0,all_slice,all_slice,all_slice].T
                var v_plot = u_np[1,all_slice,all_slice,all_slice].T
                var u_mag = np.sqrt(u_plot**2 + v_plot**2)
                pv_mesh.point_data['U_mag'] = u_mag.ravel()
                pv_mesh.point_data['U velocity'] = u_plot.ravel()
                pv_mesh.point_data['V velocity'] = v_plot.ravel()
                pv_mesh.update_frame()
            ctx.synchronize()

    ctx.synchronize()
    # Get Final U and rho
    ctx.enqueue_function[get_u_and_rho](rho.gpu(),u.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),grid_dim = grid.GRID_DIM,block_dim = grid.BLOCK_SHAPE)
    ctx.synchronize()
    

    if plot:
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
   