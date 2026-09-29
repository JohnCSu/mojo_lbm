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
                    get_D2Q9,set_exterior_walls,calculate_rho_and_velocity,set_exterior_walls_with_func,
                    UnitSystem,DoubleBufferConfig,EsotericPullConfig
                    )

from src.lbm.kernels.double_buffer import double_buffer_kernel
from src.utils import Vector,ContextTileTensor
from src.lbm.geometry.primatives import add_sphere,add_box
from src.lbm.geometry.rigidSphere import get_rigid_sphere
from src.lbm.constants import LBM_method,Bounceback_method

comptime float_dtype = DType.float32
comptime int_dtype = DType.int32
comptime float_scalar = Scalar[float_dtype]
comptime D2Q9 = get_D2Q9()
comptime D,Q = (2,9)
comptime N = get_defined_int["N", 32]()
comptime L = 0.41
comptime dx = L/float_scalar(N-1)
comptime (nx,ny,nz) = (5*N,N,1)
comptime tile_size = 1
comptime grid = LBM_Grid[D2Q9,nx,ny,nz,tile_size](dx,[0.,0.,0.])
comptime valid_bcs = {Flags.EQUILIBRIUM}
comptime config = DoubleBufferConfig(BCs = valid_bcs,DDF_shift = True)

comptime BLOCK_SHAPE = grid.BLOCK_SHAPE
comptime GRID_DIM = grid.GRID_DIM

comptime f_layout = grid.layouts.f_layout
comptime bc_layout = grid.layouts.bc_layout
comptime flag_layout = grid.layouts.flag_layout

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


def main() raises:    
    comptime assert N % tile_size == 0 , 'tile_size must divide N'
    print(grid.layouts.n_tiles_x,grid.layouts.n_tiles_y,grid.layouts.n_tiles_z)
    print('Grid Dim: ',GRID_DIM)
    print('BLOCK_SHAPE: ', BLOCK_SHAPE)
    assert N % tile_size == 0, 'Tile Size must Divide N' 
    print(grid.layouts.n_tiles_x,grid.layouts.n_tiles_y,grid.layouts.n_tiles_z)

    comptime U_phs:float_scalar = 0.2
    comptime U:float_scalar = 0.01
    comptime radius:float_scalar = 0.05
    comptime Cd:float_scalar = 5.57953523384
    comptime Cl:float_scalar = 0.010618948146
    # units = UnitSystem(U_phs,U,radius,radius/dx,1.,Re = 100.)
    var units = grid.get_UnitSystem_with_Re(U_phs,U,radius*2,Re=20.)
    var tau = units.tau
    var dt = units.dt
    print(units.tau,units.Re, units.kinematic_viscosity)

    var ctx = DeviceContext()
    
    var flags = ContextTileTensor[DType.uint8](ctx,flag_layout)
    var bc = ContextTileTensor[float_dtype](ctx,bc_layout)
    var f = ContextTileTensor[float_dtype](ctx,f_layout)
    var f_out = ContextTileTensor[float_dtype](ctx,f_layout)

    var u = ContextTileTensor[float_dtype](ctx,velocity_layout)
    var rho = ContextTileTensor[float_dtype](ctx,density_layout)

    # Set up
    comptime if not config.DDF_shift:
        f.fill(1./Float32(Q)) # Should be initialising with respective weight for each dist but should be ok as IC is fluid at rest
        f_out.fill(1./Float32(Q))
    else:
        f.fill(0.)
        f_out.fill(0.)


    # Boundary Conditions----------------------------

    var cen = 0.2//grid.dx # Ensure the center is adjustto be at a node
    print('Centre: ',[cen*grid.dx,cen*grid.dx,0.])
    var cyl = get_rigid_sphere[grid,LBM_method.DOUBLE_BUFFER,config](ctx,flags.cpu(),center = [cen*grid.dx,cen*grid.dx,0.],radius = radius)

    def inlet[float_dtype:DType,D:Int](x:Scalar[float_dtype],y:Scalar[float_dtype],z:Scalar[float_dtype],mut vel:InlineArray[Scalar[float_dtype],D]) capturing:
        comptime Um = 1.5*U_phs
        vel[0] = 4*Scalar[float_dtype](Um)*y*(L-y)/(L*L)
        vel[1] = 0.

    set_exterior_walls[grid,config](flags.cpu(),bc.cpu(),'+X',Flags.EQUILIBRIUM,[],1.)
    set_exterior_walls_with_func[grid,config,u = inlet](flags.cpu(),bc.cpu(),'-X',Flags.SOLID,units,1.)


    set_exterior_walls[grid,config](flags.cpu(),bc.cpu(),'-Y',Flags.SOLID,[0,0],1.)
    set_exterior_walls[grid,config](flags.cpu(),bc.cpu(),'+Y',Flags.SOLID,[0,0],1.)
    
    # Boundary Conditions----------------------------

    ctx.synchronize()
    # Copy To GPU()
    _ = flags.gpu()
    _ = bc.gpu()
    _ = f.gpu()
    _ = f_out.gpu()

    #Compile Functions
    comptime LBM_ = double_buffer_kernel[type_of(f_layout),type_of(bc_layout),type_of(flag_layout),grid,config]
    var LBM_func = ctx.compile_function[LBM_]()

    comptime get_u_and_rho = calculate_rho_and_velocity[type_of(f_layout),type_of(bc_layout),type_of(flag_layout),type_of(density_layout),type_of(velocity_layout),grid,config]
    var calc_rho_and_u_gpu = ctx.compile_function[get_u_and_rho]()

    ctx.synchronize()
    var u_lat_to_phys = units.U.C_lat_to_phys()

    var np = Python.import_module('numpy')

    var MAX_ITERS = get_iters(5)
    # Run Simulation
    for t in range(MAX_ITERS):
        ctx.enqueue_function[LBM_](f_out.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),tau,grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        ctx.enqueue_function[LBM_](f.gpu(),f_out.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),tau,grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        ctx.synchronize()
        cyl.bounceback[Bounceback_method.BOUZIDI](f.gpu(),flags.gpu().as_immut())
        ctx.synchronize()
        ctx.enqueue_function[get_u_and_rho](rho.gpu(),u.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        ctx.synchronize()
        var u_np = (u.buffer_to_numpy()*u_lat_to_phys).reshape(D,nx,ny,nz)
        print('step = {}, time = {} max ={} avg = {}'.format(2*t,2.*Scalar[float_dtype](t)*dt,u_np.max(),u_np.mean()))
        var forces = cyl.sum_force()
        var Fx = units.force.C_lat_to_phys()*Scalar[float_dtype](forces[0])
        var Fy = units.force.C_lat_to_phys()*Scalar[float_dtype](forces[1])
        var Cx = 2*Fx/(U_phs**2*(2*radius))
        var Cy = 2*Fy/(U_phs**2*(2*radius))
        
        print('Drag Force: {}, Target: {} Abs Error: {} Rel Error {}%'.format(Cx,Cd,abs(Cx-Cd),abs(Cd-Cx)/Cd*100) )
        print('Lift Force: {}, Target: {} Abs Error: {} Rel Error: {}%'.format(Cy,Cl,abs(Cy-Cl),abs(Cl-Cy)/Cl*100))
        ctx.synchronize()

    ctx.synchronize()
    # Get Final U and rho and drag
    ctx.enqueue_function[get_u_and_rho](rho.gpu(),u.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
    ctx.synchronize()
    cyl.bounceback[Bounceback_method.BOUZIDI](f.gpu(),flags.gpu().as_immut())
    ctx.synchronize()
    var forces = cyl.sum_force()
    var Fx = units.force.C_lat_to_phys()*Scalar[float_dtype](forces[0])
    var Fy = units.force.C_lat_to_phys()*Scalar[float_dtype](forces[1])

    var Cx = 2*Fx/(U_phs**2*(2*radius))
    var Cy = 2*Fy/(U_phs**2*(2*radius))
        
    var t = MAX_ITERS
    var u_np = (u.buffer_to_numpy()*u_lat_to_phys).reshape(D,nx,ny,nz)
    print('step = {}, time = {} max ={} avg = {}'.format(2*t,2.*Scalar[float_dtype](t)*dt,u_np.max(),u_np.mean()))
    print('Drag Force: {}, Target: {} Abs Error: {} Rel Error: {}%'.format(Cx,Cd,abs(Cx-Cd),abs(Cd-Cx)/Cd*100))
    print('Lift Force: {}, Target: {} Abs Error: {} Rel Error: {}%'.format(Cy,Cl,abs(Cy-Cl),abs(Cl-Cy)/Cl*100))
