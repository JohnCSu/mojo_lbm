from std.gpu.host import DeviceContext
from layout import TileTensor,coord
from layout.tile_layout import Layout,row_major,TensorLayout,blocked_product,col_major
from std.python import Python, PythonObject
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
from src.lbm.geometry import ImmersedObject
from src.lbm.output import calculate_drag_around_object

comptime float_dtype = DType.float32
comptime int_dtype = DType.int32
comptime float_scalar = Scalar[float_dtype]
comptime D2Q9 = get_D2Q9()
comptime D,Q = (2,9)
comptime N = 32
comptime L = 0.41
comptime dx = L/float_scalar(N-1)
comptime (nx,ny,nz) = (5*N,N,1)
comptime tile_size = 1
comptime grid = LBM_Grid[D2Q9,nx,ny,nz,tile_size](dx,[0.,0.,0.])
comptime valid_bcs = {Flags.EQUILIBRIUM}
comptime config = DoubleBufferConfig(BCs = valid_bcs,DDF_shift = True)

comptime BLOCK_SHAPE = grid.BLOCK_SHAPE
comptime GRID_DIM = grid.GRID_DIM

comptime flag_tile = col_major[tile_size,tile_size,1]()
comptime f_tile = col_major[tile_size,tile_size,1,Q]()
comptime bc_tile = col_major[tile_size,tile_size,1,D+1]()

comptime flag_tiler = col_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z]()
comptime f_tiler = col_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z,1]()
comptime bc_tiler = col_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z,1]()

# comptime flag_layout = blocked_product(flag_tile,flag_tiler)
comptime flag_layout = col_major[grid.n_tiles_x,grid.n_tiles_y,grid.n_tiles_z]()
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

    comptime U_phs:float_scalar = 0.2
    comptime U:float_scalar = 0.01
    comptime radius:float_scalar = 0.05
    comptime Cd:float_scalar = 5.57953523384
    comptime Cl:float_scalar = 0.010618948146
    # units = UnitSystem(U_phs,U,radius,radius/dx,1.,Re = 100.)
    units = grid.get_UnitSystem_with_Re(U_phs,U,radius*2,Re=20.)
    tau = units.tau
    dt = units.dt
    print(units.tau,units.Re, units.kinematic_viscosity)

    ctx = DeviceContext()
    
    flags = ContextTileTensor[DType.uint8](ctx,flag_layout)
    bc = ContextTileTensor[float_dtype](ctx,bc_layout)
    f = ContextTileTensor[float_dtype](ctx,f_layout)
    f_out = ContextTileTensor[float_dtype](ctx,f_layout)

    u = ContextTileTensor[float_dtype](ctx,velocity_layout)
    rho = ContextTileTensor[float_dtype](ctx,density_layout)

    # Set up
    comptime if not config.DDF_shift:
        f.fill(1./Float32(Q)) # Should be initialising with respective weight for each dist but should be ok as IC is fluid at rest
        f_out.fill(1./Float32(Q))
    else:
        f.fill(0.)
        f_out.fill(0.)


    # Boundary Conditions----------------------------

    cyl = ImmersedObject[grid]()

    cen = 0.2//grid.dx # Ensure the center is adjustto be at a node
    print('Centre: ',[cen*grid.dx,cen*grid.dx,0.])
    cyl.add_sphere(flags.cpu(),center = [cen*grid.dx,cen*grid.dx,0.],radius = radius )
    cyl_ids = cyl.to_ContextTileTensor(ctx)
    
    force_layout = row_major(coord[int_dtype]((cyl_ids.size(),D)))
    force_tensor = ContextTileTensor[float_dtype](ctx,force_layout)

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
    comptime LBM_ = double_buffer_kernel[f_layout,bc_layout,flag_layout,grid,config]
    LBM_func = ctx.compile_function[LBM_,LBM_]()

    comptime calculate_drag_ = calculate_drag_around_object[f_layout,flag_layout,grid,config]
    calculate_drag = ctx.compile_function[calculate_drag_,calculate_drag_]()

    comptime get_u_and_rho = calculate_rho_and_velocity[f_layout,bc_layout,flag_layout,density_layout,velocity_layout,grid,config]
    calc_rho_and_u_gpu = ctx.compile_function[get_u_and_rho,get_u_and_rho]()

    ctx.synchronize()
    u_lat_to_phys = units.U.C_lat_to_phys()

    np = Python.import_module('numpy')

    comptime MAX_ITERS = 5
    # Run Simulation
    for t in range(MAX_ITERS):
        ctx.enqueue_function(LBM_func,f_out.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),tau,grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        ctx.enqueue_function(LBM_func,f.gpu(),f_out.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),tau,grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        ctx.synchronize()
        ctx.enqueue_function(calculate_drag,f.gpu().as_immut(),flags.gpu().as_immut(),cyl_ids.gpu(),force_tensor.gpu(),grid_dim = cyl_ids.size()//256+1, block_dim = 256)
        ctx.synchronize()
        ctx.enqueue_function(calc_rho_and_u_gpu,rho.gpu(),u.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        ctx.synchronize()
        u_np = (u.buffer_to_numpy()*u_lat_to_phys).reshape(D,nx,ny,nz)
        print('step = {}, time = {} max ={} avg = {}'.format(2*t,2.*Scalar[float_dtype](t)*dt,u_np.max(),u_np.mean()))
        force_np = force_tensor.buffer_to_numpy().reshape(cyl_ids.size(),D)
        Fx,Fy = force_np.sum(axis=0)[0],force_np.sum(axis=0)[1]
        Fx,Fy = units.force.C_lat_to_phys()*Fx,units.force.C_lat_to_phys()*Fy
        Cx = float_scalar(py=2*Fx/(U_phs**2*(2*radius)))
        Cy = float_scalar(py=2*Fy/(U_phs**2*(2*radius)))
        
        print('Drag Force: {}, Target: {} Abs Error: {} Rel Error {}%'.format(Cx,Cd,abs(Cx-Cd),abs(Cd-Cx)/Cd*100) )
        print('Lift Force: {}, Target: {} Abs Error: {} Rel Error: {}%'.format(Cy,Cl,abs(Cy-Cl),abs(Cl-Cy)/Cl*100))
        ctx.synchronize()

    ctx.synchronize()
    # Get Final U and rho and drag
    ctx.enqueue_function(calc_rho_and_u_gpu,rho.gpu(),u.gpu(),f.gpu().as_immut(),bc.gpu().as_immut(),flags.gpu().as_immut(),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
    ctx.synchronize()
    ctx.enqueue_function(calculate_drag,f.gpu().as_immut(),flags.gpu().as_immut(),cyl_ids.gpu(),force_tensor.gpu(),grid_dim = cyl_ids.size()//256+1, block_dim = 256)
    force_np = force_tensor.buffer_to_numpy().reshape(cyl_ids.size(),D)
    Fx,Fy = force_np.sum(axis=0)[0],force_np.sum(axis=0)[1]
    Fx,Fy = units.force.C_lat_to_phys()*Fx,units.force.C_lat_to_phys()*Fy

    Cx = float_scalar(py=2*Fx/(U_phs**2*(2*radius)))
    Cy = float_scalar(py=2*Fy/(U_phs**2*(2*radius)))
        
    t = MAX_ITERS
    u_np = (u.buffer_to_numpy()*u_lat_to_phys).reshape(D,nx,ny,nz)
    print('step = {}, time = {} max ={} avg = {}'.format(2*t,2.*Scalar[float_dtype](t)*dt,u_np.max(),u_np.mean()))
    print('Drag Force: {}, Target: {} Abs Error: {} Rel Error: {}%'.format(Cx,Cd,abs(Cx-Cd),abs(Cd-Cx)/Cd*100))
    print('Lift Force: {}, Target: {} Abs Error: {} Rel Error: {}%'.format(Cy,Cl,abs(Cy-Cl),abs(Cl-Cy)/Cl*100))
