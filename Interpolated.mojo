'''
2D Cylinder Benchmark for Channel at Re=20 by Schäfer-Turek (DFG) Benchmark.

https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html

This benchmark tests the drag calculation against the results by DFG:

Drag coefficient: Cd = 5.57953523384
Lift coefficient: Cl = 0.010618948146

LBM with bounceback creates the notorious stair case effect (or pixelisation/voxelisation) of the smooth surface 
so calculating drag can require a very fine mesh (especially if the force is mainly due to skin friction and not pressure).
The drag force is can reach good agreement with benchmark at courser meshes as it is dominated by pressure.
The lift force requries as much finer mesh to get good agreement as drag force approximately cancel out and so is much more
sensitive to the staircase effect

Results (DDF shifting Turned on) after 800K iterations. Values may change and vary
1024 x 256:
    Drag Force: 5.644561, Target: 5.579535 Abs Error: 0.06502581 Rel Error: 1.1654341%
    Lift Force: 0.013367771, Target: 0.010618948 Abs Error: 0.0027488228 Rel Error: 25.886017% 

    TRT:
    Drag Force: 5.6207423, Target: 5.579535 Abs Error: 0.041207314 Rel Error: 0.73854387%
    Lift Force: 0.013535484, Target: 0.010618948 Abs Error: 0.0029165354 Rel Error: 27.465387%
    
2560 x 512:

    SRT:
    Drag Force: 5.58883, Target: 5.579535 Abs Error: 0.009294987 Rel Error: 0.1665907%
    Lift Force: 0.011819623, Target: 0.010618948 Abs Error: 0.001200675 Rel Error: 11.3069105%

    TRT:
    Drag Force: 5.5817976, Target: 5.579535 Abs Error: 0.0022625923 Rel Error: 0.04055163%
    Lift Force: 0.012483776, Target: 0.010618948 Abs Error: 0.0018648272 Rel Error: 17.561317%


5120 x 1024:
    Drag Force: 5.5599337, Target: 5.579535 Abs Error: 0.019601345 Rel Error: 0.35130787%
    Lift Force: 0.011516033, Target: 0.010618948 Abs Error: 0.00089708436 Rel Error: 8.447959%
    
'''
from max.gpu.host import DeviceContext
from layout import TileTensor,coord
from layout.tile_layout import Layout,row_major,TensorLayout,blocked_product,col_major
from std.python import Python, PythonObject
from std.collections import InlineArray
from src.lbm import (
                    Flags,SOLID_NODE,FLUID_NODE,
                    LBM_Grid,LBM_Config,
                    get_D2Q9,set_exterior_walls,calculate_rho_and_velocity,set_exterior_walls_with_func,
                    UnitSystem,TiledLayouts,RuntimeParams,DoubleBufferConfig,
                    Collisions,Bounceback_method
                    )

from src.utils import Vector,ContextTileTensor
from src.lbm.geometry.primatives import add_sphere,add_box
from src.lbm.geometry import RigidStationaryObject
from src.visualization import pyvista_viewer_import,grid_viewer

from src.lbm import Assembly,Solver,OutputRequest
from src.lbm.geometry.rigidSphere import get_rigid_sphere
from src.lbm.geometry.interpolated_BB import object_bounceback_kernel

comptime float_dtype = DType.float32
comptime int_dtype = DType.int32
comptime float_scalar = Scalar[float_dtype]
comptime D2Q9 = get_D2Q9()
comptime D,Q = (2,9)
comptime N = 256
comptime L = 0.41
comptime dx = L/float_scalar(N-1)
comptime (nx,ny,nz) = (5*N,N,1)
comptime tile_size = (16,16,1)
comptime grid = LBM_Grid[D2Q9,nx,ny,nz,tile_size](dx,[0.,0.,0.])

comptime config = DoubleBufferConfig(DDF_shift = False,collision_op = Collisions.SRT,BCs= {Flags.EQUILIBRIUM})

comptime BLOCK_SHAPE = grid.BLOCK_SHAPE
comptime GRID_DIM = grid.GRID_DIM


comptime all_slice = slice(None,None,None)


def main() raises:    

    print(grid.layouts.n_tiles_x,grid.layouts.n_tiles_y,grid.layouts.n_tiles_z)
    print('Grid Dim: ',GRID_DIM)
    print('BLOCK_SHAPE: ', BLOCK_SHAPE)
    
    print(grid.layouts.n_tiles_x,grid.layouts.n_tiles_y,grid.layouts.n_tiles_z)

    comptime U_phs:float_scalar = 0.2
    comptime U:float_scalar = 0.05

    comptime radius:float_scalar = 0.05
    comptime Cd:float_scalar = 5.57953523384
    comptime Cl:float_scalar = 0.010618948146
    # units = UnitSystem(U_phs,U,radius,radius/dx,1.,Re = 100.)
    units = grid.get_UnitSystem_with_tau(U_phs,0.75,radius*2,Re=20.)
    tau = units.tau
    dt = units.dt
    print(units.tau,units.Re,units.U, units.kinematic_viscosity)

    ctx = DeviceContext()
    # Set up
    assembly = Assembly[grid,config](ctx,units)
    solver  = Solver[grid,config](ctx)
    output = OutputRequest[grid,config](ctx,units)

    # Boundary Conditions----------------------------
   
    def inlet[float_dtype:DType,D:Int](x:Scalar[float_dtype],y:Scalar[float_dtype],z:Scalar[float_dtype],mut vel:InlineArray[Scalar[float_dtype],D]) capturing:
        comptime Um = 1.5*U_phs
        vel[0] = 4*Scalar[float_dtype](Um)*y*(L-y)/(L*L)
        vel[1] = 0.

    assembly.set_exterior_walls('+X',Flags.EQUILIBRIUM,[],1.,in_lattice_units = True)
    assembly.set_exterior_walls[u_func = inlet]('-X',Flags.SOLID,rho = 1.*units.density.C_lat_to_phys(),in_lattice_units = False) # Kinda annoying syntax should overlaod this
    assembly.set_exterior_walls('-Y',Flags.SOLID,[0,0],1.,in_lattice_units = True)
    assembly.set_exterior_walls('+Y',Flags.SOLID,[0,0],1.,in_lattice_units = True)
    
    
    # comptime bounceback = Bounceback_method.BOUZIDI
    # Circle
    cen = 0.2//grid.dx # Ensure the center is adjustto be at a node
    center:List[type_of(cen)] = [cen*grid.dx,cen*grid.dx,0.]

    var sphere = get_rigid_sphere[grid,config.lbm_method,config](
        ctx, assembly.flags.cpu(), center, radius,
    )

    def run_loop[MAX_ITERS:Int,N_samples:Int,bounceback:Bounceback_method](save:Bool) capturing raises -> PythonObject:
        assembly.initialize_f_at_rest()

        np = Python.import_module('numpy')
        u_np = output.velocity_as_numpy(False) # Col major

        C_arr = np.zeros(Python.tuple(N_samples,4),np.float32)
        i = 0
        # Run Simulation
        for t in range(MAX_ITERS):
            sphere.bounceback[bounceback](assembly.f.gpu(),assembly.flags.gpu())
            solver.even_step(assembly,tau)
            sphere.bounceback[bounceback](assembly.f2.value().gpu(),assembly.flags.gpu())
            solver.odd_step(assembly,tau)

            if (t % (MAX_ITERS//N_samples)) == 0:
                ctx.synchronize()
                output.velocity_frame[True](assembly)
                force = sphere.sum_force(in_lattice_units = False)
                
                Fx,Fy = force[0],force[1]
                Cx = (2*Fx/(U_phs**2*(2*radius)))
                Cy = (2*Fy/(U_phs**2*(2*radius)))
                
                Cx_rel_error = abs(Cd-Cx)/Cd*100
                Cy_rel_error = abs(Cl-Cy)/Cl*100
                C_arr[i,0] = Cx
                C_arr[i,1] = Cy
                C_arr[i,2] = Cx_rel_error
                C_arr[i,3] = Cy_rel_error

                i+=1
                print('Drag Force: {}, Target: {} Abs Error: {} Rel Error {}%'.format(Cx,Cd,abs(Cx-Cd), Cx_rel_error))
                print('Lift Force: {}, Target: {} Abs Error: {} Rel Error: {}%'.format(Cy,Cl,abs(Cy-Cl),Cy_rel_error))
                print('step = {}, time = {} max ={} avg = {}'.format(t,2.*Scalar[float_dtype](t)*dt,u_np.max(),u_np.mean()))
                
                ctx.synchronize()

        ctx.synchronize()

        if save:
            collision = 'SRT' if config.collision_op == Collisions.SRT else 'TRT'
            filename = 'Drag_{}_{}_{}.txt'.format(N,collision,'Bouzidi' if bounceback == Bounceback_method.BOUZIDI else 'MidGrid')
            np.savetxt(filename,C_arr)
        
        ctx.synchronize()
        return C_arr
    comptime MAX_ITERS = 200_000
    comptime N_samples = 100

    C_interp = run_loop[MAX_ITERS,N_samples,Bounceback_method.BOUZIDI](False)
    C_midgrid = run_loop[MAX_ITERS,N_samples,Bounceback_method.MID_GRID](False)

    all_slice = slice(None,None,None)
    
    plt = Python.import_module('matplotlib.pyplot')
    np = Python.import_module('numpy')
    Cy_err_interp = C_interp[all_slice,-1]
    Cy_err_midgrid = C_midgrid[all_slice,-1]

    x = np.arange(N_samples)*MAX_ITERS//N_samples*2
    plt.plot(x,Cy_err_interp/100.,label='Bouzidi BB')
    plt.plot(x,Cy_err_midgrid/100.,label ='MidGrid BB')
    plt.legend()
    plt.ylabel('rel Error')
    plt.yscale('log')
    plt.xlabel('Iteration Count')
    plt.title('Relative Error For Cy over time')
    plt.show()