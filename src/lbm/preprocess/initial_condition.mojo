from layout import TileTensor,LayoutTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from std.collections import InlineArray
from std.collections import Set,Dict
from src.utils import Vector,ContextTileTensor
from std.utils.numerics import nan,isnan
from src.lbm.units import UnitSystem
from src.lbm.kernels.utils.equilibrium import f_eq
from src.lbm.kernels.utils.load_and_store import store_f
from src.lbm.constants import cs_squared
from std.reflection import get_function_name

def _do_nothing[
        float_dtype:DType,
        D:Int
        ]
        (x:Scalar[float_dtype],y:Scalar[float_dtype],z:Scalar[float_dtype],
        velocity:Vector[float_dtype,D]
        ) 
        capturing -> List[Vector[float_dtype,D]]:
        du =Vector[float_dtype,D](fill=0)
        dv =Vector[float_dtype,D](fill=0)
        return [du,dv]



def initialize_fluid_at_rest[
    float_dtype:DType,
    f_origin:Origin[mut=True],
    nx:Int,ny:Int,nz:Int,
    D:Int,Q:Int,
    lattice_model:LatticeModel[D,Q,float_dtype,DType.int32],
    FLayoutType:TensorLayout,
    //,
    grid:LBM_Grid[lattice_model,nx,ny,nz,_],
    config:LBM_Config = LBM_Config(),
    *,
    f_dtype:DType = config.f_dtype.value() if config.f_dtype else float_dtype
    ]    
    (
    f:TileTensor[f_dtype,FLayoutType,f_origin],
    ) raises:

    def at_rest[
        float_dtype:DType,D:Int
        ]
        (
        x:Scalar[float_dtype],y:Scalar[float_dtype],z:Scalar[float_dtype],mut u:Vector[float_dtype,D]
        ) capturing:
        u *= 0.
    initialize_f_from_func[grid,config,u=at_rest](f,None,1,1)
         


def initialize_f_from_func[
    float_dtype:DType,
    f_origin:Origin[mut=True],
    nx:Int,ny:Int,nz:Int,
    D:Int,Q:Int,
    lattice_model:LatticeModel[D,Q,float_dtype,DType.int32],
    FLayoutType:TensorLayout,
    //,
    grid:LBM_Grid[lattice_model,nx,ny,nz,_],
    config:LBM_Config = LBM_Config(),
    *,
    u: def[float_dtype:DType,D:Int]
        (Scalar[float_dtype],Scalar[float_dtype],Scalar[float_dtype],mut Vector[float_dtype,D]) 
        capturing,

    deriv_u: Optional[def[float_dtype:DType,D:Int]
        (Scalar[float_dtype],Scalar[float_dtype],Scalar[float_dtype],Vector[float_dtype,D]) 
        capturing -> List[Vector[float_dtype,D]]] = None,

    f_dtype:DType = config.f_dtype.value() if config.f_dtype else float_dtype
    ]    
    (
    f:TileTensor[f_dtype,FLayoutType,f_origin],
    unitSystem:Optional[UnitSystem[float_dtype,D]],
    rho:Scalar[float_dtype],
    tau:Scalar[float_dtype],
    ) raises:

    comptime weights = lattice_model.weights
    comptime float_directions = lattice_model.float_directions
    # TODO: Add Parallel Code For This for very large elements
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                grid_indices =  grid.get_grid_coordinates(i,j,k)
                
                velocity = Vector[float_dtype,D](fill = 0.)
                u(grid_indices[0],grid_indices[1],grid_indices[2],velocity)
                velocity*= unitSystem.value().U.C_phys_to_lat() if unitSystem else 1.
                
                index:InlineArray[Int,3] = [i,j,k]
                
                u_dot_u = velocity.dot(velocity)
                
                comptime assert D == 2,'adding neq only works for 2D for now'

                comptime for q in range(Q):
                    f_i = f_eq[config.DDF_shift](weights[q],rho,velocity,u_dot_u,float_directions[q])
                    comptime if deriv_u:
                        comptime u_func = deriv_u.value()    
                        grad = u_func(grid_indices[0],grid_indices[1],grid_indices[2],velocity.copy())
                        if unitSystem:
                            comptime for d in range(D): # Scale Gradient to lattice units
                                grad[d] *= unitSystem.value().U.C_phys_to_lat()/unitSystem.value().L.C_phys_to_lat()
                        f_i += fi_neq(q,weights[q],rho,tau,grad,float_directions)
                    store_f[config.use_float16c](f,f_i,index,q)


def fi_neq[
    float_dtype:DType,D:Int,Q:Int
    ](
    i:Int,
    weight:Scalar[float_dtype],
    rho:Scalar[float_dtype],
    tau:Scalar[float_dtype],
    grad:List[Vector[float_dtype,D]],
    float_directions: InlineArray[Vector[float_dtype, D],Q]
    ) -> Scalar[float_dtype]:
    
    fi_neq: Scalar[float_dtype] = 0.
    
    comptime for alpha in range(D):
        comptime for beta in range(D):
            Sab = calculate_Sab(grad,alpha,beta)
            Qiab = float_directions[i][alpha]*float_directions[i][beta]
            comptime if alpha == beta:
                Qiab -= cs_squared
            fi_neq += Qiab*Sab

    comptime inv_cs_squared = 1/(cs_squared)
    fi_neq *= (-weight*rho*(tau)*inv_cs_squared)*((tau-1)/tau)
    return fi_neq

        

def calculate_Sab[float_dtype:DType,D:Int](grad:List[Vector[float_dtype,D]],a:Int,b:Int) -> Scalar[float_dtype]:
    du_a_d = grad[a]
    du_b_d = grad[b]
    return 0.5*(du_a_d[b] + du_b_d[a])

