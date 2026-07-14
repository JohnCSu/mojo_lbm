from layout import TileTensor,LayoutTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from std.collections import InlineArray
from std.collections import Set,Dict
from src.utils import Vector,ContextTileTensor
from std.utils.numerics import nan,isnan
from src.lbm.units import UnitSystem
from src.lbm.kernels.utils.equilibrium import f_eq
from src.lbm.kernels.utils.load_and_store import store_f

def initialise_f_from_func[
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
    f_dtype:DType = config.f_dtype.value() if config.f_dtype else float_dtype
    ]    
    (
    f:TileTensor[f_dtype,FLayoutType,f_origin],
    unitSystem:Optional[UnitSystem[float_dtype,D]],
    rho:Scalar[float_dtype] = 1.,
    ) raises:

    comptime weights = lattice_model.weights
    comptime float_directions = lattice_model.float_directions
    # TODO: Add Parallel Code For This for very large elements
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                velocity = Vector[float_dtype,D](fill = 0.)
                grid_indices =  grid.get_grid_coordinates(i,j,k)
                u(grid_indices[0],grid_indices[1],grid_indices[2],velocity)
                # Scale if unit system was passed in
                velocity*= unitSystem.value().U.C_phys_to_lat() if unitSystem else 1.
                u_dot_u = velocity.dot(velocity)
                index:InlineArray[Int,3] = [i,j,k]
                comptime for q in range(Q):
                    f_i = f_eq[config.DDF_shift](weights[q],rho,velocity,u_dot_u,float_directions[q])
                    store_f[config.use_float16c](f,f_i,index,q)
