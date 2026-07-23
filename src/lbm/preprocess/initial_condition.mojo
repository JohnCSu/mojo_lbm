"""Initializes the distribution function `f` from analytic velocity fields.

`initialize_fluid_at_rest` fills `f` with the equilibrium distribution for a
fluid at rest, and `initialize_f_from_func` does the same from a caller
supplied velocity function, optionally adding the non-equilibrium correction
derived from the velocity gradient.
"""
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
        """Returns a zero velocity-gradient pair for use as a default `deriv_u`.

        Args:
            x: The physical `x` coordinate (unused).
            y: The physical `y` coordinate (unused).
            z: The physical `z` coordinate (unused).
            velocity: The velocity vector (unused).

        Returns:
            A list of two zero vectors representing `du` and `dv`.
        """
        du =Vector[float_dtype,D](fill=0)
        dv =Vector[float_dtype,D](fill=0)
        return [du,dv]



def initialize_fluid_at_rest[
    float_dtype:DType,
    f_origin:Origin[mut=True],
    nx:Int,ny:Int,nz:Int,
    D:Int,Q:Int,
    lattice_model:Lattice[D,Q,float_dtype,DType.int32],
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

    """Initializes `f` with the equilibrium distribution for a fluid at rest.

    Delegates to `initialize_f_from_func` with an `at_rest` velocity function
    that zeros the velocity at every node.

    Parameters:
        grid: The compile-time `LBM_Grid` describing the domain.
        config: The `LBM_Config` used to select storage options.
        f_dtype: The storage `DType` for `f` (defaults to the config's
            `f_dtype` or `float_dtype`).

    Args:
        f: The distribution function tile tensor to fill.
    """
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
    lattice_model:Lattice[D,Q,float_dtype,DType.int32],
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

    """Initializes `f` from an analytic velocity field.

    Walks every node, evaluates the supplied `u` function in physical grid
    coordinates, scales to lattice units, and stores the equilibrium
    distribution. When `deriv_u` is supplied, also adds the non-equilibrium
    correction derived from the velocity gradient.

    Parameters:
        grid: The compile-time `LBM_Grid` describing the domain.
        config: The `LBM_Config` used to select storage options.
        u: A compile-time function that writes the physical-space velocity
            at `(x, y, z)` into a mutable `Vector`.
        deriv_u: Optional compile-time function returning the velocity
            gradient `[du, dv]` at `(x, y, z)` for the non-equilibrium
            correction (defaults to `None`).
        f_dtype: The storage `DType` for `f` (defaults to the config's
            `f_dtype` or `float_dtype`).

    Args:
        f: The distribution function tile tensor to fill.
        unitSystem: Optional unit system for converting physical velocities
            and gradients to lattice units.
        rho: The lattice density to initialize with.
        tau: The relaxation time used by the non-equilibrium correction.
    """
    comptime weights = lattice_model.weights
    comptime directions = lattice_model.directions
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

                # comptime assert D == 2,'adding neq only works for 2D for now'

                comptime for q in range(Q):
                    comptime float_direction = directions[q].cast_to[float_dtype]()
                    f_i = f_eq[config.DDF_shift](weights[q],rho,velocity,u_dot_u,float_direction)
                    comptime if deriv_u:
                        comptime u_func = deriv_u.value()
                        grad = u_func(grid_indices[0],grid_indices[1],grid_indices[2],velocity.copy())
                        if unitSystem:
                            comptime for d in range(D): # Scale Gradient to lattice units
                                grad[d] *= unitSystem.value().U.C_phys_to_lat()/unitSystem.value().L.C_phys_to_lat()
                        f_i += fi_neq[directions](q,weights[q],rho,tau,grad)
                    store_f[config.use_float16c](f,f_i,index,q)


def fi_neq[
    float_dtype:DType,int_dtype:DType,D:Int,Q:Int,//,
    directions: InlineArray[Vector[int_dtype, D],Q]
    ](
    i:Int,
    weight:Scalar[float_dtype],
    rho:Scalar[float_dtype],
    tau:Scalar[float_dtype],
    grad:List[Vector[float_dtype,D]],
    
    ) -> Scalar[float_dtype]:

    """Returns the non-equilibrium correction to `f_i` for a velocity gradient.

    Computes the second-order non-equilibrium population

    $$f_i^{neq} = -w_i \\rho \\frac{\\tau - 1}{\\tau c_s^2}
    \\sum_{\\alpha,\\beta} Q_{i\\alpha\\beta} S_{\\alpha\\beta}$$

    where `S` is the symmetric strain tensor derived from `grad` and `Q` is
    the lattice stress tensor.

    Parameters:
        float_dtype: The `DType` of the computation.
        D: The spatial dimension.
        Q: The number of discrete velocities.

    Args:
        i: The discrete velocity index.
        weight: The quadrature weight `w_i`.
        rho: The lattice density.
        tau: The relaxation time.
        grad: The velocity gradient `[du, dv]` as a list of `Vector`s.
        float_directions: The float-valued discrete velocity directions.

    Returns:
        The non-equilibrium correction `f_i^{neq}`.
    """
    fi_neq: Scalar[float_dtype] = 0.
    direction = directions[i].cast_to[float_dtype]()

    comptime for alpha in range(D):
        comptime for beta in range(D):
            Sab = calculate_Sab(grad,alpha,beta)
            Qiab = direction[i][alpha]*direction[i][beta]
            comptime if alpha == beta:
                Qiab -= cs_squared
            fi_neq += Qiab*Sab

    comptime inv_cs_squared = 1/(cs_squared)
    fi_neq *= (-weight*rho*(tau)*inv_cs_squared)*((tau-1)/tau)
    return fi_neq



def calculate_Sab[float_dtype:DType,D:Int](grad:List[Vector[float_dtype,D]],a:Int,b:Int) -> Scalar[float_dtype]:
    """Returns the symmetric strain component `S_{a,b}` from a velocity gradient.

    Parameters:
        float_dtype: The `DType` of the computation.
        D: The spatial dimension.

    Args:
        grad: The velocity gradient as a list of `Vector`s.
        a: The first strain-tensor index.
        b: The second strain-tensor index.

    Returns:
        The symmetric strain `0.5 * (grad[a][b] + grad[b][a])`.
    """
    du_a_d = grad[a]
    du_b_d = grad[b]
    return 0.5*(du_a_d[b] + du_b_d[a])
