"""Provides bounce-back and equilibrium boundary-condition kernels.

Implements wall boundary conditions for the lattice Boltzmann method,
including half-way bounce-back and equilibrium (Dirichlet) boundary
conditions.
"""
from src.utils import Vector
from layout import TileTensor,coord
from src.lbm.kernels.utils.index import get_adjacent_idx
from src.lbm.kernels.utils.load_and_store import load_f
from src.lbm.kernels.utils.equilibrium import get_f_eq_vec
from std.utils.numerics import nan,isnan
from std.math import sqrt

def moving_wall_bc[
    float_dtype:DType,int_dtype:DType,D:Int,Q:Int,//,
    directions:InlineArray[Vector[int_dtype, D], Q],
    opposite_indices:InlineArray[Scalar[int_dtype], Q],
    weights:Vector[float_dtype,Q],
    use_float16c:Bool,
    *,
    start_idx:Int = 1,
    non_temporal:Bool = False
    ]
    (
    mut f_vec:Vector[float_dtype,Q],
    pull_flags:InlineArray[UInt8,Q],
    bc:TileTensor[float_dtype,...],
    index:InlineArray[Int,3],
    grid_shape:InlineArray[Int,3],
    ): 
    """Applies the bounce-back wall boundary condition.

    For directions whose pull neighbor is a solid node, replaces the
    distribution with the bounce-back value computed from the wall
    velocity and density. Optionally includes the bounce-back
    contribution from the opposite direction.

    Parameters:
        float_dtype: The floating-point `DType` for computation.
        int_dtype: The integer `DType` for the velocity directions.
        f_dtype: The storage `DType` of the distribution function `f`.
        D: The spatial dimension.
        Q: The number of discrete velocities.
        include_bounceback: When `True`, add the bounce-back
            contribution from the opposite direction's distribution.
        directions: The compile-time discrete velocity directions.
        opposite_indices: The compile-time map from each direction to
            its opposite.
        weights: The lattice weights.
        use_float16c: When `True`, decode Float16C storage when
            loading.
        start_idx: The index of the first non-rest direction (defaults
            to 1).
        non_temporal: When `True`, issue non-temporal loads (defaults
            to `False`).

    Args:
        f_vec: The mutable distribution vector to update in place.
        pull_flags: The mutable flags gathered from pull neighbors.
        f: The distribution function tile tensor.
        flags: The `uint8` tile tensor labeling each node.
        bc: The boundary-condition tile tensor holding wall velocity
            and density.
        index: The `(x, y, z)` index of the current node.
        grid_shape: The `[nx, ny, nz]` shape of the grid.
    """

    var velocity = Vector[float_dtype,D](uninitialized = True)
    var rho:Scalar[float_dtype]

    comptime for q in range(start_idx,Q):
        comptime direction = directions[q]
        pull_index = get_adjacent_idx[shift = -1](index,grid_shape,direction) # Pulling Scheme
        if Flags.is[Flags.SOLID](pull_flags[q]):
            comptime float_direction = directions[q].cast_to[float_dtype]()
            comptime weight = weights[q]
            comptime for ii in range(D):
                velocity[ii] = bc.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2],ii)))[0]
            rho = bc.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2],D)))[0]
            f_vec[q] += 2.*3.*weight*rho*(float_direction.dot(velocity))


@always_inline
def equilibrium_bc[
    float_dtype:DType,int_dtype:DType,D:Int,Q:Int,//,
    directions:InlineArray[Vector[int_dtype, D], Q],
    weights:Vector[float_dtype,Q],
    DDF_shift:Bool,
    ]
    (
    mut f_vec:Vector[float_dtype,Q],
    pull_flags:InlineArray[UInt8,Q],
    bc:TileTensor[float_dtype,...],
    index:InlineArray[Int,3],
    grid_shape:InlineArray[Int,3],
    ):
    """Applies the equilibrium boundary condition.

    For nodes flagged as equilibrium, sets the distribution vector to
    the equilibrium distribution computed from the prescribed or
    free-stream velocity and density. `NaN` values in the
    boundary-condition tensor indicate free-stream components.

    Parameters:
        float_dtype: The floating-point `DType` for computation.
        int_dtype: The integer `DType` for the velocity directions.
        D: The spatial dimension.
        Q: The number of discrete velocities.
        directions: The compile-time discrete velocity directions.
        weights: The lattice weights.
        DDF_shift: When `True`, shift the equilibrium by the weights
            for improved numerical stability.

    Args:
        f_vec: The mutable distribution vector to set in place.
        pull_flags: The mutable flags gathered from pull neighbors.
        bc: The boundary-condition tile tensor holding the target
            velocity and density.
        index: The `(x, y, z)` index of the current node.
        grid_shape: The `[nx, ny, nz]` shape of the grid.
    """
    current_flag = pull_flags[0] # comptime assert gurantees this is the flag for the current node
    if Flags.is[Flags.EQUILIBRIUM](current_flag):
        var velocity = Vector[float_dtype,D](uninitialized = True)
        comptime for ii in range(D):
            velocity[ii] = bc.load(coord[DType.uint32]((index[0],index[1],index[2],ii)))[0]
        rho = bc.load(coord[DType.uint32]((index[0],index[1],index[2],D)))[0]
        
        rho_local,u_l = get_density_and_velocity_for_eq_BC[directions,DDF_shift](f_vec,weights,index,grid_shape)
        
        u_local = u_l if isnan(velocity[0]) else velocity # nan means the vel is free
        rho_local = rho_local if isnan(rho) else rho # Nan means density is free

        f_vec = get_f_eq_vec[directions,weights,DDF_shift](f_vec,rho_local,u_local)




@always_inline
def get_density_and_velocity_for_eq_BC[
    float_dtype:DType,D:Int,Q:Int,int_dtype:DType,//,
    directions:InlineArray[Vector[int_dtype,D],Q],
    DDF_shift:Bool = False]
    (
        f_vec:Vector[float_dtype,Q],
        weights:Vector[float_dtype,Q],
        index:InlineArray[Int,3],
        grid_shape:InlineArray[Int,3],
    )
    -> Tuple[Scalar[float_dtype],Vector[float_dtype,D]]:

    """Returns density and velocity for an equilibrium boundary node.

    Treats populations pulled from out-of-bounds neighbors as the rest value
    (the quadrature weight, or zero when `DDF_shift` is `True`) so that
    unknown directions do not corrupt the moment sums.

    Parameters:
        float_dtype: The `DType` of the computation.
        D: The spatial dimension.
        Q: The number of discrete velocities.
        int_dtype: The `DType` of the integer directions.
        directions: The compile-time discrete velocity directions.
        DDF_shift: When `True`, use the DDF-shifted rest value
            (defaults to `False`).

    Args:
        f_vec: The current distribution vector.
        weights: The quadrature weights.
        index: The `(x, y, z)` index of the central node.
        grid_shape: The `[nx, ny, nz]` shape of the grid.

    Returns:
        A tuple of `(rho, velocity)` as a scalar and a `Vector`.
    """
    var velocity = Vector[float_dtype,D](fill = 0.)
    var rho:Scalar[float_dtype] = 0

    comptime for q in range(Q):
        comptime if DDF_shift:
            rest_f:Scalar[float_dtype] = 0.
        else:
            rest_f = weights[q]

        is_oob = False
        comptime pull_direction = -directions[q]
        comptime for i in range(3):
            comptime if i < D:
                comptime pull_i = Int(pull_direction[i])
                pull_idx = index[i] + pull_i
                is_oob = (( (pull_idx < 0) or (pull_idx >= grid_shape[i]))  or is_oob)

        # We set unknown fs (i.e from out of bounds/wrapped around fs) to rest value
        fq = rest_f if is_oob else f_vec[q]
        rho += fq
        comptime float_direction = directions[q].cast_to[float_dtype]()
        velocity += fq*float_direction

    comptime if DDF_shift:
        rho += 1
    velocity /= rho
    return rho,velocity