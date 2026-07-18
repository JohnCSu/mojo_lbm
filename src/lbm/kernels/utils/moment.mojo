"""Provides moment extraction helpers for the LBM kernels.

Computes density, velocity, the non-equilibrium second-order moment, the
strain-rate tensor, and the squared Frobenius norm of the strain-rate tensor
from a vector of populations. A combined `get_density_and_velocity_for_eq_BC`
handles the equilibrium boundary-condition case where unknown populations are
replaced by their rest values.
"""
from src.utils import Vector
from src.lbm.constants import cs_squared
from .index import get_adjacent_idx,is_index_out_of_bounds

@always_inline
def get_density_and_velocity_for_eq_BC[
    float_dtype:DType,D:Int,Q:Int,int_dtype:DType,//,
    float_directions:InlineArray[Vector[float_dtype,D],Q],
    int_directions:InlineArray[Vector[int_dtype,D],Q],
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
        float_directions: The compile-time float-valued directions.
        int_directions: The compile-time integer-valued directions.
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
        comptime pull_direction = -int_directions[q]
        comptime for i in range(3):
            comptime if i < D:
                comptime pull_i = Int(pull_direction[i])
                pull_idx = index[i] + pull_i
                is_oob = (( (pull_idx < 0) or (pull_idx >= grid_shape[i]))  or is_oob)

        # We set unknown fs (i.e from out of bounds/wrapped around fs) to rest value
        fq = rest_f if is_oob else f_vec[q]
        rho += fq
        velocity += fq*float_directions[q]

    comptime if DDF_shift:
        rho += 1
    velocity /= rho
    return rho,velocity


@always_inline
def get_density[
    float_dtype:DType,Q:Int,//,
    DDF_shift:Bool = False
    ]
    (
    f_vec:Vector[float_dtype,Q],
    ) -> Scalar[float_dtype]:
    """Returns the lattice density from a distribution vector.

    Parameters:
        float_dtype: The `DType` of the computation.
        Q: The number of discrete velocities.
        DDF_shift: When `True`, add the `+1` offset used by the
            shifted-distribution form (defaults to `False`).

    Args:
        f_vec: The distribution vector.

    Returns:
        The lattice density `rho`.
    """
    comptime if DDF_shift:
        return f_vec.sum() + 1
    else:
        return f_vec.sum()


@always_inline
def get_velocity[
    float_dtype:DType,D:Int,Q:Int,//,
    float_directions:InlineArray[Vector[float_dtype,D],Q],
    ]
    (
    f_vec:Vector[float_dtype,Q],
    density:Scalar[float_dtype],
    ) -> Vector[float_dtype,D]:
    """Returns the lattice velocity from a distribution vector.

    Parameters:
        float_dtype: The `DType` of the computation.
        D: The spatial dimension.
        Q: The number of discrete velocities.
        float_directions: The compile-time float-valued directions.

    Args:
        f_vec: The distribution vector.
        density: The lattice density `rho`.

    Returns:
        The lattice velocity vector `u`.
    """
    velocity = Vector[float_dtype,D](fill =0)
    comptime for q in range(Q):
        velocity += f_vec[q]*float_directions[q]
    velocity /= density
    return velocity



@always_inline
def get_Qiab[float_dtype:DType,D:Int,Q:Int,//,
    float_directions:InlineArray[Vector[float_dtype, D], Q]]
    (f_neq:Vector[float_dtype,Q],a:Int,b:Int)
    -> Scalar[float_dtype]:

    """Returns the second-order moment $$\\sum_q f_q^{neq} e_{q,a} e_{q,b}$$.

    Parameters:
        float_dtype: The `DType` of the computation.
        D: The spatial dimension.
        Q: The number of discrete velocities.
        float_directions: The compile-time float-valued directions.

    Args:
        f_neq: The non-equilibrium distribution vector.
        a: The first velocity-component index.
        b: The second velocity-component index.

    Returns:
        The second-order moment $$Q_{a,b}$$.
    """
    Qiab:Scalar[float_dtype] = 0.
    comptime for q in range(0,Q):
        Qiab +=f_neq[q]*float_directions[q][a]*float_directions[q][b]
    return Qiab

@always_inline
def get_non_eq_second_order_moment[
    float_dtype:DType,
    int_dtype:DType,
    D:Int,
    Q:Int,
    n_stress:Int,
    //,
    float_directions:InlineArray[Vector[float_dtype, D], Q],
    stress_indices:InlineArray[InlineArray[Scalar[int_dtype],2],n_stress]
    ]
    (
        f_neq:Vector[float_dtype,Q],
    ) -> Vector[float_dtype,n_stress]:

    """Returns the symmetric non-equilibrium second-order moment vector.

    Parameters:
        float_dtype: The `DType` of the computation.
        int_dtype: The `DType` of the stress indices.
        D: The spatial dimension.
        Q: The number of discrete velocities.
        n_stress: The number of symmetric stress components; must equal
            `D * (D + 1) / 2`.
        float_directions: The compile-time float-valued directions.
        stress_indices: The compile-time symmetric stress-index pairs.

    Args:
        f_neq: The non-equilibrium distribution vector.

    Returns:
        A `Vector` of length `n_stress` holding the second-order moments.
    """
    Q_neq = Vector[float_dtype,n_stress](uninitialized=True)
    comptime assert n_stress == D*(D+1)//2
    comptime for n in range(n_stress):
        comptime alpha = Int(stress_indices[n][0])
        comptime beta  = Int(stress_indices[n][1])
        Q_neq[n] = get_Qiab[float_directions](f_neq,alpha,beta)
        # *(-1/(2*rho*cs_squared*(tau)))
    return Q_neq


@always_inline
def get_strain_rate_tensor[
    float_dtype:DType,
    n_stress:Int,//
    ](
    Q_neq:Vector[float_dtype,n_stress],rho:Scalar[float_dtype],tau:Scalar[float_dtype]
    )-> Vector[float_dtype,n_stress]:
    """Returns the strain-rate tensor from the non-equilibrium second moment.

    Computes $$S = Q^{neq} \\cdot \\frac{-1}{2 \\rho c_s^2 \\tau}$$.

    Parameters:
        float_dtype: The `DType` of the computation.
        n_stress: The number of symmetric stress components.

    Args:
        Q_neq: The non-equilibrium second-order moment vector.
        rho: The lattice density.
        tau: The relaxation time.

    Returns:
        The strain-rate tensor packed into a `Vector` of length `n_stress`.
    """
    return Q_neq*(-1/(2*rho*cs_squared*(tau)))




@always_inline
def get_strain_rate_tensor_norm_squared
    [
    float_dtype:DType,int_dtype:DType,n_stress:Int,//,
    stress_indices:InlineArray[InlineArray[Scalar[int_dtype],2],n_stress]
    ]
    (
    strain_rate_tensor:Vector[float_dtype,n_stress]
    ) -> Scalar[float_dtype]:

    """Returns the squared Frobenius norm of the strain-rate tensor.

    Doubles the contribution of off-diagonal components to account for the
    symmetry of the tensor.

    Parameters:
        float_dtype: The `DType` of the computation.
        int_dtype: The `DType` of the stress indices.
        n_stress: The number of symmetric stress components.
        stress_indices: The compile-time symmetric stress-index pairs.

    Args:
        strain_rate_tensor: The strain-rate tensor packed into a `Vector`.

    Returns:
        The squared Frobenius norm $$\\|S\\|_F^2$$.
    """
    ss = strain_rate_tensor*strain_rate_tensor
    s_norm_squared = Scalar[float_dtype](0)
    comptime for n in range(n_stress):
        comptime if stress_indices[n][0] != stress_indices[n][1]: # Alpha != Beta -> off diagonals
            s_norm_squared += ss[n]*2 # 2 as is symmetric tensor so double count off-diagonals
        else:
            s_norm_squared += ss[n]
    return s_norm_squared
