"""Provides equilibrium and non-equilibrium distribution helpers.

`f_eq` evaluates the equilibrium for a single direction, `get_f_eq_vec`
evaluates the equilibrium for all `Q` directions, and `get_f_noneq_vec`
extracts the non-equilibrium part with an optional post-collision scaling.
"""
from src.utils import Vector

@always_inline
def f_eq[dtype:DType,D:Int,//,DDF_shift:Bool = False](weight:Scalar[dtype],density:Scalar[dtype],velocity:Vector[dtype,D],u_dot_u:Scalar[dtype],direction:Vector[dtype,D]) -> Scalar[dtype]:
    """Returns the equilibrium distribution for a single discrete velocity.

    When `DDF_shift` is `True`, applies the shifted-distribution form that
    trades a `density - 1` offset for improved numerical stability with
    Float16C.

    Parameters:
        dtype: The `DType` of the computation; must be floating-point.
        D: The spatial dimension.
        DDF_shift: When `True`, use the DDF-shifted equilibrium form
            (defaults to `False`).

    Args:
        weight: The quadrature weight `w_i`.
        density: The lattice density `rho`.
        velocity: The lattice velocity vector `u`.
        u_dot_u: The squared velocity magnitude `u . u`.
        direction: The discrete velocity direction `e_i`.

    Returns:
        The equilibrium population `f_i^{eq}`.
    """
    comptime assert dtype.is_floating_point(), 'DType to BGK_collision term should be Float point like' # Weied using where statement cause compile error?
    ei_dot_u = velocity.dot(direction)
    comptime if DDF_shift:
        return weight*density*(3.*ei_dot_u + 4.5*ei_dot_u*ei_dot_u - 1.5*u_dot_u) +weight*(density - 1)
    else:
        return weight*density*(1 + 3.*ei_dot_u + 4.5*ei_dot_u*ei_dot_u - 1.5*u_dot_u)


@always_inline
def get_f_eq_vec[float_dtype:DType,D:Int,Q:Int,//,
        float_directions:InlineArray[Vector[float_dtype, D], Q],
        weights:Vector[float_dtype, Q],
        DDF_shift:Bool,
        ]
        (f_vec:Vector[float_dtype,Q],density:Scalar[float_dtype],velocity:Vector[float_dtype,D])
        -> Vector[float_dtype,Q]:
    """Returns the equilibrium distribution for all `Q` discrete velocities.

    Parameters:
        float_dtype: The `DType` of the computation.
        D: The spatial dimension.
        Q: The number of discrete velocities.
        float_directions: The compile-time float-valued directions.
        weights: The compile-time quadrature weights.
        DDF_shift: When `True`, use the DDF-shifted equilibrium form.

    Args:
        f_vec: Unused; present for API symmetry with `get_f_noneq_vec`.
        density: The lattice density `rho`.
        velocity: The lattice velocity vector `u`.

    Returns:
        A `Vector` of length `Q` holding the equilibrium populations.
    """
    var f_eq_vec =Vector[float_dtype,Q](uninitialized = True)
    u_dot_u = velocity.dot(velocity)
    comptime for q in range(Q):
        f_eq_vec[q] = f_eq[DDF_shift](weights[q],density,velocity,u_dot_u,float_directions[q])
    return f_eq_vec


@always_inline
def get_f_noneq_vec[float_dtype:DType,Q:Int,//,
        post_collision:Bool,
        ]
        (f_vec:Vector[float_dtype,Q],f_eq_vec:Vector[float_dtype,Q],tau:Scalar[float_dtype])
        -> Vector[float_dtype,Q]:
    """Returns the non-equilibrium part of `f` for all `Q` directions.

    Parameters:
        float_dtype: The `DType` of the computation.
        Q: The number of discrete velocities.
        post_collision: When `True`, scale by `tau / (tau - 1)` to recover
            the pre-collision non-equilibrium from post-collision values.

    Args:
        f_vec: The current distribution vector.
        f_eq_vec: The equilibrium distribution vector.
        tau: The relaxation time, used only when `post_collision` is `True`.

    Returns:
        A `Vector` of length `Q` holding the non-equilibrium populations.
    """
    var f_neq = f_vec - f_eq_vec
    comptime if post_collision:
        f_neq *= (tau/(tau-1)) # Post collision term
    return f_neq
