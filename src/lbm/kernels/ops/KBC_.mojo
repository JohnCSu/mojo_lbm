"""Implements the entropy-stabilized KBC collision operator for D2Q9.

Provides the Karlin-Bösch-Chikatamarla entropy-stabilized collision
operator, which adaptively blends the shear and higher-order
non-equilibrium contributions to maintain entropy balance.
"""
from src.utils import Vector
from src.lbm.kernels.utils.equilibrium import f_eq,get_f_eq_vec



# def moment_3rd_order_neq_abg[
#     float_dtype:DType,
#     int_dtype:DType,
#     D:Int,
#     Q:Int,
#     //,
#     float_directions:InlineArray[Vector[float_dtype,D],Q]
#     ]
#     (f_vec:Vector[float_dtype,Q],rho:Scalar[float_dtype],u:Vector[float_dtype,D],i:Int,a:Int,b:Int,g:Int)
#     -> Scalar[float_dtype]
#     :
#     Q_eq_abg = moment_3rd_order_eq_abg(rho,u,a,b,g)
    
#     out:Scalar[float_dtype] = 0.
#     comptime for q in range(Q):
#         comptime c_q = float_directions[q]
#         out += c_q[a]*c_q[b]*c_q[g]*f_vec[q] - Q_eq_abg
#     return out


# def moment_3rd_order_eq_abg[
#     float_dtype:DType,
#     D:Int,
#     //,
#     ]
#     (rho:Scalar[float_dtype],u:Vector[float_dtype,D],a:Int,b:Int,g:Int)
#     -> Scalar[float_dtype]
#     :
#     x1 = u[a] if b==g else 0
#     x2 = u[b] if a==g else 0
#     x3 = u[g] if a==b else 0 
#     return rho*cs_squared*(x1 + x2 + x3)


# @always_inline
# def second_order_hermite_ab[
#     float_dtype:DType,
#     int_dtype:DType,
#     D:Int,
#     n_stress:Int,
#     //,
#     stress_indices:InlineArray[InlineArray[Scalar[int_dtype],2],n_stress]
#     ]
#     (
#     float_direction_i:Vector[float_dtype, D],
#     ) -> Vector[float_dtype,n_stress]:

#     Q_i = Vector[float_dtype,n_stress](uninitialized=True)
#     comptime assert n_stress == D*(D+1)//2

#     comptime for n in range(n_stress):
#         comptime alpha = Int(stress_indices[n][0])
#         comptime beta  = Int(stress_indices[n][1])
#         Q_i[n] = float_direction_i[alpha]*float_direction_i[beta]
#         comptime if alpha == beta:
#             Q_i[n] -= cs_squared # Diagonal
#         else:
#             Q_i[n] *= 2 # Takeinto account symmetry
#     return Q_i
 

# @always_inline
# def third_order_hermite_abg_ci[
#     float_dtype:DType,
#     int_dtype:DType,
#     D:Int,
#     //,
#     ]
#     (
#     ci:Vector[float_dtype, D],
#     a:Int,b:Int,g:Int,
#     ) -> Scalar[float_dtype]:

#     Q_i_abg = ci[a]*ci[b]*ci[g]
    
#     x1 = ci[a] if b==g else 0
#     x2 = ci[b] if a==g else 0
#     x3 = ci[g] if a==b else 0 
#     return Q_i_abg - cs_squared*(x1+x2+x3)






@always_inline
def entropic_inner_product[
    float_dtype:DType,
    Q:Int,
    //,
    ]
    (
    a:Vector[float_dtype,Q],
    b:Vector[float_dtype,Q],
    f_eq:Vector[float_dtype,Q],
    ) -> Scalar[float_dtype]:
    """Computes the entropic inner product weighted by the equilibrium.

    Evaluates
    $$\\langle a, b \\rangle_f =
    \\sum_q \\frac{a_q b_q}{f_q^{\\mathrm{eq}}}$$,
    used in the KBC entropy balance estimate.

    Parameters:
        float_dtype: The floating-point `DType` for computation.
        Q: The number of discrete velocities.

    Args:
        a: The first vector.
        b: The second vector.
        f_eq: The equilibrium distribution vector used as the
            weighting.

    Returns:
        The entropic inner product as a scalar.
    """
    out:Scalar[float_dtype] = 0.
    comptime for q in range(Q):
        out += a[q]*b[q]/f_eq[q]

    return out



@always_inline
def get_shear_D2Q9[
    float_dtype:DType,N:Int,Q:Int = 9,//,
    ]
    (stress_neq:Vector[float_dtype,N]) -> Vector[float_dtype,Q]:
    """Decomposes the non-equilibrium stress into pure shear modes for D2Q9.

    Maps the stress components (xx, xy, yy) onto the nine discrete
    velocities, producing a shear-mode distribution suitable for the
    KBC collision operator.

    Parameters:
        float_dtype: The floating-point `DType` for computation.
        N: The number of independent stress components (must be 3).
        Q: The number of discrete velocities (defaults to 9).

    Args:
        stress_neq: The non-equilibrium stress tensor components.

    Returns:
        A `Q`-element vector of shear-mode distribution values.
    """
    comptime assert N == 3
    # xx - 0, xy - 1, yy - 2
    s = Vector[float_dtype,Q](uninitialized = True)
    N_xy = stress_neq[0] - stress_neq[2] # xx-yy
    s[0] = 0 # (0,0,0)    
    s[1] = N_xy*0.25 # We should assert paired up correctly
    s[2] = N_xy*0.25 # 
    
    s[3] = -N_xy*0.25
    s[4] = -N_xy*0.25

    s[5] = stress_neq[1]*0.25 # 1,1
    s[6] = stress_neq[1]*0.25 # -1,-1

    s[7] = -stress_neq[1]*0.25 # 1,-1
    s[8] = -stress_neq[1]*0.25 # -1,1

    return s



@always_inline
def KBC[
    float_dtype:DType,int_dtype:DType,D:Int,Q:Int,N:Int,//,
    directions:InlineArray[Vector[int_dtype, D], Q],
    weights:Vector[float_dtype,Q],
    DDF_shift:Bool,
    ](
    mut f_vec:Vector[float_dtype,Q],
    f_neq:Vector[float_dtype,Q],
    stress_neq:Vector[float_dtype,N],
    rho:Scalar[float_dtype],
    u:Vector[float_dtype,D],
    tau:Scalar[float_dtype],
    *,
    # min_gamma:Scalar[float_dtype] = 1.,
    # max_gamma:Scalar[float_dtype] = 3.,
    ):
    """Applies the KBC entropy-stabilized collision operator.

    Decomposes the non-equilibrium distribution into shear (`ds`) and
    higher-order (`dh`) parts, estimates the entropy balance, and
    blends them adaptively to prevent entropy growth.

    Parameters:
        float_dtype: The floating-point `DType` for computation.
        int_dtype: The integer `DType` for the velocity directions.
        D: The spatial dimension.
        Q: The number of discrete velocities.
        N: The number of independent stress components.
        directions: The compile-time discrete velocity directions.
        weights: The lattice weights.
        DDF_shift: When `True`, shift the equilibrium by the weights.

    Args:
        f_vec: The mutable distribution vector to collide in place.
        f_neq: The non-equilibrium distribution vector.
        stress_neq: The non-equilibrium stress tensor components.
        rho: The fluid density at the node.
        u: The fluid velocity at the node.
        tau: The relaxation time.
    """
    comptime assert (D==2 and Q == 9)
    comptime _eps = 1e-32
    var beta = 1/(2*tau)
    var inv_beta = 2*tau
    var f_equil = get_f_eq_vec[directions,weights,False](f_vec,rho,u)

    comptime if (D==2 and Q ==9):
        ds = rebind[Vector[float_dtype,Q]](get_shear_D2Q9(stress_neq))
        dh = f_neq - ds
        
        sp1 = entropic_inner_product(ds,dh,f_equil)
        sp2 = entropic_inner_product(dh,dh,f_equil)

        gamma = inv_beta - (2 -inv_beta) * sp1/(sp2 + _eps)
        # gamma = min(max(gamma, min_gamma), max_gamma)
        f_vec -= beta * (2*ds + gamma*dh)

    else:
        comptime assert False, 'KBC only for D2Q9 atm'
    
