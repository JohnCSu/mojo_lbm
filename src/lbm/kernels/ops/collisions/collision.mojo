"""Implements single- and multi-relaxation-time LBM collision operators.

Provides BGK (SRT), two-relaxation-time (TRT), and regularized (RLBM)
collision schemes for the lattice Boltzmann method.
"""
from src.utils import Vector
from src.lbm.constants import cs_squared
from src.lbm.kernels.utils.equilibrium import f_eq
from src.lbm.kernels.utils.checks import opposite_indices_are_adjacent,rest_direction_is_zero

@always_inline
def SRT[
    float_dtype:DType,int_dtype:DType,D:Int,Q:Int,//,DDF_shift:Bool
    ]
    (
        mut f_vec:Vector[float_dtype,Q],
        velocity:Vector[float_dtype,D],
        rho:Scalar[float_dtype],
        tau:Scalar[float_dtype],
        directions:InlineArray[Vector[int_dtype, D], Q],
        weights:Vector[float_dtype,Q]):
    """Applies the single-relaxation-time (BGK) collision operator.

    Updates the distribution vector in place using the
    Bhatnagar-Gross-Krook (BGK) approximation:
    $$f_q \\leftarrow f_q -
    \\frac{1}{\\tau}(f_q - f_q^{\\mathrm{eq}})$$.

    Parameters:
        float_dtype: The floating-point `DType` for computation.
        int_dtype: The integer `DType` for the velocity directions.
        D: The spatial dimension.
        Q: The number of discrete velocities.
        DDF_shift: When `True`, shift the equilibrium by the weights for
            improved numerical stability.

    Args:
        f_vec: The mutable distribution vector to collide in place.
        velocity: The fluid velocity at the node.
        rho: The fluid density at the node.
        tau: The relaxation time.
        directions: The discrete velocity directions.
        weights: The lattice weights.
    """
    var u_dot_u = velocity.dot(velocity)
    var inv_tau = 1./tau # This is faster by 0.4 ms on the 256^3 benchmark
    comptime for q in range(Q):
        var direction = directions[q].cast_to[float_dtype]()
        var weight = weights[q]
        f_vec[q] -= inv_tau*(f_vec[q]- f_eq[DDF_shift](weight,rho,velocity,u_dot_u,direction))

@always_inline
def TRT[
    float_dtype:DType,int_dtype:DType,D:Int,Q:Int,//,
    DDF_shift:Bool
    ](
    mut f_vec:Vector[float_dtype,Q],
    velocity:Vector[float_dtype,D],
    rho:Scalar[float_dtype],
    tau_symm:Scalar[float_dtype],
    tau_asymm:Scalar[float_dtype],
    directions:InlineArray[Vector[int_dtype, D], Q],
    weights:Vector[float_dtype,Q]):
    """Applies the two-relaxation-time collision operator.

    Splits the collision into symmetric and antisymmetric parts, each
    relaxed with its own relaxation time. Opposite directions must be
    stored at adjacent indices (q+1 is the opposite of q), and the
    rest direction must be the first element.

    Parameters:
        float_dtype: The floating-point `DType` for computation.
        int_dtype: The integer `DType` for the velocity directions.
        D: The spatial dimension.
        Q: The number of discrete velocities.
        DDF_shift: When `True`, shift the equilibrium by the weights for
            improved numerical stability.

    Args:
        f_vec: The mutable distribution vector to collide in place.
        velocity: The fluid velocity at the node.
        rho: The fluid density at the node.
        tau_symm: The relaxation time for the symmetric part.
        tau_asymm: The relaxation time for the antisymmetric part.
        directions: The discrete velocity directions.
        weights: The lattice weights.
    """
    var inv_tau_symm = 1/tau_symm
    var inv_tau_asymm = 1/tau_asymm
    var u_dot_u = velocity.dot(velocity)

    # comptime assert opposite_indices_are_adjacent(directions), 'Opposite velocity directions should be adjacent to each other e.g. q+1 = opp_q'
    # comptime assert rest_direction_is_zero(directions), 'Rest direction e.g [0,0,0] should be the first element'
    
    var direction0 = directions[0].cast_to[float_dtype]()

    # Rest direction is just regular SRT
    f_vec[0] -= inv_tau_symm*(f_vec[0]- f_eq[DDF_shift](weights[0],rho,velocity,u_dot_u,direction0))

    comptime for q in range(1,Q,2):
        comptime opp_q = q+1
        var direction = directions[q].cast_to[float_dtype]()
        var weight = weights[q]
        var opp_direction = directions[opp_q].cast_to[float_dtype]()
        
        var f_symm = (f_vec[q] + f_vec[opp_q])*0.5 # We correct shift in feq
        var f_asymm = (f_vec[q] - f_vec[opp_q])*0.5 # No shift 
                
        var f_eq_q = f_eq[DDF_shift](weight,rho,velocity,u_dot_u,direction, )
        var f_eq_oppq = f_eq[DDF_shift](weight,rho,velocity,u_dot_u,opp_direction, )

        var f_eq_symm = (f_eq_q + f_eq_oppq)*0.5
        var f_eq_asymm = (f_eq_q - f_eq_oppq)*0.5
        
        f_vec[q] -=  (inv_tau_symm*(f_symm- f_eq_symm) + inv_tau_asymm*(f_asymm - f_eq_asymm))
        f_vec[opp_q] -= (inv_tau_symm*(f_symm- f_eq_symm) + inv_tau_asymm*( (-f_asymm) - (-f_eq_asymm)))
    

    # comptime for q in range(1,Q):
    #     f_vec[q] = f_new[q] # Temp

    
@always_inline
def get_kbc_Qiab[
    float_dtype:DType,
    int_dtype:DType,
    D:Int,
    n_stress:Int,
    //,
    stress_indices:InlineArray[InlineArray[Scalar[int_dtype],2],n_stress]
    ]
    (
    float_direction_i:Vector[float_dtype, D],
    ) -> Vector[float_dtype,n_stress]:
    """Computes the second-order Hermite basis tensor for a direction.

    Evaluates
    $$H_{i,\\alpha\\beta}^{(2)} =
    c_{i\\alpha}c_{i\\beta} - c_s^2\\delta_{\\alpha\\beta}$$
    for each stress component, with symmetry factors applied.

    Parameters:
        float_dtype: The floating-point `DType` for computation.
        int_dtype: The integer `DType` for the stress indices.
        D: The spatial dimension.
        n_stress: The number of independent stress components.
        stress_indices: The compile-time index pairs `(α, β)` for each
            independent stress component.

    Args:
        float_direction_i: The floating-point velocity direction vector.

    Returns:
        A vector of `n_stress` Hermite basis values evaluated at the
        direction.
    """
    var Q_i = Vector[float_dtype,n_stress](uninitialized=True)
    comptime assert n_stress == D*(D+1)//2

    comptime for n in range(n_stress):
        comptime alpha = Int(stress_indices[n][0])
        comptime beta  = Int(stress_indices[n][1])
        Q_i[n] = float_direction_i[alpha]*float_direction_i[beta]
        comptime if alpha == beta:
            Q_i[n] -= cs_squared # Diagonal
        else:
            Q_i[n] *= 2 # Takeinto account symmetry

    return Q_i

def RLBM[
    float_dtype:DType,int_dtype:DType,D:Int,Q:Int,N:Int,//,
    stress_indices:InlineArray[InlineArray[Scalar[int_dtype],2],N],
    DDF_shift:Bool
    ](
        mut f_vec:Vector[float_dtype,Q],
        f_neq:Vector[float_dtype,Q],
        stress_neq:Vector[float_dtype,N],
        rho:Scalar[float_dtype],
        velocity:Vector[float_dtype,D],
        tau:Scalar[float_dtype],
        directions:InlineArray[Vector[int_dtype, D], Q],
        weights:Vector[float_dtype,Q],
        ):
    """Applies the regularized LBM collision operator.

    Reconstructs the non-equilibrium distribution from the
    non-equilibrium stress tensor using the second-order Hermite
    expansion, then relaxes with the single relaxation time.

    Parameters:
        float_dtype: The floating-point `DType` for computation.
        int_dtype: The integer `DType` for the velocity directions.
        D: The spatial dimension.
        Q: The number of discrete velocities.
        N: The number of independent stress components.
        stress_indices: The compile-time index pairs for each
            independent stress component.
        DDF_shift: When `True`, shift the equilibrium by the weights for
            improved numerical stability.

    Args:
        f_vec: The mutable distribution vector to collide in place.
        f_neq: The non-equilibrium distribution vector (unused by this
            operator).
        stress_neq: The non-equilibrium stress tensor components.
        rho: The fluid density at the node.
        velocity: The fluid velocity at the node.
        tau: The relaxation time.
        directions: The discrete velocity directions.
        weights: The lattice weights.
    """
    # var f_equil = Vector[float_dtype,Q](uninitialized = True)
    # var f_neq_reg = Vector[float_dtype,Q](uninitialized = True)
    var inv_tau = 1/tau
    var u_dot_u = velocity.dot(velocity)

    comptime for q in range(Q):
        var weight = weights[q]
        var float_direction = directions[q].cast_to[float_dtype]()
        var weight_div_2cs4 = weights[q]/(2*cs_squared*cs_squared)
        var Q_q = get_kbc_Qiab[stress_indices](float_direction) # Can pre compute this!
        var f_neq_reg = weight_div_2cs4*Q_q.dot(stress_neq)
        var f_equil = f_eq[DDF_shift](weight,rho,velocity,u_dot_u,float_direction)
        f_vec[q] = f_equil + (1-inv_tau)*f_neq_reg

# def central_polynomial_order_2[
#     float_dtype:DType,D:Int,Q:Int,//,
#     float_directions:InlineArray[Vector[float_dtype, D], Q]
#     ](u:Vector[float_dtype,D],i:Int,a:Int,b:Int) -> Scalar[float_dtype]:
#     comptime c = float_directions
#     Pab = ( c[i][a] - u[a]) * (c[i][b] - u[b]) - (cs_squared if a==b else Scalar[float_dtype](0.))
#     return Pab



        

# last modified by: muse-spark-1.2 on 2026/09/01
