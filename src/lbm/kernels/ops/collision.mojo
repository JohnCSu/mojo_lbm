from src.utils import Vector
from src.lbm.kernels.utils.equilibrium import f_eq

@always_inline
def SRT[
    float_dtype:DType,int_dtype:DType,D:Int,Q:Int,//,
    directions:InlineArray[Vector[int_dtype, D], Q],
    weights:Vector[float_dtype,Q],
    DDF_shift:Bool,
    ](mut f_vec:Vector[float_dtype,Q],velocity:Vector[float_dtype,D],rho:Scalar[float_dtype],tau:Scalar[float_dtype]):
    u_dot_u = velocity.dot(velocity)
    inv_tau = 1./tau # This is faster by 0.4 ms on the 256^3 benchmark
    comptime for q in range(Q):
        comptime direction = directions[q].cast_to[float_dtype]()
        comptime weight = weights[q]
        f_vec[q] -= inv_tau*(f_vec[q]- f_eq[DDF_shift](weight,rho,velocity,u_dot_u,direction))


def TRT[
    float_dtype:DType,int_dtype:DType,D:Int,Q:Int,//,
    directions:InlineArray[Vector[int_dtype, D], Q],
    weights:Vector[float_dtype,Q],
    opposite_indices:InlineArray[Scalar[int_dtype],Q],
    DDF_shift:Bool,
    ]
    (
    mut f_vec:Vector[float_dtype,Q],
    velocity:Vector[float_dtype,D],
    rho:Scalar[float_dtype],
    tau_symm:Scalar[float_dtype],
    tau_asymm:Scalar[float_dtype],
    ):        
    inv_tau_symm = 1/tau_symm
    inv_tau_asymm = 1/tau_asymm
    u_dot_u = velocity.dot(velocity)

    comptime direction0 = directions[0].cast_to[float_dtype]()
    # Rest direction is just regular SRT
    f_vec[0] -= tau_symm*(f_vec[0]- f_eq[DDF_shift](weights[0],rho,velocity,u_dot_u,direction0))

    comptime for q in range(1,Q):
        comptime direction = directions[q].cast_to[float_dtype]()
        comptime weight = weights[q]
        comptime opp_q = Int(opposite_indices[q])
        
        f_symm = (f_vec[q] + f_vec[opp_q])*0.5 # We correct shift in feq
        f_asymm = (f_vec[q] - f_vec[opp_q])*0.5 # No shift 
        
        f_eq_q = f_eq[DDF_shift](weight,rho,velocity,u_dot_u,direction)
        f_eq_oppq = f_eq[DDF_shift](weight,rho,velocity,u_dot_u,direction)

        f_eq_symm = (f_eq_q + f_eq_oppq)*0.5
        f_eq_asymm = (f_eq_q - f_eq_oppq)*0.5
        
        f_vec[q] -= (inv_tau_symm*(f_symm- f_eq_symm) + inv_tau_asymm*(f_asymm - f_eq_asymm))
    


    
