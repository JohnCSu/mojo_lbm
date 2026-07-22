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
