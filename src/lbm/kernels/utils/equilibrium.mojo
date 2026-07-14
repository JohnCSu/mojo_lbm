from src.utils import Vector

@always_inline
def f_eq[dtype:DType,D:Int,//,DDF_shift:Bool = False](weight:Scalar[dtype],density:Scalar[dtype],velocity:Vector[dtype,D],u_dot_u:Scalar[dtype],direction:Vector[dtype,D]) -> Scalar[dtype]:
    comptime assert dtype.is_floating_point(), 'DType to BGK_collision term should be Float point like' # Weied using where statement cause compile error?
    ei_dot_u = velocity.dot(direction)
    comptime if DDF_shift:
        return weight*density*(3.*ei_dot_u + 4.5*ei_dot_u*ei_dot_u - 1.5*u_dot_u) +weight*(density - 1)
    else:
        return weight*density*(1 + 3.*ei_dot_u + 4.5*ei_dot_u*ei_dot_u - 1.5*u_dot_u)