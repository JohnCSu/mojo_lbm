from src.utils import Vector

@always_inline
def f_eq[dtype:DType,D:Int,//,DDF_shift:Bool = False](weight:Scalar[dtype],density:Scalar[dtype],velocity:Vector[dtype,D],u_dot_u:Scalar[dtype],direction:Vector[dtype,D]) -> Scalar[dtype]:
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
    var f_neq = f_vec - f_eq_vec
    comptime if post_collision:
        f_neq *= (tau/(tau-1)) # Post collision term
    return f_neq