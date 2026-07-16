from src.utils import Vector
from src.lbm.constants import cs_squared

@always_inline
def get_density_and_velocity_for_eq_BC[
    float_dtype:DType,D:Int,Q:Int,//,
    DDF_shift:Bool = False]
    (
        f_vec:Vector[float_dtype,Q],
        float_directions:InlineArray[Vector[float_dtype,D],Q],
        weights:Vector[float_dtype,Q],
        index:InlineArray[Int,3],
        pull_indices:InlineArray[InlineArray[Int,3],Q]) 
    -> Tuple[Scalar[float_dtype],Vector[float_dtype,D]]:

    var velocity = Vector[float_dtype,D](fill = 0.)
    var rho:Scalar[float_dtype] = 0

    comptime for q in range(Q):
        comptime if DDF_shift:
            rest_f:Scalar[float_dtype] = 0.
        else:
            rest_f = weights[q]
            
        is_oob = False
        comptime for i in range(3):
            # So if any of the indices wraps is_oob is always i.e out of bounds
            is_oob = ((abs(pull_indices[q][i] - index[i]) > 1) or is_oob) 

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
    
    ss = strain_rate_tensor*strain_rate_tensor
    s_norm_squared = Scalar[float_dtype](0)
    comptime for n in range(n_stress):
        comptime if stress_indices[n][0] != stress_indices[n][1]: # Alpha != Beta -> off diagonals
            s_norm_squared += ss[n]*2 # 2 as is symmetric tensor so double count off-diagonals
        else:
            s_norm_squared += ss[n]
    return s_norm_squared
