from src.utils import Vector
from src.lbm.flags import cs_squared

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
def get_second_velocity_moment[
    float_dtype:DType,int_dtype:DType,D:Int,Q:Int,n_stress:Int,//,
    stress_indices:InlineArray[InlineArray[Scalar[int_dtype],2],n_stress],
    float_directions:InlineArray[Vector[float_dtype,D],Q],
    DDF_shift:Bool = False,
    ]
    (
        f_vec:Vector[float_dtype,Q],
    ) -> Vector[float_dtype,n_stress]:

    second_moment = Vector[float_dtype,n_stress](fill = 0.)
    comptime for n in range(n_stress): # This is very slow as its n_stress*Q ops
        comptime alpha = Int(stress_indices[n][0])
        comptime beta  = Int(stress_indices[n][1])
        comptime for q in range(Q):
            comptime c_ialpha_c_ibeta = float_directions[q][alpha]*float_directions[q][beta]
            second_moment[n] += f_vec[q]*c_ialpha_c_ibeta
    return second_moment

@always_inline
def get_strain_rate_tensor[
    float_dtype:DType,int_dtype:DType,D:Int,n_stress:Int,//,
    stress_indices:InlineArray[InlineArray[Scalar[int_dtype],2],n_stress],
    DDF_shift:Bool = False,
    ]
    (
        second_moment:Vector[float_dtype,n_stress],
        u:Vector[float_dtype,D],
        rho:Scalar[float_dtype],
        tau:Scalar[float_dtype],
    ) -> Vector[float_dtype,n_stress]:
    comptime assert n_stress == D*(D+1)//2
    strain_rate = Vector[float_dtype,n_stress](uninitialized =True)
    comptime if DDF_shift:
        rho_term = (rho-1)  
    else:
        rho_term = rho

    comptime for n in range(n_stress):
        comptime alpha = Int(stress_indices[n][0])
        comptime beta  = Int(stress_indices[n][1])
        strain_rate[n] = second_moment[n] - (rho_term*u[alpha]*u[beta])
        comptime if alpha == beta:
            strain_rate[n] -= (rho_term*cs_squared)
    # strain_rate /= (-2*(rho*cs_squared*(tau-0.5))) #-1.5/(rho*(tau-0.5)) # 1/(-2*(rho*cs_squared*(tau-0.5)))
    strain_rate *= -1.5/(rho*(tau-0.5))
    return strain_rate