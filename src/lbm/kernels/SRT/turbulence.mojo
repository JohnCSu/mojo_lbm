
from src.utils import Vector
from std.math import sqrt

@always_inline
def get_Smagorinsky_LES_tau[
    float_dtype:DType,int_dtype:DType,n_stress:Int,//,
    stress_indices:InlineArray[InlineArray[Scalar[int_dtype],2],n_stress]
    ]
    (strain_rate_tensor:Vector[float_dtype,n_stress],Cs:Scalar[float_dtype]) -> Scalar[float_dtype]:
    # Calculate Frobenius Norm
    ss = strain_rate_tensor*strain_rate_tensor
    s_norm = Scalar[float_dtype](0)
    comptime for n in range(n_stress):
        comptime if stress_indices[n][0] != stress_indices[n][1]: # Alpha != Beta
            s_norm += ss[n]*2
        else:
            s_norm += ss[n]

    v_eddy_lat = (Cs*Cs)*(sqrt(2*s_norm))
    return 3*v_eddy_lat # tau = v_eddt/cs^2 --> 1/cs^2 == 3 
