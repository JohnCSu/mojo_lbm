
from src.utils import Vector
from std.math import sqrt
from .moment import get_strain_rate_tensor_norm_squared
@always_inline
def get_Smagorinsky_LES_tau[
    float_dtype:DType,int_dtype:DType,n_stress:Int,//,
    stress_indices:InlineArray[InlineArray[Scalar[int_dtype],2],n_stress]
    ]
    (strain_rate_tensor:Vector[float_dtype,n_stress],Cs:Scalar[float_dtype]) -> Scalar[float_dtype]:
    # Calculate Frobenius Norm
    s_norm = get_strain_rate_tensor_norm_squared[stress_indices](strain_rate_tensor)
    v_eddy_lat = (Cs*Cs)*(sqrt(2*s_norm))
    return 3*v_eddy_lat # tau = v_eddt/cs^2 --> 1/cs^2 == 3 

