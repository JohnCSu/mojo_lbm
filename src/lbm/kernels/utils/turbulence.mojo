"""Provides the Smagorinsky large-eddy simulation (LES) relaxation-time correction.

`get_Smagorinsky_LES_tau` computes the eddy relaxation time from the
strain-rate tensor and adds it to the base SRT `tau` to close the
sub-grid-scale model.
"""
from src.utils import Vector
from std.math import sqrt
from .moment import get_strain_rate_tensor_norm_squared
@always_inline
def get_Smagorinsky_LES_tau[
    float_dtype:DType,int_dtype:DType,n_stress:Int,//,
    stress_indices:InlineArray[InlineArray[Scalar[int_dtype],2],n_stress]
    ]
    (strain_rate_tensor:Vector[float_dtype,n_stress],Cs:Scalar[float_dtype]) -> Scalar[float_dtype]:
    """Returns the LES eddy relaxation time added by the Smagorinsky model.

    Computes $$\\tau_{eddy} = 3 C_s^2 \\sqrt{2 \\|S\\|_F^2}$$, where the
    factor of 3 comes from $$1/c_s^2$$.

    Parameters:
        float_dtype: The `DType` of the computation.
        int_dtype: The `DType` of the stress indices.
        n_stress: The number of symmetric stress components.
        stress_indices: The compile-time symmetric stress-index pairs.

    Args:
        strain_rate_tensor: The strain-rate tensor packed into a `Vector`.
        Cs: The Smagorinsky constant.

    Returns:
        The eddy relaxation time `tau_eddy` in lattice units.
    """
    # Calculate Frobenius Norm
    s_norm = get_strain_rate_tensor_norm_squared[stress_indices](strain_rate_tensor)
    v_eddy_lat = (Cs*Cs)*(sqrt(2*s_norm))
    return 3*v_eddy_lat # tau = v_eddt/cs^2 --> 1/cs^2 == 3
