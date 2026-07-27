"""Defines `RuntimeParams` for GPU-passable LBM runtime constants.

`RuntimeParams` carries the Smagorinsky constant `Cs` and the TRT magic
parameter to the GPU kernel and implements the `DevicePassable` protocol for
pass-by-value semantics.
"""
from std.builtin.device_passable import DevicePassable

struct RuntimeParams[float_dtype:DType](ImplicitlyCopyable):
    """Carries the Smagorinsky constant and TRT magic parameter to the GPU.

    Implements device-passable semantics so the parameters can be passed
    by value to GPU kernels.

    Parameters:
        float_dtype: The `DType` used for the stored scalars.
    """

    comptime device_type:AnyType = Self
    comptime Float = Scalar[Self.float_dtype]
    var Cs:Self.Float
    """The Smagorinsky constant (defaults to 0.1)."""
    var TRT_magic_param:Self.Float
    """The TRT magic parameter (defaults to 3/16)."""

    def __init__(
        out self,
        *,
        Cs:Self.Float = 0.1,
        TRT_magic_param:Self.Float = 3./16.
    ):
        """Constructs a `RuntimeParams` with the given constants.

        Args:
            Cs: The Smagorinsky constant (defaults to 0.1).
            TRT_magic_param: The TRT magic parameter (defaults to 3/16).
        """
        self.Cs = Cs
        self.TRT_magic_param = TRT_magic_param

    # @staticmethod
    # def get_type_name() -> String:
    #     return String(
    #         "RuntimeParams[",
    #         reflect[type_of(Self.float_dtype)]().name(),
    #         "]")
   
    def _to_device_type(
        self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self.copy()

    def tau_asymm(self,tau:Self.Float) -> Self.Float:
        """Returns the asymmetric TRT relaxation time for a given base `tau`.

        Computes $$\\tau_{asymm} = 0.5 + \\frac{\\Lambda}{(\\tau - 0.5)}$$
        where `Lambda` is `TRT_magic_param`.

        Args:
            tau: The base symmetric relaxation time.

        Returns:
            The asymmetric relaxation time `tau_asymm`.
        """
        return 0.5 + self.TRT_magic_param/(tau-0.5)
