"""Defines `LBM_Config`, `ConfigLike`, and the `Float16C` conversion helpers.

`LBM_Config` carries the runtime toggles that parameterize an LBM run
(LES, DDF shifting, Float16C, boundary-condition set), and `Float16C`
provides the customized 16-bit floating-point format conversions described
in the FluidX3D reference.
"""
from .constants import Flags, _FlagSet
from std.collections import Set


comptime fp32 = DType.float32
"""Alias for `DType.float32` used throughout the Float16C conversions."""
comptime uint16 = DType.uint16
"""Alias for `DType.uint16` used throughout the Float16C conversions."""


trait ConfigLike:
    """Declares the float-conversion API shared by LBM configurations.

    Not a binding trait until Mojo supports treating parameters as trait
    comptime members; it currently only declares the Float16C conversion
    helpers that conforming configs must provide.
    """

    # comptime DDF_shift:Bool
    # comptime LES:Bool
    # comptime KBC:Bool
    # comptime use_float16c:Bool
    # comptime f_dtype: Optional[DType]
    # comptime INCLUDED_BCs: Set[UInt8]
    # comptime second_moment:Bool
    @staticmethod
    @always_inline
    def fp32_to_fp16c(val: Scalar[fp32]) -> Scalar[uint16]:
        """Converts an `fp32` scalar to a Float16C `uint16` scalar.

        Args:
            val: The `fp32` value to convert.

        Returns:
            The Float16C-encoded value as a `uint16`.
        """
        return Float16C.to_fp16c(val)

    @staticmethod
    @always_inline
    def fp16c_to_fp32(val: UInt16) -> Scalar[fp32]:
        """Converts a Float16C `uint16` scalar back to an `fp32` scalar.

        Args:
            val: The Float16C-encoded `uint16` value.

        Returns:
            The decoded `fp32` value.
        """
        return Float16C.to_fp32(val)

    def set_f_dtype(self, float_dtype_for_math_ops: DType) -> DType:
        """Returns the `DType` to use for distribution-function math.

        Args:
            float_dtype_for_math_ops: The fallback float `DType` used when
                the config does not override the storage type.

        Returns:
            The `DType` to use for `f` math operations.
        """
        ...
        # return Self.f_dtype.value() if Self.f_dtype is not None else float_dtype


struct LBM_Config(ConfigLike):
    """Holds the runtime toggles that parameterize an LBM run.

    Records whether DDF shifting, Smagorinsky LES, KBC, and Float16C are
    enabled, the optional storage `DType` for the distribution function, and
    the set of boundary-condition flags the run expects to encounter. Each
    member is documented alongside its declaration below.
    """

    var DDF_shift: Bool
    """Whether DDF shifting is enabled for `f`."""
    var LES: Bool
    """Whether the Smagorinsky LES model is enabled."""
    var KBC: Bool
    """Reserved for the KBC collision model; currently always `False`."""
    var use_float16c: Bool
    """Whether `f` is stored as Float16C `uint16` values."""
    var f_dtype: Optional[DType]
    """Optional override `DType` for the distribution function."""
    var INCLUDED_BCs: Set[UInt8]
    """The set of boundary-condition flags valid for this run."""
    var second_moment: Bool
    """Whether the non-equilibrium second moment is computed each step."""

    def __init__(
        out self,
        *,
        LES: Bool = False,
        BCs: Set[UInt8] = {},
        DDF_shift: Bool = False,
        use_float16c: Bool = False,
        f_dtype: Optional[DType] = None,
    ):
        """Constructs an `LBM_Config` from the supplied toggles.

        Enables the non-equilibrium second moment automatically when `LES`
        is `True`, forces `f_dtype` to `uint16` when `use_float16c` is
        `True`, and always includes `FLUID` and `SOLID` in the valid
        boundary-condition set.

        Args:
            LES: Whether to enable the Smagorinsky LES model (defaults to
                `False`).
            BCs: Additional boundary-condition flags beyond `FLUID` and
                `SOLID` (defaults to the empty set).
            DDF_shift: Whether to enable DDF shifting (defaults to `False`).
            use_float16c: Whether to store `f` as Float16C `uint16`
                (defaults to `False`).
            f_dtype: Optional override `DType` for `f` (defaults to `None`).
        """
        self.DDF_shift = DDF_shift
        self.LES = LES
        self.KBC = False
        self.second_moment = True if (LES) else False

        self.use_float16c = use_float16c
        self.f_dtype = DType.uint16 if use_float16c else f_dtype
        if self.use_float16c and self.f_dtype is not None:
            print(
                "WARNING: Float16c set to True and f_dtype was also specified."
                " Float16c overides the given dtype"
            )

        # Boundary Condition Check
        if len(BCs) == 0:
            self.INCLUDED_BCs = {Flags.FLUID, Flags.SOLID}
        else:
            __valid_bcs = materialize[_FlagSet]()
            if not BCs.issubset(__valid_bcs):
                print(
                    "Warning: Some Specified BC types are not standard: {}"
                    .format(BCs.difference(__valid_bcs))
                )
            # Ensure that fluid and solid nodes are always in the valid BC
            self.INCLUDED_BCs = {Flags.FLUID, Flags.SOLID}.union(BCs)

    def implies_f_noneq(self) -> Bool:
        """Returns `True` when the config requires the non-equilibrium part of `f`.

        Returns:
            `True` when LES is enabled, `False` otherwise.
        """
        return self.LES

    def set_f_dtype(self, float_dtype_for_math_ops: DType) -> DType:
        """Returns the `DType` to use for `f` math operations.

        Args:
            float_dtype_for_math_ops: The fallback float `DType` used when
                the config does not override the storage type.

        Returns:
            The configured `f` `DType` when set, otherwise the fallback.
        """
        return (
            self.f_dtype.value() if self.f_dtype
            is not None else float_dtype_for_math_ops
        )

    @always_inline
    def enable_float16c(mut self):
        """Enables Float16C storage and sets `f_dtype` to `uint16`."""
        self.use_float16c = True
        self.f_dtype = DType.uint16

    def enable_DDF_shift(mut self):
        """Enables DDF shifting for improved numerical stability."""
        self.DDF_shift = True

    @staticmethod
    @always_inline
    def fp32_to_fp16c(val: Scalar[fp32]) -> Scalar[uint16]:
        """Converts an `fp32` scalar to a Float16C `uint16` scalar.

        Args:
            val: The `fp32` value to convert.

        Returns:
            The Float16C-encoded value as a `uint16`.
        """
        return Float16C.to_fp16c(val)

    @staticmethod
    @always_inline
    def fp16c_to_fp32[
        dtype: DType
    ](val: Scalar[dtype]) -> Scalar[fp32] where dtype == uint16:
        """Converts a Float16C `uint16` scalar back to an `fp32` scalar.

        Parameters:
            dtype: The `DType` of the input scalar; constrained to `uint16`.

        Args:
            val: The Float16C-encoded `uint16` value.

        Returns:
            The decoded `fp32` value.
        """
        return Float16C.to_fp32(val)


from std.memory import bitcast


struct Float16C:
    """Provides conversions for the customized 16-bit float format Float16C.

    Float16C trades range for precision relative to IEEE `float16` to fit LBM
    distributions. The implementation mirrors the bit-manipulation used in
    FluidX3D.

    See:

    - [FluidX3D](https://github.com/ProjectPhysX/FluidX3D)
    - Lehmann, M., Krause, M., Amati, G., Sega, M., Harting, J. and Gekle, S.
      Accuracy and performance of the lattice Boltzmann method with 64-bit,
      32-bit, and customized 16-bit number formats. Phys. Rev. E 106, 015308
      (2022).
      [Paper](https://www.researchgate.net/publication/362275548_Accuracy_and_performance_of_the_lattice_Boltzmann_method_with_64-bit_32-bit_and_customized_16-bit_number_formats)
    """

    @staticmethod
    @always_inline
    def to_fp32[dtype: DType](val: Scalar[dtype]) -> Scalar[fp32]:
        """Decodes a Float16C `uint16` value into an `fp32` value.

        Parameters:
            dtype: The `DType` of the input scalar; constrained to `uint16`.

        Args:
            val: The Float16C-encoded `uint16` value.

        Returns:
            The decoded `fp32` value.
        """
        comptime assert dtype == uint16
        # Need to upscale first before doing the bitshifts (this is not clear in paper as upcasting is implicit and performed BEFORE the bitshift)
        e = UInt32((val & 0x7800)) >> 11
        m = UInt32((val & 0x07FF)) << 12
        v = ((Float32(m)).to_bits[DType.uint32]()) >> 23  # Wtf?

        # sign = UInt32((val&0x8000)) << 16
        # normalized = UInt32(e != 0)*(( e+112 ) << 23 | m)
        # denormalised = UInt32((e==0)&(m!=0)) * ((v-37) << 23 | ((m << ( 150-v)) & 0x007FF000))
        # out_bits = sign | normalized | denormalised

        # return bitcast[fp32](out_bits)

        return bitcast[fp32](
            UInt32((val & 0x8000)) << 16
            | UInt32(e != 0) * ((e + 112) << 23 | m)
            | UInt32((e == 0) & (m != 0))
            * ((v - 37) << 23 | ((m << (150 - v)) & 0x007FF000))
        )

    @staticmethod
    @always_inline
    def to_fp16c(val: Scalar[fp32]) -> Scalar[uint16]:
        """Encodes an `fp32` value into the Float16C `uint16` format.

        See the referenced paper for the derivation of the bit manipulation.

        Args:
            val: The `fp32` value to encode.

        Returns:
            The Float16C-encoded value as a `uint16`.
        """
        b = val.to_bits[DType.uint32]() + 0x0000_0800
        # b = b  # Add 1 to 12th bit from left
        e = (b & 0x7F80_0000) >> 23  # Exponent Bias 127
        m = b & 0x007F_FFFF  # Get Mantissa

        # sign = (b & 0x80000000 ) >> 16
        # norm =  UInt32(e > 112)* ((((e-112) << 11 ) & 0x7800) | m >> 12)
        # denorm = UInt32((e < 113)&(e > 100)) * (((( 0x007FF800 +m ) >> (124 - e) ) +1) >>1)
        # saturate = UInt32(e>127) * 0x7FFF
        # fp16c_uncompressed:UInt32 = sign | norm | denorm | saturate  # Sign Bit
        # out = bitcast[uint16,2](fp16c_uncompressed) # Mojo Syntax need to break the 32 bit number into 2 16 bit numbers and then take 0th simd element
        # return out[0]  # We keep the first 16 bits the rest are not relevant

        return bitcast[uint16, 2](
            (b & 0x80000000) >> 16
            | UInt32(e > 112) * ((((e - 112) << 11) & 0x7800) | m >> 12)
            | UInt32((e < 113) & (e > 100))
            * ((((0x007FF800 + m) >> (124 - e)) + 1) >> 1)
            | UInt32(e > 127) * 0x7FFF
        )[0]
