
from std.memory import bitcast

comptime fp32 = DType.float32
"""Alias for `DType.float32` used throughout the Float16C conversions."""
comptime uint16 = DType.uint16
"""Alias for `DType.uint16` used throughout the Float16C conversions."""

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

    
    @always_inline
    @staticmethod
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

    
    @always_inline
    @staticmethod
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
