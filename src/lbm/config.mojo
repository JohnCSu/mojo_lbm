struct LBMConfig():
    var DDF_shift:Bool
    var LES:Bool
    var KBC:Bool
    var use_float16C:Bool
    def __init__(out self):
        '''
        Default Settings
        '''
        self.DDF_shift = False
        self.LES = False
        self.KBC = False
        self.use_float16C = False

    @staticmethod
    @always_inline
    def fp32_to_fp16c(val:Float32) -> UInt16:
        return Float16C.to_fp16c(val)

    @staticmethod
    @always_inline
    def fp16c_to_fp32(val:UInt16) -> Float32 :
        return Float16C.to_fp32(val)
        


comptime fp32 = DType.float32
comptime uint16 = DType.uint16
from std.memory import bitcast
struct Float16C():
    '''
    Conversion to Float16C as seen in [Fluidx3D](https://github.com/ProjectPhysX/FluidX3D)

    From:

    Lehmann, M., Krause, M., Amati, G., Sega, M., Harting, J. and Gekle, S.
    Accuracy and performance of the lattice Boltzmann method with 64-bit, 32-bit, and customized 16-bit number formats. Phys. Rev. E 106, 015308, (2022)
    [Paper](https://www.researchgate.net/publication/362275548_Accuracy_and_performance_of_the_lattice_Boltzmann_method_with_64-bit_32-bit_and_customized_16-bit_number_formats)
    
    '''
    @staticmethod
    @always_inline
    def to_fp32(val:Scalar[uint16]) -> Scalar[fp32]:

        # Need to upscale first before doing the bitshifts (this is not clear in paper as upcasting is implicit and performed BEFORE the bitshift)
        e = UInt32((val & 0x7800)) >> 11 
        m = UInt32((val & 0x07FF)) << 12
        v = ((Float32(m)).to_bits[DType.uint32]())  >> 23  # Wtf? 

        sign = UInt32((val&0x8000)) << 16 
        normalized = UInt32(e != 0)*(( e+112 ) << 23 | m)
        denormalised = UInt32((e==0)&(m!=0)) * ((v-37) << 23 | ((m << ( 150-v)) & 0x007FF000))
        out_bits = sign | normalized | denormalised

        return bitcast[fp32](out_bits)

    @staticmethod
    @always_inline
    def to_fp16c(val:Float32) -> Scalar[uint16]:
        '''
        Dont ask me how each op works see paper for reference
        '''
        b = val.to_bits[DType.uint32]()
        b = b + 0x0000_0800 # Add 1 to 12th bit from left
        e = (b & 0x7F80_0000) >> 23 # Exponent Bias 127
        m = b & 0x007F_FFFF # Get Mantissa
        
        sign = (b & 0x80000000 ) >> 16
        norm =  UInt32(e > 112)* ((((e-112) << 11 ) & 0x7800) | m >> 12)
        denorm = UInt32((e < 113)&(e > 100)) * (((( 0x007FF800 +m ) >> (124 - e) ) +1) >>1) 
        saturate = UInt32(e>127) * 0x7FFF
        fp16c_uncompressed:UInt32 = sign | norm | denorm | saturate  # Sign Bit
        out = bitcast[uint16,2](fp16c_uncompressed) # Mojo Syntax need to break the 32 bit number into 2 16 bit numbers and then take 0th simd element
        return out[0]  # We keep the first 16 bits the rest are not relevant

