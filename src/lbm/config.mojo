
from .constants import Flags,_FlagSet
from std.collections import Set


comptime fp32 = DType.float32
comptime uint16 = DType.uint16

trait ConfigLike: 
    '''
    Not a binding trait until option for implicit parameterization or some way of
    automatically treating parameters as trait comptime members
    '''
    # comptime DDF_shift:Bool
    # comptime LES:Bool
    # comptime KBC:Bool
    # comptime use_float16c:Bool
    # comptime f_dtype: Optional[DType]
    # comptime INCLUDED_BCs: Set[UInt8]
    # comptime second_moment:Bool
    @staticmethod
    @always_inline
    def fp32_to_fp16c(val:Scalar[fp32]) -> Scalar[uint16]:
        return Float16C.to_fp16c(val)

    @staticmethod
    @always_inline
    def fp16c_to_fp32(val:UInt16) -> Scalar[fp32]: 
        return Float16C.to_fp32(val)
        
    def set_f_dtype(self,float_dtype_for_math_ops:DType) -> DType:
        ...
        # return Self.f_dtype.value() if Self.f_dtype is not None else float_dtype
    

struct LBM_Config(ConfigLike):    
    var DDF_shift:Bool
    var LES:Bool
    var KBC:Bool
    var use_float16c:Bool
    var f_dtype:Optional[DType]
    var INCLUDED_BCs: Set[UInt8]
    var second_moment:Bool

    def __init__(
        out self,
        *,
        LES:Bool = False,
        BCs:Set[UInt8] = {},
        DDF_shift:Bool = False,
        use_float16c:Bool = False,
        f_dtype:Optional[DType] = None,
        ):
        self.DDF_shift = DDF_shift        
        self.LES = LES
        self.KBC = False
        self.second_moment = True if (LES) else False

        self.use_float16c = use_float16c
        self.f_dtype = DType.uint16 if use_float16c else f_dtype
        if self.use_float16c and self.f_dtype is not None:
            print('WARNING: Float16c set to True and f_dtype was also specified. Float16c overides the given dtype')

        # Boundary Condition Check
        if len(BCs) == 0:
            self.INCLUDED_BCs = {Flags.FLUID,Flags.SOLID}
        else:
            __valid_bcs = materialize[_FlagSet]()
            if not BCs.issubset(__valid_bcs):
                print('Warning: Some Specified BC types are not standard: {}'.format(BCs.difference(__valid_bcs)))
            # Ensure that fluid and solid nodes are always in the valid BC
            self.INCLUDED_BCs = {Flags.FLUID,Flags.SOLID}.union(BCs) 
    def implies_f_noneq(self) -> Bool:
        return self.LES
    
    def set_f_dtype(self,float_dtype_for_math_ops:DType) -> DType:
        return self.f_dtype.value() if self.f_dtype is not None else float_dtype_for_math_ops



    @always_inline
    def enable_float16c(mut self):
        self.use_float16c = True
        self.f_dtype = DType.uint16

    def enable_DDF_shift(mut self):
        self.DDF_shift = True


    @staticmethod
    @always_inline
    def fp32_to_fp16c(val:Scalar[fp32]) -> Scalar[uint16]:
        return Float16C.to_fp16c(val)

    @staticmethod
    @always_inline
    def fp16c_to_fp32[dtype:DType](val:Scalar[dtype]) -> Scalar[fp32] where dtype == uint16: 
        return Float16C.to_fp32(val)
        



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
    def to_fp32[dtype:DType](val:Scalar[dtype]) -> Scalar[fp32]:
        comptime assert dtype == uint16
        # Need to upscale first before doing the bitshifts (this is not clear in paper as upcasting is implicit and performed BEFORE the bitshift)
        e = UInt32((val & 0x7800)) >> 11 
        m = UInt32((val & 0x07FF)) << 12
        v = ((Float32(m)).to_bits[DType.uint32]())  >> 23  # Wtf? 

        # sign = UInt32((val&0x8000)) << 16 
        # normalized = UInt32(e != 0)*(( e+112 ) << 23 | m)
        # denormalised = UInt32((e==0)&(m!=0)) * ((v-37) << 23 | ((m << ( 150-v)) & 0x007FF000))
        # out_bits = sign | normalized | denormalised

        # return bitcast[fp32](out_bits)

        return bitcast[fp32](
            UInt32((val&0x8000)) << 16 |
            UInt32(e != 0)*(( e+112 ) << 23 | m) |
            UInt32((e==0)&(m!=0)) * ((v-37) << 23 | ((m << ( 150-v)) & 0x007FF000))
        )
    @staticmethod
    @always_inline
    def to_fp16c(val:Scalar[fp32]) -> Scalar[uint16]:
        '''
        Dont ask me how each op works see paper for reference
        '''
        b = val.to_bits[DType.uint32]()+ 0x0000_0800
        # b = b  # Add 1 to 12th bit from left
        e = (b & 0x7F80_0000) >> 23 # Exponent Bias 127
        m = b & 0x007F_FFFF # Get Mantissa
        
        # sign = (b & 0x80000000 ) >> 16
        # norm =  UInt32(e > 112)* ((((e-112) << 11 ) & 0x7800) | m >> 12)
        # denorm = UInt32((e < 113)&(e > 100)) * (((( 0x007FF800 +m ) >> (124 - e) ) +1) >>1) 
        # saturate = UInt32(e>127) * 0x7FFF
        # fp16c_uncompressed:UInt32 = sign | norm | denorm | saturate  # Sign Bit
        # out = bitcast[uint16,2](fp16c_uncompressed) # Mojo Syntax need to break the 32 bit number into 2 16 bit numbers and then take 0th simd element
        # return out[0]  # We keep the first 16 bits the rest are not relevant

        return bitcast[uint16,2](
            (b & 0x80000000 ) >> 16 |
            UInt32(e > 112)* ((((e-112) << 11 ) & 0x7800) | m >> 12) |
            UInt32((e < 113)&(e > 100)) * (((( 0x007FF800 +m ) >> (124 - e) ) +1) >>1) |
            UInt32(e>127) * 0x7FFF
        )[0]
