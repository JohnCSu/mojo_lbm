from std.sys import size_of

from std.utils.numerics import bit_width_of
from std.sys.info import size_of
from std.time.time import perf_counter
from std.random import rand,seed
from std.memory import alloc,Pointer

def KahanBabushkaNeumaierSum[float_dtype:DType](*floats:Scalar[float_dtype]) -> Scalar[float_dtype]:
    var sum = Scalar[float_dtype](0.0)
    var c = Scalar[float_dtype](0.0)                     
    for x in floats:
        t = sum + x
        c+= (sum-t) + x if abs(sum) > abs(x) else (x-t) + sum # Branchless version
        sum = t
    return sum + c


struct EmulatedFloat[float_dtype:DType where float_dtype.is_floating_point()](ImplicitlyCopyable):
    '''
    Emulate a higher precision float using 2 lower precision floats.
    Typical is using 2 fp32 to represent a higher precision float. Useful where GPU do have limited FP64 compute.
    '''
    comptime float_scalar = Scalar[Self.float_dtype]

    var high: Self.float_scalar
    var low: Self.float_scalar

    def __init__(out self,high:Self.float_scalar = 0,low:Self.float_scalar = 0):
        self.high = self.float_scalar(high)
        self.low = self.float_scalar(low)
    
    def accumulate_floats(mut self,*floats:Self.float_scalar):
        '''
        Accumulate floats into self.
        '''
        for x in floats:
            self+=x
        
    @staticmethod
    def sum_floats(*floats:Self.float_scalar) -> Self:
        '''
        Staticmethod that Sum up floats using EmulatedFloat Struct to keep track of the error.

        Args:
            floats: VariadicList of floats.

        Returns:
            EmulatedFloat: Sum of floats using EmulatedFloat Struct summation.
        '''
        out = Self()
        for x in floats:
            out += x
        return out
    
    def __neg__(self)-> Self:
        return Self(-self.high,-self.low)
    
    def __iadd__(mut self,other:Self) -> None:
        t = (self.__add__(other))
        self.high,self.low = t.high,t.low

    def __iadd__(mut self,other:Self.float_scalar) -> None:
        t = (self.__add__(other))
        self.high,self.low = t.high,t.low

    def __isub__(mut self,other:Self) -> None:
        self.__iadd__(-other)

    def __add__(self,other:Self) -> Self:
        S,E = self.knuth_twoSum(self.high,other.high)
        e = self.low + other.low + E
        S,E = self.knuth_twoSum(S,e)
        return Self(S,E)
    
    def __add__(self,other:Self.float_scalar) -> Self:
        return self.__add__(Self(other))
    
    def __sub__(self,other:Self) -> Self:
        return self + -other

    def __sub__(self,other:Self.float_scalar) -> Self:
        return self.__add__(-other)

    def __mul__(self,other:Self) raises -> Self :
        raise Error('Not implemented!')

    def to[output_dtype:DType](self) -> Scalar[output_dtype]:
        '''
        Return the sum of the high and low values in the precision above it. The summation is done in the spcified precision.
        '''
        comptime out_float = Scalar[output_dtype]
        return out_float(self.high) + out_float(self.low)
        

    @staticmethod
    def knuth_twoSum(a:Self.float_scalar,b:Self.float_scalar) -> Tuple[Self.float_scalar,Self.float_scalar]:        
        s = a + b
        a_p = s-b
        b_p = s-a_p
        da = a-a_p
        db = b-b_p
        e = da + db
        return s,e
        

def main() raises:

    comptime SingleSingle = EmulatedFloat[DType.float32]
    comptime float32 = DType.float32

    a,b = (1e7,0.1)

    sum_64 = a+b # Float64
    
    d,e = Float32(a),Float32(b)

    sum_32 = d+e

    x = SingleSingle(d)
    y = SingleSingle(e)

    sum_Emulated = x+y
    
    sum_Emulated_to_f64 = (x+y).to[DType.float64]()

    print(sum_64,sum_32,sum_Emulated_to_f64)
    print(sum_Emulated.high,sum_Emulated.low,Float64(sum_Emulated.low))
    
    seed()
    comptime SIZE = 200000
    var float_arr = InlineArray[Float32,SIZE](uninitialized=True)
    rand(float_arr) # Randomize values
    arr = [SingleSingle(i) for i in float_arr]

    start = perf_counter()
    sum = SingleSingle()
    for i in arr:
        sum += i
    end = perf_counter()
    print(sum.to[float32]())
    print('Time to run loop: {} ns'.format(end-start))


    