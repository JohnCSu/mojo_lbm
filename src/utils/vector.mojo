from std.memory import UnsafePointer
from std.math import sqrt

struct Vector[dtype:DType, size: Int](ImplicitlyCopyable & Sized & Writable):
    '''
    Create a stack allocated vector of DType elements. Not optimised to use SIMD so you 
    should only use this for small vectors where SIMD is not worth it. Uses unrolling
    so should keep size small (<= 8 elements)

    Supports the following Ops: \\
        - add,sub,mul,div and in-place counter parts with vectors of same Type \\
        - mul and div with scalars of same DType
    '''
    # comptime dataType = InlineArray[Self.dtype,Self.size]
    comptime dataType =InlineArray[Scalar[Self.dtype],Self.size] 
    var data:InlineArray[Scalar[Self.dtype],Self.size]
    

    @always_inline
    def __init__(out self):
        '''
        Create a vector that is zero at all elements. Use keyword `uninitialized=True` for no fill
        '''
        comptime assert False, 'Do not use empty constructor'

    @always_inline
    def __init__(out self,*,uninitialized:Bool):
        self.data = InlineArray[Scalar[Self.dtype],Self.size](uninitialized = uninitialized)

    @always_inline
    def __init__(out self,*numbers:Scalar[Self.dtype]):
        '''
        Create a stack allocated vector of DType elemets using Variadic Syntax:

        Args:
            numbers: Scalar[DType] a variadic tuple of Scalars to pass into the vector.
        '''
        # assert len(numbers) == Self.size, 'Number of inputs must match'

        debug_assert[assert_mode="safe"](
            len(numbers) == Self.size,
            "InlineArray: expected ",
            Self.size,
            " numbers, received ",
            len(numbers),
        )
        self.data = InlineArray[Scalar[Self.dtype],Self.size](uninitialized = True)
        
        comptime for i in range(Self.size):
            self.data[i] = numbers[i]
    
    @always_inline
    def __init__(out self,*,fill:Scalar[Self.dtype] = 0):
        self.data = InlineArray[Scalar[Self.dtype],Self.size](fill=fill)

    @always_inline
    def __init__(out self,numbers:List[Scalar[Self.dtype]]):
        '''
        Fill Vector from List. List should match ElementType and Size of vector
        '''
        assert len(numbers) == Self.size
        self.data = InlineArray[Scalar[Self.dtype],Self.size](uninitialized = True)
        for i in range(Self.size):
            self.data[i] = numbers[i]
            
    @always_inline
    def __len__(self) -> Int:
        return Self.size

    @always_inline
    def __getitem__(self,idx:Int) -> Scalar[Self.dtype]:
        return self.data[idx]
    @always_inline
    def __setitem__(mut self,idx:Int,value:Scalar[Self.dtype]):
        # self.data[idx] = 
        self.data[idx] = value

    @always_inline
    def fill(mut self,list:List[Scalar[Self.dtype]]):
        assert len(list) == Self.size
        comptime for i in range(Self.size):
            self.data[i] = list[i]
            
    @always_inline
    def fill_and_cast_from_list[different_dtype:DType](mut self,list:List[Scalar[different_dtype]]):
        '''
        Convert a list of number not neccesarily the same dtype and cast to the vector dtype
        i.e convert a list of Int to a Vector of Floats
        '''
        assert len(list) == Self.size
        comptime for i in range(Self.size):
            self.data[i] = Scalar[Self.dtype](list[i])


    @always_inline
    def fill(mut self,value:Scalar[Self.dtype]):
        comptime for i in range(Self.size):
            self.data[i] = value

    @always_inline
    def dot(self,other:Self) -> Scalar[Self.dtype]:
        '''
        Dot Product of 2 Vectors.
        '''
        out = Scalar[Self.dtype](0)
        comptime for i in range(Self.size):
            out += self[i]*other[i]
        return out
    @always_inline
    def sum(self) -> Scalar[Self.dtype]:
        out = Scalar[Self.dtype](0)
        comptime for i in range(Self.size):
            out += self[i]
        return out
    @always_inline
    def prod(self) -> Scalar[Self.dtype]:
        out = Scalar[Self.dtype](0)
        comptime for i in range(Self.size):
            out *= self[i]
        return out
    
    @always_inline
    def norm_squared(self) -> Scalar[Self.dtype]:
        return (self*self).sum()

    @always_inline
    def norm(self) -> Scalar[Self.dtype]:
        return sqrt(self.norm_squared())


    @always_inline
    def unsafe_ptr(self) -> UnsafePointer[Self.dataType.ElementType,origin_of(self.data)]: 
        x = self.data.unsafe_ptr()
        return x

    @always_inline
    def all_true(self) -> Bool:
        '''
        Return True if all element in vector are True else False
        For Non bool vectors, anything non zero evaluates to True
        '''
        comptime for i in range(Self.size):
            if not Bool(self.data[i]):
                    return False
        return True


    def __neg__(self) -> Self:
        return Self._scalarOp[Self._mul](self,-1)

    def __add__(self,other:Self) -> Self:
        return Self._elementWise[Self._add](self,other)

    def __iadd__(mut self,other:Self):
        comptime for i in range(Self.size):
            self[i] += other[i]

    def __sub__(self,other:Self) -> Self:
        return self.__add__(-other)

    def __isub__(mut self,other:Self):
        comptime for i in range(Self.size):
            self[i] -= other[i]

    def __mul__(self,other:Self) -> Self:
            return Self._elementWise[Self._mul](self,other)

    def __imul__(mut self,other:Self):
        comptime for i in range(Self.size):
            self[i] *= other[i]

    def __imul__(mut self,other:Scalar[Self.dtype]):
        comptime for i in range(Self.size):
            self[i] *= other

    def __mul__(self,other:Scalar[Self.dtype]) -> Self:
        return Self._scalarOp[Self._mul](self,other)

    def __rmul__(self,other:Scalar[Self.dtype]) -> Self:
        return Self._scalarOp[Self._mul](self,other)

    def __pow__(self,other:Scalar[Self.dtype]) -> Self:
        return Self._scalarOp[Self._pow](self,other)
    
    def __truediv__(self,other:Self) -> Self:
        return Self._elementWise[Self._div](self,other)
        
    def __truediv__(self,other:Scalar[Self.dtype]) -> Self:
        return Self._scalarOp[Self._div](self,other)
        
    def __rtruediv__(self,other:Scalar[Self.dtype]) -> Self:
        return Self._scalarOp[Self._div,reverse = True](self,other)

    def __itruediv__(mut self,other:Self):
        comptime for i in range(Self.size):
            self[i] /= other[i]

    def __itruediv__(mut self,other:Scalar[Self.dtype]):
        comptime for i in range(Self.size):
            self[i] /= other

    def __eq__(self,other:Self) -> Vector[DType.bool,Self.size]:
        return Self._elementWise[DType.bool,Self._eq](self,other)
    
    def __ne__(self,other:Self) -> Vector[DType.bool,Self.size]:
        return Self._elementWise[DType.bool,Self._ne](self,other)
    
    def __le__(self,other:Self) -> Vector[DType.bool,Self.size]:
        return Self._elementWise[DType.bool,Self._leq](self,other)

    def __ge__(self,other:Self) -> Vector[DType.bool,Self.size]:
        return Self._elementWise[DType.bool,Self._ge](self,other)

    def __gt__(self,other:Self) -> Vector[DType.bool,Self.size]:
        return Self._elementWise[DType.bool,Self._gt](self,other)

    def __lt__(self,other:Self) -> Vector[DType.bool,Self.size]:
        return Self._elementWise[DType.bool,Self._lt](self,other)



    @always_inline
    @staticmethod
    def _elementWise[func: def(Scalar[Self.dtype],Scalar[Self.dtype]) thin -> Scalar[Self.dtype]](a:Self,b:Self) -> Self:
        out = Self(uninitialized = True)
        comptime for i in range(Self.size):
            out[i] = func(a[i],b[i])
        return out
    
    @always_inline
    @staticmethod
    def _elementWise[output_dtype:DType,func: def(Scalar[Self.dtype],Scalar[Self.dtype]) thin -> Scalar[output_dtype]](a:Self,b:Self) -> Vector[output_dtype,Self.size]:
        out = Vector[output_dtype,Self.size](uninitialized = True)
        comptime for i in range(Self.size):
            out[i] = func(a[i],b[i])
        return out

    @always_inline
    @staticmethod
    def _scalarOp[func: def(Scalar[Self.dtype],Scalar[Self.dtype]) thin ->   Scalar[Self.dtype], *, reverse:Bool = False](a:Self,b:Scalar[Self.dtype]) -> Self:
        out = Self(uninitialized = True)
        comptime for i in range(Self.size):
            comptime if reverse:
                out[i] = func(b,a[i])
            else:
                out[i] = func(a[i],b)
        return out
    
    # -------- Conditional Static Ops -----------
    @staticmethod
    @always_inline
    def _eq(a:Scalar[Self.dtype],b:Scalar[Self.dtype]) -> Scalar[DType.bool]:
        return a == b

    @staticmethod
    @always_inline
    def _ge(a:Scalar[Self.dtype],b:Scalar[Self.dtype]) -> Scalar[DType.bool]:
        return a <= b

    @staticmethod
    @always_inline
    def _gt(a:Scalar[Self.dtype],b:Scalar[Self.dtype]) -> Scalar[DType.bool]:
        return a > b

    @staticmethod
    @always_inline
    def _lt(a:Scalar[Self.dtype],b:Scalar[Self.dtype]) -> Scalar[DType.bool]:
        return a < b
    @always_inline
    @staticmethod
    def _leq(a:Scalar[Self.dtype],b:Scalar[Self.dtype]) -> Scalar[DType.bool]:
        return a <= b

    @always_inline
    @staticmethod
    def _ne(a:Scalar[Self.dtype],b:Scalar[Self.dtype]) -> Bool:
        return a != b
    

    # -------- Arithmetic Static Ops -----------
    @always_inline
    @staticmethod
    def _add(a:Scalar[Self.dtype],b:Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        return a+b

    @always_inline
    @staticmethod
    def _sub(a:Scalar[Self.dtype],b:Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        return a-b

    @always_inline
    @staticmethod
    def _mul(a:Scalar[Self.dtype],b:Scalar[Self.dtype]) -> Scalar[Self.dtype]:
            return a*b

    @always_inline
    @staticmethod
    def _div(a:Scalar[Self.dtype],b:Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        return a/b
 
    @always_inline
    @staticmethod
    def _pow(a:Scalar[Self.dtype],b:Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        return a**b
    
        


