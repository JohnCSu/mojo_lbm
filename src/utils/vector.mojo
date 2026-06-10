struct Vector[dtype:DType, size: Int](ImplicitlyCopyable & Sized & Equatable & Writable):
    '''
    Create a stack allocated vector of DType elements. Not optimised to use SIMD so you 
    should only use this for small vectors where SIMD is not worth it. Uses unrolling
    so should keep size small (<= 8 elements)

    Supports the following Ops: \\
        - add,sub,mul,div and in-place counter parts with vectors of same Type \\
        - mul and div with scalars of same DType
    '''
    # comptime dataType = InlineArray[Self.dtype,Self.size]
    var data:InlineArray[Scalar[Self.dtype],Self.size]

    @always_inline
    def __init__(out self,*numbers:Scalar[Self.dtype]):
        '''
        Create a stack allocated vector of DType elemets using Variadic Syntax:

        Args:
            numbers: Scalar[DType] a variadic list of Scalars to pass into the list
        '''
        assert len(numbers) == Self.size, 'Number of inputs must match'
        self.data = InlineArray[Scalar[Self.dtype],Self.size](uninitialized = True)
        
        for i in range(Self.size):
            self.data[i] = numbers[i]
        
    @always_inline
    def __init__(out self,*,fill:Scalar[Self.dtype] = 0):
        self.data = InlineArray[Scalar[Self.dtype],Self.size](fill=fill)

    @always_inline
    def __init__(out self,numbers:List[Scalar[Self.dtype]]):
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
    def fill_from_list(mut self,list:List[Scalar[Self.dtype]]):
        assert len(list) == Self.size
        comptime for i in range(Self.size):
            self.data[i] = list[i]
        
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
            out += Self._mul(self[i],other[i])
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
    
    # def write_to(self, mut writer: Some[Writer]):
    #     # s:String  = '['
    #     # for i in range(Self.size):
    #     #     s += ' {},'.format(self[i])
    #     # s+= ']'
    #     writer.write('[',self.data,']')

    @always_inline
    @staticmethod
    def _elementWise[func: def(Scalar[Self.dtype],Scalar[Self.dtype]) thin -> Scalar[Self.dtype]](a:Self,b:Self) -> Self:
        out = Self()
        comptime for i in range(Self.size):
            out[i] = func(a[i],b[i])
        return out
        
    @always_inline
    @staticmethod
    def _scalarOp[func: def(Scalar[Self.dtype],Scalar[Self.dtype]) thin ->   Scalar[Self.dtype], *, reverse:Bool = False](a:Self,b:Scalar[Self.dtype]) -> Self:
        out = Self()
        comptime for i in range(Self.size):
            comptime if reverse:
                out[i] = func(b,a[i])
            else:
                out[i] = func(a[i],b)
        return out

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
    def _eq(a:Scalar[Self.dtype],b:Scalar[Self.dtype]) -> Bool:
        return a == b
    
    @always_inline
    @staticmethod
    def _neq(a:Scalar[Self.dtype],b:Scalar[Self.dtype]) -> Bool:
        return a != b
    
    def __eq__(self,other:Self) -> Bool:
        comptime for i in range(Self.size):
            if self[i] != other[i]:
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

    def __mul__(self,other:Scalar[Self.dtype]) -> Self:
        return Self._scalarOp[Self._mul](self,other)

    def __rmul__(self,other:Scalar[Self.dtype]) -> Self:
        return Self._scalarOp[Self._mul](self,other)


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


