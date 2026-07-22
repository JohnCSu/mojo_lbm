"""Defines a stack-allocated, value-semantic vector for small fixed-size tuples.

The `Vector` struct stores its elements in an `InlineArray` and unrolls
element-wise operations at compile time, so it is intended for short vectors
(eight elements or fewer) where SIMD optimization is not worth the overhead.
"""
from std.memory import UnsafePointer
from std.math import sqrt

struct Vector[dtype:DType, size: Int](ImplicitlyCopyable & Sized & Writable):
    """Models a stack-allocated vector of fixed compile-time length.

    Behaves like a numeric value: it is implicitly copyable and supports the
    standard arithmetic, comparison, and in-place operators against another
    `Vector` of the same type or against a scalar of the same `DType`.
    Element-wise operations are unrolled at compile time, so keep `size`
    small (eight or fewer) for the best codegen.

    Parameters:
        dtype: The element type of the vector.
        size: The number of elements stored in the vector.

    Examples:

    ```mojo
    var v = Vector[DType.float32, 3](1.0, 2.0, 3.0)
    var w = Vector[DType.float32, 3](fill=0.0)
    w += v
    print(w.sum())  # 6.0
    ```
    """
    # comptime dataType = InlineArray[Self.dtype,Self.size]
    comptime dataType =InlineArray[Scalar[Self.dtype],Self.size]
    var data:InlineArray[Scalar[Self.dtype],Self.size]
    """The underlying element storage as an inline array."""


    @always_inline
    def __init__(out self):
        """Constructs an empty vector.

        This overload always aborts. Use the `uninitialized`, variadic, fill,
        or `List` overload instead.
        """
        comptime assert False, 'Do not use empty constructor'

    @always_inline
    def __init__(out self,*,uninitialized:Bool):
        """Constructs a vector with uninitialized storage.

        Args:
            uninitialized: Pass `True` to skip zero-filling the elements.
        """
        self.data = InlineArray[Scalar[Self.dtype],Self.size](uninitialized = uninitialized)

    @always_inline
    def __init__(out self,*numbers:Scalar[Self.dtype]):
        """Constructs a vector from a variadic list of scalars.

        Args:
            numbers: A variadic tuple of scalars to assign element by element.
                The number of values must match `size`.
        """
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
        """Constructs a vector filled with a single scalar value.

        Args:
            fill: The value to broadcast to every element (defaults to 0).
        """
        self.data = InlineArray[Scalar[Self.dtype],Self.size](fill=fill)

    @always_inline
    def __init__(out self,numbers:List[Scalar[Self.dtype]]):
        """Constructs a vector from a `List` of scalars.

        The list length must match the vector size.

        Args:
            numbers: The list of scalars to copy element by element.
        """
        assert len(numbers) == Self.size
        self.data = InlineArray[Scalar[Self.dtype],Self.size](uninitialized = True)
        for i in range(Self.size):
            self.data[i] = numbers[i]

    @always_inline
    def __len__(self) -> Int:
        """Returns the number of elements in the vector."""
        return Self.size

    @always_inline
    def __getitem__(self,idx:Int) -> Scalar[Self.dtype]:
        """Returns the element at the given index.

        Args:
            idx: The element index.

        Returns:
            The scalar stored at position `idx`.
        """
        return self.data[idx]
    @always_inline
    def __setitem__(mut self,idx:Int,value:Scalar[Self.dtype]):
        """Assigns a scalar to the element at the given index.

        Args:
            idx: The element index.
            value: The scalar to store.
        """
        # self.data[idx] =
        self.data[idx] = value

    @always_inline
    def fill(mut self,list:List[Scalar[Self.dtype]]):
        """Copies elements from a list into the vector.

        Args:
            list: The list of scalars to copy. Its length must match `size`.
        """
        assert len(list) == Self.size
        comptime for i in range(Self.size):
            self.data[i] = list[i]

    @always_inline
    def fill_and_cast_from_list[different_dtype:DType](mut self,list:List[Scalar[different_dtype]]):
        """Casts and copies elements from a list of a different `DType`.

        Useful for converting a list of integers into a vector of floats, for
        example.

        Args:
            list: The list of scalars to cast and copy. Its length must match
                `size`.
        """
        assert len(list) == Self.size
        comptime for i in range(Self.size):
            self.data[i] = Scalar[Self.dtype](list[i])

    def cast_to[target_dtype:DType](self) -> Vector[target_dtype,Self.size]:
        var out = Vector[target_dtype,Self.size](uninitialized = True)
        comptime for i in range(Self.size):
            out[i] = Scalar[target_dtype](self[i])
        return out 
        


    @always_inline
    def fill(mut self,value:Scalar[Self.dtype]):
        """Broadcasts a scalar to every element of the vector.

        Args:
            value: The scalar to assign to each element.
        """
        comptime for i in range(Self.size):
            self.data[i] = value

    @always_inline
    def dot(self,other:Self) -> Scalar[Self.dtype]:
        """Returns the dot product of two vectors.

        Args:
            other: The vector to dot with `self`.

        Returns:
            The scalar dot product.
        """
        out = Scalar[Self.dtype](0)
        comptime for i in range(Self.size):
            out += self[i]*other[i]
        return out
    @always_inline
    def sum(self) -> Scalar[Self.dtype]:
        """Returns the sum of all elements."""
        out = Scalar[Self.dtype](0)
        comptime for i in range(Self.size):
            out += self[i]
        return out
    @always_inline
    def prod(self) -> Scalar[Self.dtype]:
        """Returns the product of all elements.

        Note that the accumulator starts at 0, so an empty product still
        returns 0 rather than 1.
        """
        out = Scalar[Self.dtype](0)
        comptime for i in range(Self.size):
            out *= self[i]
        return out

    @always_inline
    def norm_squared(self) -> Scalar[Self.dtype]:
        """Returns the squared L2 norm of the vector."""
        return (self*self).sum()

    @always_inline
    def norm(self) -> Scalar[Self.dtype]:
        """Returns the L2 norm of the vector."""
        return sqrt(self.norm_squared())


    @always_inline
    def unsafe_ptr(self) -> UnsafePointer[Self.dataType.ElementType,origin_of(self.data)]:
        """Returns an unsafe pointer to the underlying element storage.

        Returns:
            A pointer to the first element of the inline array.
        """
        x = self.data.unsafe_ptr()
        return x

    @always_inline
    def all_true(self) -> Bool:
        """Returns `True` if every element is truthy, otherwise `False`.

        For non-boolean vectors, any non-zero value is treated as `True`.

        Returns:
            `True` when every element is truthy, `False` otherwise.
        """
        comptime for i in range(Self.size):
            if not Bool(self.data[i]):
                    return False
        return True


    def __neg__(self) -> Self:
        """Returns the element-wise negation of the vector."""
        return Self._scalarOp[Self._mul](self,-1)

    def __add__(self,other:Self) -> Self:
        """Returns the element-wise sum of two vectors."""
        return Self._elementWise[Self._add](self,other)

    def __iadd__(mut self,other:Self):
        """Adds another vector element-wise into this vector in place."""
        comptime for i in range(Self.size):
            self[i] += other[i]

    def __sub__(self,other:Self) -> Self:
        """Returns the element-wise difference of two vectors."""
        return self.__add__(-other)

    def __isub__(mut self,other:Self):
        """Subtracts another vector element-wise from this vector in place."""
        comptime for i in range(Self.size):
            self[i] -= other[i]

    def __mul__(self,other:Self) -> Self:
            """Returns the element-wise product of two vectors."""
            return Self._elementWise[Self._mul](self,other)

    def __imul__(mut self,other:Self):
        """Multiplies another vector element-wise into this vector in place."""
        comptime for i in range(Self.size):
            self[i] *= other[i]

    def __imul__(mut self,other:Scalar[Self.dtype]):
        """Scales this vector by a scalar in place."""
        comptime for i in range(Self.size):
            self[i] *= other

    def __mul__(self,other:Scalar[Self.dtype]) -> Self:
        """Returns the scalar-vector product."""
        return Self._scalarOp[Self._mul](self,other)

    def __rmul__(self,other:Scalar[Self.dtype]) -> Self:
        """Returns the scalar-vector product for left-hand scalar operands."""
        return Self._scalarOp[Self._mul](self,other)

    def __pow__(self,other:Scalar[Self.dtype]) -> Self:
        """Returns the element-wise power of the vector by a scalar."""
        return Self._scalarOp[Self._pow](self,other)

    def __truediv__(self,other:Self) -> Self:
        """Returns the element-wise quotient of two vectors."""
        return Self._elementWise[Self._div](self,other)

    def __truediv__(self,other:Scalar[Self.dtype]) -> Self:
        """Returns the vector scaled by the reciprocal of a scalar."""
        return Self._scalarOp[Self._div](self,other)

    def __rtruediv__(self,other:Scalar[Self.dtype]) -> Self:
        """Returns the reciprocal scalar broadcast divided by the vector."""
        return Self._scalarOp[Self._div,reverse = True](self,other)

    def __itruediv__(mut self,other:Self):
        """Divides this vector by another element-wise in place."""
        comptime for i in range(Self.size):
            self[i] /= other[i]

    def __itruediv__(mut self,other:Scalar[Self.dtype]):
        """Divides this vector by a scalar in place."""
        comptime for i in range(Self.size):
            self[i] /= other

    def __eq__(self,other:Self) -> Vector[DType.bool,Self.size]:
        """Returns the element-wise equality comparison as a boolean vector."""
        return Self._elementWise[DType.bool,Self._eq](self,other)

    def __ne__(self,other:Self) -> Vector[DType.bool,Self.size]:
        """Returns the element-wise inequality comparison as a boolean vector."""
        return Self._elementWise[DType.bool,Self._ne](self,other)

    def __le__(self,other:Self) -> Vector[DType.bool,Self.size]:
        """Returns the element-wise less-or-equal comparison as a boolean vector."""
        return Self._elementWise[DType.bool,Self._leq](self,other)

    def __ge__(self,other:Self) -> Vector[DType.bool,Self.size]:
        """Returns the element-wise greater-or-equal comparison as a boolean vector."""
        return Self._elementWise[DType.bool,Self._ge](self,other)

    def __gt__(self,other:Self) -> Vector[DType.bool,Self.size]:
        """Returns the element-wise greater-than comparison as a boolean vector."""
        return Self._elementWise[DType.bool,Self._gt](self,other)

    def __lt__(self,other:Self) -> Vector[DType.bool,Self.size]:
        """Returns the element-wise less-than comparison as a boolean vector."""
        return Self._elementWise[DType.bool,Self._lt](self,other)



    @always_inline
    @staticmethod
    def _elementWise[func: def(Scalar[Self.dtype],Scalar[Self.dtype]) thin -> Scalar[Self.dtype]](a:Self,b:Self) -> Self:
        """Applies a binary function element-wise to two vectors.

        Parameters:
            func: The binary function applied to each pair of elements.

        Args:
            a: The left-hand vector.
            b: The right-hand vector.

        Returns:
            A vector whose elements are `func(a[i], b[i])`.
        """
        out = Self(uninitialized = True)
        comptime for i in range(Self.size):
            out[i] = func(a[i],b[i])
        return out

    @always_inline
    @staticmethod
    def _elementWise[output_dtype:DType,func: def(Scalar[Self.dtype],Scalar[Self.dtype]) thin -> Scalar[output_dtype]](a:Self,b:Self) -> Vector[output_dtype,Self.size]:
        """Applies a binary function element-wise to two vectors with a custom output type.

        Parameters:
            output_dtype: The `DType` of the resulting vector.
            func: The binary function applied to each pair of elements.

        Args:
            a: The left-hand vector.
            b: The right-hand vector.

        Returns:
            A vector of `output_dtype` whose elements are `func(a[i], b[i])`.
        """
        out = Vector[output_dtype,Self.size](uninitialized = True)
        comptime for i in range(Self.size):
            out[i] = func(a[i],b[i])
        return out

    @always_inline
    @staticmethod
    def _scalarOp[func: def(Scalar[Self.dtype],Scalar[Self.dtype]) thin ->   Scalar[Self.dtype], *, reverse:Bool = False](a:Self,b:Scalar[Self.dtype]) -> Self:
        """Applies a binary function between a vector and a scalar broadcast.

        Parameters:
            func: The binary function applied to each element and the scalar.
            reverse: When `True`, calls `func(b, a[i])` instead of
                `func(a[i], b)` (defaults to `False`).

        Args:
            a: The vector.
            b: The scalar to broadcast against every element.

        Returns:
            A vector whose elements are `func(a[i], b)` (or `func(b, a[i])`
            when `reverse` is `True`).
        """
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
        """Returns the equality of two scalars as a boolean scalar."""
        return a == b

    @staticmethod
    @always_inline
    def _ge(a:Scalar[Self.dtype],b:Scalar[Self.dtype]) -> Scalar[DType.bool]:
        """Returns the less-or-equal comparison of two scalars as a boolean scalar.

        """
        return a >= b

    @staticmethod
    @always_inline
    def _gt(a:Scalar[Self.dtype],b:Scalar[Self.dtype]) -> Scalar[DType.bool]:
        """Returns the greater-than comparison of two scalars as a boolean scalar."""
        return a > b

    @staticmethod
    @always_inline
    def _lt(a:Scalar[Self.dtype],b:Scalar[Self.dtype]) -> Scalar[DType.bool]:
        """Returns the less-than comparison of two scalars as a boolean scalar."""
        return a < b
    @always_inline
    @staticmethod
    def _leq(a:Scalar[Self.dtype],b:Scalar[Self.dtype]) -> Scalar[DType.bool]:
        """Returns the less-or-equal comparison of two scalars as a boolean scalar."""
        return a <= b

    @always_inline
    @staticmethod
    def _ne(a:Scalar[Self.dtype],b:Scalar[Self.dtype]) -> Bool:
        """Returns the inequality of two scalars as a boolean."""
        return a != b


    # -------- Arithmetic Static Ops -----------
    @always_inline
    @staticmethod
    def _add(a:Scalar[Self.dtype],b:Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        """Returns the sum of two scalars."""
        return a+b

    @always_inline
    @staticmethod
    def _sub(a:Scalar[Self.dtype],b:Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        """Returns the difference of two scalars."""
        return a-b

    @always_inline
    @staticmethod
    def _mul(a:Scalar[Self.dtype],b:Scalar[Self.dtype]) -> Scalar[Self.dtype]:
            """Returns the product of two scalars."""
            return a*b

    @always_inline
    @staticmethod
    def _div(a:Scalar[Self.dtype],b:Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        """Returns the quotient of two scalars."""
        return a/b

    @always_inline
    @staticmethod
    def _pow(a:Scalar[Self.dtype],b:Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        """Returns `a` raised to the power `b`."""
        return a**b


