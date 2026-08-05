"""Defines comptime aliases for runtime row-major layout types.

Exposes ready-to-use `TensorLayout` types for 1D, 2D, and 3D row-major
layouts parameterized by `Int32` coordinates, so callers can avoid
re-spelling the `row_major(coord[...])` boilerplate.
"""
# RowMajor Runtime types

from layout import row_major,coord,col_major

def rowMajor1D[int_dtype:DType]() -> type_of( row_major(coord[int_dtype]((1,))) ):
    """Returns a 1D row-major `Int`-dtype layout instance.

    Parameters:
        int_dtype: The `DType` used for the layout coordinates.

    Returns:
        A 1D row-major layout instance.
    """
    return row_major(coord[int_dtype]((1,)))

def rowMajor2D[int_dtype:DType]() -> type_of(row_major(coord[int_dtype]((1,2))) ):
    """Returns a 2D row-major `Int`-dtype layout instance.

    Parameters:
        int_dtype: The `DType` used for the layout coordinates.

    Returns:
        A 2D row-major layout instance.
    """
    return row_major(coord[int_dtype]((1,2)))


def col_major1D[int_dtype:DType = DType.int32](n:Int) -> type_of( col_major(coord[int_dtype]((1,))) ):
    """Returns a 1D row-major `Int`-dtype layout instance.

    Parameters:
        int_dtype: The `DType` used for the layout coordinates.

    Returns:
        A 1D row-major layout instance.
    """
    return col_major(coord[int_dtype]((n,)))

def col_major2D[int_dtype:DType = DType.int32](rows:Int,cols:Int) -> type_of(col_major(coord[int_dtype]((1,2))) ):
    """Returns a 2D row-major `Int`-dtype layout instance.

    Parameters:
        int_dtype: The `DType` used for the layout coordinates.

    Returns:
        A 2D row-major layout instance.
    """
    return col_major(coord[int_dtype]((rows,cols)))



comptime RuntimeColMajor1DType = type_of(col_major1D(1))
comptime RuntimeColMajor2DType = type_of(col_major2D(1,2))




