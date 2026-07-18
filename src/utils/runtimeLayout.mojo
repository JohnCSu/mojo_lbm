"""Defines comptime aliases for runtime row-major layout types.

Exposes ready-to-use `TensorLayout` types for 1D, 2D, and 3D row-major
layouts parameterized by `Int32` coordinates, so callers can avoid
re-spelling the `row_major(coord[...])` boilerplate.
"""
# RowMajor Runtime types
from layout import TileTensor, row_major, col_major, coord

comptime Runtime_rowMajor_1D_Type = type_of(row_major(coord[DType.int32]((1,))))
"""The runtime type of a 1D row-major `Int32` layout."""
comptime Runtime_rowMajor_2D_Type = type_of(
    row_major(coord[DType.int32]((1, 2)))
)
"""The runtime type of a 2D row-major `Int32` layout."""
comptime Runtime_rowMajor_3D_Type = type_of(
    row_major(coord[DType.int32]((1, 2, 3)))
)
"""The runtime type of a 3D row-major `Int32` layout."""
