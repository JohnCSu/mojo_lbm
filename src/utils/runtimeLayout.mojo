# RowMajor Runtime types
from layout import TileTensor,row_major,col_major,coord
comptime Runtime_rowMajor_1D_Type = type_of(row_major(coord[DType.int32]((1,))))
comptime Runtime_rowMajor_2D_Type = type_of(row_major(coord[DType.int32]((1,2))))
comptime Runtime_rowMajor_3D_Type = type_of(row_major(coord[DType.int32]((1,2,3))))