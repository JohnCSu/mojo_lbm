from src.utils import Vector

@always_inline
def get_adjacent_idx[int_dtype:DType,//,D:Int,shift:Int = 1](index:InlineArray[Int,3],grid_shape:InlineArray[Int,3],direction:Vector[int_dtype,D],) -> InlineArray[Int,3]:
    comptime assert D <= 3 
    adj_index = InlineArray[Int,3](fill = 0 )
    comptime for d in range(D):
        adj_index[d] = (index[d] + shift*Int(direction[d])) % grid_shape[d]
    return adj_index

@always_inline
def get_adjacent_idx[int_dtype:DType,//,D:Int,shift:Scalar[int_dtype] = 1](index:InlineArray[Scalar[int_dtype],3],grid_shape:InlineArray[Int,3],direction:Vector[int_dtype,D],) -> InlineArray[Int,3]:
    comptime assert not int_dtype.is_floating_point()
    comptime assert D <= 3 
    adj_index = InlineArray[Int,3](fill = 0 )
    comptime for d in range(D):
        adj_index[d] = Int(index[d] + shift*direction[d]) % grid_shape[d]
    return adj_index