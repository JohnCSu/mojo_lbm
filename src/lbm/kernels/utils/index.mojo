from src.utils import Vector
from layout import TileTensor,coord
from layout.tile_layout import TensorLayout

@always_inline
def get_indices_and_flags
    [
    int_dtype:DType,
    Q:Int,D:Int,
    flagLayoutType:TensorLayout,
    //,
    directions:InlineArray[Vector[int_dtype, D], Q],
    shift:Int,
    ]
    (
    flags:TileTensor[DType.uint8,flagLayoutType,ImmutAnyOrigin],
    index:InlineArray[Int,3],
    grid_shape:InlineArray[Int,3]
    ) 
    -> Tuple[InlineArray[InlineArray[Int,3],Q],InlineArray[UInt8,Q]]:
    comptime assert flags.flat_rank == 3
    var neighbor_flags = InlineArray[UInt8,Q](uninitialized = True)
    var neighbor_indices = InlineArray[InlineArray[Int,3],Q](uninitialized = True)
    comptime for q in range(Q):
        comptime direction = directions[q]
        neighbor_indices[q] = get_adjacent_idx[D,shift](index,grid_shape,direction) # Pulling Scheme
        neighbor_flags[q] = flags.load(coord[DType.int32]((neighbor_indices[q][0],neighbor_indices[q][1],neighbor_indices[q][2])))[0]
    
    return neighbor_indices^,neighbor_flags^

comptime get_pulled_indices_and_flags = get_indices_and_flags[_,-1] # Need directions 
'''Convience function for pull method of getting flags and indices. Pass lattice_model.directions to complete parameterization.'''

comptime get_pushed_indices_and_flags = get_indices_and_flags[_,1] # Need directions 
'''Convience function for push method of getting flags and indices. Pass lattice_model.directions to complete parameterization.'''



@always_inline
def is_index_out_of_bounds(index:InlineArray[Int,3],grid_shape:InlineArray[Int,3]) -> Bool:
    is_oob = False
    comptime for i in range(3):
        is_oob = True if (index[i] >= grid_shape[i] or index[i] < 0) else is_oob
    return is_oob

@always_inline
def is_index_valid(index:InlineArray[Int,3],grid_shape:InlineArray[Int,3]) -> Bool:
    return not is_index_out_of_bounds(index,grid_shape)



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