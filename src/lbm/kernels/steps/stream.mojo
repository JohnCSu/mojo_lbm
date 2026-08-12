from std.gpu import block_dim,block_idx,thread_idx,grid_dim,barrier
from layout import TileTensor,LayoutTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout,col_major
from layout.tile_tensor import stack_allocation

from src.lbm import LBM_Grid,LBM_Config,Lattice,GridLike,LBM_method
from src.utils import Vector,ContextTileTensor

from src.lbm.kernels.utils.index import get_adjacent_idx
from src.lbm.kernels.ops.load_and_store import esoteric_pull_load_f_vec,double_buffer_pull_load_f,set_adjacent_flags

@always_inline
def stream[
    float_dtype:DType,
    f_dtype:DType,
    FlayoutType:TensorLayout,
    FlaglayoutType:TensorLayout,
    //,
    grid:Some[GridLike],
    config:LBM_Config,
    *,
    is_even_time_step:Optional[Bool] = None,
    ]
    (
    mut f_vec:Vector[float_dtype,grid.Q],
    mut pull_flags:InlineArray[UInt8,grid.Q],
    f:TileTensor[f_dtype,FlayoutType,_],
    flags:TileTensor[DType.uint8,FlaglayoutType,_],
    current_flag:UInt8,
    index:InlineArray[Int,3],
    grid_shape:InlineArray[Int,3],
    ):
    set_f_vector_and_flags[
        config.lbm_method,
        grid.lattice.directions,
        grid.lattice.opposite_indices,
        config.use_float16c,
        config.implies_get_adjacent_flags(),
        is_even_time_step = is_even_time_step
        ](
        f_vec,
        pull_flags,
        f,
        flags,
        current_flag,
        index,
        grid_shape,
        )


@always_inline
def set_f_vector_and_flags[ 
    float_dtype:DType,
    f_dtype:DType,
    int_dtype:DType,
    Q:Int,
    D:Int,
    FlayoutType:TensorLayout,
    FlaglayoutType:TensorLayout,
    //,
    lbm_method:LBM_method,
    directions:InlineArray[Vector[int_dtype, D], Q],
    opposite_indices:InlineArray[Scalar[int_dtype], Q],
    use_float16c:Bool,
    implies_get_adjacent_flags:Bool,
    *,
    is_even_time_step:Optional[Bool] = None,
    ]
    (
    mut f_vec:Vector[float_dtype,Q],
    mut pull_flags:InlineArray[UInt8,Q],
    f:TileTensor[f_dtype,FlayoutType,_],
    flags:TileTensor[DType.uint8,FlaglayoutType,_],
    current_flag:UInt8,
    index:InlineArray[Int,3],
    grid_shape:InlineArray[Int,3],
    ):
    pull_flags[0] = current_flag
    comptime if lbm_method == lbm_method.ESOTERIC_PULL:
        comptime assert is_even_time_step is not None, 'If lbm_method is set to esoteric_pull, is_even_time_step must be defined'
        f_vec = esoteric_pull_load_f_vec[float_dtype,directions,is_even_time_step.value(),use_float16c](f,index,grid_shape)
        comptime if implies_get_adjacent_flags: # If we have set include moving boundary or double buffer
            set_adjacent_flags[directions](pull_flags,flags,index,grid_shape)
        
    elif lbm_method == lbm_method.DOUBLE_BUFFER:
        set_adjacent_flags[directions](pull_flags,flags,index,grid_shape)
        f_vec = double_buffer_pull_load_f[float_dtype,directions,opposite_indices,use_float16c](f,pull_flags,index,grid_shape)
    else:
        comptime assert False, 'lbm_method not valid'