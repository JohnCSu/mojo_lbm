from std.gpu import block_dim,block_idx,thread_idx,grid_dim
from max.gpu.sync import barrier
from layout import TileTensor,LayoutTensor
from layout.tile_layout import Layout,row_major,Coord,TensorLayout,col_major
from layout.tile_tensor import stack_allocation

from src.lbm import LBM_Grid,LBM_Config,Lattice,GridLike,LBM_method
from src.utils import Vector,ContextTileTensor

from src.lbm.kernels.utils.index import get_adjacent_idx
from src.lbm.kernels.ops.load_and_store import esoteric_pull_load_f_vec,double_buffer_pull_load_f_vec,set_adjacent_flags

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
    ):

    var grid_shape = materialize[grid.shape]()
    var directions = materialize[grid.lattice.directions]()
    var opposite_indices = materialize[grid.lattice.opposite_indices]()

    pull_flags[0] = current_flag
    comptime if config.lbm_method == LBM_method.ESOTERIC_PULL:
        comptime assert is_even_time_step is not None, 'If lbm_method is set to esoteric_pull, is_even_time_step must be defined'
        f_vec = esoteric_pull_load_f_vec[float_dtype,is_even_time_step.value(),config.use_float16c](f,index,grid_shape,directions)
        comptime if config.implies_get_adjacent_flags(): # If we have set include moving boundary or double buffer
            set_adjacent_flags(pull_flags,flags,index,grid_shape,directions)
        
    elif config.lbm_method == LBM_method.DOUBLE_BUFFER:
        set_adjacent_flags(pull_flags,flags,index,grid_shape,directions)
        f_vec = double_buffer_pull_load_f_vec[float_dtype,config.use_float16c](f,pull_flags,index,grid_shape,directions,opposite_indices,)
    else:
        comptime assert False, 'lbm_method not valid'
