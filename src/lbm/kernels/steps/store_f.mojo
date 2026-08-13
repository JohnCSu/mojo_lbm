from layout import TileTensor,LayoutTensor,coord
from layout.tile_tensor import stack_allocation
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from src.lbm import LBM_Grid,LBM_Config,Flags,GridLike,LBM_method
from src.utils import Vector
from src.lbm.kernels.utils.index import get_adjacent_idx
from src.lbm.kernels.ops.load_and_store import esoteric_pull_store_f_vec
from src.lbm.kernels.utils.load_and_store import store_f

from src.utils.custom_fp import Float16C

@always_inline
def store_f_vec_to_global[
    f_dtype:DType,
    f_layout:TensorLayout,
    f_origin:Origin[mut=True],
    //,
    grid:Some[GridLike],
    config:LBM_Config,
    *,
    is_even_time_step:Optional[Bool] = None,
    non_temporal:Bool = False,
    ]
    (
    f:TileTensor[f_dtype,f_layout,f_origin],
    f_vec:Vector[grid.float_dtype,grid.Q],
    index:InlineArray[Int,3],
    ):
    comptime lattice = grid.lattice
    comptime grid_shape = grid.shape
    comptime Q = grid.Q
    
    comptime if config.lbm_method == LBM_method.DOUBLE_BUFFER:
        comptime for q in range(Q):
            store_f[config.use_float16c,non_temporal](f,f_vec[q],index,q)
    elif config.lbm_method == LBM_method.ESOTERIC_PULL:
        comptime assert is_even_time_step is not None
        esoteric_pull_store_f_vec[lattice.directions,is_even_time_step.value(),config.use_float16c,non_temporal = non_temporal](f,f_vec,index,grid_shape)
    else:
        comptime assert False, 'Invalid lbm method used'


