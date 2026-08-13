
from layout import TileTensor,LayoutTensor,coord
from layout.tile_tensor import stack_allocation
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from src.lbm import LBM_Grid,LBM_Config,Flags,LBM_method
from src.utils import Vector

from src.lbm.kernels.utils.load_and_store import load_f,store_f
from src.lbm.kernels.ops.load_and_store import esoteric_pull_load_single_f

def load_single_f[
    f_dtype:DType,
    f_layout:TensorLayout,
    //,
    grid:Some[GridLike],
    config:LBM_Config,
    *,
    is_even_time_step:Optional[Bool] = None,
    non_temporal:Bool = False,
    ]( 
    f:TileTensor[f_dtype,f_layout,_],
    index:InlineArray[Int,3],
    q:Int,
    ) -> Scalar[grid.float_dtype]:
    comptime load_f_val = load_f[grid.float_dtype,config.use_float16c,non_temporal]
    comptime assert f.rank == 4
    comptime if config.lbm_method == LBM_method.DOUBLE_BUFFER:
        return load_f_val(f,index,q)

    elif config.lbm_method == LBM_method.ESOTERIC_PULL:
        comptime assert is_even_time_step is not None, 'is_even_time_step cannot be none for esoteric pull config'
        comptime lattice = grid.lattice
        return esoteric_pull_load_single_f[is_even_time_step.value(),grid.float_dtype,lattice.directions,config.use_float16c](f,index,q,grid.shape)
    else:
        comptime assert False, 'Invalid lbm method used' 