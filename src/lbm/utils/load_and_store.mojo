from layout import TileTensor,LayoutTensor,coord
from layout.tile_tensor import stack_allocation
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from src.lbm import LBM_Grid,LBM_Config

@always_inline
def store_f[
        f_dtype:DType,
        FlayoutType:TensorLayout,
        float_dtype:DType,
        //,
        use_float16c:Bool = False,
        non_temporal:Bool = False
        ]
        (
        f:TileTensor[f_dtype,FlayoutType,MutAnyOrigin],
        val:Scalar[float_dtype],
        index:InlineArray[Int,3],
        q:Int
        ):
        comptime assert FlayoutType.rank == 4, 'For all LBM grids we use i,j,k,q indexing'
        comptime if use_float16c:
            comptime assert f_dtype == DType.uint16
            f_next = Float32(val)
            f_as_uint = Scalar[f_dtype](LBM_Config.fp32_to_fp16c(f_next))
            f.store[non_temporal = non_temporal](coord = coord[DType.uint32]((index[0],index[1],index[2],q)),value = f_as_uint)
            # You can add your own logic here by adding an elif statement to the comptime conditional
        else:
            comptime assert f_dtype == float_dtype
            f.store[non_temporal = non_temporal](coord = coord[DType.uint32]((index[0],index[1],index[2],q)),value = Scalar[f_dtype](val))


@always_inline
def load_f[
        f_dtype:DType,
        FlayoutType:TensorLayout,
        //,
        float_dtype:DType,
        use_float16c:Bool = False,
        non_temporal:Bool = False,
        ]
        (
        f:TileTensor[f_dtype,FlayoutType,ImmutAnyOrigin],
        index:InlineArray[Int,3],q:Int
        ) -> Scalar[float_dtype]:
        comptime to_compute_float = Scalar[float_dtype]
        comptime assert FlayoutType.rank == 4, 'For all LBM grids we use i,j,k,q indexing'
        comptime if use_float16c:
                comptime assert f_dtype == DType.uint16, 'Float16C requires the f tiletensors to be uint16 dtype'
                pulled_f = to_compute_float(LBM_Config.fp16c_to_fp32( f.load[non_temporal = non_temporal](coord[DType.uint32]((index[0],index[1],index[2],q)))[0] ))
            else:
                comptime assert f_dtype == float_dtype
                pulled_f = Scalar[float_dtype](f.load[non_temporal = non_temporal](coord[DType.uint32]((index[0],index[1],index[2],q)))[0])

        return pulled_f