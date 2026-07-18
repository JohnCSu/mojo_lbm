"""Provides distribution-function load and store helpers for the LBM kernels.

The `load_f` and `store_f` functions wrap `TileTensor` access with optional
Float16C conversion and non-temporal hints, centralizing the `(x, y, z, q)`
indexing convention used throughout the solver.
"""
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
        """Stores a single distribution value at `(x, y, z, q)`.

        Converts `val` to Float16C when `use_float16c` is `True`, otherwise
        stores it as `f_dtype`. The store uses the `(x, y, z, q)` indexing
        convention required by all LBM layouts.

        Parameters:
            f_dtype: The storage `DType` of `f`.
            FlayoutType: The compile-time layout of `f`; must be rank 4.
            float_dtype: The `DType` of `val`.
            use_float16c: When `True`, encode `val` as Float16C before
                storing (defaults to `False`).
            non_temporal: When `True`, issue a non-temporal store (defaults
                to `False`).

        Args:
            f: The distribution function tile tensor.
            val: The scalar value to store.
            index: The `(x, y, z)` lattice index.
            q: The discrete velocity index.
        """
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
        """Loads a single distribution value from `(x, y, z, q)`.

        Decodes Float16C storage to `float_dtype` when `use_float16c` is
        `True`, otherwise casts the stored value to `float_dtype`.

        Parameters:
            f_dtype: The storage `DType` of `f`.
            FlayoutType: The compile-time layout of `f`; must be rank 4.
            float_dtype: The `DType` of the returned scalar.
            use_float16c: When `True`, decode the loaded value from
                Float16C (defaults to `False`).
            non_temporal: When `True`, issue a non-temporal load (defaults
                to `False`).

        Args:
            f: The distribution function tile tensor.
            index: The `(x, y, z)` lattice index.
            q: The discrete velocity index.

        Returns:
            The loaded distribution value as a `Scalar[float_dtype]`.
        """
        comptime to_compute_float = Scalar[float_dtype]
        comptime assert FlayoutType.rank == 4, 'For all LBM grids we use i,j,k,q indexing'
        comptime if use_float16c:
                comptime assert f_dtype == DType.uint16, 'Float16C requires the f tiletensors to be uint16 dtype'
                pulled_f = to_compute_float(LBM_Config.fp16c_to_fp32( f.load[non_temporal = non_temporal](coord[DType.uint32]((index[0],index[1],index[2],q)))[0] ))
            else:
                comptime assert f_dtype == float_dtype
                pulled_f = Scalar[float_dtype](f.load[non_temporal = non_temporal](coord[DType.uint32]((index[0],index[1],index[2],q)))[0])

        return pulled_f
