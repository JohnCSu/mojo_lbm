"""Provides distribution-function load and store helpers for the LBM kernels.

The `load_f` and `store_f` functions wrap `TileTensor` access with optional
Float16C conversion and non-temporal hints, centralizing the `(x, y, z, q)`
indexing convention used throughout the solver.
"""
from layout import TileTensor,LayoutTensor,coord
from layout.tile_tensor import stack_allocation
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from src.lbm import LBM_Grid,LBM_Config
from src.utils import Vector
from .index import get_adjacent_idx

from src.utils.custom_fp import Float16C

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
            f_as_uint = Scalar[f_dtype](Float16C.to_fp16c(f_next))
            f.store[non_temporal = non_temporal](coord = coord[DType.uint32]((index[0],index[1],index[2],q)),value = f_as_uint)
            # You can add your own logic here by adding an elif statement to the comptime conditional
        else:
            comptime assert f_dtype == float_dtype
            f.store[non_temporal = non_temporal](coord = coord[DType.uint32]((index[0],index[1],index[2],q)),value = Scalar[f_dtype](val))


@always_inline
def load_f[
        f_dtype:DType,
        # FlayoutType:TensorLayout,
        //,
        float_dtype:DType,
        use_float16c:Bool = False,
        non_temporal:Bool = False,
        ]
        (
        f:TileTensor[f_dtype,...],
        index:InlineArray[Int,3],
        q:Int
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
        comptime assert f.rank == 4, 'For all LBM grids we use i,j,k,q indexing'
        comptime if use_float16c:
                comptime assert f_dtype == DType.uint16, 'Float16C requires the f tiletensors to be uint16 dtype'
                pulled_f = to_compute_float(Float16C.to_fp32( f.load[non_temporal = non_temporal](coord[DType.uint32]((index[0],index[1],index[2],q)))[0] ))

            else:
                comptime assert f_dtype == float_dtype
                pulled_f = Scalar[float_dtype](f.load[non_temporal = non_temporal](coord[DType.uint32]((index[0],index[1],index[2],q)))[0])

        return pulled_f


@always_inline
def get_flags
    [
    int_dtype:DType,
    Q:Int,D:Int,
    flagLayoutType:TensorLayout,
    //,
    directions:InlineArray[Vector[int_dtype, D], Q],
    shift:Int,
    *,
    start_idx:Int = 0,
    ]
    (
    flags:TileTensor[DType.uint8,flagLayoutType,ImmutAnyOrigin],
    index:InlineArray[Int,3],
    grid_shape:InlineArray[Int,3]
    )
    -> InlineArray[UInt8,Q] where start_idx >= 0:
    """Gathers the neighbor indices and flags around a lattice node.

    For each discrete velocity, computes the neighbor index with the given
    `shift` and loads the corresponding flag value.

    Parameters:
        int_dtype: The `DType` of the integer directions.
        Q: The number of discrete velocities per node.
        D: The spatial dimension.
        flagLayoutType: The compile-time layout of `flags`.
        directions: The compile-time discrete velocity directions.
        shift: The shift applied to each direction (`-1` for pull, `1` for
            push).

    Args:
        flags: The `uint8` tile tensor labeling each node.
        index: The `(x, y, z)` index of the central node.
        grid_shape: The `[nx, ny, nz]` shape of the grid.

    Returns:
        A tuple of `(neighbor_indices, neighbor_flags)` as `InlineArray`s
        of length `Q`.
    """
    comptime assert flags.rank == 3
    var neighbor_flags = InlineArray[UInt8,Q](uninitialized = True)
    
    comptime for q in range(start_idx,Q):
        comptime direction = directions[q]
        neighbor_index = get_adjacent_idx[shift = shift](index,grid_shape,direction) # Pulling Scheme
        neighbor_flags[q] = flags.load(coord[DType.int32]((neighbor_index[0],neighbor_index[1],neighbor_index[2])))[0]

    return neighbor_flags^


def esoteric_pull_load_f_vec[
    f_dtype:DType,
    int_dtype:DType,
    Q:Int,
    D:Int,
    f_layout:TensorLayout,
    //,
    float_dtype:DType,
    directions:InlineArray[Vector[int_dtype, D], Q],
    is_even_time_step:Bool,
    use_float16c:Bool,
    non_temporal:Bool = False
    ]
    (
    f:TileTensor[f_dtype,f_layout,_],
    index:InlineArray[Int,3],
    grid_shape:InlineArray[Int,3],
    ) -> Vector[float_dtype,Q]:
    # We always pull the 0th idx
    comptime load_f_from_xyzq = load_f[float_dtype,use_float16c,non_temporal]
    # comptime load_f_from_xyzq = load_f[f_dtype,non_temporal = non_temporal] # We load raw values regardles of dtype
    # f_vec = Vector[f_dtype,Q](uninitialized = True)
    f_vec = Vector[float_dtype,Q](uninitialized = True)
    f_vec[0] = load_f_from_xyzq(f,index,0)

    comptime if is_even_time_step:
    #     # Pull Positive from current node and pull negatives using standard pull scheme
        comptime for pos_q in range(1,Q-1,2):
            comptime neg_q = pos_q + 1
            comptime direction = directions[neg_q]
            pull_index = get_adjacent_idx[shift = -1](index,grid_shape,direction) # Pulling Scheme
            f_vec[pos_q] = load_f_from_xyzq(f,index,pos_q)
            f_vec[neg_q] =  load_f_from_xyzq(f,pull_index,neg_q)

    else:
        comptime for pos_q in range(1,Q-1,2):
            comptime neg_q = pos_q + 1
            # Using Push Scheme along positive directions and store in negative dir. For pos_q we get the value at current index in neg_q
            comptime direction = directions[pos_q]
            push_index = get_adjacent_idx[shift = 1](index,grid_shape,direction) # Pulling Scheme

            f_vec[pos_q] = load_f_from_xyzq(f,index,neg_q)
            f_vec[neg_q] = load_f_from_xyzq(f,push_index,pos_q)

    return f_vec



def esoteric_pull_store_f_vec[
    f_dtype:DType,
    int_dtype:DType,
    Q:Int,
    D:Int,
    f_layout:TensorLayout,
    float_dtype:DType,
    //,
    directions:InlineArray[Vector[int_dtype, D], Q],
    is_even_time_step:Bool,
    use_float16c:Bool,
    non_temporal:Bool = False
    ]
    (
    f:TileTensor[f_dtype,f_layout,MutAnyOrigin],
    f_vec:Vector[float_dtype,Q],
    index:InlineArray[Int,3],
    grid_shape:InlineArray[Int,3],
    ): 

    store_f[use_float16c,non_temporal](f,f_vec[0],index,0)

    comptime if is_even_time_step:
        # Store f back to Global
        #  WE stroe the negative directions in to the positve current index
        comptime for neg_q in range(2,Q,2):
            comptime pos_q = neg_q -1
            comptime direction = directions[neg_q]
            pull_index = get_adjacent_idx[shift = -1](index,grid_shape,direction) # Get the original index

            store_f[use_float16c,non_temporal](f,f_vec[pos_q],pull_index,neg_q) # We store it in the pull direction place
            store_f[use_float16c,non_temporal](f,f_vec[neg_q],index,pos_q)
            
            
    else:
        # We store Negatives at their respective locations at index
        comptime for neg_q in range(2,Q,2):
            comptime pos_q = neg_q -1
            # We store Positives in their push directions
            comptime direction = directions[pos_q]
            push_index = get_adjacent_idx[shift = 1](index,grid_shape,direction) # Get the original index
            
            store_f[use_float16c,non_temporal](f,f_vec[pos_q],push_index,pos_q) # We store it in the pull direction place
            store_f[use_float16c,non_temporal](f,f_vec[neg_q],index,neg_q)



@always_inline
def double_buffer_pull_load_f[
    int_dtype:DType,f_dtype:DType,D:Int,Q:Int,//,
    float_dtype:DType,
    directions:InlineArray[Vector[int_dtype, D], Q],
    use_float16c:Bool,
    *,
    non_temporal:Bool = False
    ]
    (
    f:TileTensor[f_dtype,...,address_space = AddressSpace.GENERIC],
    index:InlineArray[Int,3],
    grid_shape:InlineArray[Int,3],
    ) -> Vector[float_dtype,Q]: 
    var f_vec = Vector[float_dtype,Q](uninitialized = True)
    comptime load_f_from_xyzq = load_f[float_dtype,use_float16c,non_temporal]
    comptime for q in range(Q):
        comptime direction = directions[q]
        pull_index = get_adjacent_idx[shift = -1](index,grid_shape,direction) # Pulling Scheme
        f_vec[q] =  load_f_from_xyzq(f,pull_index,q)

    return f_vec 