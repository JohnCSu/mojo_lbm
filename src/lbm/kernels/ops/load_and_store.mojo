
from layout import TileTensor,LayoutTensor,coord
from layout.tile_tensor import stack_allocation
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from src.lbm import LBM_Grid,LBM_Config,Flags,LBM_method
from src.utils import Vector
from src.lbm.kernels.utils.checks import opposite_indices_are_adjacent
from src.lbm.kernels.utils.index import get_adjacent_idx
from src.lbm.kernels.utils.load_and_store import load_f,store_f
from src.utils.custom_fp import Float16C

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
    """Loads the distribution vector using the esoteric pull scheme.

    On even time steps, loads positive directions from the current node
    and negative directions from the pull neighbor. On odd time steps,
    the assignment is reversed, swapping the roles of positive and
    negative directions.

    Parameters:
        f_dtype: The storage `DType` of the distribution function.
        int_dtype: The integer `DType` for the velocity directions.
        Q: The number of discrete velocities.
        D: The spatial dimension.
        f_layout: The compile-time `Layout` of the distribution
            function.
        float_dtype: The compute `DType` for the returned vector.
        directions: The compile-time discrete velocity directions.
        is_even_time_step: When `True`, use the even-step loading
            pattern.
        use_float16c: When `True`, decode Float16C storage.
        non_temporal: When `True`, issue non-temporal loads (defaults
            to `False`).

    Args:
        f: The distribution function tile tensor.
        index: The `(x, y, z)` index of the current node.
        grid_shape: The `[nx, ny, nz]` shape of the grid.

    Returns:
        The loaded distribution vector of length `Q`.
    """
    # We always pull the 0th idx
    comptime assert opposite_indices_are_adjacent(directions),'For esoteric pull methods, opposite indices must be adjacent and positive directions are assumed to be odd indices'
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
    f_origin:Origin[mut=True],
    //,
    directions:InlineArray[Vector[int_dtype, D], Q],
    is_even_time_step:Bool,
    use_float16c:Bool,
    non_temporal:Bool = False
    ]
    (
    f:TileTensor[f_dtype,f_layout,f_origin],
    f_vec:Vector[float_dtype,Q],
    index:InlineArray[Int,3],
    grid_shape:InlineArray[Int,3],
    ):
    """Stores the distribution vector using the esoteric pull scheme.

    On even time steps, stores negative-direction values at the pull
    neighbor while positive-direction values stay at the current node.
    On odd time steps, the assignment is reversed.

    Parameters:
        f_dtype: The storage `DType` of the distribution function.
        int_dtype: The integer `DType` for the velocity directions.
        Q: The number of discrete velocities.
        D: The spatial dimension.
        f_layout: The compile-time `Layout` of the distribution
            function.
        float_dtype: The compute `DType` of the distribution vector.
        f_origin: The mutable origin of the distribution function
            tensor.
        directions: The compile-time discrete velocity directions.
        is_even_time_step: When `True`, use the even-step storage
            pattern.
        use_float16c: When `True`, encode to Float16C storage.
        non_temporal: When `True`, issue non-temporal stores (defaults
            to `False`).

    Args:
        f: The mutable distribution function tile tensor.
        f_vec: The distribution vector to store.
        index: The `(x, y, z)` index of the current node.
        grid_shape: The `[nx, ny, nz]` shape of the grid.
    """
    store_f[use_float16c,non_temporal](f,f_vec[0],index,0)
    comptime assert opposite_indices_are_adjacent(directions),'For esoteric pull methods, opposite indices must be adjacent and positive directions are assumed to be odd indices'
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
    int_dtype:DType,
    f_dtype:DType,
    D:Int,
    Q:Int,
    //,
    float_dtype:DType,
    directions:InlineArray[Vector[int_dtype, D], Q],
    opposite_indices:InlineArray[Scalar[int_dtype], Q],
    use_float16c:Bool,
    *,
    non_temporal:Bool = False,
    ]
    (
    f:TileTensor[f_dtype,...,address_space = AddressSpace.GENERIC],
    pull_flags:InlineArray[UInt8,Q],
    index:InlineArray[Int,3],
    grid_shape:InlineArray[Int,3],
    ) -> Vector[float_dtype,Q]:
    """Loads the full distribution vector using the double-buffer pull
    scheme.

    For each discrete velocity, loads the distribution value from the
    pull neighbor (the node shifted by `-direction`).

    Parameters:
        int_dtype: The integer `DType` for the velocity directions.
        f_dtype: The storage `DType` of the distribution function.
        D: The spatial dimension.
        Q: The number of discrete velocities.
        float_dtype: The compute `DType` for the returned vector.
        directions: The compile-time discrete velocity directions.
        use_float16c: When `True`, decode Float16C storage.
        non_temporal: When `True`, issue non-temporal loads (defaults
            to `False`).

    Args:
        f: The distribution function tile tensor.
        index: The `(x, y, z)` index of the current node.
        grid_shape: The `[nx, ny, nz]` shape of the grid.

    Returns:
        The loaded distribution vector of length `Q`.
    """
    var f_vec = Vector[float_dtype,Q](uninitialized = True)
    comptime load_f_from_xyzq = load_f[float_dtype,use_float16c,non_temporal]
    comptime for q in range(Q):
        comptime direction = directions[q]
        comptime opp_q = Int(opposite_indices[q])
        pull_index = get_adjacent_idx[shift = -1](index,grid_shape,direction) # Pulling Scheme
        f_vec[q] =  load_f_from_xyzq(f,pull_index,q)

        if pull_flags[q] == Flags.SOLID:
            f_vec[q] = load_f_from_xyzq(f,index,opp_q) # Bounceback
        else:
            f_vec[q] = load_f_from_xyzq(f,pull_index,q)

    return f_vec 


def load_single_f[
    int_dtype:DType,
    f_dtype:DType,
    D:Int,
    Q:Int,
    //,
    lbm_method:LBM_method,
    float_dtype:DType,
    directions:InlineArray[Vector[int_dtype, D], Q],
    opposite_indices:InlineArray[Scalar[int_dtype], Q],
    use_float16c:Bool,
    *,
    is_even_time_step:Optional[Bool] = None,
    non_temporal:Bool = False,
    ]( 
    f:TileTensor[f_dtype,...,address_space = AddressSpace.GENERIC],
    index:InlineArray[Int,3],
    q:Int,
    grid_shape:InlineArray[Int,3],
    ) -> Scalar[float_dtype]:
    comptime load_f_val = load_f[float_dtype,use_float16c,non_temporal]
    
    comptime if lbm_method == LBM_method.DOUBLE_BUFFER:
        return load_f_val(f,index,q)

    elif lbm_method == LBM_method.ESOTERIC_PULL:
        comptime assert is_even_time_step is not None, 'is_even_time_step cannot be none for esoteric pull config'

        if q == 0:
            return load_f_val(f,index,0)

        is_pos_q = ((q % 2) == 1) # Odd indices are positive, Even Indices are negative
        comptime if is_even_time_step.value():
            neg_q = q+1 if is_pos_q else q
            direction = directions[neg_q]
            pull_index = get_adjacent_idx[shift = -1](index,grid_shape,direction) # Case if q is neg
            index_to_load = index if is_pos_q else pull_index
            q_to_load = q

        else:
            pos_q = q if is_pos_q else q-1
            direction = directions[pos_q]
            push_index = get_adjacent_idx[shift = 1](index,grid_shape,direction)
            index_to_load = index if is_pos_q else push_index
            q_to_load = q+1 if is_pos_q else q-1
        
        return load_f_val(f,index_to_load,q_to_load)

    else:
        comptime assert False, 'Invalid lbm method used' 


@always_inline
def set_adjacent_flags[
    int_dtype:DType,
    Q:Int,
    D:Int,
    FlaglayoutType:TensorLayout,
    //,
    directions:InlineArray[Vector[int_dtype, D], Q],
    start_idx:Int = 1,
    *,
    end_idx:Int = Q,
    shift:Int = -1
    ](
    mut pull_flags:InlineArray[UInt8,Q],
    flags:TileTensor[DType.uint8,FlaglayoutType,_],
    index:InlineArray[Int,3],
    grid_shape:InlineArray[Int,3]
    ):

    comptime for q in range(start_idx,end_idx):
        comptime direction = directions[q]
        pull_index = get_adjacent_idx[shift](index,grid_shape,direction) # Pulling Scheme
        pull_flags[q] = flags.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2])))[0]
    