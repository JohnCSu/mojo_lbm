"""Provides index arithmetic for the LBM kernels.

Includes bounds checks, adjacent-index computation with pull/push shifts,
and a combined gather of neighbor indices and flags used by the streaming
step.
"""
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
    comptime assert flags.flat_rank == 3
    var neighbor_flags = InlineArray[UInt8,Q](uninitialized = True)
    var neighbor_indices = InlineArray[InlineArray[Int,3],Q](uninitialized = True)
    comptime for q in range(Q):
        comptime direction = directions[q]
        neighbor_indices[q] = get_adjacent_idx[shift = shift](index,grid_shape,direction) # Pulling Scheme
        neighbor_flags[q] = flags.load(coord[DType.int32]((neighbor_indices[q][0],neighbor_indices[q][1],neighbor_indices[q][2])))[0]

    return neighbor_indices^,neighbor_flags^

comptime get_pulled_indices_and_flags = get_indices_and_flags[_,-1] # Need directions
"""Convenience alias of `get_indices_and_flags` with `shift = -1` (pull scheme).

Pass `lattice.directions` to complete parameterization.
"""

comptime get_pushed_indices_and_flags = get_indices_and_flags[_,1] # Need directions
"""Convenience alias of `get_indices_and_flags` with `shift = 1` (push scheme).

Pass `lattice.directions` to complete parameterization.
"""



@always_inline
def is_index_out_of_bounds(index:InlineArray[Int,3],grid_shape:InlineArray[Int,3]) -> Bool:
    """Returns `True` when `index` lies outside the grid.

    Args:
        index: The `(x, y, z)` index to test.
        grid_shape: The `[nx, ny, nz]` shape of the grid.

    Returns:
        `True` when any axis is out of bounds, `False` otherwise.
    """
    is_oob = False
    comptime for i in range(3):
        is_oob = True if (index[i] >= grid_shape[i] or index[i] < 0) else is_oob
    return is_oob

@always_inline
def is_index_valid(index:InlineArray[Int,3],grid_shape:InlineArray[Int,3]) -> Bool:
    """Returns `True` when `index` lies inside the grid.

    Args:
        index: The `(x, y, z)` index to test.
        grid_shape: The `[nx, ny, nz]` shape of the grid.

    Returns:
        `True` when the index is in bounds, `False` otherwise.
    """
    return not is_index_out_of_bounds(index,grid_shape)



@always_inline
def get_adjacent_idx[int_dtype:DType,D:Int,//,shift:Int = 1](index:InlineArray[Int,3],grid_shape:InlineArray[Int,3],direction:Vector[int_dtype,D],) -> InlineArray[Int,3]:
    """Returns the neighbor of `index` along `direction` with wrap-around.

    Parameters:
        int_dtype: The `DType` of the direction vector.
        D: The spatial dimension.
        shift: The scalar multiplier applied to `direction` (defaults to 1).

    Args:
        index: The `(x, y, z)` index of the central node.
        grid_shape: The `[nx, ny, nz]` shape of the grid.
        direction: The direction vector to step along.

    Returns:
        The wrapped `(x, y, z)` index of the neighbor.
    """
    comptime assert D <= 3
    adj_index = InlineArray[Int,3](fill = 0 )
    comptime for d in range(D):
        adj_index[d] = (index[d] + shift*Int(direction[d])) % grid_shape[d]
    return adj_index

@always_inline
def get_adjacent_idx[int_dtype:DType,//,D:Int,shift:Scalar[int_dtype] = 1](index:InlineArray[Scalar[int_dtype],3],grid_shape:InlineArray[Int,3],direction:Vector[int_dtype,D],) -> InlineArray[Int,3]:
    """Returns the neighbor of `index` along `direction` with wrap-around.

    Overload that accepts `Scalar`-typed indices and a scalar shift.

    Parameters:
        int_dtype: The `DType` of the index and direction values; must not
            be a floating-point type.
        D: The spatial dimension.
        shift: The scalar multiplier applied to `direction` (defaults to 1).

    Args:
        index: The `(x, y, z)` index of the central node.
        grid_shape: The `[nx, ny, nz]` shape of the grid.
        direction: The direction vector to step along.

    Returns:
        The wrapped `(x, y, z)` index of the neighbor.
    """
    comptime assert not int_dtype.is_floating_point()
    comptime assert D <= 3
    adj_index = InlineArray[Int,3](fill = 0 )
    comptime for d in range(D):
        adj_index[d] = Int(index[d] + shift*direction[d]) % grid_shape[d]
    return adj_index
