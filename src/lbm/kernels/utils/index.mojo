"""Provides index arithmetic for the LBM kernels.

Includes bounds checks, adjacent-index computation with pull/push shifts,
and a combined gather of neighbor indices and flags used by the streaming
step.
"""
from src.utils import Vector
from layout import TileTensor,coord
from layout.tile_layout import TensorLayout


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
