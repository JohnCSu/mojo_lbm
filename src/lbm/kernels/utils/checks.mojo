"""Provides compile-time assertions for validating the discrete velocity
set.

Checks that the lattice velocity directions satisfy the adjacency and
rest-direction conventions required by the TRT and esoteric-pull
kernels.
"""
from src.utils import Vector

def opposite_indices_are_adjacent[int_dtype:DType,D:Int,Q:Int,//](directions:InlineArray[Vector[int_dtype, D], Q]) -> Bool:
    """Verifies that opposite velocity directions are stored at adjacent
    indices.

    Parameters:
        int_dtype: The integer `DType` for the velocity directions.
        D: The spatial dimension.
        Q: The number of discrete velocities.

    Args:
        directions: The compile-time discrete velocity directions.

    Returns:
        `True` if each direction at index `q` (q >= 1) has its opposite
        at `q + 1`, `False` otherwise.
    """
    comptime for q in range(1,Q-1,2):
        dir_q = directions[q]
        opp_dir_q = directions[q+1]
        # print(q,q+1,dir_q,opp_dir_q)
        if (dir_q + opp_dir_q).sum() != 0: # If not equal to zero then the adjacent direction is not the opposite direction
            return False
    # All must hold the condition to return True
    return True


def rest_direction_is_zero[int_dtype:DType,D:Int,Q:Int,//](directions:InlineArray[Vector[int_dtype, D], Q]) -> Bool:
    """Verifies that the rest direction is the zero vector.

    Parameters:
        int_dtype: The integer `DType` for the velocity directions.
        D: The spatial dimension.
        Q: The number of discrete velocities.

    Args:
        directions: The compile-time discrete velocity directions.

    Returns:
        `True` if `directions[0]` is the zero vector, `False`
        otherwise.
    """
    return False if directions[0].any_true() else True


@always_inline
def is_valid_thread(index:InlineArray[Int,3],grid_shape:InlineArray[Int,3],flag:UInt8) -> Bool:
    valid_index = (index[0] < grid_shape[0]) and (index[1] < grid_shape[1]) and (index[2] < grid_shape[2])

    node_to_be_excluded = Flags.is[Flags.SOLID](flag) or Flags.has[Flags.EXCLUDE](flag) 

    return  valid_index and not node_to_be_excluded
