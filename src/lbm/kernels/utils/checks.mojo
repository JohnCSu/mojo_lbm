from src.utils import Vector

def opposite_indices_are_adjacent[int_dtype:DType,D:Int,Q:Int,//](directions:InlineArray[Vector[int_dtype, D], Q]) -> Bool:
        comptime for q in range(1,Q-1,2):
            dir_q = directions[q]
            opp_dir_q = directions[q+1]
            # print(q,q+1,dir_q,opp_dir_q)
            if (dir_q + opp_dir_q).sum() != 0: # If not equal to zero then the adjacent direction is not the opposite direction
                return False
        # All must hold the condition to return True
        return True


def rest_direction_is_zero[int_dtype:DType,D:Int,Q:Int,//](directions:InlineArray[Vector[int_dtype, D], Q]) -> Bool:
    return False if directions[0].any_true() else True