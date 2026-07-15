from std.gpu import block_dim,block_idx,thread_idx,grid_dim,barrier
from layout import TileTensor,LayoutTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout,col_major
from layout.tile_tensor import stack_allocation
from std.gpu.memory import AddressSpace
from .index import is_index_valid

@always_inline
def get_velocity_gradient[
    float_dtype:DType,sharedType:TensorLayout,flagType:TensorLayout,//,
    dx:Scalar[float_dtype]
    ]
    (
    shared_u:TileTensor[float_dtype,sharedType,_,address_space = AddressSpace.SHARED],
    flags:TileTensor[DType.uint8,flagType,ImmutAnyOrigin,],
    local_index:InlineArray[Int,3],
    global_index:InlineArray[Int,3],
    grid_shape:InlineArray[Int,3],
    velocity_direction:Int,
    axis:Int,
    ) -> Scalar[float_dtype]:
    '''
    TODO: Check for periodicity (Assume one sided finite difference at boundary currently)
    '''
    comptime assert shared_u.flat_rank == 4 
    comptime assert flags.rank == 3
    comptime inv_dx = 1/dx

    var adj_index = local_index.copy()
    var adj_global_index = global_index.copy()
    var left_grad:Scalar[float_dtype] =0.
    var left_dx:Scalar[float_dtype] = 0.

    var right_grad:Scalar[float_dtype] =0.
    var right_dx:Scalar[float_dtype] = 0.

    var u2 = shared_u[local_index[0],local_index[1],local_index[2],velocity_direction]
    
    adj_index[axis] -= 1 # shared always be between (and incl) 0 and tile_size
    adj_global_index[axis] = (global_index[axis] - 1)
    var adj_is_valid_left = is_index_valid(adj_global_index,grid_shape)
    if adj_is_valid_left:
        var u1 = shared_u[adj_index[0],adj_index[1],adj_index[2],velocity_direction]
        var f1 = flags.load(coord[DType.int32]((adj_global_index[0],adj_global_index[1],adj_global_index[2])))[0]
        left_grad,left_dx = get_adj_finite_difference[dx,'left'](adj_index,adj_global_index,f1,u1,u2,grid_shape)

    adj_index[axis] += 2
    adj_global_index[axis] = (global_index[axis] + 1)

    var adj_is_valid_right = is_index_valid(adj_global_index,grid_shape)
    if adj_is_valid_right:
        u3 = shared_u[adj_index[0],adj_index[1],adj_index[2],velocity_direction] 
        f3 = flags.load(coord[DType.int32]((adj_global_index[0],adj_global_index[1],adj_global_index[2])))[0]
        right_grad,right_dx = get_adj_finite_difference[dx,'right'](adj_index,adj_global_index,f3,u3,u2,grid_shape)

    total_dx = left_dx + right_dx

    left_weight = 1 - left_dx/total_dx
    right_weight = (1 - left_weight)

    return right_grad*right_weight + left_grad*left_weight if adj_is_valid_left or adj_is_valid_right else 0.


@always_inline
def get_adj_finite_difference[
    float_dtype:DType,//,
    dx:Scalar[float_dtype],
    side:StaticString = 'right',
    ]
    (
    adj_index:InlineArray[Int,3],
    adj_global_index:InlineArray[Int,3],
    adj_flag:UInt8,
    adj_u:Scalar[float_dtype],
    u:Scalar[float_dtype],
    grid_shape:InlineArray[Int,3])
    -> Tuple[Scalar[float_dtype],Scalar[float_dtype]]:
    comptime assert side in {'right','left'}
    grad_dx = 0.5*dx if adj_flag == SOLID_NODE else dx
    comptime if side == 'left': # u_cur - u_1
        grad = forward_difference(u,adj_u,1/grad_dx)
    else: # u_adj - u_cur
        grad = forward_difference(adj_u,u,1/grad_dx)

    return grad,grad_dx

@always_inline
def forward_difference[
    float_dtype:DType
    ]
    (
    u2:Scalar[float_dtype],
    u1:Scalar[float_dtype],
    inv_dx:Scalar[float_dtype]) -> Scalar[float_dtype]:
    return (u2-u1)*inv_dx


