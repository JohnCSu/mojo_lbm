from layout import TileTensor,LayoutTensor,coord
from layout.tile_tensor import stack_allocation
from layout.tile_layout import Layout,row_major,Coord,TensorLayout

@always_inline
def get_global_index_for_shared_memory[D:Int,tile_size:Int]
                    (
                        local_index:InlineArray[Int,3],
                        block_index:InlineArray[Int,3],
                        tiler_shape:InlineArray[Int,3],
                    ) -> InlineArray[Int,3]:
    '''
    Assumes Halo Region of 1
    '''
    
    comptime shift:InlineArray[Int,3] = [1,1 if D >= 2 else 0, 1 if D == 3 else 0]

    adj_local_index = InlineArray[Int,3](fill =0)
    adj_block_index = InlineArray[Int,3](fill =0)
    
    comptime for d in range(D):
        shifted_index = (local_index[d]-shift[d])
        adj_local_index[d] = shifted_index % tile_size # Modulo as we flip back
        sign = -1 if shifted_index < 0 else 1
        next_block =  shifted_index < 0 or shifted_index >= tile_size        
        adj_block_index[d] = (block_index[d] + (sign if next_block else 0)) % tiler_shape[d]

    global_index = InlineArray[Int,3](fill=0)
    comptime for d in range(D):
        global_index[d] = adj_local_index[d] + adj_block_index[d]*tile_size
        
    return global_index


@always_inline
def sync_load_rank4_tensor_to_shared_with_halo[
    float_dtype:DType,
    sharedLayoutType:TensorLayout,
    srcLayoutType:TensorLayout,
    srcOrigin:Origin,
    //,
    tile_size:Int,
    D:Int,
    ]
    (
    shared_tile:TileTensor[float_dtype,sharedLayoutType, MutExternalOrigin, address_space=AddressSpace.SHARED],
    src_tensor:TileTensor[float_dtype,srcLayoutType,srcOrigin],
    tid:Int,
    block_index:InlineArray[Int,3],
    tiler_shape:InlineArray[Int,3],
    ):
    comptime assert shared_tile.rank == 4 and shared_tile.flat_rank == 4
    comptime assert src_tensor.rank == 4 and (src_tensor.flat_rank == 4 or src_tensor.flat_rank == 8)

    comptime is_nested = src_tensor.rank != src_tensor.flat_rank
    comptime src_N = src_tensor.static_shape[6]*src_tensor.static_shape[7] if is_nested else src_tensor.static_shape[3]
    comptime N = shared_tile.static_shape[3] # Guranteed to be non-nested layout from assertion above
    comptime assert N == src_N, 'The last dimension of the shared tile and soruce tensor must be <= to last dimension of src_tensor'
    
    comptime SHARED_x = tile_size + 2
    comptime SHARED_y = tile_size + 2 if D >= 2 else 1 
    comptime SHARED_z = tile_size + 2 if D == 3 else 1
    comptime NUM_THREADS = tile_size**D
    comptime NUM_SHARED_XYZ_POINTS = SHARED_x*SHARED_y*SHARED_z

    var shared_local_index = InlineArray[Int,3](uninitialized = True)
    for i in range(tid,NUM_SHARED_XYZ_POINTS,NUM_THREADS): # loop only iterates 1-2 per thread
        # Indexes for shared array
        shared_local_index[0] = i % SHARED_x
        shared_local_index[1] = (i % (SHARED_x*SHARED_y))//SHARED_x 
        shared_local_index[2] = i // (SHARED_x * SHARED_y)
        # Index for the current i threadindex
        shared_global_index = get_global_index_for_shared_memory[D,tile_size](shared_local_index,block_index,tiler_shape)
        comptime for n in range(N):
            val = src_tensor.load(coord[DType.int32]((shared_global_index[0],shared_global_index[1],shared_global_index[2],n)))[0]
            shared_tile[shared_local_index[0],shared_local_index[1],shared_local_index[2],n] = val
    