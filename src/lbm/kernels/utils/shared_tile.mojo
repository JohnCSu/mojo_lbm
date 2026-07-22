"""Provides shared-memory tile loading helpers for the LBM kernels.

`get_global_index_for_shared_memory` maps a thread-local shared-memory index
to a global lattice index assuming a halo of one, and
`sync_load_rank4_tensor_to_shared_with_halo` cooperatively loads a rank-4
tensor into shared memory with that halo.
"""
from layout import TileTensor,LayoutTensor,coord
from layout.tile_tensor import stack_allocation
from layout.tile_layout import Layout,row_major,Coord,TensorLayout

@always_inline
def get_global_index_for_shared_memory[D:Int,tile_shape:Tuple[Int,Int,Int]]
                    (
                        local_index:InlineArray[Int,3],
                        block_index:InlineArray[Int,3],
                        tiler_shape:InlineArray[Int,3],
                    ) -> InlineArray[Int,3]:
    """Returns the global lattice index for a shared-memory local index.

    Assumes a halo of one along each active axis, so a `local_index` of `0`
    maps to the previous tile's last element and a `local_index` of
    `tile_shape[d] + 1` maps to the next tile's first element.

    Parameters:
        D: The spatial dimension of the grid.
        tile_shape: The per-axis tile sizes used for tiled layouts.

    Args:
        local_index: The shared-memory local index.
        block_index: The `(bx, by, bz)` block (tiler) index.
        tiler_shape: The `[n_tiles_x, n_tiles_y, n_tiles_z]` tiler shape.

    Returns:
        The global `(x, y, z)` lattice index.
    """

    comptime shift:InlineArray[Int,3] = [1,1 if D >= 2 else 0, 1 if D == 3 else 0]

    adj_local_index = InlineArray[Int,3](fill =0)
    adj_block_index = InlineArray[Int,3](fill =0)

    comptime for d in range(D):
        shifted_index = (local_index[d]-shift[d])
        adj_local_index[d] = shifted_index % tile_shape[d]
        sign = -1 if shifted_index < 0 else 1
        next_block =  shifted_index < 0 or shifted_index >= tile_shape[d]
        adj_block_index[d] = (block_index[d] + (sign if next_block else 0)) % tiler_shape[d]

    global_index = InlineArray[Int,3](fill=0)
    comptime for d in range(D):
        global_index[d] = adj_local_index[d] + adj_block_index[d]*tile_shape[d]

    return global_index


@always_inline
def sync_load_rank4_tensor_to_shared_with_halo[
    float_dtype:DType,
    sharedLayoutType:TensorLayout,
    srcLayoutType:TensorLayout,
    srcOrigin:Origin,
    //,
    tile_shape:Tuple[Int,Int,Int],
    D:Int,
    ]
    (
    mut shared_tile:TileTensor[float_dtype,sharedLayoutType, MutExternalOrigin, address_space=AddressSpace.SHARED],
    src_tensor:TileTensor[float_dtype,srcLayoutType,srcOrigin],
    tid:Int,
    block_index:InlineArray[Int,3],
    tiler_shape:InlineArray[Int,3],
    ):
    """Cooperatively loads a rank-4 tensor tile into shared memory with a halo.

    Each thread loads the points whose linear index equals `tid` modulo the
    number of threads, walking the `(SHARED_x, SHARED_y, SHARED_z)` halo-padded
    tile and the last dimension of length `N` with a comptime-unrolled loop.

    Parameters:
        float_dtype: The `DType` of the tensor elements.
        sharedLayoutType: The compile-time layout of the shared tile; must be
            rank 4 and non-nested.
        srcLayoutType: The compile-time layout of the source tensor; must be
            rank 4 and either flat-rank 4 or 8.
        srcOrigin: The origin of the source tensor.
        tile_shape: The per-axis tile sizes used for the halo-padded shared tile.
        D: The spatial dimension of the grid.

    Args:
        shared_tile: The shared-memory tile tensor to fill.
        src_tensor: The source tensor to load from.
        tid: The calling thread's linear thread index.
        block_index: The `(bx, by, bz)` block (tiler) index.
        tiler_shape: The `[n_tiles_x, n_tiles_y, n_tiles_z]` tiler shape.
    """
    comptime assert shared_tile.rank == 4 and shared_tile.flat_rank == 4
    comptime assert src_tensor.rank == 4 and (src_tensor.flat_rank == 4 or src_tensor.flat_rank == 8)

    comptime is_nested = src_tensor.rank != src_tensor.flat_rank
    comptime src_N = src_tensor.static_shape[6]*src_tensor.static_shape[7] if is_nested else src_tensor.static_shape[3]
    comptime N = shared_tile.static_shape[3]
    comptime assert N == src_N, 'The last dimension of the shared tile and soruce tensor must be <= to last dimension of src_tensor'

    comptime SHARED_x = tile_shape[0] + 2
    comptime SHARED_y = tile_shape[1] + 2 if D >= 2 else 1
    comptime SHARED_z = tile_shape[2] + 2 if D == 3 else 1
    comptime NUM_THREADS = tile_shape[0]*tile_shape[1]*tile_shape[2]
    comptime NUM_SHARED_XYZ_POINTS = SHARED_x*SHARED_y*SHARED_z
    var shared_local_index = InlineArray[Int,3](uninitialized = True)
    for i in range(tid,NUM_SHARED_XYZ_POINTS,NUM_THREADS):
        shared_local_index[0] = i % SHARED_x
        shared_local_index[1] = (i % (SHARED_x*SHARED_y))//SHARED_x
        shared_local_index[2] = i // (SHARED_x * SHARED_y)
        shared_global_index = get_global_index_for_shared_memory[D,tile_shape](shared_local_index,block_index,tiler_shape)
        comptime for n in range(N):
            val = src_tensor.load(coord[DType.int32]((shared_global_index[0],shared_global_index[1],shared_global_index[2],n)))[0]
            shared_tile[shared_local_index[0],shared_local_index[1],shared_local_index[2],n] = val
