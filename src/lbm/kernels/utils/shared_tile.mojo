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