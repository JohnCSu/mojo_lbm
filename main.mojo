from std.gpu.host import DeviceContext
from std.sys import has_accelerator
from std.gpu import block_idx,thread_idx
from layout import TileTensor,coord
from layout.tile_layout import Layout,row_major

from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from std.collections import InlineArray
from std.memory import Pointer
from std.collections import Set,Dict

from flags import *
from contextTensor import ContextTileTensor
from LBM import set_outer_walls,LBM_kernel,LBM_Grid,get_D2Q9,LBM_kernel

comptime float_dtype = DType.float32
comptime int_dtype = DType.int32
comptime float_scalar = Scalar[float_dtype]
comptime D2Q9 = get_D2Q9[DType.float32,DType.int32]()
comptime D,Q = (2,9)
comptime N = 5
comptime (nx,ny,nz) = (N,N,1)
comptime num_points = nx*ny*nz

comptime THREADS_PER_BLOCK = 32
comptime BLOCK_GRID = (nx // THREADS_PER_BLOCK + 1,ny // THREADS_PER_BLOCK + 1,nz // THREADS_PER_BLOCK + 1 )# Plus one

# This can be stored in LBM Grid
comptime flag_layout = row_major[nx,ny,nz]()
comptime FlagLayoutType = type_of(flag_layout)
comptime f_layout = row_major[D,nx,ny,nz]()
comptime FLayoutType = type_of(f_layout)

comptime bc_layout = row_major[nx,ny,nz,D+1]()
comptime BCLayoutType = type_of(bc_layout)

def main() raises:
    # print(D2Q9.directions)
    # print(D2Q9.opposite_indices)
    ctx = DeviceContext()
    var dx = 1/float_scalar(N)
    var grid = LBM_Grid[D2Q9,nx,ny,nz](dx)
    
    flags = ContextTileTensor[DType.uint8](ctx,flag_layout)
    bc = ContextTileTensor[float_dtype](ctx,bc_layout)
    f = ContextTileTensor[float_dtype](ctx,f_layout)
    f_out = ContextTileTensor[float_dtype](ctx,f_layout)

    set_outer_walls[D,nx,ny,nz](flags.cpu(),bc.cpu(),'-Y',SOLID_NODE,[1,0],1.)
    set_outer_walls[D,nx,ny,nz](flags.cpu(),bc.cpu(),'+Y',SOLID_NODE,[0,0],1.)
    set_outer_walls[D,nx,ny,nz](flags.cpu(),bc.cpu(),'-X',SOLID_NODE,[0,0],1.)
    set_outer_walls[D,nx,ny,nz](flags.cpu(),bc.cpu(),'+X',SOLID_NODE,[0,0],1.)
    print(flags.last_used_cpu)
  
    flags.copy_cpu_to_gpu()
    bc.copy_cpu_to_gpu()
    f.copy_cpu_to_gpu()
    f_out.copy_cpu_to_gpu()

    ctx.synchronize()
    
    LBM_func = ctx.compile_function[LBM_kernel[D2Q9],LBM_kernel[D2Q9]]()
    ctx.enqueue_function(LBM_func,f_out.gpu(),f.gpu(),bc.gpu(),flags.gpu(),grid.shape,grid_dim = (1,1,1),block_dim = (1,1,1))
    


    # f_buffer =  ctx.enqueue_create_host_buffer[grid.float_dtype](grid.f_field_size)
    # f_out_buffer =  ctx.enqueue_create_host_buffer[grid.float_dtype](grid.f_field_size)

    # flag_buffer = ctx.enqueue_create_host_buffer[DType.uint8](grid.num_points)
    # bc_buffer = ctx.enqueue_create_host_buffer[grid.float_dtype](grid.bc_field_size)

    # ctx.synchronize()
    # # flags = TileTensor[DType.uint8,RowMajorType,MutAnyOrigin](flag_buffer,layout)
    
    # # Do BC on this
    # flags = TileTensor[DType.uint8](flag_buffer,flag_layout)
    # bc = TileTensor[float_dtype](bc_buffer,bc_layout)
    # f = TileTensor[float_dtype](f_buffer,f_layout)
    # Make buffers to GPU

    # f_buffer_gpu =  ctx.enqueue_create_buffer[grid.float_dtype](grid.f_field_size)
    # f_out_buffer_gpu =  ctx.enqueue_create_buffer[grid.float_dtype](grid.f_field_size)

    # flag_buffer_gpu = ctx.enqueue_create_buffer[DType.uint8](grid.num_points)
    # bc_buffer_gpu = ctx.enqueue_create_buffer[grid.float_dtype](grid.bc_field_size)

    # ctx.synchronize()

    # # Copy Buffers from cpu to GPU

    # ctx.enqueue_copy(dst_buf = f_buffer_gpu,src_buf = f_buffer)
    # ctx.enqueue_copy(dst_buf = f_out_buffer_gpu,src_buf = f_out_buffer)
    # ctx.enqueue_copy(dst_buf = flag_buffer_gpu,src_buf = flag_buffer)
    # ctx.enqueue_copy(dst_buf = bc_buffer_gpu,src_buf = bc_buffer)

    # ctx.synchronize()

