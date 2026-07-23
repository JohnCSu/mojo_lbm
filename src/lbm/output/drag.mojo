from std.gpu import block_dim,block_idx,thread_idx,grid_dim,barrier
from layout import TileTensor,LayoutTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout,col_major
from src.lbm.kernels.utils.index import get_adjacent_idx,is_index_valid
from src.utils import Vector
from src.lbm.kernels.utils.load_and_store import load_f,store_f,esoteric_pull_load_f_vec,esoteric_pull_store_f_vec
from src.lbm import LBM_Grid,LBM_Config,Lattice

def rowMajor1D[int_dtype:DType]() -> type_of( row_major(coord[int_dtype]((1,))) ):
    """Returns a 1D row-major `Int`-dtype layout instance.

    Parameters:
        int_dtype: The `DType` used for the layout coordinates.

    Returns:
        A 1D row-major layout instance.
    """
    return row_major(coord[int_dtype]((1,)))

def rowMajor2D[int_dtype:DType]() -> type_of(row_major(coord[int_dtype]((1,2))) ):
    """Returns a 2D row-major `Int`-dtype layout instance.

    Parameters:
        int_dtype: The `DType` used for the layout coordinates.

    Returns:
        A 2D row-major layout instance.
    """
    return row_major(coord[int_dtype]((1,2)))


def calculate_drag_around_object[
    FLayout:Layout[...] ,
    FlagLayout:Layout[...],
    grid: LBM_Grid,
    config:LBM_Config,
    *,
    f_dtype:DType = grid.float_dtype if config.f_dtype is None else config.f_dtype.value()
    ](
        f:TileTensor[f_dtype,type_of(FLayout),ImmutAnyOrigin],
        flags:TileTensor[DType.uint8,type_of(FlagLayout),ImmutAnyOrigin],
        fluid_boundary:TileTensor[grid.int_dtype,type_of(rowMajor1D[grid.int_dtype]()),MutAnyOrigin],
        force_tensor:TileTensor[grid.float_dtype,type_of(rowMajor2D[grid.int_dtype]()),MutAnyOrigin],
    ):

    """Computes the drag force on the fluid nodes adjacent to an immersed object.

    Iterates over the linear fluid boundary indices, gathers the push-scheme
    neighbor flags, and accumulates the momentum-exchange contribution

    $$F = \\sum_q 2 f_{link} e_q$$

    for every direction `q` whose push neighbor is solid. The result is
    written into `force_tensor[tid, i]` for each dimension `i`.

    Parameters:
        FLayout: The compile-time `Layout` of the distribution function `f`.
        FlagLayout: The compile-time `Layout` of the `uint8` flag tensor.
        grid: The compile-time `LBM_Grid` describing the domain.
        config: The `LBM_Config` used to select storage options.
        f_dtype: The storage `DType` for `f` (defaults to the config's
            `f_dtype` or `float_dtype`).

    Args:
        f: The input distribution function tile tensor (rank 4).
        flags: The `uint8` tile tensor labeling each node (rank 3).
        fluid_boundary: The 1D tile tensor of linear fluid boundary indices.
        force_tensor: The 2D output tile tensor of per-node force vectors.
    """
    comptime D = grid.D
    comptime Q = grid.Q
    comptime float_dtype = grid.float_dtype
    comptime int_dtype = grid.int_dtype
    comptime lattice = grid.lattice
    comptime tile_shape = grid.tile_shape
    # Should be a 1D based kernel loop
    tid = block_dim.x * block_idx.x + thread_idx.x
    if tid < fluid_boundary.layout.size():
        crd = FlagLayout.idx2crd[out_dtype = int_dtype](Int(fluid_boundary[tid])).flatten()
        grid_index = InlineArray[Int,3](uninitialized = True)

        comptime if FlagLayout.rank*2 == FlagLayout.flat_rank:
            comptime for i in range(3):
                loc_x = Int(crd[2*i].value()) # local
                til_x = Int(crd[(2*i)+1].value())
                grid_index[i] = tile_shape[i]*til_x + loc_x
        else:
            comptime assert FlagLayout.rank == FlagLayout.flat_rank
            comptime for i in range(3):
                grid_index[i] = Int(crd[i].value())

        comptime grid_shape:InlineArray[Int,3] = grid.shape
        comptime opposite_index = lattice.opposite_indices

        if grid_index[0] < grid_shape[0] and grid_index[1] < grid_shape[1] and grid_index[2] < grid_shape[2]:
            var push_flags = InlineArray[UInt8,Q](uninitialized = True)
            var push_indices = InlineArray[InlineArray[Int,3],Q](uninitialized = True)

            # Gather Neighboring Flags in PUSH direction not Pull
            comptime for q in range(Q):
                comptime direction = lattice.directions[q]
                push_indices[q] = get_adjacent_idx[1](grid_index,grid_shape,direction) # push Scheme as
                push_flags[q] = flags.load(coord[DType.uint32]((push_indices[q][0],push_indices[q][1],push_indices[q][2])))[0]

            # Compute Forces
            var force_vec = Vector[float_dtype,D](fill = 0.)
            comptime for q in range(Q):
                comptime direction = lattice.directions[q]
                comptime float_direction = lattice.directions[q].cast_to[float_dtype]()
                if push_flags[q] == SOLID_NODE:
                    var f_local = load_f[float_dtype,config.use_float16c](f,grid_index,q)
                    comptime if config.DDF_shift:
                        comptime weight = lattice.weights[q]
                        f_link = f_local + weight
                    else:
                        f_link = f_local
                    force_vec += (2*f_link)*float_direction # Only stationary wall for now

            # push to global
            comptime for i in range(D): # Overwrite
                force_tensor[tid,i] = force_vec[i]
