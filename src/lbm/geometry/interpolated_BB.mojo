"""Computes drag force on immersed objects using momentum exchange.

Iterates over fluid boundary nodes adjacent to solid objects and
accumulates the momentum-exchange force contributions.
"""
from std.gpu import block_dim,block_idx,thread_idx,grid_dim,barrier
from layout import TileTensor,LayoutTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout,col_major
from src.lbm.kernels.utils.index import get_adjacent_idx,is_index_valid
from src.utils import Vector
from src.lbm.kernels.utils.load_and_store import load_f,store_f,esoteric_pull_load_f_vec,esoteric_pull_store_f_vec
from src.lbm import LBM_Grid,LBM_Config,Lattice
from src.lbm import constants
from src.utils.runtimeLayouts import RuntimeColMajor1DType,RuntimeColMajor2DType

from src.lbm.kernels.ops import wall_bc,equilibrium_bc,SRT,TRT,KBC,RLBM
from src.lbm.kernels.utils.moment import (
                                            get_density,
                                            get_velocity,
                                        )
from src.lbm.constants import Bounceback_method


def idx_to_ijk[
    int_dtype:DType,FlagLayoutType:TensorLayout,//
    ](
    fluid_idx:Scalar[int_dtype],
    flags:TileTensor[DType.uint8,FlagLayoutType,_],
    tile_shape:Tuple[Int,Int,Int],
    ) -> InlineArray[Int,3]:

    index = InlineArray[Int,3](uninitialized = True)
    crd = flags.layout.idx2crd[out_dtype = int_dtype](Int(fluid_idx)).flatten()
    comptime if FlagLayoutType.rank*2 == FlagLayoutType.flat_rank:
        comptime for i in range(3):
            loc_x = Int(crd[2*i].value()) # local
            til_x = Int(crd[(2*i)+1].value())
            index[i] = tile_shape[i]*til_x + loc_x
    else:
        comptime assert FlagLayoutType.rank == FlagLayoutType.flat_rank
        comptime for i in range(3):
            index[i] = Int(crd[i].value())
    return index


def object_bounceback_kernel[
    bounceback_method:Bounceback_method,
    FLayoutType:TensorLayout,
    FlagLayoutType:TensorLayout,
    grid: LBM_Grid,
    config:LBM_Config[_],
    ](
        f_in:TileTensor[config.set_f_dtype(grid.float_dtype),FLayoutType,MutAnyOrigin],
        force_tensor:TileTensor[grid.float_dtype,RuntimeColMajor2DType,MutAnyOrigin],
        flags:TileTensor[DType.uint8,FlagLayoutType,ImmutAnyOrigin],
        fluid_boundaries:TileTensor[grid.int_dtype,RuntimeColMajor1DType,ImmutAnyOrigin],
        lattice_links:TileTensor[grid.int_dtype,RuntimeColMajor1DType,ImmutAnyOrigin],
        link_distances:TileTensor[grid.float_dtype,RuntimeColMajor1DType,ImmutAnyOrigin],
        compute_force:Scalar[DType.bool],
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
    comptime grid_shape:InlineArray[Int,3] = grid.shape
    comptime opposite_index = lattice.opposite_indices
    comptime weights = lattice.weights
    comptime directions = lattice.directions
    comptime float_directions = lattice.float_directions

    comptime assert config.lbm_method == constants.DOUBLE_BUFFER
    # Should be a 1D based kernel loop

    # Each thread updates their corresponding link and write to f_in in-place
    tid = block_dim.x * block_idx.x + thread_idx.x
    if tid < fluid_boundaries.layout.size():
            
        fluid_idx = fluid_boundaries[tid]
        
        index = idx_to_ijk(fluid_idx,flags,tile_shape)
        
        if index[0] < grid_shape[0] and index[1] < grid_shape[1] and index[2] < grid_shape[2]:
            i = Int(lattice_links[tid])
            opp_i = Int(opposite_index[i])

            direction = directions[i]
            q_dist = link_distances[tid]
            
            f_into_wall = load_f[float_dtype,config.DDF_shift](f_in,index,i) # About to be bounced back value
            
            comptime if bounceback_method == Bounceback_method.BOUZIDI:
                if q_dist > 0.5: # We need the f at the boundary leaving the wall and opposite direction i       
                    f_out_of_wall =  load_f[float_dtype,config.DDF_shift](f_in,index,opp_i)
                    f_bb = 0.5/q_dist*f_into_wall + (2*q_dist-1)/(2*q_dist)*f_out_of_wall
                else:
                    # we go double pull
                    # f_into_wall = load_f[float_dtype,config.DDF_shift](f_in,index,opp_i)
                    xff_index = get_adjacent_idx[-1](index,grid_shape,direction) # xff is in opp direction to i direction
                    f_at_xff = load_f[float_dtype,config.DDF_shift](f_in,xff_index,opp_i)
                    f_bb = 2*q_dist*f_into_wall + (1-2*q_dist)*f_at_xff

                store_f[config.use_float16c](f_in,f_bb,index,i)

            else: # Standard Mid Grid Bounceback
                f_bb = f_into_wall

            if compute_force:
                link_force = float_directions[i]*(f_into_wall + f_bb)
                comptime for d in range(D):
                    force_tensor[tid,d] = link_force[d]

