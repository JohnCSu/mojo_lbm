"""Computes drag force on immersed objects using momentum exchange.

Iterates over fluid boundary nodes adjacent to solid objects and
accumulates the momentum-exchange force contributions.
"""
from std.gpu import block_dim,block_idx,thread_idx,grid_dim
from max.gpu.sync import barrier
from layout import TileTensor,LayoutTensor
from layout.tile_layout import Layout,row_major,Coord,TensorLayout,col_major
from src.lbm.kernels.utils.index import get_adjacent_idx,is_index_valid
from src.utils import Vector
from src.lbm.kernels.utils.load_and_store import load_f,store_f
from src.lbm import LBM_Grid,LBM_Config,Lattice
from src.lbm import constants
from src.utils.runtimeLayouts import RuntimeColMajor1DType,RuntimeColMajor2DType


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

    var index = InlineArray[Int,3](uninitialized = True)
    var crd = flags.layout.idx2crd[out_dtype = int_dtype](Int(fluid_idx)).flatten()
    comptime if FlagLayoutType.rank*2 == FlagLayoutType.flat_rank:
        comptime for i in range(3):
            var loc_x = Int(crd[2*i].value()) # local
            var til_x = Int(crd[(2*i)+1].value())
            index[i] = tile_shape[i]*til_x + loc_x
    else:
        comptime assert FlagLayoutType.rank == FlagLayoutType.flat_rank
        comptime for i in range(3):
            index[i] = Int(crd[i].value())
    return index^


def linkwise_bounceback_kernel[
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
        q_clamp:Scalar[grid.float_dtype],
    ):

    """Computes the drag force on the fluid nodes adjacent to an immersed object.

    Iterates over the linear fluid boundary indices, gathers the push-scheme
    neighbor flags, and accumulates the momentum-exchange contribution

    $$F = \\sum_q 2 f_{link} e_q$$

    for every direction `q` whose push neighbor is solid. The result is
    written into `force_tensor[tid, i]` for each dimension `i`.

    Parameters:
        bounceback_method: The bounce-back method.
        FLayoutType: The compile-time layout of the distribution function.
        FlagLayoutType: The compile-time layout of the flag tensor.
        grid: The compile-time `LBM_Grid` describing the domain.
        config: The compile-time `LBM_Config` for the run.

    Args:
        f_in: The input distribution function tile tensor.
        force_tensor: The 2D output tile tensor of per-node force vectors.
        flags: The `uint8` tile tensor labeling each node.
        fluid_boundaries: The 1D tile tensor of linear fluid boundary
            indices.
        lattice_links: The 1D tile tensor of lattice link indices.
        link_distances: The 1D tile tensor of wall distances.
        compute_force: Whether to compute the force.
        q_clamp: The minimum wall distance.
    """
    comptime D = grid.D
    comptime Q = grid.Q
    comptime float_dtype = grid.float_dtype
    comptime int_dtype = grid.int_dtype
    comptime lattice = grid.lattice
    comptime tile_shape = grid.tile_shape
    var grid_shape:InlineArray[Int,3] = materialize[grid.shape]()
    var opposite_index = materialize[lattice.opposite_indices]()
    var weights = materialize[lattice.weights]()
    var directions = materialize[lattice.directions]()
    var float_directions = materialize[lattice.float_directions]()

    comptime assert config.lbm_method == constants.DOUBLE_BUFFER
    # Should be a 1D based kernel loop

    # Each thread updates their corresponding link and write to f_in in-place
    var tid = block_dim.x * block_idx.x + thread_idx.x
    if tid < fluid_boundaries.layout.size():
            
        var fluid_idx = fluid_boundaries[tid]
        
        var index = idx_to_ijk(fluid_idx,flags,tile_shape)
        
        if index[0] < grid_shape[0] and index[1] < grid_shape[1] and index[2] < grid_shape[2]:
            
            var i = Int(lattice_links[tid])
            var opp_i = Int(opposite_index[i])

            var direction = directions[i]
            var q_dist = link_distances[tid]
            q_dist = max(q_dist,q_clamp)
            
            var f_into_wall = load_f[float_dtype,config.use_float16c](f_in,index,i) # About to be bounced back value

            var f_bb: Scalar[float_dtype]
            comptime if bounceback_method == Bounceback_method.BOUZIDI:
                if q_dist > 0.5: # We need the f at the boundary leaving the wall and opposite direction i       
                    var f_out_of_wall =  load_f[float_dtype,config.use_float16c](f_in,index,opp_i)
                    f_bb = 0.5/q_dist*f_into_wall + (2*q_dist-1)/(2*q_dist)*f_out_of_wall
                else:
                    # we go double pull
                    # f_into_wall = load_f[float_dtype,config.DDF_shift](f_in,index,opp_i)
                    var xff_index = get_adjacent_idx[-1](index,grid_shape,direction) # xff is in opp direction to i direction
                    var f_at_xff = load_f[float_dtype,config.use_float16c](f_in,xff_index,opp_i)
                    f_bb = 2*q_dist*f_into_wall + (1-2*q_dist)*f_at_xff

                store_f[config.use_float16c](f_in,f_bb,index,i)

            else: # Standard Mid Grid Bounceback
                f_bb = f_into_wall

            if compute_force:
                var link_force = float_directions[i]*(f_into_wall + f_bb)
                comptime for d in range(D):
                    force_tensor[tid,d] = link_force[d]

# last modified by: muse-spark-1.2 on 2026/09/01
