"""Computes drag force on immersed objects using momentum exchange.

Iterates over fluid boundary nodes adjacent to solid objects and
accumulates the momentum-exchange force contributions.
"""
from std.gpu import block_dim,block_idx,thread_idx,grid_dim
from max.gpu.sync import barrier
from layout import TileTensor,LayoutTensor
from std.utils.coord import dyn_coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout,col_major
from src.lbm.kernels.utils.index import get_adjacent_idx,is_index_valid
from src.utils import Vector
from src.lbm.kernels.utils.load_and_store import load_f,store_f
from src.lbm import LBM_Grid,LBM_Config,Lattice
from src.lbm import constants
from src.utils.runtimeLayouts import RuntimeColMajor1DType,RuntimeColMajor2DType

from src.lbm.kernels.steps import stream,collide,apply_boundary_conditions,store_f_vec_to_global,load_single_f
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
    comptime if FlagLayoutType.rank*2 == FlagLayoutType.flat_rank and (FlagLayoutType.rank == 3):
        comptime for i in range(3):
            var loc_x = Int(crd[2*i].value()) # local
            var til_x = Int(crd[(2*i)+1].value())
            index[i] = tile_shape[i]*til_x + loc_x
    else:
        comptime assert FlagLayoutType.rank == FlagLayoutType.flat_rank
        comptime for i in range(3):
            index[i] = Int(crd[i].value())
    return index^


def nodewise_bounceback_kernel[
    bounceback_method:Bounceback_method,
    FLayoutType:TensorLayout,
    FlagLayoutType:TensorLayout,
    BClayoutType:TensorLayout,
    grid: LBM_Grid,
    config:LBM_Config[_],
    *,
    is_even_time_step:Optional[Bool] = None,
    ](
        f_out:TileTensor[config.set_f_dtype(grid.float_dtype),FLayoutType,MutAnyOrigin],
        force_tensor:TileTensor[grid.float_dtype,RuntimeColMajor2DType,MutAnyOrigin],
        
        f_in:TileTensor[config.set_f_dtype(grid.float_dtype),FLayoutType,ImmutAnyOrigin],
        flags:TileTensor[DType.uint8,FlagLayoutType,ImmutAnyOrigin],
        bc:TileTensor[grid.float_dtype,BClayoutType,MutAnyOrigin if config.implies_bc_is_mutable() else ImmutAnyOrigin],
        tau:Scalar[grid.float_dtype],
        
        
        # fluid_boundary_ids:TileTensor[grid.int_dtype,RuntimeColMajor1DType,ImmutAnyOrigin],
        # CSR Inputs (uncompressed, we can also do this as fluid ids and compress lattiice links into a single uint32)
        fluid_id_row_offsets:TileTensor[grid.int_dtype,RuntimeColMajor1DType,ImmutAnyOrigin], # Row Offsets
        lattice_links:TileTensor[grid.int_dtype,RuntimeColMajor1DType,ImmutAnyOrigin], # Col Indices
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
        BClayoutType: The compile-time layout of the boundary-condition
            tensor.
        grid: The compile-time `LBM_Grid` describing the domain.
        config: The compile-time `LBM_Config` for the run.
        is_even_time_step: Whether this is an even time step.

    Args:
        f_out: The output distribution function tile tensor.
        force_tensor: The 2D output tile tensor of per-node force vectors.
        f_in: The input distribution function tile tensor.
        flags: The `uint8` tile tensor labeling each node.
        bc: The boundary-condition tile tensor.
        tau: The relaxation time.
        fluid_boundary_ids: The 1D tile tensor of linear fluid boundary
            indices.
        lattice_links: The 1D tile tensor of lattice link bitmasks.
        fluid_rowoffsets: The row offsets for the fluid boundary links.
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
    var fluid_id = block_dim.x * block_idx.x + thread_idx.x
            
    # var fluid_idx = fluid_boundary_ids[tid]
    
    var index = idx_to_ijk(fluid_id,flags,tile_shape)
    
    if index[0] < grid_shape[0] and index[1] < grid_shape[1] and index[2] < grid_shape[2]:
        
        var q_vec = Vector[float_dtype,Q](uninitialized=True)

        var row_start = fluid_id_row_offsets[fluid_id]
        var row_end = fluid_id_row_offsets[fluid_id+1]

        var coord_index = dyn_coord[DType.int32]((index[0],index[1],index[2]))
        var flag = flags.load(coord_index)[0]
        var f_vec = Vector[float_dtype,Q](fill = 0)
        var pull_flags = InlineArray[UInt8,Q](uninitialized=True)
        
        # We load all streamed values
        stream[grid,config](f_vec,pull_flags,f_in,flags,flag,index)
        
        var rho = get_density[config.DDF_shift](f_vec)
        var u = get_velocity(f_vec,rho, directions)

        var force_vec = Vector[float_dtype,D](fill = 0)
        var dm:Scalar[float_dtype] =0

        var moving_wall_term:Scalar[float_dtype] = 0.

        for i in range(row_start,row_end):
            var q = Int(lattice_links[i])
            var q_dist = link_distances[i]
            q_dist = max(q_dist,q_clamp)
            # Do bouceback here
            var opp_q = Int(opposite_index[q])
            comptime if bounceback_method == Bounceback_method.BOUZIDI:
                # We need the prestreamed values but we have the streamed values with midgrid Bounceback
                var f_into_wall = f_vec[opp_q] # This value has been bounced back
                var f_bb: Scalar[float_dtype]
                if q_dist > 0.5: # We need the f at the boundary leaving the wall and opposite direction i
                    var f_out_of_wall =  load_f[float_dtype,config.DDF_shift](f_in,index,opp_q)
                    # f_out_of_wall =  load_single_f[grid,config](f_in,index,opp_q)
                    f_bb = 0.5/q_dist*f_into_wall + (2*q_dist-1)/(2*q_dist)*f_out_of_wall
                else:
                    # This value has been streamed to the current node
                    var f_at_xff = f_vec[q]
                    f_bb = 2*q_dist*f_into_wall + (1-2*q_dist)*f_at_xff
                
                f_vec[opp_q] = f_bb + moving_wall_term

                var link_force = float_directions[q]*(f_into_wall + f_bb)
                force_vec += link_force
                dm += (f_bb - f_into_wall) # Measure the change in mass due to interpolation

        if compute_force:
            comptime for d in range(D):
                force_tensor[fluid_id,d] = force_vec[d]
        # Conserve Mass
        f_vec[0] += dm

        # comptime assert 
        apply_boundary_conditions[grid,config,exclude_moving_wall = True](f_vec,f_in,bc,flags,pull_flags,index,tau)
        collide[grid,config](f_vec,f_in,bc,flags,pull_flags,index,tau)
    
        # Save here
        store_f_vec_to_global[grid,config,is_even_time_step = is_even_time_step](f_out,f_vec,index)

# last modified by: muse-spark-1.2 on 2026/09/01
