from layout import TileTensor,LayoutTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from src.lbm import LBM_Config,Lattice,GridLike,LBM_Grid,RuntimeParams
from src.lbm.constants import SOLID_NODE,FLUID_NODE,Flags,cs_squared,Collisions
from src.lbm.kernels.utils.index import get_adjacent_idx
from src.lbm.kernels.ops import wall_bc,equilibrium_bc
from src.utils import Vector,ContextTileTensor

@always_inline
def apply_boundary_conditions[
    FlayoutType:TensorLayout,
    BClayoutType:TensorLayout,
    FlaglayoutType:TensorLayout,
    //,
    grid: Some[GridLike],
    config:LBM_Config,
    ](
    mut f_vec:Vector[grid.float_dtype,grid.Q],
    f:TileTensor[config.set_f_dtype(grid.float_dtype),FlayoutType,_],
    bc:TileTensor[grid.float_dtype,BClayoutType,_],
    flags:TileTensor[DType.uint8,FlaglayoutType,_],
    pull_flags:InlineArray[UInt8,grid.Q],
    index:InlineArray[Int,3],
    tau:Scalar[grid.float_dtype],
    ):
    comptime float_dtype = grid.float_dtype
    comptime int_dtype = grid.int_dtype
    comptime lattice = grid.lattice
    comptime weights = lattice.weights
    comptime directions = lattice.directions
    comptime opposite_indices = lattice.opposite_indices
    comptime grid_shape:InlineArray[Int,3] = grid.shape
    
    # Bounce Back AND PULL FLAGS
    comptime if config.include_moving_boundary:
        wall_bc[directions,opposite_indices,weights,config.use_float16c](f_vec,pull_flags,bc,index,grid_shape)
    # Equilibrium BC
    comptime if Flags.EQUILIBRIUM in config.INCLUDED_BCs:
        equilibrium_bc[directions,weights,config.DDF_shift](f_vec,pull_flags,bc,index,grid_shape)
