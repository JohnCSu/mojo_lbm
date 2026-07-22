"""Defines `TiledGridLayouts`, which derives tiled buffer layouts from an `LBM_Grid`.

Computes the flag, distribution-function, and boundary-condition tile and
tiler layouts for a grid by composing column-major tiles with a column-major
tiler via `blocked_product`.
"""
# from std.utils.coord import Coordlike
from layout import TileTensor, coord,CoordLike,ComptimeInt
from layout.tile_layout import (
    Layout,
    row_major,
    TensorLayout,
    blocked_product,
    col_major,
    
)
from src.lbm import LBM_Grid

struct TiledGridLayouts[grid: LBM_Grid]:
    """Computes tiled tensor layouts for the flag, `f`, and `bc` fields.

    Each layout is the `blocked_product` of a column-major tile and a
    column-major tiler, sized so the tiler dimensions track the grid's tile
    counts and the tile dimensions track the per-tile element counts.

    Parameters:
        grid: The compile-time `LBM_Grid` the layouts are derived from.
    """

    comptime D: Int = Self.grid.D
    comptime Q: Int = Self.grid.Q
    comptime grid_shape = Self.grid.shape

    comptime x_tile = Self.grid.x_tile
    comptime y_tile = Self.grid.y_tile
    comptime z_tile = Self.grid.z_tile

    comptime _flag_tile = col_major[Self.x_tile, Self.y_tile, Self.z_tile]()
    comptime _f_tile = col_major[
        Self.x_tile, Self.y_tile, Self.z_tile, Self.Q
    ]()
    comptime _bc_tile = col_major[
        Self.x_tile, Self.y_tile, Self.z_tile, Self.D + 1
    ]()

    comptime _rank_3_tiler = col_major[
        Self.grid.n_tiles_x, Self.grid.n_tiles_y, Self.grid.n_tiles_z
    ]()
    comptime _rank_4_tiler = col_major[
        Self.grid.n_tiles_x, Self.grid.n_tiles_y, Self.grid.n_tiles_z, 1
    ]()

    comptime flag_layout = blocked_product(Self._flag_tile, Self._rank_3_tiler)
    """The tiled layout for the flag field."""
    comptime f_layout = Self.create_tiled_rank_4_tensor[Self.Q]()
    """The tiled layout for the distribution function field."""
    comptime bc_layout = Self.create_tiled_rank_4_tensor[Self.D+1]()
    """The tiled layout for the boundary-condition field."""

    comptime untiled_flag_layout = col_major[Self.grid_shape[0],Self.grid_shape[1],Self.grid_shape[2]]
    comptime untiled_f_layout = Self.create_untiled_rank_4_tensor[Self.Q]()
    comptime untiled_bc_layout = Self.create_untiled_rank_4_tensor[Self.D+1]()

    def __init__(out self):
        pass

    @staticmethod
    def create_tiled_rank_4_tensor[
        last_dim_size:Int
        ]()
        -> type_of(blocked_product(col_major[Self.x_tile, Self.y_tile, Self.z_tile,last_dim_size](),Self._rank_4_tiler )):
        return blocked_product(col_major[Self.x_tile, Self.y_tile, Self.z_tile,last_dim_size](),Self._rank_4_tiler )

    @staticmethod
    def create_untiled_rank_4_tensor[
        last_dim_size:Int
        ]
        () 
        -> type_of(col_major[Self.grid_shape[0],Self.grid_shape[1],Self.grid_shape[2],last_dim_size]):
        return col_major[Self.grid_shape[0],Self.grid_shape[1],Self.grid_shape[2],last_dim_size]
