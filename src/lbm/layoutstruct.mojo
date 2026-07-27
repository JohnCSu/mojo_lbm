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

struct TiledLayouts[
    D: Int,
    Q: Int,
    grid_shape: InlineArray[Int, 3],
    tile_shape: Tuple[Int,Int,Int]
    ](ImplicitlyCopyable):
    """Computes tiled tensor layouts for the flag, `f`, and `bc` fields.

    Each layout is the `blocked_product` of a column-major tile and a
    column-major tiler, sized so the tiler dimensions track the grid's tile
    counts and the tile dimensions track the per-tile element counts.

    Parameters:
        grid: The compile-time `LBM_Grid` the layouts are derived from.
    """

    comptime x_tile = Self.tile_shape[0]
    comptime y_tile = Self.tile_shape[1]
    comptime z_tile = Self.tile_shape[2]

    comptime n_tiles_x = Self.grid_shape[0] // Self.tile_shape[0]
    comptime n_tiles_y = Self.grid_shape[1] // Self.tile_shape[1] if Self.D >= 2 else 1
    comptime n_tiles_z = Self.grid_shape[2] // Self.tile_shape[2] if Self.D == 3 else 1

    comptime _flag_tile = col_major[Self.x_tile, Self.y_tile, Self.z_tile]()
    comptime _f_tile = col_major[
        Self.x_tile, Self.y_tile, Self.z_tile, Self.Q
    ]()
    comptime _bc_tile = col_major[
        Self.x_tile, Self.y_tile, Self.z_tile, Self.D + 1
    ]()

    comptime _rank_3_tiler = col_major[
        Self.n_tiles_x, Self.n_tiles_y, Self.n_tiles_z
    ]()
    comptime _rank_4_tiler = col_major[
        Self.n_tiles_x, Self.n_tiles_y, Self.n_tiles_z, 1
    ]()

    comptime flag_layout = blocked_product(Self._flag_tile, Self._rank_3_tiler)
    """The tiled layout for the flag field."""
    comptime f_layout = Self.create_tiled_rank_4_layout[Self.Q]()
    """The tiled layout for the distribution function field."""
    comptime bc_layout = Self.create_tiled_rank_4_layout[Self.D+1]()
    """The tiled layout for the boundary-condition field."""

    comptime untiled_flag_layout = col_major[Self.grid_shape[0],Self.grid_shape[1],Self.grid_shape[2]]()
    comptime untiled_f_layout = Self.create_col_major_rank_4_layout[Self.Q]()
    comptime untiled_bc_layout = Self.create_col_major_rank_4_layout[Self.D+1]()

    comptime density_layout = row_major[Self.grid_shape[0],Self.grid_shape[1],Self.grid_shape[2]]()
    comptime velocity_layout = row_major[Self.grid_shape[0],Self.grid_shape[1],Self.grid_shape[2],Self.D]()
    
    # def __init__(out self):
    #     pass

    @staticmethod
    def create_tiled_rank_4_layout[
        last_dim_size:Int
        ]()
        -> type_of(blocked_product(col_major[Self.x_tile, Self.y_tile, Self.z_tile,last_dim_size](),Self._rank_4_tiler )):
        return blocked_product(col_major[Self.x_tile, Self.y_tile, Self.z_tile,last_dim_size](),Self._rank_4_tiler )

    @staticmethod
    def create_col_major_rank_4_layout[
        last_dim_size:Int
        ]
        () 
        -> type_of(col_major[Self.grid_shape[0],Self.grid_shape[1],Self.grid_shape[2],last_dim_size]()):
        return col_major[Self.grid_shape[0],Self.grid_shape[1],Self.grid_shape[2],last_dim_size]()

    @staticmethod
    def create_col_major_rank_3_layout() 
        -> type_of(col_major[Self.grid_shape[0],Self.grid_shape[1],Self.grid_shape[2]]()):
        return col_major[Self.grid_shape[0],Self.grid_shape[1],Self.grid_shape[2]]()

    @staticmethod
    def create_row_major_rank_3_layout() 
        -> type_of(row_major[Self.grid_shape[0],Self.grid_shape[1],Self.grid_shape[2]]()):
        return row_major[Self.grid_shape[0],Self.grid_shape[1],Self.grid_shape[2]]()

    @staticmethod
    def create_row_major_rank_4_layout[
        last_dim_size:Int
        ]
        () 
        -> type_of(row_major[Self.grid_shape[0],Self.grid_shape[1],Self.grid_shape[2],last_dim_size]()):
        return row_major[Self.grid_shape[0],Self.grid_shape[1],Self.grid_shape[2],last_dim_size]()
