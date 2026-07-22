"""Defines `TiledGridLayouts`, which derives tiled buffer layouts from an `LBM_Grid`.

Computes the flag, distribution-function, and boundary-condition tile and
tiler layouts for a grid by composing column-major tiles with a column-major
tiler via `blocked_product`.
"""
from layout import TileTensor, coord
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

    comptime tile_size = Self.grid.tile_size
    comptime x_tile = Self.tile_size
    comptime y_tile = Self.tile_size if Self.D >= 2 else 1
    comptime z_tile = Self.tile_size if Self.D >= 3 else 1

    comptime _flag_tile = col_major[Self.x_tile, Self.y_tile, Self.z_tile]()
    comptime _f_tile = col_major[
        Self.x_tile, Self.y_tile, Self.z_tile, 1, Self.Q
    ]()
    comptime _bc_tile = col_major[
        Self.x_tile, Self.y_tile, Self.z_tile, 1, Self.D + 1
    ]()

    comptime _flag_tiler = col_major[
        Self.grid.n_tiles_x, Self.grid.n_tiles_y, Self.grid.n_tiles_z
    ]()
    comptime _f_tiler = col_major[
        Self.grid.n_tiles_x, Self.grid.n_tiles_y, Self.grid.n_tiles_z, 1
    ]()
    comptime _bc_tiler = col_major[
        Self.grid.n_tiles_x, Self.grid.n_tiles_y, Self.grid.n_tiles_z, 1
    ]()

    comptime flag_layout = blocked_product(Self._flag_tile, Self._flag_tiler)
    """The tiled layout for the flag field."""
    comptime f_layout = blocked_product(Self._f_tile, Self._f_tiler)
    """The tiled layout for the distribution function field."""
    comptime bc_layout = blocked_product(Self._bc_tile, Self._bc_tiler)
    """The tiled layout for the boundary-condition field."""

    def __init__(out self):
        pass