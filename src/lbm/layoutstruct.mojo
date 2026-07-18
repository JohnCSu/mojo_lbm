from layout import TileTensor,coord
from layout.tile_layout import Layout,row_major,TensorLayout,blocked_product,col_major
from src.lbm import LBM_Grid

struct TiledGridLayouts[grid:LBM_Grid]:  
    comptime D:Int = Self.grid.D
    comptime Q:Int = Self.grid.Q

    comptime tile_size = Self.grid.tile_size
    comptime x_tile = Self.tile_size
    comptime y_tile = Self.tile_size if Self.D >= 2 else 1
    comptime z_tile = Self.tile_size if Self.D >= 3 else 1

    comptime _flag_tile = col_major[Self.x_tile,Self.y_tile,Self.z_tile]()
    comptime _f_tile = col_major[Self.x_tile,Self.y_tile,Self.z_tile,1,Self.Q]()
    comptime _bc_tile = col_major[Self.x_tile,Self.y_tile,Self.z_tile,1,Self.D+1]()

    comptime _flag_tiler = col_major[Self.grid.n_tiles_x,Self.grid.n_tiles_y,Self.grid.n_tiles_z]()
    comptime _f_tiler = col_major[Self.grid.n_tiles_x,Self.grid.n_tiles_y,Self.grid.n_tiles_z,1]()
    comptime _bc_tiler = col_major[Self.grid.n_tiles_x,Self.grid.n_tiles_y,Self.grid.n_tiles_z,1]()


    comptime flag_layout = blocked_product(Self._flag_tile,Self._flag_tiler)
    comptime f_layout = blocked_product(Self._f_tile,Self._f_tiler)
    comptime bc_layout = blocked_product(Self._bc_tile,Self._bc_tiler)