from layout import Coord, coord, Idx, print_layout,TileTensor,LayoutTensor
from layout.layout import Layout as LegacyLayout
from layout.layout import blocked_product as legacy_blocked_product
from layout.tile_layout import (
    Layout,
    blocked_product,
    col_major,
    row_major,
    zipped_divide,
)
comptime tile = col_major[4,4]()
    # Define a 2x5 tiler
comptime tiler = row_major[2, 2]()
comptime blocked = blocked_product(tile, tiler)
comptime legacyblocked = blocked.to_layout()

def main() raises:
    print("blocked product")
    # Define 3x2 tile   
    print("Tile:")
    print_layout(tile.to_layout())
    print("\nTiler:")
    print_layout(tiler.to_layout())
    print("\nTiled layout:")
    print_layout(blocked.to_layout())
    print()
    var storage = InlineArray[Float32, blocked.size()](uninitialized=True)
    for i in range((blocked.size())):
        storage[i] = Float32(i)

    tensor = TileTensor(storage,blocked)
    
    leg_tensor = LayoutTensor[DType.float32,legacyblocked](tensor.ptr)


    comptime for i in range(tensor.rank):
        print(tensor.static_shape[1+i*2])
    # x = tensor.to_layout_tensor()
    # x = tensor.to_layout_tensor()
    print(leg_tensor[0,0])
    print(leg_tensor[0,1])
    print(leg_tensor[1,0])
    print(leg_tensor[1,1])

    print(tensor[0,0,0,0])
    print(tensor[0,0,1,0])
    print(tensor[1,0,0,0])
    print(tensor[1,0,1,0])

    comptime nx,ny,nz = (2048,2048,1)
    comptime D,Q = (2,9)
    comptime tile_size = 16

    comptime assert (nx % tile_size) == 0 ,'Grid must be a multiple of tilesize'
    comptime assert nx == ny and nz == 1,'Benchmark is for a 2D square grid'
    comptime n_tiles = nx//tile_size
    
    comptime flag_tile = col_major[tile_size,tile_size,1]()
    
    
    
    leg_f_tile = LegacyLayout.row_major(tile_size,tile_size,1,Q)
    leg_f_tiler = LegacyLayout.row_major(n_tiles,n_tiles,1,Q)

    leg_f_layout = legacy_blocked_product(leg_f_tile^,leg_f_tiler^)    

    comptime f_tile_AoS = col_major[tile_size,tile_size,1,Q]()
    # comptime f_tile2_AoS = row_major[tile_size,tile_size,1,Q]()
    comptime f_tiler_AoS = row_major[n_tiles,n_tiles,1,1]()

    comptime f_tile_SoA = col_major[1,tile_size,tile_size,1]()
    comptime f_tiler_SoA = row_major[Q,n_tiles,n_tiles,1]()

    comptime row_major_f = row_major[Q,nx,ny,1]()

    comptime bc_tile = col_major[tile_size,tile_size,1,D+1]()
        
    comptime flag_tiler = row_major[n_tiles,n_tiles,1]()
    
    comptime bc_tiler = row_major[n_tiles,n_tiles,1,D+1]()

    comptime flag_layout = blocked_product(flag_tile,flag_tiler)


    comptime f_layout_SoA = blocked_product(f_tile_SoA,f_tiler_SoA)
    comptime f_layout_AoS = blocked_product(f_tile_AoS,f_tiler_AoS)

    print('Row Major: ',row_major_f.size(), 'SoA Size: ', f_layout_SoA.size(),'AoS Size: ',f_layout_AoS.size() ) #, ' AoS Legacy: ', leg_f_layout.size())
    
    comptime bc_layout = blocked_product(bc_tile,bc_tiler)

    comptime density_layout = row_major[nx,ny,nz]()
    comptime velocity_layout = row_major[D,nx,ny,nz]()

    print("Row major strides:", row_major_f.shape_coord())
    print("SoA strides:", f_layout_SoA.shape_coord()) 
    print("AoS strides:", f_layout_AoS.shape_coord())
    # print("AoS size (bytes):", )
    