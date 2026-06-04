from layout import Coord, coord, Idx, print_layout,TileTensor,LayoutTensor
from layout.layout import Layout as LegacyLayout
from layout.tile_layout import (
    Layout,
    blocked_product,
    col_major,
    row_major,
    zipped_divide,
)
comptime tile = col_major[2, 2]()
    # Define a 2x5 tiler
comptime tiler = row_major[2, 3]()
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