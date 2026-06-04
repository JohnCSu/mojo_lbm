from layout import Coord, coord, Idx, print_layout,TileTensor,LayoutTensor
from layout.layout import (
    Layout,
    blocked_product,
    zipped_divide,
    
)
# comptime tile = col_major[2, 2]()
#     # Define a 2x5 tiler
# comptime tiler = row_major[2, 3]()
# comptime blocked = blocked_product(tile, tiler)
comptime tile = Layout.row_major(2, 2)
# Define a 2x5 tiler
comptime tiler = Layout.row_major(2, 3)
comptime blocked = blocked_product(tile.copy(), tiler.copy())
def main() raises:
    # Define 2x3 tile
    # print("Tile:")
    # print_layout(tile)
    # print("\nTiler:")
    # print_layout(tiler)
    # print("\nTiled layout:")
    # print(blocked)
    var storage = InlineArray[Float32, blocked.size()](uninitialized=True)
    for i in range(comptime (blocked.size())):
        storage[i] = Float32(i)

    tensor = LayoutTensor[DType.float32,blocked](storage)

    # x = tensor.to_layout_tensor()
    print(tensor[0,0])
    print(tensor[0,1])
    print(tensor[1,0])
    print(tensor[1,1])

    print(tensor[0,0,0,0])
    print(tensor[0,0,1,0])
    print(tensor[1,0,0,0])
    print(tensor[1,0,1,0])
    # print(tensor[1, 0, 0,0])