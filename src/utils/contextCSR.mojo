from .contextTileTensor import ContextTileTensor
from std.gpu.host import DeviceContext
from std.gpu import HostBuffer,DeviceBuffer
from layout import TileTensor,row_major,col_major,coord,Coord
from layout.tile_layout import Layout
from std.builtin.variadics import TypeList
struct ContextCSR[int_dtype:DType = DType.int32]():
    # comptime dim = len(Self.ints)
    # var rows
    # var 

    # comptime 
    comptime dim = 3
    comptime RowMajor1D_Type = type_of( row_major(coord[DType.int32]((3,)) ) ) # the 3 is a dummy variable to capture the runtime type
    var deviceContext:DeviceContext
    var shape: Tuple[Int,Int]
    # var rows:Int
    # var cols:Int
    var row_offsets:ContextTileTensor[Self.int_dtype,Self.RowMajor1D_Type]
    var col_indices:ContextTileTensor[Self.int_dtype,Self.RowMajor1D_Type]
    var nnz:Int
    def __init__(out self,ctx:DeviceContext,shape:Tuple[Int,Int],mut indices:List[Tuple[Int,Int]] ) raises : 
        self.deviceContext = ctx

        self.shape = shape
        n_rows,n_cols = shape
        self.nnz = len(indices)

        self.row_offsets = ContextTileTensor[Self.int_dtype](ctx, layout = row_major(coord[DType.int32]((n_rows + 1,)) ))
        self.col_indices = ContextTileTensor[Self.int_dtype](ctx, layout = row_major(coord[DType.int32]((self.nnz,)) ))
    
        # Sort the rows    
        def cmp(src:Tuple[Int,Int],other:Tuple[Int,Int]) capturing -> Bool:
            return src[0] < src[0] if src[0] != other[0] else src[1] < src[1]

        sort[cmp_fn = cmp,stable=True](Span(indices))

        self.row_offsets.cpu()[0] = 0

        current_row = 0
        offset_count = 0
        row_inc = 1
        for i,(row,col) in enumerate(indices):
            self.col_indices.cpu()[i] = Scalar[Self.int_dtype](col)
            offset_count += 1
            if row != current_row:
                self.row_offsets.cpu()[row_inc] = Scalar[Self.int_dtype](offset_count)
                row_inc += 1
                offset_count = 0 
             

    @staticmethod
    def from_3d_indices[*T:ImplicitlyCopyable & Intable](ctx:DeviceContext,shape:Tuple[Int,Int,Int],indices: List[ Tuple[Int,Int,Int]],tile_size:Int) raises -> Self:
        '''
        Convert 3D indices to 1D flattened index. Asssume Row_major for now
        '''
        # We need to convert the 3D indices into a 1D format
        
        if not (1 <= len(shape) <= 3):
            raise Error()

        # comptime assert T = Tuple[]

        grid_dim = 3
        num_rows = tile_size**3
        indices_1D = [x + y*shape[0] + z*(shape[0]*shape[1]) for x,y,z in indices]
        return Self()


        




        



