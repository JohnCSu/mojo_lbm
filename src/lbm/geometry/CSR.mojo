from max.gpu.host import DeviceContext,DeviceBuffer
from layout import TileTensor,LayoutTensor,row_major
from layout.tile_layout import TensorLayout
from std.utils.coord import dyn_coord
from layout import Idx, All

trait CSRLike(Copyable):
    comptime int_dtype: DType
    comptime float_dtype:DType
    comptime ValueArrayType:Copyable & Sized 

    def rows[origin: Origin, //](ref[origin] self) -> Span[Scalar[Self.int_dtype], origin]:
        ...

    def cols[origin: Origin, //](ref[origin] self) -> Span[Scalar[Self.int_dtype], origin]:
        ...

    def row_offsets[origin: Origin, //](ref[origin] self) -> Span[Scalar[Self.int_dtype], origin]:
        ...

    def values[origin:Origin,//](ref[origin] self) -> ref[origin] Dict[String, Self.ValueArrayType]:
        ...

    def nnz(self) -> Int:
        return len(self.rows())

    def nnz_rows(self) -> Int:
        return len(self.row_offsets()) - 1


    def keys(self) -> List[String]:
        ...

    def shape(self) -> Tuple[Int,Int]:
        ...


def cmp[int_dtype: DType](a: Tuple[Scalar[int_dtype], Scalar[int_dtype], Int], b: Tuple[Scalar[int_dtype], Scalar[int_dtype], Int]) capturing -> Bool:
    if a[0] != b[0]:
        return a[0] < b[0]
    return a[1] < b[1]


def is_non_negative[origin:Origin,int_dtype:DType](arr: Span[Scalar[int_dtype], origin]) raises:
    for x in arr:
        if x < 0:
            raise Error('Negative integers are not allowed')
    
def error_check[int_dtype:DType,row_origin:Origin,col_origin:Origin](rows: Span[Scalar[int_dtype], row_origin],cols: Span[Scalar[int_dtype], col_origin]) raises:
    if len(rows) != len(cols):
        raise Error('rows and cols must be the same length')
    is_non_negative(rows)
    is_non_negative(cols)


struct CSR[int_dtype_: DType, float_dtype_:DType](Movable, CSRLike):
    comptime int_dtype = Self.int_dtype_
    comptime float_dtype = Self.float_dtype_
    comptime ValueArrayType = List[Scalar[Self.float_dtype]]
    var _rows: List[Scalar[Self.int_dtype]]
    var _cols: List[Scalar[Self.int_dtype]]
    var _argsort: List[Int]
    var _row_offsets: List[Scalar[Self.int_dtype]]
    var _values: Dict[String, Self.ValueArrayType ]
    var _unique_rows: List[Scalar[Self.int_dtype]]
    var _shape: Tuple[Int, Int]

    def __init__[row_origin: Origin, col_origin: Origin, //](
        out self,
        shape: Tuple[Int, Int],
        rows: Span[Scalar[Self.int_dtype], row_origin],
        cols: Span[Scalar[Self.int_dtype], col_origin],
    ) raises:
        self._shape = shape

        var indices = [i for i in range(len(rows))]
        var tuple_list = [x for x in zip(rows, cols, indices)]
        
        error_check(rows,cols)
        sort[cmp[Self.int_dtype]](Span[mut=True](tuple_list))

        self._values = {}
        self._rows = [x[0] for x in tuple_list]
        self._cols = [x[1] for x in tuple_list]
        self._argsort = [x[2] for x in tuple_list]
        self._row_offsets = [0]
        self._unique_rows = []

        if len(tuple_list) > 0:
            var current_row = tuple_list[0][0]
            self._unique_rows.append(current_row)
            for i, (row, _, _) in enumerate(tuple_list[1:]):
                if row != current_row:
                    self._row_offsets.append(Scalar[Self.int_dtype](i + 1))
                    current_row = row
                    self._unique_rows.append(current_row)
            self._row_offsets.append(Scalar[Self.int_dtype](len(tuple_list)))

    def rows[origin: Origin, //](ref[origin] self) -> Span[Scalar[Self.int_dtype], origin]:
        return rebind[Span[Scalar[Self.int_dtype], origin]](Span(self._rows))

    def unique_rows[origin: Origin, //](ref[origin] self) -> Span[Scalar[Self.int_dtype], origin]:
        return rebind[Span[Scalar[Self.int_dtype], origin]](Span(self._unique_rows))

    def cols[origin: Origin, //](ref[origin] self) -> Span[Scalar[Self.int_dtype], origin]:
        return rebind[Span[Scalar[Self.int_dtype], origin]](Span(self._cols))

    def row_offsets[origin: Origin, //](ref[origin] self) -> Span[Scalar[Self.int_dtype], origin]:
        return rebind[Span[Scalar[Self.int_dtype], origin]](Span(self._row_offsets))

    def values[origin: Origin, //](ref[origin] self) -> ref[origin] Dict[String, Self.ValueArrayType]:
        return rebind[Pointer[Dict[String, Self.ValueArrayType], origin]](
            Pointer(to=self._values)
        )[]

    def add_value(mut self, key: String, value: Span[Scalar[Self.float_dtype], ...], *, sort: Bool = True) raises:
        if len(value) != self.nnz():
            raise Error('Value spans lengths must match number of nnz (i.e number of column indices) ')
        
        if sort:
            self.values()[key] = [value[i] for i in   self._argsort]
        else:
            self.values()[key] = [value[i] for i in range(len(value))]

    def get_value(self, key: String) raises -> Span[Scalar[Self.float_dtype], origin_of(self.values()[key])]:
        return Span(self.values()[key])

    def keys(self) -> List[String]:
        return [key for key in self.values().keys()]

    def shape(self) -> Tuple[Int,Int]:
        return self._shape

    def to_context(self,deviceContext:DeviceContext) raises -> Context_CSR[Self.int_dtype,Self.float_dtype]:
        # var x = Context_CSR(deviceContext,self)
        return Context_CSR(deviceContext,self)


from src.utils import ContextTileTensor,RuntimeColMajor1DType,col_major1D,RuntimeColMajor2DType,col_major2D
from std.memory import ArcPointer


struct Context_CSR[int_dtype:DType,float_dtype:DType](Copyable):
    # comptime int_dtype = Self.csrType.int_dtype
    # comptime float_dtype = Self.csrType.float_dtype
    comptime ValueArrayType = ContextTileTensor[Self.float_dtype,RuntimeColMajor1DType]
    var row_offsets:ContextTileTensor[Self.int_dtype,RuntimeColMajor1DType]
    var rows:ContextTileTensor[Self.int_dtype,RuntimeColMajor1DType]
    var cols:ContextTileTensor[Self.int_dtype,RuntimeColMajor1DType]
    var unique_rows:ContextTileTensor[Self.int_dtype,RuntimeColMajor1DType]
    var values:Dict[String,ArcPointer[ContextTileTensor[Self.float_dtype,RuntimeColMajor1DType]]]

    var _shape:Tuple[Int,Int]
    var deviceContext:DeviceContext
    
    def __init__(out self,deviceContext:DeviceContext,csr:CSR[Self.int_dtype,Self.float_dtype]) raises:
        self.deviceContext = deviceContext

        self.row_offsets = self.span_to_CTT[Self.int_dtype](deviceContext,csr.row_offsets())
        self.cols = self.span_to_CTT[Self.int_dtype](deviceContext,csr.cols())
        self.rows = self.span_to_CTT[Self.int_dtype](deviceContext,csr.rows())
        self.unique_rows = self.span_to_CTT[Self.int_dtype](deviceContext,csr.unique_rows())

        var total_N = len(csr.values())*csr.nnz(
        )
        if total_N <= 0:
            raise Error('There must be atleast one values array in the passed in csr')

        self.values = {}
        self._shape = csr.shape()
        var value_keys = csr.values().keys()
        
        for key in value_keys:
            var value_span = Span(csr.values()[key])
            self.add_value(key,rebind[Span[Scalar[Self.float_dtype],value_span.origin]](value_span))


    @staticmethod
    def span_to_CTT[src_dtype:DType,//,dtype:DType](deviceContext:DeviceContext,src:Span[Scalar[src_dtype],...]) raises -> ContextTileTensor[dtype,RuntimeColMajor1DType]:
        var layout = col_major1D(len(src))
        var s =  ContextTileTensor[dtype](deviceContext,layout)
        s.fill(rebind[Span[Scalar[dtype],src.origin]](src))
        return s^

    def get_value(ref self,key:String) raises -> ArcPointer[ContextTileTensor[Self.float_dtype,RuntimeColMajor1DType]]:
        return self.values[key]

    def value_layout(self) -> RuntimeColMajor1DType:
        return col_major1D(self.nnz())

    def add_value[origin:Origin,//](mut self,key:String,src:Span[Scalar[Self.float_dtype],origin]) raises:
        if key in self.values:
            raise Error('Key already Exists')

        self.values[key] = ArcPointer(ContextTileTensor[Self.float_dtype](self.deviceContext,self.value_layout()))
        self.values[key][].fill(src)

    def shape(self) -> Tuple[Int,Int]:
        return self._shape

    def n_rows(self) -> Int:
        return self.unique_rows.size()

    def n_cols(self) -> Int:
        return self.shape()[1]

    def nnz(self) -> Int:
        return len(self.cols)
    
    def create_2D_value(self,rows:Int) raises -> ContextTileTensor[Self.float_dtype,RuntimeColMajor2DType]:
        return ContextTileTensor[Self.float_dtype](self.deviceContext,col_major2D(self.nnz(),rows),fill = Scalar[Self.float_dtype](0.))
    

    def create_row_nd_tensor(self,d:Int) raises -> ContextTileTensor[Self.float_dtype,RuntimeColMajor2DType]:
        return ContextTileTensor[Self.float_dtype](self.deviceContext,col_major2D(self.n_rows(),d),fill = Scalar[Self.float_dtype](0.))
        

# struct CSR_GPU[float_dtype: DType](Movable):
#     comptime int_dtype: DType = DType.int32
#     # comptime OffsetLayout = type_of(row_major(dyn_coord[Self.int_dtype]((1,))))
#     comptime RowMajor1DType = type_of(row_major(dyn_coord[Self.int_dtype]((1,))))
#     # comptime RowMajor1DType = type_of(row_major(dyn_coord[Self.int_dtype]((1,))))
#     var row_offsets_buffer:DeviceBuffer[Self.int_dtype]
#     var cols_buffer:DeviceBuffer[Self.int_dtype]
#     var _values_offsets: Dict[String, Int] 
#     var _shape:Tuple[Int,Int]

#     var deviceContext:DeviceContext
#     var _all_values_buffer:DeviceBuffer[Self.float_dtype]
#     def __init__(out self,deviceContext:DeviceContext,csr:Some[CSRLike]) raises:
#         self.deviceContext = deviceContext
#         self.row_offsets_buffer = rebind[DeviceBuffer[Self.int_dtype]](
#             self.create_buffer_and_copy(deviceContext,csr.row_offsets())
#         )
#         self.cols_buffer = rebind[DeviceBuffer[Self.int_dtype]](
#             self.create_buffer_and_copy(deviceContext,csr.cols())
#         )
#         var total_N = len(csr.values())*csr.nnz(

#         )

#         if total_N <= 0:
#             raise Error('There must be atleast one values array in the passed in csr')

#         self._all_values_buffer = deviceContext.enqueue_create_buffer[Self.float_dtype](total_N)
#         self._values_offsets = {} 
#         self._shape = csr.shape()

#         self._copy_values_to_buffer(csr)


 


#     @staticmethod
#     def create_buffer_and_copy[origin:Origin,//,dtype:DType](deviceContext:DeviceContext,src:Span[Scalar[dtype],origin]) raises -> DeviceBuffer[dtype]:
#         var buffer = deviceContext.enqueue_create_buffer[dtype](len(src))
#         deviceContext.enqueue_copy(buffer,src)
#         return buffer^

#     def _copy_values_to_buffer(mut self,csr:Some[CSRLike]) raises :
#         ref values = csr.values()
#         var offset = 0
#         for key in values.keys():
#             self._values_offsets[key] = offset
#             var sub_buffer = self._all_values_buffer.create_sub_buffer[Self.float_dtype](offset,self.nnz()) 
#             var value_span = Span(unsafe_ptr =  Pointer(to = values[key]),length = len(values[key]))
#             self.deviceContext.enqueue_copy(sub_buffer,rebind[Span[Scalar[Self.float_dtype],value_span.origin]](value_span))
#             offset += self.nnz()


#     def get_value_buffer(mut self,key:String) raises ->  DeviceBuffer[Self.float_dtype]:
#         var offset = self._values_offsets[key]
#         return self._all_values_buffer.create_sub_buffer[Self.float_dtype](offset,self.nnz())

#     def value_layout(self) -> type_of(row_major(dyn_coord[Self.int_dtype]((self.nnz(),) ))):
#         return row_major( dyn_coord[Self.int_dtype]((self.nnz(),) ) )

#     def row_offsets(self) raises -> TileTensor[Self.int_dtype, Self.RowMajor1DType, origin_of(self.row_offsets_buffer)]:
#         return TileTensor(
#             self.row_offsets_buffer,
#             row_major(dyn_coord[Self.int_dtype]((len(self.row_offsets_buffer),))),
#         )

#     def cols(self) raises -> TileTensor[Self.int_dtype, Self.RowMajor1DType, origin_of(self.cols_buffer)]:
#         return TileTensor(
#             self.cols_buffer,
#             row_major(dyn_coord[Self.int_dtype]((len(self.cols_buffer),))),
#         )


#     def shape(self) -> Tuple[Int,Int]:
#         return self._shape

#     def nnz(self) -> Int:
#         return len(self.cols_buffer)