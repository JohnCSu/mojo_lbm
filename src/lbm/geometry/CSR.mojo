from max.gpu.host import DeviceContext,DeviceBuffer
from layout import TileTensor,LayoutTensor,row_major
from layout.tile_layout import TensorLayout
from std.utils.coord import dyn_coord


trait CSRLike(Movable):
    comptime int_dtype: DType
    comptime float_dtype:DType

    def rows[origin: Origin, //](ref[origin] self) -> Span[Scalar[Self.int_dtype], origin]:
        ...

    def cols[origin: Origin, //](ref[origin] self) -> Span[Scalar[Self.int_dtype], origin]:
        ...

    def row_offsets[origin: Origin, //](ref[origin] self) -> Span[Scalar[Self.int_dtype], origin]:
        ...


    def values[origin:Origin,//](ref[origin] self) -> ref[origin] Dict[String, List[Scalar[Self.float_dtype]]]:
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
    var _rows: List[Scalar[Self.int_dtype]]
    var _cols: List[Scalar[Self.int_dtype]]
    var _argsort: List[Int]
    var _row_offsets: List[Scalar[Self.int_dtype]]
    var _values: Dict[String, List[Scalar[Self.float_dtype]]]
    
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

        if len(tuple_list) > 0:
            var current_row = tuple_list[0][0]
            for i, (row, _, _) in enumerate(tuple_list[1:]):
                if row != current_row:
                    self._row_offsets.append(Scalar[Self.int_dtype](i + 1))
                    current_row = row
            self._row_offsets.append(Scalar[Self.int_dtype](len(tuple_list)))

    def rows[origin: Origin, //](ref[origin] self) -> Span[Scalar[Self.int_dtype], origin]:
        return rebind[Span[Scalar[Self.int_dtype], origin]](Span(self._rows))

    def cols[origin: Origin, //](ref[origin] self) -> Span[Scalar[Self.int_dtype], origin]:
        return rebind[Span[Scalar[Self.int_dtype], origin]](Span(self._cols))

    def row_offsets[origin: Origin, //](ref[origin] self) -> Span[Scalar[Self.int_dtype], origin]:
        return rebind[Span[Scalar[Self.int_dtype], origin]](Span(self._row_offsets))

    def values[origin: Origin, //](ref[origin] self) -> ref[origin] Dict[String, List[Scalar[Self.float_dtype]]]:
        return rebind[Pointer[Dict[String, List[Scalar[Self.float_dtype]]], origin]](
            Pointer(to=self._values)
        )[]

    def add_value(mut self, key: String, value: Span[Scalar[Self.float_dtype], ...], *, sort: Bool = True) raises:
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

    def to_gpu(self,deviceContext:DeviceContext) raises -> CSR_GPU[Self.float_dtype]:
        return CSR_GPU[Self.float_dtype](deviceContext,self)


struct CSR_GPU[float_dtype: DType](Movable):
    comptime int_dtype: DType = DType.int32
    # comptime OffsetLayout = type_of(row_major(dyn_coord[Self.int_dtype]((1,))))
    comptime RowMajor1DType = type_of(row_major(dyn_coord[Self.int_dtype]((1,))))

    var row_offsets_buffer:DeviceBuffer[Self.int_dtype]
    var cols_buffer:DeviceBuffer[Self.int_dtype]
    var values_buffer: Dict[String, DeviceBuffer[Self.float_dtype]]
    var _shape:Tuple[Int,Int]
    var deviceContext:DeviceContext

    def __init__(out self,deviceContext:DeviceContext,csr:Some[CSRLike]) raises:
        self.deviceContext = deviceContext
        self.row_offsets_buffer = rebind[DeviceBuffer[Self.int_dtype]](
            self.create_buffer_and_copy(deviceContext,csr.row_offsets())
        )
        self.cols_buffer = rebind[DeviceBuffer[Self.int_dtype]](
            self.create_buffer_and_copy(deviceContext,csr.cols())
        )
        self.values_buffer = {}
        for entry in csr.values().items():
            self.values_buffer[entry.key] = rebind[DeviceBuffer[Self.float_dtype]](
                self.create_buffer_and_copy(deviceContext,Span(entry.value))
            )
        
        self._shape = csr.shape()

    @staticmethod
    def create_buffer_and_copy[origin:Origin,//,dtype:DType](deviceContext:DeviceContext,src:Span[Scalar[dtype],origin]) raises -> DeviceBuffer[dtype]:
        var buffer = deviceContext.enqueue_create_buffer[dtype](len(src))
        deviceContext.enqueue_copy(buffer,src)
        return buffer^

    def row_offsets(self) raises -> TileTensor[Self.int_dtype, Self.RowMajor1DType, origin_of(self.row_offsets_buffer)]:
        return TileTensor(
            self.row_offsets_buffer,
            row_major(dyn_coord[Self.int_dtype]((len(self.row_offsets_buffer),))),
        )

    def cols(self) raises -> TileTensor[Self.int_dtype, Self.RowMajor1DType, origin_of(self.cols_buffer)]:
        return TileTensor(
            self.cols_buffer,
            row_major(dyn_coord[Self.int_dtype]((len(self.cols_buffer),))),
        )

    def get_value(mut self, key: String) raises -> TileTensor[Self.float_dtype, Self.RowMajor1DType,origin_of(self.values_buffer[key])]:
        var layout = row_major(dyn_coord[Self.int_dtype]((len(self.values_buffer[key]),)))
        return TileTensor(self.values_buffer[key],layout)

    def add_value(mut self,key:String) raises:
        if key not in self.values_buffer:

            self.values_buffer[key] = self.deviceContext.enqueue_create_buffer[Self.float_dtype](self.nnz())
        else:
            raise Error(t'Assigned key: {key} exists')

    def add_value[origin: Origin, //](mut self,key:String,src:Span[Scalar[Self.float_dtype], origin]) raises:
        if key not in self.values_buffer:
            if len(src) != self.nnz():
                raise Error(t'input span has length of {len(src)} but nnz for CSR is {self.nnz()}')

            self.values_buffer[key] = self.create_buffer_and_copy(self.deviceContext,src)
        else:
            raise Error(t'Assigned key: {key} exists')
            
    def keys(self) -> List[String]:
        return [key for key in self.values_buffer.keys()]
    
    def shape(self) -> Tuple[Int,Int]:
        return self._shape

    def nnz(self) -> Int:
        return len(self.cols_buffer)