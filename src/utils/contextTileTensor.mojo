from std.gpu.host import DeviceContext
from std.gpu import HostBuffer,DeviceBuffer
from layout import TileTensor,row_major,col_major,coord,Coord
from std.utils import IndexList
from layout.tile_layout import Layout,TensorLayout
from std.collections import Set
from std.python import Python, PythonObject


def get_shape_and_stride[LayoutType:TensorLayout]() ->Tuple[IndexList[LayoutType.rank],IndexList[LayoutType.rank]]:
    comptime assert LayoutType.rank == LayoutType.flat_rank or LayoutType.rank*2 == LayoutType.flat_rank
    comptime is_nested = LayoutType.rank*2 == LayoutType.flat_rank
    shape = IndexList[LayoutType.rank](fill = 0)
    stride = IndexList[LayoutType.rank](fill = 0)

    comptime for i in range(LayoutType.rank):
        comptime if is_nested:
            shape[i] = LayoutType.static_shape[i*2] *LayoutType.static_shape[i*2+1]
            stride[i] = LayoutType.static_stride[i*2] *LayoutType.static_stride[i*2+1]
        else:
            shape[i] = LayoutType.static_shape[i]
            stride[i] = LayoutType.static_stride[i]
    return (shape,stride)

struct ContextTileTensor[dtype:DType,LayoutType:TensorLayout](Movable):
    '''
    A simple container for storing both host and device buffers tied to a deviceContext and tying to a specific tiletensor.
    Uses `.cpu()` and `.gpu()` method to call tiletensor views of underlying data which are created created on the fly to
    prevent accidental sync issues. getter methods for host and device buffers can also be accessed through 
    `.cpu_buffer()` and `.gpu_buffer()` respectively.

    If the last access is different to the current access (e.g. first access was .cpu and then .gpu) then a synchronization
    and copy from the last used buffer to new buffer is forced. This can be turned off by setting copy_on_switch attribute

    Parameters:
        dtype: DType of tileTensor.
        LayoutType: TensorLayout: a compile time tile_layout layout that defines the view of the TileTensor. Can be 
            Inferred from __init__.
    '''
    # comptime LayoutType = type_of(Self.layout)
    
    # comptime assert Self.layout.all_dims_known
    
    comptime rank = Self.LayoutType.rank
    comptime flat_rank = Self.LayoutType.rank
    
    comptime __shape_and_stride = get_shape_and_stride[Self.LayoutType]()
    comptime logical_shape = Self.__shape_and_stride[0]
    comptime row_major_layout = row_major(Coord(Self.logical_shape))
    comptime col_major_layout = col_major(Coord(Self.logical_shape))

    var deviceContext:DeviceContext
    var _cpu_buffer:HostBuffer[Self.dtype]
    var _gpu_buffer:DeviceBuffer[Self.dtype]

    var last_used_cpu: Optional[Bool]
    var copy_on_switch:Bool
    var synchronize_on_copy:Bool
    var layout:Self.LayoutType
    var _size: Int 
    var last_device_used:Optional[String]

    var _extra_row_host_buffer:Optional[HostBuffer[Self.dtype]]

    def __init__(out self,deviceContext:DeviceContext,layout:Self.LayoutType,*,fill:Optional[Scalar[Self.dtype]] = None,synchronize_on_copy:Bool = False,copy_on_switch:Bool = True) raises:
        '''
        Initialise ContextTileTensor with DeviceContext and layout. LayoutType is inferred from layout passed in.

        Args:
            deviceContext: DeviceContext.
            layout: Compile Time Layout of tileTensor.
            synchronize_on_copy: A Bool to determine whether to synchronize the DeviceContext after a copy between buffers. Keyword Only, Default False.
            copy_on_switch: A Bool to indicate that a copy should be performed between host and device buffer whenever the cpu()/gpu() is called after gpu()/cpu()
                            respectively. Keyword Only, Default is True.
        
        '''
        self.layout = layout
        self.deviceContext = deviceContext
        self._size = layout.size()

        self._cpu_buffer = deviceContext.enqueue_create_host_buffer[Self.dtype](self._size)
        self._gpu_buffer = deviceContext.enqueue_create_buffer[Self.dtype](self._size)
        


        self.last_used_cpu = None
        self.last_device_used = None
        self.copy_on_switch = copy_on_switch
        self.synchronize_on_copy = synchronize_on_copy

        self._extra_row_host_buffer = None

        if fill:
            self.fill(fill.value())

    def __len__(self) -> Int:
        return self.size()

    @always_inline
    def size(self) -> Int:
        return self._size

    @always_inline
    def fill(mut self,value:Scalar[Self.dtype]) raises:
        self._cpu_buffer.enqueue_fill(value) # Weird bug where a memory spike occures when gpu_buffer ie enqued
        self.deviceContext.enqueue_copy(dst_buf= self._gpu_buffer,src_buf = self._cpu_buffer)
        self.synchronize()
        # _ = self.gpu()

    def synchronize(self) raises:
        self.deviceContext.synchronize()

    def copy_cpu_to_gpu(mut self) raises:
        self.deviceContext.enqueue_copy(dst_buf= self._gpu_buffer,src_buf = self._cpu_buffer)
        if self.synchronize_on_copy:
            self.synchronize()

    def copy_gpu_to_cpu(mut self) raises:
        self.deviceContext.enqueue_copy(dst_buf= self._cpu_buffer,src_buf = self._gpu_buffer)
        if self.synchronize_on_copy:
            self.synchronize()
    
    @always_inline
    def cpu_buffer(mut self) raises -> HostBuffer[Self.dtype]:
        self._check_last_used_device(currentDevice = 'cpu')
        return self._cpu_buffer
    @always_inline
    def gpu_buffer(mut self) raises -> DeviceBuffer[Self.dtype]:
        self._check_last_used_device(currentDevice = 'gpu')
        return self._gpu_buffer

    @always_inline
    def cpu(mut self) raises -> TileTensor[Self.dtype,Self.LayoutType,origin_of(self._cpu_buffer)]:
        self._check_last_used_device(currentDevice = 'cpu')
        return TileTensor(self._cpu_buffer,self.layout)
    @always_inline
    def gpu(mut self) raises -> TileTensor[Self.dtype,Self.LayoutType,origin_of(self._gpu_buffer)]:
        self._check_last_used_device(currentDevice = 'gpu')
        return TileTensor(self._gpu_buffer,self.layout)

    def buffer_to_numpy(mut self) raises -> PythonObject:
        '''
        returns a view of the host buffer as a 1D Numpy Array. Unsafe Pointer Used!
        '''
        return contextTensor_to_numpy(self)


    # def copy_to_row_major(mut self) raises -> TileTensor[Self.dtype,type_of(Self.row_major_layout),origin_of(self._extra_row_host_buffer.value())]:
    #     '''
    #     copy cpu buffer to a row_major equivalent layout
    #     '''
    # Worlk
    #     # This keeps track
        # if self._extra_row_host_buffer is None:
            # self._extra_row_host_buffer = self.deviceContext.enqueue_create_host_buffer[Self.dtype](self._size)
        
        # row_tensor = TileTensor(self._extra_row_host_buffer.value(),Self.row_major_layout)
        # row_tensor.copy_from(self.cpu())
        # return row_tensor

    def _check_last_used_device(mut self,currentDevice:String) raises:
        if currentDevice not in Set[String]('cpu','gpu'):
            raise Error('Device String either cpu or gpu')

        if self.last_device_used is None: # Initial access of buffer
            self.last_device_used = currentDevice

        if currentDevice != self.last_device_used.value():
            if self.copy_on_switch:
                if currentDevice == 'cpu':
                    self.copy_gpu_to_cpu()
                else: # Current Device is thereor gpu
                    self.copy_cpu_to_gpu()
            self.last_device_used = currentDevice


def contextTensor_to_numpy[dtype:DType,layoutType:TensorLayout,synchronize:Bool = True](mut contextTensor:ContextTileTensor[dtype,layoutType]) raises -> PythonObject:
    '''
    Zero Copy view of the Host Buffer of Context Tensor as a numpy array, by first syncing device and host buffers and passing an unsafe pointer to
    numpy.
    
    as of now returns the buffer as a 1D numpy array (as tile layouts may not be strictly row or column major)

    NOTE: Changes to the array will affect 
    '''
    np = Python.import_module('numpy')
    ctypes = Python.import_module("ctypes")

    ctypes_dict = {
        DType.bool: ctypes.c_bool,
        DType.int8: ctypes.c_int8,
        DType.int16: ctypes.c_int16,
        DType.int32: ctypes.c_int32,
        DType.int64: ctypes.c_int64,
        DType.uint8: ctypes.c_uint8,
        DType.uint16: ctypes.c_uint16,
        DType.uint32: ctypes.c_uint32,
        DType.uint64: ctypes.c_uint64,
        DType.float32: ctypes.c_float,
        DType.float64: ctypes.c_double,
    }

    c_dtype = ctypes_dict[dtype]
    
    flag_ptr = contextTensor.cpu_buffer().unsafe_ptr()
    comptime if synchronize:
        contextTensor.synchronize()
    address = Int(flag_ptr) # Need to get the pointer address as Int type
    p_int = ctypes.POINTER(c_dtype) # Set Dtype
    np_ptr = ctypes.cast(address, p_int)
    np_arr = np.ctypeslib.as_array(np_ptr, shape=Python.tuple(contextTensor.size()))
    return np_arr
