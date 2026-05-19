from std.gpu.host import DeviceContext
from std.gpu import HostBuffer,DeviceBuffer
from layout import TileTensor,row_major,coord,Coord
from layout.tile_layout import Layout,TensorLayout
from std.collections import Set

struct ContextTileTensor[dtype:DType,LayoutType:TensorLayout]():
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
    comptime TensorType = TileTensor[Self.dtype,Self.LayoutType,MutAnyOrigin]
    var deviceContext:DeviceContext
    var _cpu_buffer:HostBuffer[Self.dtype]
    var _gpu_buffer:DeviceBuffer[Self.dtype]

    var last_used_cpu: Optional[Bool]
    var copy_on_switch:Bool
    var synchronize_on_copy:Bool
    var layout:Self.LayoutType
    var size: Int 
    var last_device_used:Optional[String]

    def __init__(out self,deviceContext:DeviceContext,layout:Self.LayoutType,*,synchronize_on_copy:Bool = False,copy_on_switch:Bool = True) raises:
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
        self.size = layout.size()

        self._cpu_buffer = deviceContext.enqueue_create_host_buffer[Self.dtype](self.size)
        self._gpu_buffer = deviceContext.enqueue_create_buffer[Self.dtype](self.size)

        self.last_used_cpu = None
        self.last_device_used = None
        self.copy_on_switch = copy_on_switch
        self.synchronize_on_copy = synchronize_on_copy


    def fill(mut self,value:Scalar[Self.dtype]) raises:
        self._cpu_buffer.enqueue_fill(value)
        self._gpu_buffer.enqueue_fill(value)

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

    def cpu_buffer(mut self) raises -> HostBuffer[Self.dtype]:
        self._check_last_used_device(currentDevice = 'cpu')
        return self._cpu_buffer

    def gpu_buffer(mut self) raises -> DeviceBuffer[Self.dtype]:
        self._check_last_used_device(currentDevice = 'gpu')
        return self._gpu_buffer


    def _check_last_used_device(mut self,currentDevice:String) raises:
        if currentDevice not in Set[String]('cpu','gpu'):
            raise Error('Device String either cpu or gpu')

        if self.last_device_used is None:
            self.last_device_used = currentDevice

        if currentDevice != self.last_device_used.value():
            if self.copy_on_switch:
                if currentDevice == 'cpu':
                    self.copy_gpu_to_cpu()
                else: # Current Device is thereor gpu
                    self.copy_cpu_to_gpu()
            self.last_device_used = currentDevice



    def cpu(mut self) raises -> TileTensor[Self.dtype,Self.LayoutType,origin_of(self._cpu_buffer)]:
        self._check_last_used_device(currentDevice = 'cpu')
        return TileTensor(self._cpu_buffer,self.layout)

    def gpu(mut self) raises -> TileTensor[Self.dtype,Self.LayoutType,origin_of(self._gpu_buffer)]:
        self._check_last_used_device(currentDevice = 'gpu')
        return TileTensor(self._gpu_buffer,self.layout)