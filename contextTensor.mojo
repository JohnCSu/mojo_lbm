from std.gpu.host import DeviceContext
from std.gpu import HostBuffer,DeviceBuffer
from layout import TileTensor,row_major,coord,Coord
from layout.tile_layout import Layout,TensorLayout

struct ContextTileTensor[dtype:DType,LayoutType:TensorLayout]():
    '''
    A simple container for storing both host and device buffers tied to a deviceContext and viewiing as a tiletensor. Uses .cpu() and .gpu() method to call tiletensor views of 
    underlying data which are created created on the fly to prevent accidental sync issues.

    If the last access is different to the current access (e.g. first access was .cpu and then .gpu) then a synchronization and copy from the last used buffer to new buffer
    is forced. This can be turned off by setting copy_on_switch flag

    Parameters:
        dtype:DType of tileTensor.W
        LayoutType: TensorLayout: a compile time tile_layout layout that defines the view of the TileTensor. Can be Inferred from __init__.
    '''
    # comptime LayoutType = type_of(Self.layout)
    
    # comptime assert Self.layout.all_dims_known
    
    comptime rank = Self.LayoutType.rank
    comptime _mutOrigin = MutOrigin
    comptime TensorType = TileTensor[Self.dtype,Self.LayoutType,MutAnyOrigin]
    var deviceContext:DeviceContext
    var cpu_buffer:HostBuffer[Self.dtype]
    var gpu_buffer:DeviceBuffer[Self.dtype]

    var last_used_cpu: Optional[Bool]
    var copy_on_switch:Bool
    var synchronize_on_copy:Bool
    var layout:Self.LayoutType
    var size: Int 
    
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

        self.cpu_buffer = deviceContext.enqueue_create_host_buffer[Self.dtype](self.size)
        self.gpu_buffer = deviceContext.enqueue_create_buffer[Self.dtype](self.size)

        self.last_used_cpu = None

        self.copy_on_switch = copy_on_switch
        self.synchronize_on_copy = synchronize_on_copy
    def synchronize(self) raises:
        self.deviceContext.synchronize()

    def copy_cpu_to_gpu(mut self) raises:
        self.deviceContext.enqueue_copy(dst_buf= self.gpu_buffer,src_buf = self.cpu_buffer)
        if self.synchronize_on_copy:
            self.synchronize()
    def copy_gpu_to_cpu(mut self) raises:
        self.deviceContext.enqueue_copy(dst_buf= self.cpu_buffer,src_buf = self.gpu_buffer)
        if self.synchronize_on_copy:
            self.synchronize()

    def cpu(mut self) raises -> TileTensor[Self.dtype,Self.LayoutType,origin_of(self.cpu_buffer)]:
        #
        # Track last time the data was accessed by GPU or CPU and then update state
        if self.last_used_cpu is None: # Handles Initial Case where GPU or CPU may be accessed first
            self.last_used_cpu = True

        if not self.last_used_cpu: # GPU Tensor was used
            if self.copy_on_switch:
                self.copy_gpu_to_cpu()
            self.last_used_cpu = True
        return TileTensor(self.cpu_buffer,self.layout)

    def gpu(mut self) raises -> TileTensor[Self.dtype,Self.LayoutType,origin_of(self.gpu_buffer)]:
        if self.last_used_cpu is None: # Handles Initial Case where GPU or CPU may be accessed first
            self.last_used_cpu = False

        if self.last_used_cpu: # CPU Tensor was last called
            if self.copy_on_switch:
                self.copy_cpu_to_gpu()
            self.last_used_cpu = False
        return TileTensor(self.gpu_buffer,self.layout)