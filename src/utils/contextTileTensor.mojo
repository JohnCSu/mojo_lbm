"""Defines `ContextTileTensor`, a host/device buffer pair tied to a `DeviceContext`.

The container keeps a host buffer and a device buffer in sync and exposes
them as `TileTensor` views through the `.cpu()` and `.gpu()` accessors, only
copying between buffers when the accessor switches the active device.
"""
from std.gpu.host import DeviceContext
from std.gpu import HostBuffer, DeviceBuffer
from layout import TileTensor, row_major, col_major, coord, Coord
from std.utils import IndexList
from layout.tile_layout import Layout, TensorLayout
from std.collections import Set
from std.python import Python, PythonObject


def get_shape_and_stride[
    LayoutType: TensorLayout
]() -> Tuple[IndexList[LayoutType.rank], IndexList[LayoutType.rank]]:
    """Returns the logical shape and stride for a possibly-nested tile layout.

    When the layout is nested (rank doubled by a tile/tiler product), each
    logical axis collapses its two flat axes by multiplying their shapes and
    strides. Non-nested layouts are returned unchanged.

    Parameters:
        LayoutType: The compile-time `TensorLayout` to summarize.

    Returns:
        A tuple of `(shape, stride)` as `IndexList`s of length `LayoutType.rank`.
    """
    comptime assert (
        LayoutType.rank == LayoutType.flat_rank
        or LayoutType.rank * 2 == LayoutType.flat_rank
    )
    comptime is_nested = LayoutType.rank * 2 == LayoutType.flat_rank
    shape = IndexList[LayoutType.rank](fill=0)
    stride = IndexList[LayoutType.rank](fill=0)

    comptime for i in range(LayoutType.rank):
        comptime if is_nested:
            shape[i] = (
                LayoutType.static_shape[i * 2]
                * LayoutType.static_shape[i * 2 + 1]
            )
            stride[i] = (
                LayoutType.static_stride[i * 2]
                * LayoutType.static_stride[i * 2 + 1]
            )
        else:
            shape[i] = LayoutType.static_shape[i]
            stride[i] = LayoutType.static_stride[i]
    return (shape, stride)


struct ContextTileTensor[dtype: DType, LayoutType: TensorLayout](Movable):
    """Manages paired host and device buffers tied to a `DeviceContext`.

    Holds a `HostBuffer` and a `DeviceBuffer` for the same logical tensor and
    exposes them on demand as `TileTensor` views via `.cpu()` and `.gpu()`.
    A copy between the host and device buffers is performed only when the
    accessor switches the active device, so repeated reads of the same buffer
    are free. Buffer copies can be disabled with `copy_on_switch=False`.

    Parameters:
        dtype: The `DType` of the tile tensor elements.
        LayoutType: The compile-time `TensorLayout` that defines the view of
            the `TileTensor`. May be inferred from the `layout` argument to
            `__init__`.

    Examples:

    ```mojo
    var a = ContextTileTensor(ctx, layout)
    var cpu_tensor = a.cpu()  # No copy: initial access
    # CPU work here...
    var gpu_tensor = a.gpu()  # Copy host -> device
    # GPU work here...
    var gpu_tensor2 = a.gpu() # No copy: previous access was GPU
    var cpu_tensor2 = a.cpu() # Copy device -> host
    ```
    """

    # comptime LayoutType = type_of(Self.layout)

    # comptime assert Self.layout.all_dims_known

    comptime rank = Self.LayoutType.rank
    comptime flat_rank = Self.LayoutType.rank

    comptime __shape_and_stride = get_shape_and_stride[Self.LayoutType]()
    comptime logical_shape = Self.__shape_and_stride[0]
    comptime row_major_layout = row_major(Coord(Self.logical_shape))
    comptime col_major_layout = col_major(Coord(Self.logical_shape))

    var deviceContext: DeviceContext
    """The device context used to allocate and synchronize the buffers."""
    var _cpu_buffer: HostBuffer[Self.dtype]
    var _gpu_buffer: DeviceBuffer[Self.dtype]

    var last_used_cpu: Optional[Bool]
    var copy_on_switch: Bool
    """Whether to copy between buffers when the active device switches."""
    var synchronize_on_copy: Bool
    """Whether to synchronize the device context after each cross-device copy."""
    var layout: Self.LayoutType
    var _size: Int
    var last_device_used: Optional[String]
    """Tracks the most recently accessed device (`'cpu'` or `'gpu'`)."""

    var _extra_row_host_buffer: Optional[HostBuffer[Self.dtype]]

    def __init__(
        out self,
        deviceContext: DeviceContext,
        layout: Self.LayoutType,
        *,
        fill: Optional[Scalar[Self.dtype]] = None,
        synchronize_on_copy: Bool = False,
        copy_on_switch: Bool = True,
    ) raises:
        """Initializes the host and device buffers for a given layout.

        Allocates a `HostBuffer` and a `DeviceBuffer` of the size required by
        `layout` and records the device context used for synchronization and
        cross-buffer copies.

        Args:
            deviceContext: The device context that owns the buffers.
            layout: The compile-time layout of the tile tensor.
            fill: When not `None`, fills both buffers with this scalar.
            synchronize_on_copy: Whether to synchronize the device context
                after a copy between buffers (keyword-only, defaults to
                `False`).
            copy_on_switch: Whether to copy between host and device buffers
                when `cpu()`/`gpu()` is called after `gpu()`/`cpu()`
                (keyword-only, defaults to `True`).
        """
        self.layout = layout
        self.deviceContext = deviceContext
        self._size = layout.size()

        self._cpu_buffer = deviceContext.enqueue_create_host_buffer[Self.dtype](
            self._size
        )
        self._gpu_buffer = deviceContext.enqueue_create_buffer[Self.dtype](
            self._size
        )

        self.last_used_cpu = None
        self.last_device_used = None
        self.copy_on_switch = copy_on_switch
        self.synchronize_on_copy = synchronize_on_copy

        self._extra_row_host_buffer = None

        if fill:
            self.fill(fill.value())

    def __len__(self) -> Int:
        """Returns the number of elements in the tensor."""
        return self.size()

    @always_inline
    def size(self) -> Int:
        """Returns the number of elements in the tensor."""
        return self._size

    @always_inline
    def fill(mut self, value: Scalar[Self.dtype]) raises:
        """Fills both the host and device buffers with a scalar value.

        Args:
            value: The scalar to broadcast to every element.
        """
        self._cpu_buffer.enqueue_fill(
            value
        )  # Weird bug where a memory spike occures when gpu_buffer ie enqued
        self.deviceContext.enqueue_copy(
            dst_buf=self._gpu_buffer, src_buf=self._cpu_buffer
        )
        self.synchronize()
        # _ = self.gpu()

    def synchronize(self) raises:
        """Synchronizes the underlying device context."""
        self.deviceContext.synchronize()

    def copy_cpu_to_gpu(mut self) raises:
        """Copies the host buffer into the device buffer."""
        self.deviceContext.enqueue_copy(
            dst_buf=self._gpu_buffer, src_buf=self._cpu_buffer
        )
        if self.synchronize_on_copy:
            self.synchronize()

    def copy_gpu_to_cpu(mut self) raises:
        """Copies the device buffer into the host buffer."""
        self.deviceContext.enqueue_copy(
            dst_buf=self._cpu_buffer, src_buf=self._gpu_buffer
        )
        if self.synchronize_on_copy:
            self.synchronize()

    @always_inline
    def cpu_buffer(mut self) raises -> HostBuffer[Self.dtype]:
        """Returns the host buffer, switching the active device to CPU if needed.

        Returns:
            The host `HostBuffer`.
        """
        self._check_last_used_device(currentDevice="cpu")
        return self._cpu_buffer

    @always_inline
    def gpu_buffer(mut self) raises -> DeviceBuffer[Self.dtype]:
        """Returns the device buffer, switching the active device to GPU if needed.

        Returns:
            The device `DeviceBuffer`.
        """
        self._check_last_used_device(currentDevice="gpu")
        return self._gpu_buffer

    @always_inline
    def cpu(
        mut self,
    ) raises -> TileTensor[
        Self.dtype, Self.LayoutType, origin_of(self._cpu_buffer)
    ]:
        """Returns a `TileTensor` view of the host buffer.

        Switches the active device to CPU and copies from the device buffer
        when the previous access was GPU and `copy_on_switch` is `True`.

        Returns:
            A `TileTensor` view of the host buffer.
        """
        self._check_last_used_device(currentDevice="cpu")
        return TileTensor(self._cpu_buffer, self.layout)

    @always_inline
    def gpu(
        mut self,
    ) raises -> TileTensor[
        Self.dtype, Self.LayoutType, origin_of(self._gpu_buffer)
    ]:
        """Returns a `TileTensor` view of the device buffer.

        Switches the active device to GPU and copies from the host buffer
        when the previous access was CPU and `copy_on_switch` is `True`.

        Returns:
            A `TileTensor` view of the device buffer.
        """
        self._check_last_used_device(currentDevice="gpu")
        return TileTensor(self._gpu_buffer, self.layout)

    def buffer_to_numpy(mut self) raises -> PythonObject:
        """Returns a 1D NumPy view of the host buffer.

        Returns:
            A NumPy array sharing memory with the host buffer.
        """
        return contextTensor_to_numpy(self)

    # def copy_to_row_major(mut self) raises -> TileTensor[Self.dtype,type_of(Self.row_major_layout),origin_of(self._extra_row_host_buffer.value())]:
    #     '''
    #     copy cpu buffer to a row_major equivalent layout
    #     '''
    # Worilk
    #     # This keeps track
    # if self._extra_row_host_buffer is None:
    # self._extra_row_host_buffer = self.deviceContext.enqueue_create_host_buffer[Self.dtype](self._size)

    # row_tensor = TileTensor(self._extra_row_host_buffer.value(),Self.row_major_layout)
    # row_tensor.copy_from(self.cpu())
    # return row_tensor

    def _check_last_used_device(mut self, currentDevice: String) raises:
        if currentDevice not in Set[String]("cpu", "gpu"):
            raise Error("Device String either cpu or gpu")

        if self.last_device_used is None:  # Initial access of buffer
            self.last_device_used = currentDevice

        if currentDevice != self.last_device_used.value():
            if self.copy_on_switch:
                if currentDevice == "cpu":
                    self.copy_gpu_to_cpu()
                else:  # Current Device is thereor gpu
                    self.copy_cpu_to_gpu()
            self.last_device_used = currentDevice


def contextTensor_to_numpy[
    dtype: DType, layoutType: TensorLayout, synchronize: Bool = True
](
    mut contextTensor: ContextTileTensor[dtype, layoutType]
) raises -> PythonObject:
    """Returns a zero-copy 1D NumPy view of a `ContextTileTensor` host buffer.

    Synchronizes the device and host buffers (when `synchronize` is `True`) and
    hands NumPy an unsafe pointer into the host buffer. The returned array is
    one-dimensional because tile layouts are not strictly row- or column-major.

    Changes to the returned array mutate the underlying host buffer.

    Parameters:
        dtype: The `DType` of the tensor elements.
        layoutType: The compile-time layout of the tensor.
        synchronize: When `True`, synchronizes the device context before
            building the view (defaults to `True`).

    Args:
        contextTensor: The context tensor to view.

    Returns:
        A 1D NumPy array sharing memory with the host buffer.
    """
    np = Python.import_module("numpy")
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
    address = Int(flag_ptr)  # Need to get the pointer address as Int type
    p_int = ctypes.POINTER(c_dtype)  # Set Dtype
    np_ptr = ctypes.cast(address, p_int)
    np_arr = np.ctypeslib.as_array(
        np_ptr, shape=Python.tuple(contextTensor.size())
    )
    return np_arr
