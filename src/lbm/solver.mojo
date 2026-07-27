
"""Defines the LBM solver traits and the double-buffer solver implementation.

`SolverLike` declares the compile-time interface that all LBM solvers must
satisfy. `DoubleBufferSolver` implements the double-buffer SRT time-stepping
loop, compiling the kernel on construction and exposing `even_step` and
`odd_step` methods that the caller alternates between.
"""

# from layout import TileTensor, coord,CoordLike,ComptimeInt
from layout.tile_layout import Layout,row_major,Coord,TensorLayout,col_major
from src.lbm.constants import Lbm_methods,Collisions
from src.lbm import LBM_Grid,LBM_Config,TiledLayouts,calculate_rho_and_velocity
from src.lbm import kernels
from layout import TileTensor
# from layout import Idx
from std.gpu.host import DeviceContext,DeviceFunction
from src.lbm import GridLike,ConfigLike,Assembly


trait SolverLike:
    """Declares the compile-time interface for an LBM solver.

    Conforming types expose the LBM method name, the grid, the
    `LBM_Config`, and the tiled layouts needed by the time-stepping
    kernels.
    """

    comptime lbm_method:StaticString
    """The LBM streaming method."""
    comptime gridType:GridLike
    """The compile-time type of the grid."""
    comptime grid:Self.gridType
    """The compile-time grid instance."""
    comptime config:LBM_Config[Self.lbm_method]
    """The compile-time `LBM_Config`."""

    # comptime layouts = TiledLayouts[Self.grid]()

    # comptime output_velocity_func = calculate_rho_and_velocity[
    #     Self.layouts.f_layout,
    #     Self.layouts.bc_layout,
    #     Self.layouts.flag_layout,
    #     Self.layouts.rho_layout,
    #     Self.layouts.velocity_layout,
    #     Self.grid,
    #     Self.config,
    #     ]

    # comptime compiled_velocity_func_type = type_of( DeviceContext().compile_function[Self.output_velocity_func,Self.output_velocity_func]() )
    

    # @staticmethod
    # def get_compiled_output_rho_and_velocity_kernel(ctx:DeviceContext) raises -> Self.compiled_velocity_func_type:
    #     return ctx.compile_function[Self.output_velocity_func,Self.output_velocity_func]()    
    
    # @staticmethod
    # def get_rho_and_velocity(
    #     ctx:DeviceContext,
    #     compiled_kernel:Self.compiled_velocity_func_type,
    #     density:TileTensor[Self.grid.float_dtype,type_of(Self.layouts.rho_layout),MutAnyOrigin],
    #     velocity:TileTensor[Self.grid.float_dtype,type_of(Self.layouts.velocity_layout),MutAnyOrigin],
        
    #     f:TileTensor[Self.config.set_f_dtype(Self.grid.float_dtype),type_of(Self.layouts.f_layout),_],
    #     bc:TileTensor[Self.grid.float_dtype,type_of( Self.layouts.bc_layout),_],
    #     flags:TileTensor[DType.uint8,type_of(Self.layouts.flag_layout),_],
    #     *,
    #     block_dim:Optional[Tuple[Int,Int,Int]] = None,
    #     grid_dim:Optional[Tuple[Int,Int,Int]] = None,
    #     GPU:Bool = True
    #     ) raises:
    #     GRID_DIM = grid_dim.value() if grid_dim else Self.grid.GRID_DIM
    #     BLOCK_SHAPE = block_dim.value() if block_dim else Self.grid.BLOCK_SHAPE
    #     ctx.synchronize()
    #     ctx.enqueue_function(compiled_kernel,density,velocity,f.as_immut(),bc.as_immut(),flags.as_immut(),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
 
struct DoubleBufferSolver[grid_:LBM_Grid,config_:LBM_Config[Lbm_methods.DOUBLE_BUFFER]](SolverLike):
    """Implements the double-buffer SRT LBM time-stepping loop.

    Compiles the double-buffer kernel on construction and exposes
    `even_step` and `odd_step` methods that read from `f_in` and write
    to `f_out`, allowing the caller to swap buffers between time steps.

    Parameters:
        grid_: The compile-time `LBM_Grid` describing the domain.
        config_: The compile-time `LBM_Config` for the double-buffer method.
    """

    comptime gridType = type_of(Self.grid_)
    comptime configType = type_of(Self.config_)

    comptime grid = Self.grid_
    comptime config = Self.config_
    comptime lbm_method = Self.config.lbm_method
    comptime layouts = Self.grid.layouts

    comptime double_buffer = kernels.double_buffer_kernel[
        Self.layouts.f_layout,Self.layouts.bc_layout,Self.layouts.flag_layout,Self.grid,Self.config
        ]


    comptime compiled_type = type_of(DeviceContext().compile_function[Self.double_buffer,Self.double_buffer]())

    var deviceContext:DeviceContext
    """The device context used for kernel compilation and enqueue."""
    var compiled_double_buffer_kernel:Self.compiled_type
    """The compiled double-buffer kernel."""

    var BLOCK_SHAPE:Tuple[Int,Int,Int]
    """The GPU block shape for the kernel launch."""
    var GRID_DIM:Tuple[Int,Int,Int]
    """The GPU grid dimensions for the kernel launch."""

    def __init__(out self,deviceContext:DeviceContext,*,block_dim:Optional[Tuple[Int,Int,Int]]=None,grid_dim:Optional[Tuple[Int,Int,Int]] = None,) raises:
        """Constructs a `DoubleBufferSolver` and compiles the kernel.

        Args:
            deviceContext: The device context for kernel compilation.
            block_dim: Optional override for the GPU block dimensions
                (defaults to `None`).
            grid_dim: Optional override for the GPU grid dimensions
                (defaults to `None`).
        """
        self.deviceContext = deviceContext
        self.compiled_double_buffer_kernel = self.deviceContext.compile_function[Self.double_buffer,Self.double_buffer]()    
        
        self.GRID_DIM = grid_dim.value() if grid_dim else Self.grid.GRID_DIM
        self.BLOCK_SHAPE = block_dim.value() if block_dim else Self.grid.BLOCK_SHAPE

    def get_compiled_kernel(self) raises -> type_of(self.deviceContext.compile_function[Self.double_buffer,Self.double_buffer]()):
        """Recompiles the double-buffer kernel on demand.

        Returns:
            A new compiled kernel instance.
        """
        return self.deviceContext.compile_function[Self.double_buffer,Self.double_buffer]()    

    @always_inline
    def double_buffer_step[f_dtype:DType,float_dtype:DType,f_out_origin:Origin[mut=True],f_layout_type:TensorLayout,bc_layout_type:TensorLayout,flags_layout_type:TensorLayout,//](
        self,
        f_out:TileTensor[f_dtype,f_layout_type,f_out_origin],
        f_in:TileTensor[f_dtype,f_layout_type,_],
        bc:TileTensor[float_dtype,bc_layout_type,_],
        flags:TileTensor[DType.uint8,flags_layout_type,_],
        tau:Scalar[float_dtype],
        ) raises:
        """Enqueues one double-buffer LBM time step on the GPU.

        Parameters:
            f_dtype: The storage `DType` of `f_in` and `f_out`.
            float_dtype: The float `DType` used for `bc` and `tau`.
            f_out_origin: The mutability origin of `f_out`.
            f_layout_type: The compile-time layout of the distribution
                function.
            bc_layout_type: The compile-time layout of `bc`.
            flags_layout_type: The compile-time layout of `flags`.

        Args:
            f_out: The output distribution function tile tensor.
            f_in: The input distribution function tile tensor.
            bc: The boundary-condition tile tensor.
            flags: The `uint8` flag tile tensor.
            tau: The base SRT relaxation time.
        """
        comptime assert Self.lbm_method == Lbm_methods.DOUBLE_BUFFER
        self.deviceContext.enqueue_function(self.compiled_double_buffer_kernel,f_out,f_in.as_immut(),bc.as_immut(),flags.as_immut(),tau,grid_dim = self.GRID_DIM,block_dim = self.BLOCK_SHAPE)

    @always_inline
    def even_step(self,mut assembly:Assembly,tau:Scalar[assembly.grid.float_dtype]) raises:
        """Runs one even time step using `f1` as input and `f2` as output.

        Args:
            assembly: The `Assembly` providing the GPU buffers.
            tau: The SRT relaxation time.
        """
        f_in,f_out,bc,flags = assembly.get_gpu_tensors_for_double_buffer()
        self.double_buffer_step(f_out,f_in,bc,flags,tau)

    @always_inline
    def odd_step(self,mut assembly:Assembly,tau:Scalar[assembly.grid.float_dtype]) raises:
        """Runs one odd time step using `f2` as input and `f1` as output.

        Args:
            assembly: The `Assembly` providing the GPU buffers.
            tau: The SRT relaxation time.
        """
        f_in,f_out,bc,flags = assembly.get_gpu_tensors_for_double_buffer()
        self.double_buffer_step(f_in,f_out,bc,flags,tau)
