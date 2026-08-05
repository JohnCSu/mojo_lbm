
"""Defines the LBM solver traits and the double-buffer solver implementation.

`SolverLike` declares the compile-time interface that all LBM solvers must
satisfy. `DoubleBufferSolver` implements the double-buffer SRT time-stepping
loop, compiling the kernel on construction and exposing `even_step` and
`odd_step` methods that the caller alternates between.
"""
from layout.tile_layout import Layout,row_major,Coord,TensorLayout,col_major
from src.lbm.constants import Lbm_methods,Collisions
from src.lbm import LBM_Grid,LBM_Config,TiledLayouts,calculate_rho_and_velocity
from src.lbm import kernels
from layout import TileTensor
from std.gpu.host import DeviceContext,DeviceFunction
from src.lbm import GridLike,ConfigLike,Assembly
from std.utils import Variant

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
 
struct Solver[grid_:LBM_Grid,config_:LBM_Config[Lbm_methods.DOUBLE_BUFFER]](SolverLike):
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
        type_of(Self.layouts.f_layout),type_of(Self.layouts.bc_layout),type_of(Self.layouts.flag_layout),Self.grid,Self.config
        ]

    comptime esoteric_pull_odd_step = kernels.esoteric_pull_kernel[
        False,type_of(Self.layouts.f_layout),type_of(Self.layouts.bc_layout),type_of(Self.layouts.flag_layout),Self.grid,Self.config,]
        
    comptime esoteric_pull_even_step = kernels.esoteric_pull_kernel[
        True,type_of(Self.layouts.f_layout),type_of(Self.layouts.bc_layout),type_of(Self.layouts.flag_layout),Self.grid,Self.config]

    var deviceContext:DeviceContext
    """The device context used for kernel compilation and enqueue."""
    
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
        
        self.GRID_DIM = grid_dim.value() if grid_dim else Self.grid.GRID_DIM
        self.BLOCK_SHAPE = block_dim.value() if block_dim else Self.grid.BLOCK_SHAPE

    def get_compiled_kernel(self) raises -> type_of(self.deviceContext.compile_function[Self.double_buffer]()):
        """Recompiles the double-buffer kernel on demand.

        Returns:
            A new compiled kernel instance.
        """
        return self.deviceContext.compile_function[Self.double_buffer]()    


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
        self.deviceContext.enqueue_function[Self.double_buffer](f_out,f_in.as_immut(),bc.as_immut(),flags.as_immut(),tau,grid_dim = self.GRID_DIM,block_dim = self.BLOCK_SHAPE)


    @always_inline
    def esoteric_pull_step[
        f_dtype:DType,float_dtype:DType,f_out_origin:Origin[mut=True],f_layout_type:TensorLayout,bc_layout_type:TensorLayout,flags_layout_type:TensorLayout,//,
        is_even_step:Bool,
        ](
        self,
        f:TileTensor[f_dtype,f_layout_type,f_out_origin],
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
            is_even_step: When `True`, runs the even step of the esoteric pull kernel.

        Args:
            f: The distribution function tile tensor.
            bc: The boundary-condition tile tensor.
            flags: The `uint8` flag tile tensor.
            tau: The base SRT relaxation time.
        """
        comptime if is_even_step:
            self.deviceContext.enqueue_function[Self.esoteric_pull_even_step](f,bc.as_immut(),flags.as_immut(),tau,grid_dim = self.GRID_DIM,block_dim = self.BLOCK_SHAPE)
        else:
            self.deviceContext.enqueue_function[Self.esoteric_pull_odd_step](f,bc.as_immut(),flags.as_immut(),tau,grid_dim = self.GRID_DIM,block_dim = self.BLOCK_SHAPE)



    @always_inline
    def even_step(self,mut assembly:Assembly,tau:Scalar[assembly.grid.float_dtype]) raises:
        """Runs one even time step using `f1` as input and `f2` as output.

        Args:
            assembly: The `Assembly` providing the GPU buffers.
            tau: The SRT relaxation time.
        """

        comptime if self.lbm_method == Lbm_methods.DOUBLE_BUFFER:
            f_in,f_out,bc,flags = assembly.get_gpu_tensors_for_double_buffer()
            self.double_buffer_step(f_out,f_in,bc,flags,tau)
        else:
            f_in,bc,flags = assembly.get_gpu_tensors_for_esoteric_pull()
            self.esoteric_pull_step[True](f_in,bc,flags,tau)

    @always_inline
    def odd_step(self,mut assembly:Assembly,tau:Scalar[assembly.grid.float_dtype]) raises:
        """Runs one odd time step using `f2` as input and `f1` as output.

        Args:
            assembly: The `Assembly` providing the GPU buffers.
            tau: The SRT relaxation time.
        """
        comptime if self.lbm_method == Lbm_methods.DOUBLE_BUFFER:
            f_in,f_out,bc,flags = assembly.get_gpu_tensors_for_double_buffer()
            self.double_buffer_step(f_in,f_out,bc,flags,tau)
        else:
            f_in,bc,flags = assembly.get_gpu_tensors_for_esoteric_pull()
            self.esoteric_pull_step[False](f_in,bc,flags,tau)
