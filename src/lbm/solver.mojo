
# from layout import TileTensor, coord,CoordLike,ComptimeInt
from src.lbm.constants import Lbm_methods,Collisions
from src.lbm import LBM_Grid,LBM_Config,TiledLayouts,calculate_rho_and_velocity
from src.lbm import kernels
from layout import TileTensor
# from layout import Idx
from std.gpu.host import DeviceContext,DeviceFunction
from src.lbm import GridLike,ConfigLike


trait SolverLike:
    comptime lbm_method:StaticString
    comptime gridType:GridLike
    comptime grid:Self.gridType
    comptime config:LBM_Config[Self.lbm_method]

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
    comptime gridType = type_of(Self.grid_)
    comptime configType = type_of(Self.config_)

    comptime grid = Self.grid_
    comptime config = Self.config_
    comptime lbm_method = Self.config.lbm_method
    comptime layouts = Self.grid.layouts
    
    comptime double_buffer = kernels.double_buffer_kernel[
        Self.layouts.f_layout,Self.layouts.bc_layout,Self.layouts.flag_layout,Self.grid,Self.config
        ]
    
    var deviceContext:DeviceContext
    # This is a test will probs change in future. Insane Type Hacks
    comptime compiled_type = type_of(DeviceContext().compile_function[Self.double_buffer,Self.double_buffer]())
    var compiled_double_buffer_kernel:Self.compiled_type

    var BLOCK_SHAPE:Tuple[Int,Int,Int]
    var GRID_DIM:Tuple[Int,Int,Int]

    def __init__(out self,deviceContext:DeviceContext,*,block_dim:Optional[Tuple[Int,Int,Int]]=None,grid_dim:Optional[Tuple[Int,Int,Int]] = None,) raises:
        self.deviceContext = deviceContext
        self.compiled_double_buffer_kernel = self.deviceContext.compile_function[Self.double_buffer,Self.double_buffer]()    
        
        self.GRID_DIM = grid_dim.value() if grid_dim else Self.grid.GRID_DIM
        self.BLOCK_SHAPE = block_dim.value() if block_dim else Self.grid.BLOCK_SHAPE


    def get_compiled_kernel(self) raises -> type_of(self.deviceContext.compile_function[Self.double_buffer,Self.double_buffer]()):
        return self.deviceContext.compile_function[Self.double_buffer,Self.double_buffer]()    

    
    @always_inline
    def step[f_dtype:DType,//,GPU:Bool = True](
        self,
        f_out:TileTensor[f_dtype,type_of(Self.layouts.f_layout),_],
        f_in:TileTensor[f_dtype,type_of(Self.layouts.f_layout),_],
        bc:TileTensor[Self.grid.float_dtype,type_of( Self.layouts.bc_layout),_],
        flags:TileTensor[DType.uint8,type_of(Self.layouts.flag_layout),_],
        tau:Scalar[Self.grid.float_dtype],
        ) raises:

        self.deviceContext.enqueue_function(self.compiled_double_buffer_kernel,f_out,f_in.as_immut(),bc.as_immut(),flags.as_immut(),tau,grid_dim = self.GRID_DIM,block_dim = self.BLOCK_SHAPE)


