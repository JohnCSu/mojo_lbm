# from layout import TileTensor, coord,CoordLike,ComptimeInt
from layout.tile_layout import Layout,row_major,Coord,TensorLayout,col_major
from src.lbm.constants import Lbm_methods,Collisions
from src.lbm import LBM_Grid,LBM_Config,TiledLayouts,calculate_rho_and_velocity
from src.lbm.output import calculate_rho_and_velocity,calculate_Q_criterion

from src.lbm import kernels
from layout import TileTensor
# from layout import Idx
from std.gpu.host import DeviceContext,DeviceFunction
from src.lbm import GridLike,ConfigLike,Assembly
from src.utils import ContextTileTensor,Vector


struct OutputRequest[grid_:LBM_Grid,config_:LBM_Config](Movable):
    '''
    For Ease lets have it as row major output --> Numpy easy viewing
    '''
    comptime grid = Self.grid_
    comptime float_dtype = Self.grid.float_dtype
    comptime config = Self.config_
    comptime lbm_method = Self.config.lbm_method
    comptime layouts = Self.grid.layouts

    comptime velocity_layout = Self.layouts.create_row_major_rank_4_layout[Self.grid.D]()
    comptime density_layout = Self.layouts.create_row_major_rank_3_layout()
    comptime Q_criterion_layout = Self.layouts.create_row_major_rank_3_layout()

    comptime __output_velocity_func = calculate_rho_and_velocity[
        Self.layouts.f_layout,
        Self.layouts.bc_layout,
        Self.layouts.flag_layout,
        Self.density_layout,
        Self.velocity_layout,
        Self.grid,
        Self.config,
        ]

    comptime __Q_criterion_func = calculate_Q_criterion[
        Self.Q_criterion_layout,
        Self.layouts.f_layout,
        Self.layouts.bc_layout,
        Self.layouts.flag_layout,
        Self.velocity_layout,
        Self.grid,
        Self.config,
    ]

    comptime compiled_velocity_func_type = type_of( DeviceContext().compile_function[Self.__output_velocity_func,Self.__output_velocity_func]() )
    comptime compiled_Q_criterion_func_type = type_of( DeviceContext().compile_function[Self.__Q_criterion_func,Self.__Q_criterion_func]() )

    var velocity: Optional[ContextTileTensor[Self.float_dtype,type_of(Self.velocity_layout)]]
    var density: Optional[ContextTileTensor[Self.float_dtype,type_of(Self.density_layout)]]
    var Q_criterion:Optional[ContextTileTensor[Self.float_dtype,type_of(Self.Q_criterion_layout)]]

    var deviceContext:DeviceContext
    var unitSystem:UnitSystem[Self.float_dtype,Self.grid.D]

    var compiled_density_and_velocity_func:Self.compiled_velocity_func_type
    var compiled_Q_criterion_func:Self.compiled_Q_criterion_func_type

    def __init__(out self,deviceContext:DeviceContext,unitSystem:UnitSystem[Self.float_dtype,Self.grid.D],*,velocity:Bool = True,density:Bool = True,Q_criterion:Bool = False) raises:

        self.deviceContext = deviceContext
        self.velocity = None
        self.density = None
        self.Q_criterion = None
        self.unitSystem = unitSystem

        self.compiled_density_and_velocity_func = deviceContext.compile_function[Self.__output_velocity_func,Self.__output_velocity_func]()    
        
        self.compiled_Q_criterion_func = deviceContext.compile_function[Self.__Q_criterion_func,Self.__Q_criterion_func]()    

        if velocity:
            self.velocity = self.velocity.Element(deviceContext,self.velocity_layout)
        
        if density:
            self.density = self.density.Element(deviceContext,self.density_layout)
        
        if Q_criterion:
            if not velocity:
                raise Error('Q criterion, the velocity must be on')
            self.Q_criterion = self.Q_criterion.Element(deviceContext,self.Q_criterion_layout)


    @staticmethod
    def get_rho_and_velocity[
        f_dtype:DType,
        float_dtype:DType,
        density_origin:Origin[mut=True],
        velocity_origin:Origin[mut=True],
        f_layout_type:TensorLayout,
        bc_layout_type:TensorLayout,
        flags_layout_type:TensorLayout,//
        ](
        deviceContext:DeviceContext,
        compiled_func:Self.compiled_velocity_func_type,

        density:TileTensor[float_dtype,type_of(Self.density_layout),density_origin],
        velocity:TileTensor[float_dtype,type_of(Self.velocity_layout),velocity_origin],
        
        f:TileTensor[f_dtype,f_layout_type,_],
        bc:TileTensor[float_dtype,bc_layout_type,_],
        flags:TileTensor[DType.uint8,flags_layout_type,_],
        *,
        block_dim:Optional[Tuple[Int,Int,Int]] = None,
        grid_dim:Optional[Tuple[Int,Int,Int]] = None,
        GPU:Bool = True
        ) raises:
        GRID_DIM = grid_dim.value() if grid_dim else Self.grid.GRID_DIM
        BLOCK_SHAPE = block_dim.value() if block_dim else Self.grid.BLOCK_SHAPE
        deviceContext.enqueue_function(compiled_func,density,velocity,f.as_immut(),bc.as_immut(),flags.as_immut(),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)


    @staticmethod
    def get_Q_criterion[
        f_dtype:DType,
        float_dtype:DType,
        Q_origin:Origin[mut=True],
        f_layout_type:TensorLayout,
        bc_layout_type:TensorLayout,
        flags_layout_type:TensorLayout,//
        ](
        deviceContext:DeviceContext,
        compiled_func:Self.compiled_Q_criterion_func_type,

        Q:TileTensor[float_dtype,type_of(Self.Q_criterion_layout),Q_origin],
        f:TileTensor[f_dtype,f_layout_type,_],
        bc:TileTensor[float_dtype,bc_layout_type,_],
        flags:TileTensor[DType.uint8,flags_layout_type,_],
        velocity:TileTensor[float_dtype,type_of(Self.velocity_layout),_],
        *,
        block_dim:Optional[Tuple[Int,Int,Int]] = None,
        grid_dim:Optional[Tuple[Int,Int,Int]] = None,
        GPU:Bool = True
        ) raises:

        GRID_DIM = grid_dim.value() if grid_dim else Self.grid.GRID_DIM
        BLOCK_SHAPE = block_dim.value() if block_dim else Self.grid.BLOCK_SHAPE
        deviceContext.enqueue_function(compiled_func,Q,f.as_immut(),bc.as_immut(),flags.as_immut(),velocity.as_immut(),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)


    def velocity_frame[after_odd_step:Bool](mut self,mut assembly:Assembly[Self.grid,Self.config]) raises:
    
        f1,f2,bc,flags = assembly.get_gpu_tensors_for_double_buffer()
        comptime if after_odd_step:
            self.get_rho_and_velocity(self.deviceContext,self.compiled_density_and_velocity_func,self.density.value().gpu(),self.velocity.value().gpu(),f1,bc,flags)
        else:
            self.get_rho_and_velocity(self.deviceContext,self.compiled_density_and_velocity_func,self.density.value().gpu(),self.velocity.value().gpu(),f2,bc,flags)
    
    def Q_criterion_frame[after_odd_step:Bool](mut self,mut assembly:Assembly[Self.grid,Self.config]) raises:

        f1,f2,bc,flags = assembly.get_gpu_tensors_for_double_buffer()
        comptime if after_odd_step:
            self.get_Q_criterion(self.deviceContext,self.compiled_Q_criterion_func,self.Q_criterion.value().gpu(),f1,bc,flags,self.velocity.value().gpu())
        else:
            self.get_Q_criterion(self.deviceContext,self.compiled_Q_criterion_func,self.Q_criterion.value().gpu(),f2,bc,flags,self.velocity.value().gpu())


    


