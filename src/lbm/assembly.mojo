
# from layout import TileTensor, coord,CoordLike,ComptimeInt
from src.lbm.constants import Lbm_methods,Collisions
from src.lbm import LBM_Grid,LBM_Config,TiledLayouts,calculate_rho_and_velocity
from src.lbm import kernels
from layout import TileTensor
# from layout import Idx
from std.gpu.host import DeviceContext,DeviceFunction
from src.lbm import GridLike,ConfigLike
from src.utils import ContextTileTensor,Vector
from src.lbm import UnitSystem
from src.lbm.preprocess import initialize_fluid_at_rest,initialize_f_from_func,set_exterior_walls,set_exterior_walls_with_func
from std.utils.numerics import nan,isnan

trait AssemblyLike:
    comptime lbm_method:StaticString
    comptime gridType:GridLike
    comptime grid:Self.gridType
    comptime config:LBM_Config[Self.lbm_method]
    comptime float_dtype:DType
    comptime f_dtype:DType

struct Assembly[gridType_:GridLike,//,grid_:gridType_,config_:LBM_Config](AssemblyLike & Movable):
    comptime gridType = Self.gridType_
    comptime grid = Self.grid_
    comptime config = Self.config_
    comptime lbm_method = Self.config.lbm_method
    comptime layouts = Self.grid.layouts
    comptime float_dtype = Self.grid.float_dtype
    comptime f_dtype = Self.config.set_f_dtype(Self.float_dtype)

    var f: ContextTileTensor[Self.f_dtype,type_of(Self.layouts.f_layout)]
    var f2: Optional[ContextTileTensor[Self.f_dtype,type_of(Self.layouts.f_layout)]]
    var bc: ContextTileTensor[Self.float_dtype,type_of(Self.layouts.bc_layout)]
    var flags: ContextTileTensor[DType.uint8,type_of(Self.layouts.flag_layout)]
    var unitSystem:UnitSystem[Self.float_dtype,Self.grid.D]
    var deviceContext:DeviceContext

    def __init__(out self,deviceContext:DeviceContext,unitSystem:UnitSystem[Self.float_dtype,Self.grid.D]) raises:

        self.deviceContext = deviceContext
        self.unitSystem = unitSystem
        self.flags = ContextTileTensor[DType.uint8](deviceContext,self.layouts.flag_layout)
        self.bc = ContextTileTensor[self.float_dtype](deviceContext,self.layouts.bc_layout)
        self.f = ContextTileTensor[self.f_dtype](deviceContext,self.layouts.f_layout)

        if Self.lbm_method == Lbm_methods.ESOTERIC_PULL:
            self.f2 = None
        else:
            self.f2 = ContextTileTensor[self.f_dtype](deviceContext,self.layouts.f_layout)

    def get_cpu_tensors_for_esoteric_pull(mut self) 
        raises -> Tuple[
            TileTensor[Self.f_dtype,type_of(Self.layouts.f_layout),origin_of(self.f._cpu_buffer)],
            TileTensor[Self.float_dtype,type_of(Self.layouts.bc_layout),origin_of(self.bc._cpu_buffer)],
            TileTensor[DType.uint8,type_of(Self.layouts.flag_layout),origin_of(self.flags._cpu_buffer)],
            ]:

        if self.lbm_method != Lbm_methods.ESOTERIC_PULL:
            raise Error('This function is only valid if the lbm method is Esoteric Pull')

        return (self.f.cpu(),self.bc.cpu(),self.flags.cpu())

    def get_gpu_tensors_for_esoteric_pull(mut self) 
        raises -> Tuple[
            TileTensor[Self.f_dtype,type_of(Self.layouts.f_layout),origin_of(self.f._gpu_buffer)],
            TileTensor[Self.float_dtype,type_of(Self.layouts.bc_layout),origin_of(self.bc._gpu_buffer)],
            TileTensor[DType.uint8,type_of(Self.layouts.flag_layout),origin_of(self.flags._gpu_buffer)],
            ]:

        if self.lbm_method != Lbm_methods.ESOTERIC_PULL:
            raise Error('This function is only valid if the lbm method is Esoteric Pull')

        return (self.f.gpu(),self.bc.gpu(),self.flags.gpu())


    def get_gpu_tensors_for_double_buffer(mut self) 
        raises -> Tuple[
            TileTensor[Self.f_dtype,type_of(Self.layouts.f_layout),origin_of(self.f._gpu_buffer)],
            TileTensor[Self.f_dtype,type_of(Self.layouts.f_layout),origin_of(self.f2.value()._gpu_buffer)],
            TileTensor[Self.float_dtype,type_of(Self.layouts.bc_layout),origin_of(self.bc._gpu_buffer)],
            TileTensor[DType.uint8,type_of(Self.layouts.flag_layout),origin_of(self.flags._gpu_buffer)],
            ]:

        if self.lbm_method != Lbm_methods.DOUBLE_BUFFER:
            raise Error('This function is only valid if the lbm method is Esoteric Pull')

        if not self.f2:
            raise Error('with LBM == Double Buffer f2 should have a value inside it')

        return (self.f.gpu(),self.f2.value().gpu(),self.bc.gpu(),self.flags.gpu())

    def get_cpu_tensors_for_double_buffer(mut self) 
        raises -> Tuple[
            TileTensor[Self.f_dtype,type_of(Self.layouts.f_layout),origin_of(self.f._cpu_buffer)],
            TileTensor[Self.f_dtype,type_of(Self.layouts.f_layout),origin_of(self.f2.value()._cpu_buffer)],
            TileTensor[Self.float_dtype,type_of(Self.layouts.bc_layout),origin_of(self.bc._cpu_buffer)],
            TileTensor[DType.uint8,type_of(Self.layouts.flag_layout),origin_of(self.flags._cpu_buffer)],
            ]:

        if self.lbm_method != Lbm_methods.DOUBLE_BUFFER:
            raise Error('This function is only valid if the lbm method is Esoteric Pull')

        if not self.f2:
            raise Error('with LBM == Double Buffer f2 should have a value inside it')

        return (self.f.cpu(),self.f2.value().cpu(),self.bc.cpu(),self.flags.cpu())

    
    def initialize_f[
        *,
        u: Optional[def[float_dtype:DType,D:Int]
        (Scalar[float_dtype],Scalar[float_dtype],Scalar[float_dtype],mut Vector[float_dtype,D])
        capturing] = None,
        deriv_u: Optional[def[float_dtype:DType,D:Int]
        (Scalar[float_dtype],Scalar[float_dtype],Scalar[float_dtype],Vector[float_dtype,D])
        capturing -> List[Vector[float_dtype,D]]] = None,
        ](mut self,rho:Scalar[self.float_dtype],tau:Scalar[self.float_dtype]) raises
        :

        comptime assert True if (u and deriv_u) or (u) else False,'if deriv u is passed, u must also be passed'
        comptime if u:
            initialize_f_from_func[Self.grid,Self.config,u=u.value(),deriv_u = deriv_u](self.f.cpu(),rho,tau,self.unitSystem)
        else:
            initialize_fluid_at_rest[Self.grid,Self.config](self.f.cpu())

    def set_exterior_walls[
        *,
        u_func: Optional[def[float_dtype:DType,D:Int]
        (Scalar[float_dtype],Scalar[float_dtype],Scalar[float_dtype],mut InlineArray[Scalar[float_dtype],D])
        capturing] = None
        ](
        mut self,
        side:String,
        boundary_type:Scalar[DType.uint8],
        value_in_lattice_coords:Bool,
        u:List[Scalar[self.float_dtype]] = [],
        rho:Scalar[self.float_dtype] = nan[self.float_dtype](),
        ) raises:

        var unitSystem:Optional[type_of(self.unitSystem)]
        if value_in_lattice_coords:
            unitSystem = None
        else:
            unitSystem = self.unitSystem

        comptime if u_func: # To Do unify these 2 functions into one
            set_exterior_walls_with_func[self.grid,self.config, u = u_func.value()](self.flags.cpu(),self.bc.cpu(),side,boundary_type,unitSystem,rho)
        else:
            set_exterior_walls[self.grid,self.config](self.flags.cpu(),self.bc.cpu(),side,boundary_type,u,rho,unitSystem)


