
"""Defines `Assembly`, which manages the host/device LBM buffer pair.

`Assembly` allocates the flag, `bc`, and one or two `f` buffers tied to a
`DeviceContext`, provides accessor methods for the GPU and CPU tensors, and
exposes convenience methods for initializing the distribution function and
setting boundary conditions.
"""

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
    """Declares the compile-time interface for an LBM buffer assembly.

    Conforming types expose the LBM method name, grid, `LBM_Config`,
    and the float and distribution function dtypes.
    """

    comptime lbm_method:StaticString
    """The LBM streaming method."""
    comptime gridType:GridLike
    """The compile-time type of the grid."""
    comptime grid:Self.gridType
    """The compile-time grid instance."""
    comptime config:LBM_Config[Self.lbm_method]
    """The compile-time `LBM_Config`."""
    comptime float_dtype:DType
    """The `DType` of the float-valued fields."""
    comptime f_dtype:DType
    """The storage `DType` of the distribution function."""

struct Assembly[gridType_:GridLike,//,grid_:gridType_,config_:LBM_Config](AssemblyLike & Movable):
    """Manages the host/device buffers for the flag, `f`, and `bc` fields.

    Allocates paired host and device buffers for the flag, boundary-condition,
    and one or two distribution-function fields. Provides accessor methods for
    the GPU and CPU tensor views and convenience methods for distribution
    initialization and exterior-wall boundary conditions.

    Parameters:
        gridType_: The compile-time `GridLike` type.
        grid_: The compile-time grid instance.
        config_: The compile-time `LBM_Config`.
    """

    comptime gridType = Self.gridType_
    comptime grid = Self.grid_
    comptime config = Self.config_
    comptime lbm_method = Self.config.lbm_method
    comptime layouts = Self.grid.layouts
    comptime float_dtype = Self.grid.float_dtype
    comptime f_dtype = Self.config.set_f_dtype(Self.float_dtype)

    var f: ContextTileTensor[Self.f_dtype,type_of(Self.layouts.f_layout)]
    """The primary distribution function buffer."""
    var f2: Optional[ContextTileTensor[Self.f_dtype,type_of(Self.layouts.f_layout)]]
    """The secondary distribution function buffer (double-buffer only)."""
    var bc: ContextTileTensor[Self.float_dtype,type_of(Self.layouts.bc_layout)]
    """The boundary-condition buffer."""
    var flags: ContextTileTensor[DType.uint8,type_of(Self.layouts.flag_layout)]
    """The flag buffer (`uint8` per node)."""
    var unitSystem:UnitSystem[Self.float_dtype,Self.grid.D]
    """The unit system used for physical-to-lattice conversion."""
    var deviceContext:DeviceContext
    """The device context used for buffer allocation and synchronization."""

    def __init__(out self,deviceContext:DeviceContext,unitSystem:UnitSystem[Self.float_dtype,Self.grid.D]) raises:
        """Constructs an `Assembly` and allocates the required buffers.

        Allocates `f`, `bc`, and `flags` buffers. Allocates `f2` only when
        the LBM method is not esoteric-pull.

        Args:
            deviceContext: The device context for buffer allocation.
            unitSystem: The unit system for physical-to-lattice conversion.
        """

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
        """Returns the CPU `TileTensor` views for the esoteric-pull buffers.

        Returns:
            A tuple of `(f, bc, flags)` as CPU tile tensors.

        Raises:
            Error: If the LBM method is not esoteric-pull.
        """

        if self.lbm_method != Lbm_methods.ESOTERIC_PULL:
            raise Error('This function is only valid if the lbm method is Double Buffer')

        return (self.f.cpu(),self.bc.cpu(),self.flags.cpu())

    def get_gpu_tensors_for_esoteric_pull(mut self) 
        raises -> Tuple[
            TileTensor[Self.f_dtype,type_of(Self.layouts.f_layout),origin_of(self.f._gpu_buffer)],
            TileTensor[Self.float_dtype,type_of(Self.layouts.bc_layout),origin_of(self.bc._gpu_buffer)],
            TileTensor[DType.uint8,type_of(Self.layouts.flag_layout),origin_of(self.flags._gpu_buffer)],
            ]:
        """Returns the GPU `TileTensor` views for the esoteric-pull buffers.

        Returns:
            A tuple of `(f, bc, flags)` as GPU tile tensors.

        Raises:
            Error: If the LBM method is not esoteric-pull.
        """

        if self.lbm_method != Lbm_methods.ESOTERIC_PULL:
            raise Error('This function is only valid if the lbm method is Double Buffer')

        return (self.f.gpu(),self.bc.gpu(),self.flags.gpu())


    def get_gpu_tensors_for_double_buffer(mut self) 
        raises -> Tuple[
            TileTensor[Self.f_dtype,type_of(Self.layouts.f_layout),origin_of(self.f._gpu_buffer)],
            TileTensor[Self.f_dtype,type_of(Self.layouts.f_layout),origin_of(self.f2.value()._gpu_buffer)],
            TileTensor[Self.float_dtype,type_of(Self.layouts.bc_layout),origin_of(self.bc._gpu_buffer)],
            TileTensor[DType.uint8,type_of(Self.layouts.flag_layout),origin_of(self.flags._gpu_buffer)],
            ]:
        """Returns the GPU `TileTensor` views for the double-buffer scheme.

        Returns:
            A tuple of `(f, f2, bc, flags)` as GPU tile tensors.

        Raises:
            Error: If the LBM method is not double-buffer.
            Error: If `f2` has not been allocated.
        """

        if self.lbm_method != Lbm_methods.DOUBLE_BUFFER:
            raise Error('This function is only valid if the lbm method is Double Buffer')

        if not self.f2:
            raise Error('with LBM == Double Buffer f2 should have a value inside it')
        else:
            return (self.f.gpu(),self.f2.value().gpu(),self.bc.gpu(),self.flags.gpu())

    def get_cpu_tensors_for_double_buffer(mut self) 
        raises -> Tuple[
            TileTensor[Self.f_dtype,type_of(Self.layouts.f_layout),origin_of(self.f._cpu_buffer)],
            TileTensor[Self.f_dtype,type_of(Self.layouts.f_layout),origin_of(self.f2.value()._cpu_buffer)],
            TileTensor[Self.float_dtype,type_of(Self.layouts.bc_layout),origin_of(self.bc._cpu_buffer)],
            TileTensor[DType.uint8,type_of(Self.layouts.flag_layout),origin_of(self.flags._cpu_buffer)],
            ]:
        """Returns the CPU `TileTensor` views for the double-buffer scheme.

        Returns:
            A tuple of `(f, f2, bc, flags)` as CPU tile tensors.

        Raises:
            Error: If the LBM method is not double-buffer.
            Error: If `f2` has not been allocated.
        """

        if self.lbm_method != Lbm_methods.DOUBLE_BUFFER:
            raise Error('This function is only valid if the lbm method is Double Buffer')

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
        """Initializes the distribution function from a velocity field or at rest.

        Parameters:
            u: Optional velocity function; required if `deriv_u` is supplied
                (defaults to `None`).
            deriv_u: Optional velocity gradient function for the
                non-equilibrium correction (defaults to `None`).

        Args:
            rho: The lattice density to initialize with.
            tau: The relaxation time used by the non-equilibrium correction.
        """

        comptime assert True if (u and deriv_u) or (u) else False,'if deriv u is passed, u must also be passed'
        comptime if u:
            initialize_f_from_func[Self.grid,Self.config,u=u.value(),deriv_u = deriv_u](self.f.cpu(),rho,tau,self.unitSystem)
        else:
            initialize_fluid_at_rest[Self.grid,Self.config](self.f.cpu())


    def initialize_f_at_rest(mut self) raises:
        """Initializes `f` with the equilibrium distribution for a fluid at rest."""
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
        u:List[Scalar[self.float_dtype]] = [],
        rho:Scalar[self.float_dtype] = nan[self.float_dtype](),
        *,
        in_lattice_units:Bool,

        ) raises:
        """Applies a boundary condition to one exterior wall.

        Parameters:
            u_func: Optional spatially varying velocity function (defaults to
                `None`).

        Args:
            side: The wall to write, one of `'-X'`, `'+X'`, `'-Y'`, `'+Y'`,
                `'-Z'`, `'+Z'`.
            boundary_type: The flag value to write at the target wall.
            u: The velocity components for the wall (defaults to the empty
                list).
            rho: The density for the wall (defaults to `NaN`).
            in_lattice_units: When `True`, velocities and density are already
                in lattice units.
        """

        var unitSystem:Optional[type_of(self.unitSystem)]
        if in_lattice_units:
            unitSystem = None
        else:
            unitSystem = self.unitSystem

        comptime if u_func: # To Do unify these 2 functions into one
            set_exterior_walls_with_func[self.grid,self.config, u = u_func.value()](self.flags.cpu(),self.bc.cpu(),side,boundary_type,unitSystem,rho)
        else:
            set_exterior_walls[self.grid,self.config](self.flags.cpu(),self.bc.cpu(),side,boundary_type,u,rho,unitSystem)


