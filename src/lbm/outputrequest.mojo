"""Defines `OutputRequest` for computing and storing reduced output fields.

`OutputRequest` compiles the density/velocity and Q-criterion extraction
kernels, allocates output buffers with row-major layouts for convenient
NumPy viewing, and provides methods to extract a frame from the GPU buffers.
"""
# from layout import TileTensor, coord,CoordLike,ComptimeInt
from layout.tile_layout import Layout,row_major,Coord,TensorLayout,col_major
from src.lbm.constants import Lbm_methods,Collisions
from src.lbm import LBM_Grid,LBM_Config,TiledLayouts,calculate_rho_and_velocity
from src.lbm.output import calculate_rho_and_velocity,calculate_Q_criterion
# from src.lbm.output.velocity import calculate_rho_and_velocity_temp
# from src.lbm.output.Q_criterion import calculate_Q_criterion_temp
from src.lbm import kernels
from layout import TileTensor
# from layout import Idx
from std.gpu.host import DeviceContext,DeviceFunction
from src.lbm import GridLike,ConfigLike,Assembly
from src.utils import ContextTileTensor,Vector
from std.python import Python, PythonObject

struct OutputRequest[grid_:LBM_Grid,config_:LBM_Config](Movable):
    """Holds compiled output kernels and buffers for density, velocity, and Q-criterion.

    Uses row-major output layouts for straightforward NumPy viewing.
    """
    comptime grid = Self.grid_
    comptime float_dtype = Self.grid.float_dtype
    comptime config = Self.config_
    comptime lbm_method = Self.config.lbm_method
    comptime layouts = Self.grid.layouts

    comptime velocity_layout = Self.layouts.create_col_major_rank_4_layout[Self.grid.D]()
    comptime density_layout = Self.layouts.create_col_major_rank_3_layout()
    comptime Q_criterion_layout = Self.layouts.create_col_major_rank_3_layout()

    comptime output_density_and_velocity_func = calculate_rho_and_velocity[
        type_of(Self.layouts.f_layout),
        type_of(Self.layouts.bc_layout),
        type_of(Self.layouts.flag_layout),
        type_of(Self.density_layout),
        type_of(Self.velocity_layout),
        Self.grid,
        Self.config,
        ]

    comptime output_Q_criterion_func = calculate_Q_criterion[
        type_of(Self.Q_criterion_layout),
        type_of(Self.layouts.f_layout),
        type_of(Self.layouts.bc_layout),
        type_of(Self.layouts.flag_layout),
        type_of(Self.velocity_layout),
        Self.grid,
        Self.config,
    ]

    var velocity: Optional[ContextTileTensor[Self.float_dtype,type_of(Self.velocity_layout)]]
    """Optional buffer for the velocity field (rank 4, row-major)."""
    var density: Optional[ContextTileTensor[Self.float_dtype,type_of(Self.density_layout)]]
    """Optional buffer for the density field (rank 3, row-major)."""
    var Q_criterion:Optional[ContextTileTensor[Self.float_dtype,type_of(Self.Q_criterion_layout)]]
    """Optional buffer for the Q-criterion field (rank 3, row-major)."""

    var deviceContext:DeviceContext
    """The device context used for kernel compilation and buffer allocation."""
    var unitSystem:UnitSystem[Self.float_dtype,Self.grid.D]
    """The unit system for converting lattice to physical units."""
    



    def __init__(out self,deviceContext:DeviceContext,unitSystem:UnitSystem[Self.float_dtype,Self.grid.D],*,velocity:Bool = True,density:Bool = True,Q_criterion:Bool = False) raises:
        """Constructs an `OutputRequest` and compiles the extraction kernels.

        Args:
            deviceContext: The device context for kernel compilation.
            unitSystem: The unit system for unit conversion.
            velocity: When `True`, allocate the velocity output buffer
                (defaults to `True`).
            density: When `True`, allocate the density output buffer
                (defaults to `True`).
            Q_criterion: When `True`, allocate the Q-criterion output buffer
                (defaults to `False`).

        Raises:
            Error: If `Q_criterion` is `True` but `velocity` is `False`.
        """

        self.deviceContext = deviceContext
        self.velocity = None
        self.density = None
        self.Q_criterion = None
        self.unitSystem = unitSystem

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
        density:TileTensor[float_dtype,type_of(Self.density_layout),density_origin],
        velocity:TileTensor[float_dtype,type_of(Self.velocity_layout),velocity_origin],
        f:TileTensor[f_dtype,f_layout_type,_],
        bc:TileTensor[float_dtype,bc_layout_type,_],
        flags:TileTensor[DType.uint8,flags_layout_type,_],
        *,
        block_dim:Optional[Tuple[Int,Int,Int]] = None,
        grid_dim:Optional[Tuple[Int,Int,Int]] = None,
        ) raises:
        GRID_DIM = grid_dim.value() if grid_dim else Self.grid.GRID_DIM
        BLOCK_SHAPE = block_dim.value() if block_dim else Self.grid.BLOCK_SHAPE
        deviceContext.enqueue_function[Self.output_density_and_velocity_func](density,velocity,f.as_immut(),bc.as_immut(),flags.as_immut(),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)


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
        Q:TileTensor[float_dtype,type_of(Self.Q_criterion_layout),Q_origin],
        f:TileTensor[f_dtype,f_layout_type,_],
        bc:TileTensor[float_dtype,bc_layout_type,_],
        flags:TileTensor[DType.uint8,flags_layout_type,_],
        velocity:TileTensor[float_dtype,type_of(Self.velocity_layout),_],
        *,
        block_dim:Optional[Tuple[Int,Int,Int]] = None,
        grid_dim:Optional[Tuple[Int,Int,Int]] = None,
        ) raises:
        GRID_DIM = grid_dim.value() if grid_dim else Self.grid.GRID_DIM
        BLOCK_SHAPE = block_dim.value() if block_dim else Self.grid.BLOCK_SHAPE
        deviceContext.enqueue_function[Self.output_Q_criterion_func](Q,f.as_immut(),bc.as_immut(),flags.as_immut(),velocity.as_immut(),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)


    def velocity_frame[after_odd_step:Bool](mut self,mut assembly:Assembly[Self.grid,Self.config]) raises:
        """Computes and stores the density and velocity fields for the frame.

        Loads `f` from either `f1` (after an odd step) or `f2` (after an
        even step) and writes the results into the preallocated `density`
        and `velocity` buffers.

        Parameters:
            after_odd_step: When `True`, read from `f1`; otherwise from
                `f2`.

        Args:
            assembly: The `Assembly` providing the GPU buffers.
        """
        f1,f2,bc,flags = assembly.get_gpu_tensors_for_double_buffer()
        
        comptime if after_odd_step:
            self.get_rho_and_velocity(self.deviceContext,self.density.value().gpu(),self.velocity.value().gpu(),f1,bc,flags)
        else:
            self.get_rho_and_velocity(self.deviceContext,self.density.value().gpu(),self.velocity.value().gpu(),f2,bc,flags)
    

    def Q_criterion_frame[after_odd_step:Bool](mut self,mut assembly:Assembly[Self.grid,Self.config]) raises:
        """Computes and stores the Q-criterion field for the frame.

        Loads `f` from either `f1` (after an odd step) or `f2` (after an
        even step) and writes the Q-criterion into the preallocated
        `Q_criterion` buffer. Requires that the velocity field has already
        been computed.

        Parameters:
            after_odd_step: When `True`, read from `f1`; otherwise from
                `f2`.

        Args:
            assembly: The `Assembly` providing the GPU buffers.
        """
        f1,f2,bc,flags = assembly.get_gpu_tensors_for_double_buffer()

        comptime if after_odd_step:
            self.get_Q_criterion(self.deviceContext,self.Q_criterion.value().gpu(),f1,bc,flags,self.velocity.value().gpu())
        else:
            self.get_Q_criterion(self.deviceContext,self.Q_criterion.value().gpu(),f2,bc,flags,self.velocity.value().gpu())


    
    def velocity_as_numpy(mut self,convert_from_lattice_units:Bool,squeeze:Bool =True) raises -> PythonObject:
        """Returns the velocity buffer as a NumPy array.

        Converts the row-major GPU buffer to a NumPy array shaped
        `(nx, ny, nz, D)` with Fortran ordering, optionally converting
        from lattice to physical units and squeezing unit dimensions.

        Args:
            convert_from_lattice_units: When `True`, multiply by the
                velocity conversion factor.
            squeeze: When `True`, squeeze unit dimensions (defaults to
                `True`).

        Returns:
            The velocity field as a NumPy array.
        """

        # Weshould use reflection to consoldate this
        var shape = Python.tuple(Self.grid.shape[0],Self.grid.shape[1],Self.grid.shape[2],Self.grid.D)
        velocity_buffer = ((self.velocity.value()).buffer_to_numpy()).reshape(shape,order = 'F')

        if convert_from_lattice_units:
            velocity_buffer *= self.unitSystem.U.C_lat_to_phys()
        
        if squeeze:
            return velocity_buffer.squeeze()
        else:
            return velocity_buffer


    def Q_criterion_as_numpy(mut self,convert_from_lattice_units:Bool,squeeze:Bool =True) raises -> PythonObject:
        """Returns the Q-criterion buffer as a NumPy array.

        Converts the row-major GPU buffer to a NumPy array shaped
        `(nx, ny, nz)` with Fortran ordering, optionally converting from
        lattice to physical units and squeezing unit dimensions.

        Args:
            convert_from_lattice_units: When `True`, multiply by the
                Q-criterion conversion factor.
            squeeze: When `True`, squeeze unit dimensions (defaults to
                `True`).

        Returns:
            The Q-criterion field as a NumPy array.
        """
        
        var shape = Python.tuple(Self.grid.shape[0],Self.grid.shape[1],Self.grid.shape[2])
        # shape.append(Self.grid.D)

        Q_criterion_np = ((self.Q_criterion.value()).buffer_to_numpy()).reshape(shape,order = 'F')
        
        if convert_from_lattice_units:
            Q_criterion_np *= (self.unitSystem.U.C_lat_to_phys()/self.unitSystem.L.C_lat_to_phys())
        
        if squeeze:
            return Q_criterion_np.squeeze()
        else:
            return Q_criterion_np


    
