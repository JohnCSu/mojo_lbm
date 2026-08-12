"""Defines `ImmersedObject`, which collects fluid boundary nodes for force kernels.

`ImmersedObject` wraps the linear memory indices of the fluid nodes adjacent
to an immersed solid and provides a method for uploading those indices into
a `ContextTileTensor` so the GPU force kernel can consume them.
"""
from std.memory import Pointer
from layout import TileTensor,CoordLike
from layout.tile_layout import TensorLayout,Layout
from .primatives import add_box,add_sphere,get_sphere_boundary_indices
from src.lbm import LBM_Grid,Lattice,LBM_Config,UnitSystem
from layout import TileTensor,row_major,col_major,coord
from src.utils import ContextTileTensor
from std.gpu.host import DeviceContext
from src.utils import Vector
from src.utils.runtimeLayouts import RuntimeColMajor1DType,RuntimeColMajor2DType,col_major1D,col_major2D
from .interpolated_BB import linkwise_bounceback_kernel
from src.lbm.constants import Bounceback_method,LBM_method
from std.python import Python, PythonObject

struct RigidStationaryObject[
    grid: LBM_Grid,
    lbm_method:LBM_method,
    config:LBM_Config[lbm_method],
    ](Movable):
    """Collects the fluid boundary nodes adjacent to an immersed solid.

    Stores the linear memory indices of the fluid nodes that touch the solid
    surface so the drag kernel can iterate over them. The indices are
    produced by `get_sphere_boundary_indices` and uploaded to the GPU via
    `to_ContextTileTensor`.

    Parameters:
        grid: The compile-time `LBM_Grid` describing the domain.
    """
    comptime float_dtype = Self.grid.float_dtype
    comptime int_dtype = Self.grid.int_dtype
    comptime Int_Scalar = Scalar[Self.int_dtype]

    var unique_fluid_ids:List[Self.Int_Scalar]
    """The linear memory indices of the fluid nodes adjacent to the solid."""

    var fluid_boundaries_list:List[Self.Int_Scalar]
    var lattice_links_list:List[Self.Int_Scalar]
    var link_distances_list:List[Scalar[Self.float_dtype]]
    
    var fluid_boundaries:ContextTileTensor[Self.int_dtype,RuntimeColMajor1DType]
    var lattice_links:ContextTileTensor[Self.int_dtype,RuntimeColMajor1DType]
    var link_distances:ContextTileTensor[Self.float_dtype,RuntimeColMajor1DType]
    var link_forces:ContextTileTensor[Self.float_dtype,RuntimeColMajor2DType]
    var deviceContext:DeviceContext
    var units:Optional[UnitSystem[Self.float_dtype,Self.grid.D]]
    # var bounceback_method
    def __init__(
        out self,
        deviceContext:DeviceContext,
        var unique_fluid_ids:List[Self.Int_Scalar],
        var fluid_boundaries_list:List[Self.Int_Scalar],
        var lattice_links_list:List[Self.Int_Scalar],
        var link_distances_list:List[Scalar[Self.float_dtype]],
        var units:Optional[UnitSystem[Self.float_dtype,Self.grid.D]] = None
        )
        raises:
        """Constructs an `ImmersedObject` from pre-computed boundary indices.

        Args:
            deviceContext: The device context used for buffer allocation.
            fluid_boundary_list: The list of linear fluid boundary indices
                (ownership is transferred).
            links_fluid_list: The list of fluid node indices for each link
                (ownership is transferred).
            links_direction_list: The list of direction indices for each link
                (ownership is transferred).
            links_q_dist: The list of quarter-way distances for each link
                (ownership is transferred).
        """
        self.unique_fluid_ids = unique_fluid_ids^
        self.deviceContext = deviceContext
        self.fluid_boundaries_list = fluid_boundaries_list^
        self.lattice_links_list =lattice_links_list^
        self.link_distances_list= link_distances_list^
        self.units = units

        if len(self.fluid_boundaries_list) != len(self.lattice_links_list) or len(self.link_distances_list) != len(self.fluid_boundaries_list):
            raise Error('links_fluid_list, links_direction_list and links_q_dist, should all be the same length')
        
        self.fluid_boundaries = self.list_to_1D_ContextTileTensor(deviceContext,self.fluid_boundaries_list)
        self.lattice_links = self.list_to_1D_ContextTileTensor(deviceContext,self.lattice_links_list)
        self.link_distances = self.list_to_1D_ContextTileTensor(deviceContext,self.link_distances_list)
        n = self.fluid_boundaries.size()
        self.link_forces = self.create_NxM_ContextTileTensor[self.float_dtype](deviceContext,n,Self.grid.D)

    @staticmethod
    def list_to_1D_ContextTileTensor[dtype:DType,//]
        (
        deviceContext:DeviceContext,
        ls:List[Scalar[dtype]]
        ) raises
        -> ContextTileTensor[dtype,RuntimeColMajor1DType]
        :
        N = Int(len(ls))
        layout  = col_major1D(N)
        out = ContextTileTensor[dtype](deviceContext,layout)
        out.cpu_buffer().enqueue_copy_from(src = Span(ls))
        return out^ # Must take ownership of ContextTileTensor

    @staticmethod
    def create_NxM_ContextTileTensor[dtype:DType](
        deviceContext:DeviceContext,
        n:Int,
        m:Int,
        ) raises
        -> ContextTileTensor[dtype,RuntimeColMajor2DType]:

        layout  = col_major2D(n,m)
        out = ContextTileTensor[dtype](deviceContext,layout,fill = Scalar[dtype](0))
        return out^ # Must take ownership of ContextTileTensor


    @staticmethod
    def from_stl(filename:String) raises:
        pass

    def num_links(self) -> Int:
        return self.fluid_boundaries.size()

    def bounceback[
        FLayoutType:TensorLayout,
        FlagLayoutType:TensorLayout,
        f_dtype:DType,
        //,
        bounceback:Bounceback_method
        ](
        mut self,
        f_in:TileTensor[f_dtype,FLayoutType,MutAnyOrigin],
        flags:TileTensor[DType.uint8,FlagLayoutType,_],
        compute_force:Bool = True,
        q_min:Scalar[Self.grid.float_dtype] = 0.,
        ) raises:

        comptime kernel = linkwise_bounceback_kernel[bounceback,FLayoutType,FlagLayoutType,Self.grid,Self.config]
        self.deviceContext.enqueue_function[kernel](
            f_in,
            self.link_forces.gpu(),
            flags.as_immut(),
            self.fluid_boundaries.gpu().as_immut(),
            self.lattice_links.gpu().as_immut(),
            self.link_distances.gpu().as_immut(),
            Scalar[DType.bool](compute_force),
            q_min,
            grid_dim = self.num_links()//256 + 1,
            block_dim = 256,
            )

    def sum_force(mut self,*,in_lattice_units:Bool = True) raises -> InlineArray[Scalar[Self.float_dtype],Self.grid.D]:
        summed_force = InlineArray[Scalar[Self.float_dtype],Self.grid.D](uninitialized = True)
        comptime float = Scalar[Self.float_dtype]
        np = Python.import_module('numpy')

        force_np = self.link_forces.buffer_to_numpy().reshape(self.num_links(),self.grid.D,order = 'F')
        
        comptime for d in range(self.grid.D):
            F_i = force_np.sum(axis=0)[d]
            if in_lattice_units:
                summed_force[d] = float(py=F_i)
            else:
                summed_force[d] = float(py=self.units.value().force.C_lat_to_phys()*F_i)

        return summed_force















