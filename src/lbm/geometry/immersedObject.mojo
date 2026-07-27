"""Defines `ImmersedObject`, which collects fluid boundary nodes for force kernels.

`ImmersedObject` wraps the linear memory indices of the fluid nodes adjacent
to an immersed solid and provides a method for uploading those indices into
a `ContextTileTensor` so the GPU force kernel can consume them.
"""
from std.memory import Pointer
from layout import TileTensor,CoordLike
from layout.tile_layout import TensorLayout,Layout
from .primatives import add_box,add_sphere,get_sphere_boundary_indices
from src.lbm import LBM_Grid,Lattice
from layout import TileTensor,row_major,col_major,coord
from src.utils import ContextTileTensor
from std.gpu.host import DeviceContext
from src.utils import Vector


struct RigidImmersedObject[
    grid:LBM_Grid
    ]():
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
    

    var fluid_boundary_list:List[Self.Int_Scalar]
    """The linear memory indices of the fluid nodes adjacent to the solid."""

    var center_of_mass:Vector[Self.float_dtype,Self.grid.D]
    """The center of mass of the immersed object."""
    var mass:Scalar[Self.float_dtype]
    """The mass of the immersed object."""

    def __init__(out self):
        """Constructs an empty `ImmersedObject` with default values.

        Sets mass to 1, center of mass to zero, and the fluid boundary list
        to empty.
        """
        self.fluid_boundary_list = []
        self.mass = 1.
        self.center_of_mass = Vector[Self.float_dtype,Self.grid.D](fill=0.)

    def __init__(out self,var fluid_boundary_list:List[Self.Int_Scalar],mass:Scalar[Self.float_dtype] = 1.,center_of_mass:Vector[Self.float_dtype,Self.grid.D] = Vector[Self.float_dtype,Self.grid.D](fill=0.), ):
        """Constructs an `ImmersedObject` from pre-computed boundary indices.

        Args:
            fluid_boundary_list: The list of linear fluid boundary indices
                (ownership is transferred).
            mass: The mass of the object (defaults to 1).
            center_of_mass: The center of mass (defaults to zero).
        """
        self.fluid_boundary_list = fluid_boundary_list^
        self.mass = mass
        self.center_of_mass = center_of_mass

    def add_sphere[FlagLayoutType:TensorLayout,flag_origin:Origin[mut=True]](
    mut self,
    flags:TileTensor[DType.uint8,FlagLayoutType,flag_origin],
    center:List[Scalar[Self.float_dtype]],
    radius:Scalar[Self.float_dtype],
    ) raises :
        """Embeds a sphere into the grid and records its fluid boundary nodes.

        Parameters:
            FlagLayoutType: The compile-time layout of `flags`.
            flag_origin: The origin of the `flags` tile tensor.

        Args:
            flags: The `uint8` tile tensor labeling each node.
            center: The physical `(x, y, z)` coordinates of the sphere center.
            radius: The physical radius of the sphere.
        """
        self.fluid_boundary_list = get_sphere_boundary_indices[self.grid](flags,center,radius)

    def to_ContextTileTensor(
        self,
        deviceContext:DeviceContext
        )
        raises
        -> ContextTileTensor[Self.int_dtype,type_of( row_major(coord[Self.int_dtype]((1,))) ) ]:

        """Uploads the fluid boundary indices to a 1D `ContextTileTensor`.

        Args:
            deviceContext: The device context that owns the buffers.

        Returns:
            A `ContextTileTensor` holding the linear fluid boundary indices.
        """
        N = Int(len(self.fluid_boundary_list))
        layout = row_major( coord[Self.int_dtype]((N,) ))
        out = ContextTileTensor[Self.int_dtype](deviceContext,layout)
        out.cpu_buffer().enqueue_copy_from(src = Span(self.fluid_boundary_list))
        return out^ # Must take ownership of ContextTileTensor
