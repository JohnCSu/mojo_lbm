"""Defines `LBM_Grid` and the `GridLike` trait that describe an LBM domain.

`LBM_Grid` is the compile-time-parameterized description of the simulation
domain: dimension, lattice model, grid shape, tile size, derived block and
grid dimensions for GPU launch, and per-node field sizes. Helper functions
in this module validate the grid shape against its lattice model and compute
the GPU block and grid dimensions from the tile size.
"""
from std.gpu.host import DeviceContext
from layout import TileTensor, LayoutTensor, coord
from layout.tile_layout import Layout, row_major, Coord, TensorLayout
from std.collections import InlineArray
from std.collections import Set, Dict
from src.utils import Vector, ContextTileTensor
from std.utils.numerics import nan, isnan
from .units import UnitSystem
from std.utils import Variant


comptime Int_Or_Tuple_Of_Ints = Variant[Int,Tuple[Int,Int,Int]]

def set_tile_shape(x:Int_Or_Tuple_Of_Ints,D:Int) -> Tuple[Int,Int,Int]:
    if x.isa[Int]():
        tile_size = x[Int]
        return (tile_size,tile_size if D >= 2 else 1, tile_size if D == 3 else 1)
    else:
        return x[Tuple[Int,Int,Int]]


def set_default_block_shape[tile_shape:Tuple[Int,Int,Int],D:Int]() -> Tuple[Int,Int,Int]:
    comptime if tile_shape == (1,1,1):
        if D == 1:
            return (256,1,1)
        elif D == 2:
            return (16,16,1)
        else:
            return (8,8,4)
    else:
        comptime for i in range(3):
            comptime assert tile_shape[i] > 1 if i < D else tile_shape[i] == 1
        return tile_shape


def set_grid_dims[nx:Int,ny:Int,nz:Int,block_shape:Tuple[Int,Int,Int]]() -> Tuple[Int,Int,Int]:
    comptime assert (
                (nx % block_shape[0] == 0 or nx == 1)
            and (ny % block_shape[1] == 0 or ny == 1)
            and (nz % block_shape[2] == 0 or nz == 1)
        ),'The grid shape along nx,ny and nz should be either equal to 1 or divisible by the block shape'
    
    return (nx//block_shape[0],ny//block_shape[1],nz//block_shape[2])


trait GridLike:
    """Declares the compile-time shape and lattice description of an LBM grid.

    Conforming types expose the float and integer dtypes, dimension `D`,
    velocity count `Q`, per-axis extents, tile size, and the compile-time
    `Lattice` used by the solver.
    """

    comptime float_dtype: DType
    comptime int_dtype: DType
    comptime D: Int
    comptime Q: Int
    comptime nx: Int
    comptime ny: Int
    comptime nz: Int
    comptime tile_shape: Tuple[Int,Int,Int]
    comptime lattice: Lattice[
        Self.D, Self.Q, Self.float_dtype, Self.int_dtype
    ]
    comptime shape: InlineArray[Int, 3]
    

struct LBM_Grid[
    float_dtype_: DType,
    int_dtype_: DType,
    D_: Int,
    Q_: Int,
    //,
    lattice_: Lattice[D_, Q_, float_dtype_, int_dtype_],
    nx_: Int,
    ny_: Int,
    nz_: Int,
    tile_shape_:Int_Or_Tuple_Of_Ints,
    
    
](ImplicitlyCopyable & GridLike):
    """Describes an LBM simulation domain and its GPU launch parameters.

    Carries the lattice model, grid shape, and tile size as compile-time
    parameters, and derives the GPU block and grid dimensions, the tile
    counts per axis, and the per-node field sizes (f, velocity, bc) from
    them. The runtime state records the lattice spacing, origin, and
    physical domain extents.

    Parameters:
        float_dtype_: The float `DType` used by the solver.
        int_dtype_: The integer `DType` used for indices and directions.
        D_: The spatial dimension of the grid.
        Q_: The number of discrete velocities per node.
        lattice_: The compile-time `Lattice` for the grid.
        nx_: The number of lattice nodes along `x`.
        ny_: The number of lattice nodes along `y`.
        nz_: The number of lattice nodes along `z`.
        tile_size_: The tile size used for tiled layouts and GPU blocking.
    """
    
    comptime float_dtype: DType = Self.float_dtype_
    comptime int_dtype: DType = Self.int_dtype_
    comptime D: Int = Self.D_
    comptime Q: Int = Self.Q_
    comptime nx: Int = Self.nx_
    comptime ny: Int = Self.ny_
    comptime nz: Int = Self.nz_
    comptime tile_size: Int = 8
    comptime tile_shape = set_tile_shape(Self.tile_shape_,Self.D)

    comptime lattice = Self.lattice_

    # comptime float_dtype:DType = Self.float_dtype
    comptime Float_Scalar = Scalar[Self.float_dtype]
    comptime BLOCK_SHAPE = set_default_block_shape[Self.tile_shape,Self.D]()
    comptime GRID_DIM = set_grid_dims[Self.nx,Self.ny,Self.nz,Self.BLOCK_SHAPE]()
    comptime THREADS_PER_BLOCK = Self.BLOCK_SHAPE[0] * Self.BLOCK_SHAPE[1] * Self.BLOCK_SHAPE[2]

    comptime n_tiles_x = Self.nx // Self.tile_shape[0]
    comptime n_tiles_y = Self.ny // Self.tile_shape[1] if Self.D >= 2 else 1
    comptime n_tiles_z = Self.nz // Self.tile_shape[2] if Self.D == 3 else 1
    comptime x_tile = Self.tile_shape[0]
    comptime y_tile = Self.tile_shape[1]
    comptime z_tile = Self.tile_shape[2]


    var dx: Self.Float_Scalar
    """The lattice spacing in physical units."""
    var domain_size: Tuple[
        Self.Float_Scalar, Self.Float_Scalar, Self.Float_Scalar
    ]
    """The physical extents of the domain along each axis."""
    var area: Self.Float_Scalar
    """The area of one lattice cell (`dx**2`)."""
    var volume: Self.Float_Scalar
    """The volume of one lattice cell (`dx**3`)."""
    comptime shape: InlineArray[Int, 3] = [Self.nx,Self.ny,Self.nz]
    """The `[nx, ny, nz]` node counts per axis."""
    var num_points: Int
    """The total number of lattice nodes."""
    var f_field_size: Int
    """The total number of stored `f` values (`Q * num_points`)."""
    var vel_field_size: Int
    """The total number of stored velocity components (`D * num_points`)."""
    var bc_field_size: Int
    """The total number of stored boundary-condition values (`(D+1) * num_points`)."""
    var origin: InlineArray[Self.Float_Scalar, 3]
    """The physical coordinate of the `(0, 0, 0)` node."""

    def __init__(
        out self,
        dx: Self.Float_Scalar,
        origin: InlineArray[Self.Float_Scalar, 3] = [0.0, 0.0, 0.0],
    ):
        """Constructs an `LBM_Grid` with a lattice spacing and origin.

        Args:
            dx: The lattice spacing in physical units.
            origin: The physical coordinate of the `(0, 0, 0)` node
                (defaults to `[0., 0., 0.]`).
        """
        check_model_match_dim[Self.D, Self.nx, Self.ny, Self.nz]()
        self.dx = dx
        self.area = dx**2
        self.volume = dx**3
        
        self.num_points = Self.nx * Self.ny * Self.nz
        self.f_field_size = Self.Q * self.num_points
        self.vel_field_size = Self.D * self.num_points
        self.bc_field_size = (Self.D + 1) * self.num_points
        self.domain_size = (
            Self.Float_Scalar(Self.nx - 1) * dx,
            Self.Float_Scalar(Self.ny - 1) * dx,
            Self.Float_Scalar(Self.nz - 1) * dx,
        )
        self.origin = origin

    def get_grid_coordinates(
        self, i: Int, j: Int, k: Int
    ) -> InlineArray[Scalar[Self.float_dtype], 3]:
        """Returns the physical coordinates of a lattice index triplet.

        Args:
            i: The `x`-axis lattice index.
            j: The `y`-axis lattice index.
            k: The `z`-axis lattice index.

        Returns:
            The physical `(x, y, z)` coordinates of the node.
        """
        out: InlineArray[Scalar[Self.float_dtype], 3] = [0, 0, 0]
        grid_index = (i, j, k)
        comptime for i in range(3):
            out[i] = (
                Scalar[Self.float_dtype](grid_index[i]) * self.dx
                + self.origin[i]
            )
        return out

    def get_UnitSystem_with_Re(
        self,
        U_phys: Self.Float_Scalar,
        U_lattice: Self.Float_Scalar,
        L_phys: Self.Float_Scalar,
        Re: Self.Float_Scalar,
        density: Self.Float_Scalar = 1.0,
    ) -> UnitSystem[Self.float_dtype, Self.D]:
        """Builds a `UnitSystem` from a target Reynolds number.

        Args:
            U_phys: The physical velocity scale.
            U_lattice: The lattice velocity.
            L_phys: The physical length scale.
            Re: The target Reynolds number.
            density: The fluid density (defaults to 1).

        Returns:
            A `UnitSystem` configured for this grid's dimension and dtype.
        """
        L_lattice = L_phys / self.dx
        kinematic_viscosity = U_phys * L_phys / Re
        return UnitSystem[Self.float_dtype, Self.D](
            U_phys, U_lattice, L_phys, L_lattice, density, kinematic_viscosity
        )

    def get_UnitSystem(
        self,
        U_phys: Self.Float_Scalar,
        U_lattice: Self.Float_Scalar,
        L_phys: Self.Float_Scalar,
        kinematic_viscosity: Self.Float_Scalar,
        density: Self.Float_Scalar = 1.0,
    ) -> UnitSystem[Self.float_dtype, Self.D]:
        """Builds a `UnitSystem` from a kinematic viscosity.

        Args:
            U_phys: The physical velocity scale.
            U_lattice: The lattice velocity.
            L_phys: The physical length scale.
            kinematic_viscosity: The kinematic viscosity of the fluid.
            density: The fluid density (defaults to 1).

        Returns:
            A `UnitSystem` configured for this grid's dimension and dtype.
        """
        L_lattice = L_phys / self.dx
        return UnitSystem[Self.float_dtype, Self.D](
            U_phys, U_lattice, L_phys, L_lattice, density, kinematic_viscosity
        )


def set_block_shape_and_grid_dim[
    nx: Int, ny: Int, nz: Int, D: Int, tile_size: Int
]() -> Tuple[Tuple[Int, Int, Int], Tuple[Int, Int, Int]]:
    """Returns the GPU block and grid dimensions for a given grid shape.

    For `tile_size > 1`, the block shape is `(tile_size, tile_size, tile_size)`
    (clamped to 1 along inactive axes) and the grid shape is the per-axis
    tile count. For `tile_size == 1`, the block shape is chosen so a 2D block
    has 256 threads and a 3D block has 512 threads, and the grid shape is
    derived by ceiling-division.

    Parameters:
        nx: The number of lattice nodes along `x`.
        ny: The number of lattice nodes along `y`.
        nz: The number of lattice nodes along `z`.
        D: The spatial dimension of the grid.
        tile_size: The tile size used for tiled layouts and GPU blocking.

    Returns:
        A tuple of `(block_shape, grid_dim)` as 3-tuples of `Int`.
    """
    comptime assert (
        (nx % tile_size == 0 or nx == 1)
        and (ny % tile_size == 0 or ny == 1)
        and (nz % tile_size == 0 or nz == 1)
    ), "Tile size must divide nx,ny and nz"
    comptime assert tile_size >= 1
    comptime if tile_size > 1:
        block_shape: Tuple[Int, Int, Int] = (
            tile_size,
            tile_size if D >= 2 else 1,
            tile_size if D == 3 else 1,
        )
        grid_dim: Tuple[Int, Int, Int] = (
            nx // tile_size,
            ny // tile_size if D >= 2 else 1,
            nz // tile_size if D == 3 else 1,
        )

    else:
        if D == 1:
            g_dim = 256
        elif D == 2:
            g_dim = 16  # 2D Block has 256 Threads
        else:
            g_dim = 8  # 3D block has 512 threads

        def calc_grid_dim(n: Int, g: Int) -> Int:
            return n // g if n % g == 0 else n // g + 1

        block_shape: Tuple[Int, Int, Int] = (
            g_dim,
            g_dim if D >= 2 else 1,
            g_dim if D == 3 else 1,
        )

        grid_dim: Tuple[Int, Int, Int] = (
            calc_grid_dim(nx, block_shape[0]),
            calc_grid_dim(ny, block_shape[1]),
            calc_grid_dim(nz, block_shape[2]),
        )

    return block_shape, grid_dim


def check_model_match_dim[D: Int, nx: Int, ny: Int, nz: Int]():
    """Asserts that the lattice dimension matches the active grid axes.

    Counts the number of axes with length greater than one and asserts that
    it equals `D`, so a 2D lattice model cannot be paired with a 3D grid.

    Parameters:
        D: The spatial dimension of the lattice model.
        nx: The number of lattice nodes along `x`.
        ny: The number of lattice nodes along `y`.
        nz: The number of lattice nodes along `z`.
    """
    comptime assert 1 <= D <= 3
    comptime assert nx > 0 and ny > 0 and nz > 0
    comptime grid_D = (1 if nx > 1 else 0) + (1 if ny > 1 else 0) + (
        1 if nz > 1 else 0
    )
    comptime assert D == grid_D, (
        "The given dimension of the Lattice does not match that of the"
        " dimension of the grid"
    )
