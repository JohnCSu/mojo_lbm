"""Applies boundary conditions to the exterior walls of an LBM domain.

Provides `set_exterior_walls` for uniform velocity/density walls and
`set_exterior_walls_with_func` for spatially varying velocity walls. Both
write the flag value and the boundary-condition velocities and density into
the provided `TileTensor` views, optionally converting from physical to
lattice units via a `UnitSystem`.
"""
from std.gpu.host import DeviceContext
from layout import TileTensor,LayoutTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from std.collections import InlineArray
from std.collections import Set,Dict
from src.utils import Vector,ContextTileTensor
from std.utils.numerics import nan,isnan
from src.lbm import UnitSystem
from src.lbm import LBM_Grid,Lattice


def set_exterior_walls[float_dtype:DType,
                    flag_origin:Origin[mut=True],
                    bc_origin:Origin[mut=True],
                    nx:Int,ny:Int,nz:Int,
                    D:Int,Q:Int,
                    lattice_model:Lattice[D,Q,float_dtype,DType.int32],
                    FlagLayoutType:TensorLayout,
                    BCLayoutType:TensorLayout,
                    //,
                    grid:LBM_Grid[lattice_model,nx,ny,nz,_],
                    config:LBM_Config = LBM_Config()
                    ]
                    (flags:TileTensor[DType.uint8,FlagLayoutType,flag_origin],
                            bc:TileTensor[float_dtype,BCLayoutType,bc_origin],
                            side:String,
                            boundary_type:Scalar[DType.uint8],
                            u:List[Scalar[float_dtype]] = [],
                            rho:Scalar[float_dtype] = nan[float_dtype](),
                            unitSystem:Optional[UnitSystem[float_dtype,D]] = None) raises:
    """Applies a uniform boundary condition to one exterior wall of the grid.

    Writes the flag value and the velocity/density into the supplied `flags`
    and `bc` tensors for every node on the chosen wall. Velocity and density
    values are interpreted in lattice units by default; passing a
    `unitSystem` converts them from physical units first.

    Parameters:
        grid: The compile-time `LBM_Grid` describing the domain.
        config: The `LBM_Config` whose `INCLUDED_BCs` constrains the valid
            `boundary_type` values.

    Args:
        flags: The `uint8` tile tensor labeling each node (e.g. `0` for
            fluid).
        bc: The tile tensor storing the velocity and density for each node.
        side: The wall to write, one of `'-X'`, `'+X'`, `'-Y'`, `'+Y'`,
            `'-Z'`, `'+Z'`.
        boundary_type: The flag value to write at the target wall.
        u: The velocity components for the wall. An empty list means the
            velocity is free (defaults to the empty list).
        rho: The density for the wall. `NaN` means the density is free
            (defaults to `NaN`).
        unitSystem: Optional unit system. When supplied, `u` and `rho` are
            interpreted in physical units and converted to lattice units.

    Raises:
        Error: If both `u` and `rho` are left unspecified.
        Error: If the velocity list length does not match the grid dimension.
        Error: If `boundary_type` is not in `config.INCLUDED_BCs`.
        Error: If `side` is not one of the six valid strings.
    """
    comptime assert float_dtype.is_floating_point()
    comptime assert FlagLayoutType.rank == 3 and BCLayoutType.rank == 4

    VALID_BOUNDARIES = materialize[config.INCLUDED_BCs]()

    axes:Dict[String,Int] = {'X':0,
                    'Y':1,
                    'Z':2,}
    valid_strings:Set[String] = {'-X','+X','-Y','+Y','-Z','+Z'}

    u_is_empty = (len(u) == 0)
    if u_is_empty and isnan(rho):
        raise Error('Either velocity or density or both have to be specified. Both cant be left as None')

    velocity = [nan[float_dtype]() for _ in range(D)] if u_is_empty else u.copy()
    density:Scalar[float_dtype] = rho

    if (boundary_type ==  SOLID_NODE) and (u_is_empty or isnan(rho)):
        raise Error('For Solid Type you must specify both u and rho')

    if len(velocity) != D:
        raise Error('Input velocity list was of length {} but Grid is {} Dimensional'.format(len(velocity),D))

    if boundary_type not in VALID_BOUNDARIES:
        raise Error('Input Boundary Type was {} but valid boundary types are: {}'.format(boundary_type,VALID_BOUNDARIES))

    if side not in valid_strings:
        raise Error('Side not valid. Input was {} but expects {}'.format(side,valid_strings))

    if unitSystem: # if not None then implies bc give are not in
        density *=unitSystem.value().density.C_phys_to_lat()
        velocity = [unitSystem.value().U.C_phys_to_lat()*u for u in velocity]

    axis = axes[String(side[byte = 1])]
    end_values = [nx,ny,nz]

    if side[byte = 0] == '-':
        fixed = 0
    else:
        fixed = end_values[axis] - 1
    if axis == 0: # X-axis, fix x and loop
        x = fixed
        for y in range(ny):
            for z in range(nz):
                flags.store(coord[DType.int32]((x,y,z)),flags.ElementType(boundary_type))
                # flags_lt[fixed,y,z] = flags.ElementType(boundary_type)
                comptime for i in range(D):
                    bc.store(coord[DType.int32]((x,y,z,i)),velocity[i])
                bc.store(coord[DType.int32]((x,y,z,D)),density)
    elif axis == 1:
        y = fixed
        for x in range(nx):
            for z in range(nz):
                flags.store(coord[DType.int32]((x,y,z)),flags.ElementType(boundary_type))
                comptime for i in range(D):
                    bc.store(coord[DType.int32]((x,y,z,i)),velocity[i])
                bc.store(coord[DType.int32]((x,y,z,D)),density)
    else: # Loop Z-face
        z = fixed
        for x in range(nx):
            for y in range(ny):
                flags.store(coord[DType.int32]((x,y,z)),flags.ElementType(boundary_type))
                comptime for i in range(D):
                    bc.store(coord[DType.int32]((x,y,z,i)),velocity[i])
                bc.store(coord[DType.int32]((x,y,z,D)),density)


def set_exterior_walls_with_func[
                    float_dtype:DType,
                    flag_origin:Origin[mut=True],
                    bc_origin:Origin[mut=True],
                    nx:Int,ny:Int,nz:Int,
                    D:Int,Q:Int,
                    latticeModel:Lattice[D,Q,float_dtype,DType.int32],
                    FlagLayoutType:TensorLayout,
                    BCLayoutType:TensorLayout,
                    //,
                    grid:LBM_Grid[latticeModel,nx,ny,nz,_],
                    config:LBM_Config = LBM_Config(),
                    *,
                    u: def[float_dtype:DType,D:Int]
                        (Scalar[float_dtype],Scalar[float_dtype],Scalar[float_dtype],mut InlineArray[Scalar[float_dtype],D])
                        capturing
                    # u: Optional[def[float_dtype:DType,D:Int]
                    #     (Scalar[float_dtype],Scalar[float_dtype],Scalar[float_dtype],mut InlineArray[Scalar[float_dtype],D])
                    #     capturing] = None
                    ]
                    (flags:TileTensor[DType.uint8,FlagLayoutType,flag_origin],
                            bc:TileTensor[float_dtype,BCLayoutType,bc_origin],
                            side:String,
                            boundary_type:Scalar[DType.uint8],
                            unitSystem:Optional[UnitSystem[float_dtype,D]],
                            rho:Scalar[float_dtype] = nan[float_dtype](),
                            ) raises:
    """Applies a spatially varying velocity boundary condition to one wall.

    Calls the supplied `u` function at every node on the chosen wall to
    compute the velocity in physical grid coordinates, then writes the flag,
    velocity, and density into the `flags` and `bc` tensors. When a
    `unitSystem` is supplied, velocities and density are converted from
    physical to lattice units.

    Parameters:
        grid: The compile-time `LBM_Grid` describing the domain.
        config: The `LBM_Config` whose `INCLUDED_BCs` constrains the valid
            `boundary_type` values.
        u: A compile-time function that takes the physical `(x, y, z)`
            coordinates and writes the velocity into a mutable
            `InlineArray[Scalar[float_dtype], D]`.

    Args:
        flags: The `uint8` tile tensor labeling each node.
        bc: The tile tensor storing the velocity and density for each node.
        side: The wall to write, one of `'-X'`, `'+X'`, `'-Y'`, `'+Y'`,
            `'-Z'`, `'+Z'`.
        boundary_type: The flag value to write at the target wall.
        unitSystem: The unit system used to convert velocities and density
            from physical to lattice units.
        rho: The density for the wall. `NaN` means the density is free
            (defaults to `NaN`).

    Raises:
        Error: If a solid boundary is requested without a specified `rho`.
        Error: If `boundary_type` is not in `config.INCLUDED_BCs`.
        Error: If `side` is not one of the six valid strings.
    """
    comptime assert float_dtype.is_floating_point()
    comptime assert FlagLayoutType.rank == 3 and BCLayoutType.rank == 4
    # comptime assert u is not None
    comptime u_func = u
    VALID_BOUNDARIES = materialize[config.INCLUDED_BCs]()

    axes:Dict[String,Int] = {'X':0,
                    'Y':1,
                    'Z':2,}
    valid_strings:Set[String] = {'-X','+X','-Y','+Y','-Z','+Z'}

    density:Scalar[float_dtype] = rho

    if (boundary_type ==  SOLID_NODE) and (isnan(rho)):
        raise Error('For Solid Type you must specify both u and rho')

    if boundary_type not in VALID_BOUNDARIES:
        raise Error('Input Boundary Type was {} but valid boundary types are: {}'.format(boundary_type,VALID_BOUNDARIES))

    if side not in valid_strings:
        raise Error('Side not valid. Input was {} but expects {}'.format(side,valid_strings))

    if unitSystem: # if not None then implies bc give are not in
        density *=unitSystem.value().density.C_phys_to_lat()
        # velocity = [unitSystem.value().U.C_phys_to_lat()*u for u in velocity]

    axis = axes[String(side[byte = 1])]
    end_values = [nx,ny,nz]

    if side[byte = 0] == '-':
        fixed = 0
    else:
        fixed = end_values[axis] - 1

    conversion_factor = unitSystem.value().U.C_phys_to_lat() if unitSystem else 1.
    if axis == 0: # X-axis, fix x and loop
        x = fixed
        for y in range(ny):
            for z in range(nz):
                flags.store(coord[DType.int32]((x,y,z)),flags.ElementType(boundary_type))
                velocity = InlineArray[Scalar[float_dtype],D](fill =0)
                grid_coords =  grid.get_grid_coordinates(x,y,z)
                u_func(grid_coords[0],grid_coords[1],grid_coords[2],velocity)
                comptime for i in range(D):
                    bc.store(coord[DType.int32]((x,y,z,i)),velocity[i]*(conversion_factor) )
                bc.store(coord[DType.int32]((x,y,z,D)),density)
    elif axis == 1:
        y = fixed
        for x in range(nx):
            for z in range(nz):
                flags.store(coord[DType.int32]((x,y,z)),flags.ElementType(boundary_type))
                velocity = InlineArray[Scalar[float_dtype],D](fill =0)
                grid_coords =  grid.get_grid_coordinates(x,y,z)
                u_func(grid_coords[0],grid_coords[1],grid_coords[2],velocity)
                comptime for i in range(D):
                    bc.store(coord[DType.int32]((x,y,z,i)),velocity[i]*(conversion_factor) )
                bc.store(coord[DType.int32]((x,y,z,D)),density)
    else: # Loop Z-face
        z = fixed
        for x in range(nx):
            for y in range(ny):
                flags.store(coord[DType.int32]((x,y,z)),flags.ElementType(boundary_type))
                velocity = InlineArray[Scalar[float_dtype],D](fill =0)
                grid_coords =  grid.get_grid_coordinates(x,y,z)
                u_func(grid_coords[0],grid_coords[1],grid_coords[2],velocity)
                comptime for i in range(D):
                    bc.store(coord[DType.int32]((x,y,z,i)),velocity[i]*(conversion_factor) )
                bc.store(coord[DType.int32]((x,y,z,D)),density)
