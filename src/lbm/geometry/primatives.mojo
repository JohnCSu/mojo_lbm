"""Provides geometry primitives for marking solid nodes on the flag field.

The functions here embed simple shapes (spheres, circles, boxes) into an LBM
domain by writing `Flags.SOLID` into the flag tile tensor for every node
inside the shape, and `get_sphere_boundary_indices` collects the fluid nodes
adjacent to a sphere for use by force computations.
"""
from src.lbm import LBM_Grid,Lattice
from src.lbm.constants import Flags
from std.gpu.host import DeviceContext
from layout import TileTensor,LayoutTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from src.utils import Vector
from std.collections import Set


def get_sphere_boundary_indices[
    flag_origin:Origin[mut=True],
    FlagLayoutType:TensorLayout,
    //,
    grid:LBM_Grid
    ]
    (
        flags:TileTensor[DType.uint8,FlagLayoutType,flag_origin],
        center:List[Scalar[grid.float_dtype]],
        radius:Scalar[grid.float_dtype],
        ) raises -> List[Scalar[grid.int_dtype]]:
    """Marks a sphere solid and returns the linear indices of adjacent fluid nodes.

    Writes `Flags.SOLID` into `flags` for every node inside the sphere, then
    collects the linear memory indices of every fluid node whose
    neighborhood touches the solid. Assumes a column-major flag layout for
    the linear-index computation.

    Parameters:
        flag_origin: The origin of the `flags` tile tensor.
        FlagLayoutType: The compile-time layout of `flags`.
        grid: The compile-time `LBM_Grid` describing the domain.

    Args:
        flags: The `uint8` tile tensor labeling each node.
        center: The physical `(x, y, z)` coordinates of the sphere center.
        radius: The physical radius of the sphere.

    Returns:
        A list of linear memory indices of the fluid nodes adjacent to the
        sphere.

    Raises:
        Error: If `center` does not have exactly three elements.
    """
    comptime D = grid.D
    comptime Q = grid.Q
    comptime float_dtype = grid.float_dtype
    comptime int_dtype = grid.int_dtype
    comptime latticeModel = grid.lattice

    if len(center) != 3:
        raise Error('centre must be a list of 3 floats got a len of {} instead'.format(len(center)))
    var bounding_box:List[List[Int]] = []

    for i in range(3):
        if i < D:
            a_min,a_max = center[i] - radius, center[i] + radius
            n_min,n_max = max(0,Int((a_min-grid.origin[i])//grid.dx)), min(grid.shape[i],Int((a_max-grid.origin[i])//grid.dx + 1))
            bounding_box.append([n_min,n_max+1])
        else:
            bounding_box.append([0,1])

    comptime vec3 = Vector[float_dtype,3]
    comptime float = Scalar[float_dtype]
    var center_vec = vec3(center)

    def inside_boundary(coord_vec:vec3,center_vec:vec3,radius:float) -> Bool:
        return ((coord_vec - center_vec)**2).sum() <= radius**2

    candidate_indices:Set[Tuple[Int,Int,Int]] = {}
    for nx in range(bounding_box[0][0],bounding_box[0][1]):
        for ny in range(bounding_box[1][0],bounding_box[1][1]):
            for nz in range(bounding_box[2][0],bounding_box[2][1]):
                coord_vec = index_to_coord((nx,ny,nz),grid.dx,grid.origin)
                if inside_boundary(coord_vec,center_vec,radius):
                    flags.store(coord[DType.int32]((nx,ny,nz)),value = Flags.SOLID)
                else:
                    candidate_indices.add((nx,ny,nz))

    indices =  List[Scalar[int_dtype]](length = len(candidate_indices),fill = (-1))

    num_boundary_indices = 0
    for idx in candidate_indices:
        x,y,z = idx
        crd = coord[int_dtype]((x,y,z))
        for q in range(Q):
            test_direction:InlineArray[Int,3] = [x,y,z]
            comptime for i in range(D):
                test_direction[i] += Int(latticeModel.directions[q][i])
            coord_test = index_to_coord((test_direction[0],test_direction[1],test_direction[2]),grid.dx,grid.origin)
            if inside_boundary(coord_test,center_vec,radius):
                indices[num_boundary_indices] = flags.layout[linear_idx_type = int_dtype](crd)
                num_boundary_indices += 1
                break

    indices.shrink(num_boundary_indices)
    return (indices)^


def index_to_coord[float_dtype:DType]
    (
        grid_index:Tuple[Int,Int,Int],
        grid_spacing:Scalar[float_dtype],
        origin:InlineArray[Scalar[float_dtype],3]
    ) -> Vector[float_dtype,3]:
    """Converts a lattice index triplet to physical coordinates.

    Parameters:
        float_dtype: The `DType` of the returned vector.

    Args:
        grid_index: The `(i, j, k)` lattice indices.
        grid_spacing: The lattice spacing `dx`.
        origin: The physical coordinate of the `(0, 0, 0)` node.

    Returns:
        The physical `(x, y, z)` coordinates of the node.
    """
    out = Vector[float_dtype,3](fill =0)
    comptime for i in range(3):
        out[i] = Scalar[float_dtype](grid_index[i])*grid_spacing + origin[i]
    return out



def add_sphere[
    flag_origin:Origin[mut=True],
    FlagLayoutType:TensorLayout,
    //,
    grid:LBM_Grid]
    (flags:TileTensor[DType.uint8,FlagLayoutType,flag_origin],center:List[Scalar[grid.float_dtype]],radius:Scalar[grid.float_dtype]) raises:
    """Marks a sphere (circle in 2D, ball in 3D) solid in the flag field.

    Writes `Flags.SOLID` into `flags` for every node whose physical coordinate
    lies within `radius` of `center`.

    Parameters:
        grid: The compile-time `LBM_Grid` describing the domain.

    Args:
        flags: The `uint8` tile tensor labeling each node.
        center: The physical `(x, y, z)` coordinates of the sphere center.
        radius: The physical radius of the sphere.

    Raises:
        Error: If `center` does not have exactly three elements.
    """
    comptime D = grid.D
    comptime float_dtype = grid.float_dtype
    comptime assert FlagLayoutType.rank == 3
    if len(center) != 3:
        raise Error('centre must be a list of 3 floats got a len of {} instead'.format(len(center)))
    var bounding_box:List[List[Int]] = []


    for i in range(3):
        if i < D:
            a_min,a_max = center[i] - radius, center[i] + radius
            n_min,n_max = max(0,Int((a_min-grid.origin[i])//grid.dx)), min(grid.shape[i],Int((a_max-grid.origin[i])//grid.dx + 1))
            bounding_box.append([n_min,n_max])
        else:
            bounding_box.append([0,1])

    comptime vec3 = Vector[float_dtype,3]
    comptime float = Scalar[float_dtype]
    var center_vec = vec3(center)
    var coord_vec = vec3(fill=0.)

    for nx in range(bounding_box[0][0],bounding_box[0][1]):
        for ny in range(bounding_box[1][0],bounding_box[1][1]):
            for nz in range(bounding_box[2][0],bounding_box[2][1]):
                dx,dy,dz = (float(nx)*grid.dx,float(ny)*grid.dx,float(nz)*grid.dx)
                coord_vec[0] = dx + grid.origin[0]
                coord_vec[1] = dy + grid.origin[1]
                coord_vec[2] = dz + grid.origin[2]
                if ((coord_vec - center_vec)**2).sum() <= radius**2:
                    flags.store(coord[DType.int32]((nx,ny,nz)),value = Flags.SOLID)

def add_circle[
    flag_origin:Origin[mut=True],
    FlagLayoutType:TensorLayout,
    //,
    grid:LBM_Grid
    ]
    (flags:TileTensor[DType.uint8,FlagLayoutType,flag_origin],center:List[Scalar[grid.float_dtype]],radius:Scalar[grid.float_dtype]) raises:

    """Alias of `add_sphere` constrained to 2D grids.

    Parameters:
        grid: The compile-time `LBM_Grid` describing the domain.

    Args:
        flags: The `uint8` tile tensor labeling each node.
        center: The physical `(x, y, z)` coordinates of the circle center.
        radius: The physical radius of the circle.
    """
    comptime assert grid.lattice.D == 2,'Circle only valid for 2D'
    return add_sphere[grid](flags,center,radius)



def add_box[
    flag_origin:Origin[mut=True],
    FlagLayoutType:TensorLayout,
    //,
    grid:LBM_Grid
    ]
    (flags:TileTensor[DType.uint8,FlagLayoutType,flag_origin],center:List[Scalar[grid.float_dtype]],box_radius:List[Scalar[grid.float_dtype]]) raises:
    """Marks an axis-aligned box solid in the flag field.

    Writes `Flags.SOLID` into `flags` for every node whose physical coordinate
    lies within `box_radius` of `center` along each axis.

    Parameters:
        grid: The compile-time `LBM_Grid` describing the domain.

    Args:
        flags: The `uint8` tile tensor labeling each node.
        center: The physical `(x, y, z)` coordinates of the box center.
        box_radius: The half-extents of the box along each axis.

    Raises:
        Error: If `center` does not have exactly three elements.
        Error: If `box_radius` does not have exactly three elements.
    """
    comptime D = grid.D
    comptime float_dtype = grid.float_dtype
    comptime assert FlagLayoutType.rank == 3
    if len(center) != 3:
        raise Error('centre must be a list of 3 floats got a len of {} instead'.format(len(center)))

    if len(box_radius) != 3:
        raise Error('lengths must be a list of 3 floats got a len of {} instead'.format(len(box_radius)))
    var bounding_box:List[List[Int]] = []


    for i in range(3):
        if i < D:
            a_min,a_max = center[i] - box_radius[i], center[i] + box_radius[i]
            n_min,n_max = max(0,Int((a_min-grid.origin[i])//grid.dx)), min(grid.shape[i],Int((a_max-grid.origin[i])//grid.dx + 1))
            bounding_box.append([n_min,n_max])
        else:
            bounding_box.append([0,1])

    comptime vec3 = Vector[float_dtype,3]
    comptime float = Scalar[float_dtype]
    var center_vec = vec3(center)
    var coord_vec = vec3(fill=0.)
    var box_radius_vec = vec3(box_radius)

    for nx in range(bounding_box[0][0],bounding_box[0][1]):
        for ny in range(bounding_box[1][0],bounding_box[1][1]):
            for nz in range(bounding_box[2][0],bounding_box[2][1]):
                dx,dy,dz = (float(nx)*grid.dx,float(ny)*grid.dx,float(nz)*grid.dx)
                coord_vec[0] = dx + grid.origin[0]
                coord_vec[1] = dy + grid.origin[1]
                coord_vec[2] = dz + grid.origin[2]
                if check_box_axis(coord_vec,center_vec,box_radius_vec):
                    flags.store(coord[DType.int32]((nx,ny,nz)),value = Flags.SOLID)


def check_box_axis[float_dtype:DType,//](point:Vector[float_dtype,3],center:Vector[float_dtype,3],box_radius:Vector[float_dtype,3]) -> Bool:
    """Returns `True` when `point` lies inside an axis-aligned box.

    The box is centered at `center` with half-extents `box_radius` along each
    axis.

    Parameters:
        float_dtype: The `DType` of the vectors (inferred).

    Args:
        point: The point to test.
        center: The box center.
        box_radius: The half-extents of the box along each axis.

    Returns:
        `True` when `point` is inside the box, `False` otherwise.
    """
    if (center-box_radius <= point).all_true() and (point <= (center + box_radius)).all_true():
        return True
    else:
        return False
