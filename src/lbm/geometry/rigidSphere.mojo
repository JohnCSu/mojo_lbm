from src.lbm import LBM_Grid,Lattice,GridLike,LBM_Config
from src.lbm.constants import Flags,LBM_method
from std.gpu.host import DeviceContext
from layout import TileTensor,LayoutTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from src.utils import Vector
from std.collections import Set
from std.math import sqrt
from . import RigidStationaryObject
from src.lbm.kernels.utils.index import index_to_coord
# from CSR import BitmaskCSR


def get_rigid_sphere[
    flag_origin:Origin[mut=True],
    FlagLayoutType:TensorLayout,
    //,
    grid:LBM_Grid,
    lbm_method:LBM_method,
    config:LBM_Config[lbm_method]
    ]
    (
        ctx:DeviceContext,
        flags:TileTensor[DType.uint8,FlagLayoutType,flag_origin],
        center:List[Scalar[grid.float_dtype]],
        radius:Scalar[grid.float_dtype],
        ) raises -> RigidStationaryObject[grid,lbm_method,config]:
    """Marks a sphere solid and returns the fluid-boundary links as a `BitmaskCSR`.

    Writes `Flags.SOLID` into `flags` for every node inside the sphere and
    `Flags.FLUID_BOUNDARY` (value `2`) into `flags` for every fluid node
    whose neighborhood touches the solid. Collects the `(boundary_row, q)`
    link pairs into a `BitmaskCSR` of shape `(num_boundary_nodes, Q)` where
    `num_boundary_nodes` is the count of fluid boundary nodes (typically
    tens, not the whole grid). The CSR row index is the *packed* index into
    `boundary_global_ids` (row 0 = first boundary node, row 1 = second,
    ...), and the column index is the discrete velocity `q` whose neighbor
    is a solid node. `nnz` is the total number of `(boundary_node, q)`
    links.

    Parameters:
        flag_origin: The origin of the `flags` tile tensor.
        FlagLayoutType: The compile-time layout of `flags`.
        grid: The `LBM_Grid` describing the domain.

    Args:
        ctx: The `DeviceContext` that owns the `BitmaskCSR` buffers.
        flags: The `uint8` tile tensor labeling each node (mutated in place).
        center: The physical `(x, y, z)` coordinates of the sphere center.
        radius: The physical radius of the sphere.

    Returns:
        The `BitmaskCSR[grid.int_dtype]` over `(num_boundary_nodes, Q)`
        holding the `(boundary_row, q)` link pairs.

    Raises:
        Error: If `center` does not have exactly three elements.
        Error: If the lattice `Q` exceeds `32` (the `BitmaskCSR` column cap).
    """
    comptime D = grid.D
    comptime Q = grid.Q
    comptime float_dtype = grid.float_dtype
    comptime int_dtype = grid.int_dtype
    comptime latticeModel = grid.lattice

    if Q > 32:
        raise Error('BitmaskCSR supports at most 32 columns, got Q='+String(Q))
    if len(center) != 3:
        raise Error('centre must be a list of 3 floats got a len of {} instead'.format(len(center)))

    # Bounding box — expanded by 1 on each side so the search-radius
    # captures boundary fluid nodes that sit just outside the solid region.
    var bounding_box:List[List[Int]] = []
    for i in range(3):
        if i < D:
            a_min,a_max = center[i] - radius, center[i] + radius
            n_min,n_max = max(0,Int((a_min-grid.origin[i])//grid.dx)-1), min(grid.shape[i],Int((a_max-grid.origin[i])//grid.dx + 1)+1)
            bounding_box.append([n_min,n_max+1])
        else:
            bounding_box.append([0,1])

    comptime vec3 = Vector[float_dtype,3]
    comptime float_scalar = Scalar[float_dtype]
    var center_vec = vec3(center)

    def inside_boundary(coord_vec:vec3,center_vec:vec3,radius:float_scalar) -> Bool:
        return ((coord_vec - center_vec)**2).sum() <= radius**2

    fluid_boundary_candidate:Set[Tuple[Int,Int,Int]] = {}
    # Phase 1: mark SOLID nodes inside the sphere within the bounding box
    for nx in range(bounding_box[0][0],bounding_box[0][1]):
        for ny in range(bounding_box[1][0],bounding_box[1][1]):
            for nz in range(bounding_box[2][0],bounding_box[2][1]):
                coord_vec = index_to_coord((nx,ny,nz),grid.dx,grid.origin)
                if inside_boundary(coord_vec,center_vec,radius):
                    flags.store(coord[DType.int32]((nx,ny,nz)),value = Flags.SOLID)
                else:
                    fluid_boundary_candidate.add((nx,ny,nz))

    unique_fluid_id:List[Scalar[grid.int_dtype]] = []
    fluid_boundary_id:List[Scalar[grid.int_dtype]] =[]
    lattice_direction:List[Scalar[grid.int_dtype]] =[]
    q_dists:List[Scalar[grid.float_dtype]] = []

    for (nx,ny,nz) in fluid_boundary_candidate:
        crd = coord[int_dtype]((nx,ny,nz))
        fluid_id = flags.layout[linear_idx_type = int_dtype](crd)
        xf = vec3(grid.get_grid_coordinates(nx,ny,nz))
        id_has_been_checked = False

        for q in range(Q):
            test_direction:InlineArray[Int,3] = [nx,ny,nz]
            comptime for i in range(D):
                test_direction[i] += Int(latticeModel.directions[q][i])
            coord_test = index_to_coord((test_direction[0],test_direction[1],test_direction[2]),grid.dx,grid.origin)
             
            if inside_boundary(coord_test,center_vec,radius):
                if not id_has_been_checked :
                    id_has_been_checked = True
                    unique_fluid_id.append(Scalar[int_dtype](fluid_id))

                fluid_boundary_id.append(Scalar[int_dtype](fluid_id))
                lattice_direction.append(Scalar[int_dtype](q))
                
                xs = coord_test
                q_dists.append(get_q_for_sphere(xf,xs,center_vec,radius))

    sphere = RigidStationaryObject[grid,lbm_method,config](ctx,unique_fluid_id^,fluid_boundary_id^,lattice_direction^,q_dists^)
    return sphere^


def get_q_for_sphere[
    float_dtype:DType
    ](
    xf:Vector[float_dtype,3],xs:Vector[float_dtype,3],center:Vector[float_dtype,3],radius:Scalar[float_dtype]
    ) -> Scalar[float_dtype]:
    """Returns the ray-sphere intersection parameter `t` in `[0, 1]`.

    Solves `|xf + t*(xs-xf) - center|^2 = radius^2` for `t`, picking the
    root that lies within the fluid-to-solid link segment `[0, 1]`. The
    `t` returned is the fraction of the link that lies in the fluid before
    hitting the sphere surface — the standard q-distance used by
    bounce-back boundary conditions.

    Args:
        xf: The physical coordinate of the fluid boundary node.
        xs: The physical coordinate of the solid neighbor.
        center: The physical sphere center.
        radius: The physical sphere radius.

    Returns:
        The link-surface intersection parameter `t` in `[0, 1]`.
    """
    fluid_to_cent = xf - center
    solid_to_fluid = xs - xf

    a = solid_to_fluid.norm_squared()
    b = fluid_to_cent.dot(solid_to_fluid)*2
    c = fluid_to_cent.norm_squared() - radius*radius

    disc = b*b - 4*a*c
    sqrt_disc = sqrt(disc)

    t1 = (-b + sqrt_disc) / (2*a)
    t2 = (-b - sqrt_disc) / (2*a)

    # `t1` is the larger root (>= ~1), `t2` is the smaller (in (0, 1] for a
    # boundary fluid node with a solid neighbor inside the sphere). Pick
    # the one that actually lies on the link interval — usually `t2`.
    return t1 if (t1 >= 0. and t1 <= 1.) else t2