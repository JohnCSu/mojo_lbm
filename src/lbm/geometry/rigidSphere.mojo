from src.lbm import LBM_Grid,Lattice,GridLike
from src.lbm.constants import Flags
from std.gpu.host import DeviceContext
from layout import TileTensor,LayoutTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from src.utils import Vector
from std.collections import Set
from std.math import sqrt
from . import RigidStationaryObject
from src.lbm.kernels.utils.index import index_to_coord
from CSR import BitmaskCSR


def get_rigid_sphere[
    flag_origin:Origin[mut=True],
    FlagLayoutType:TensorLayout,
    //,
    grid:LBM_Grid
    ]
    (
        ctx:DeviceContext,
        flags:TileTensor[DType.uint8,FlagLayoutType,flag_origin],
        center:List[Scalar[grid.float_dtype]],
        radius:Scalar[grid.float_dtype],
        ) raises -> RigidStationaryObject[grid]:
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

    The caller-supplied lists are appended to in place so the raw
    pre-sort link data is available alongside the CSR for upload to the
    device:
      - `boundary_global_ids[k]` is the linear memory index of the k-th
        boundary node (one entry per CSR row).
      - `row_indices[i]` / `col_indices[i]` are the raw pre-sort `(row, q)`
        link pairs used to build the CSR (in iter-insertion order).
      - `q_dists[i]` is the per-link ray-sphere distance; pass
        `csr.values_tensor(Span(q_dists), sort=True)` to align it with the
        CSR's sorted set-bit traversal order.

    Parameters:
        flag_origin: The origin of the `flags` tile tensor.
        FlagLayoutType: The compile-time layout of `flags`.
        grid: The `LBM_Grid` describing the domain.

    Args:
        ctx: The `DeviceContext` that owns the `BitmaskCSR` buffers.
        flags: The `uint8` tile tensor labeling each node (mutated in place).
        center: The physical `(x, y, z)` coordinates of the sphere center.
        radius: The physical radius of the sphere.
        boundary_global_ids: Pre-initialized empty list that is appended to
            in place with the linear memory index of each fluid boundary
            node, in raster-scan order. Row `k` of the returned CSR
            corresponds to `boundary_global_ids[k]`.
        row_indices: Pre-initialized empty list that is appended to in place
            with the packed row index of each link's owner boundary node.
        col_indices: Pre-initialized empty list that is appended to in place
            with the `q` direction index of each link.
        q_dists: Pre-initialized empty list that is appended to in place with
            the per-link ray-sphere distance (in `[0, 1]`).

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

    # Phase 1: mark SOLID nodes inside the sphere within the bounding box
    for nx in range(bounding_box[0][0],bounding_box[0][1]):
        for ny in range(bounding_box[1][0],bounding_box[1][1]):
            for nz in range(bounding_box[2][0],bounding_box[2][1]):
                coord_vec = index_to_coord((nx,ny,nz),grid.dx,grid.origin)
                if inside_boundary(coord_vec,center_vec,radius):
                    flags.store(coord[DType.int32]((nx,ny,nz)),value = Flags.SOLID)

    # Phase 2: walk the bounding box in raster order. A node is a boundary
    # node iff it is itself outside the sphere AND at least one lattice
    # neighbor over `q in 0..Q-1` is inside (solid). Register each boundary
    # node exactly once with a packed local index, tag the flag, and emit
    # all `(local_idx, q)` links with their `q_dist`.
    boundary_global_ids:List[Scalar[grid.int_dtype]] = []
    fluid_boundary_id:List[Scalar[grid.int_dtype]] =[]
    lattice_direction:List[Scalar[grid.int_dtype]] =[]
    q_dists:List[Scalar[grid.float_dtype]] = []

    for nx in range(bounding_box[0][0],bounding_box[0][1]):
        for ny in range(bounding_box[1][0],bounding_box[1][1]):
            for nz in range(bounding_box[2][0],bounding_box[2][1]):
                coord_self = index_to_coord((nx,ny,nz),grid.dx,grid.origin)
                if inside_boundary(coord_self,center_vec,radius):
                    continue  # solid node, not a boundary

                crd = coord[int_dtype]((nx,ny,nz))
                global_id = flags.layout[linear_idx_type = int_dtype](crd)

                has_solid = False
                for q in range(Q):
                    test_direction:InlineArray[Int,3] = [nx,ny,nz]
                    comptime for i in range(D):
                        test_direction[i] += Int(latticeModel.directions[q][i])
                    coord_test = index_to_coord((test_direction[0],test_direction[1],test_direction[2]),grid.dx,grid.origin)
                    if inside_boundary(coord_test,center_vec,radius):
                        has_solid = True
                        break

                if not has_solid:
                    continue

                flags.store(coord[DType.int32]((nx,ny,nz)),value = Flags.FLUID_BOUNDARY)
                boundary_global_ids.append(global_id)
                local_idx = len(boundary_global_ids) - 1

                xf = vec3(grid.get_grid_coordinates(nx,ny,nz))
                for q in range(Q):
                    test_direction:InlineArray[Int,3] = [nx,ny,nz]
                    comptime for i in range(D):
                        test_direction[i] += Int(latticeModel.directions[q][i])
                    coord_test = index_to_coord((test_direction[0],test_direction[1],test_direction[2]),grid.dx,grid.origin)
                    if inside_boundary(coord_test,center_vec,radius):
                        xs = coord_test
                        fluid_boundary_id.append(Scalar[int_dtype](local_idx))
                        lattice_direction.append(Scalar[int_dtype](q))
                        q_dists.append(get_q_for_sphere(xf,xs,center_vec,radius))

    # var shape = (len(boundary_global_ids), Q)
    # var csr = BitmaskCSR[int_dtype](ctx, shape, Span(fluid_boundary_id), Span(lattice_direction))
    sphere = RigidStationaryObject[grid](ctx,boundary_global_ids^,fluid_boundary_id^,lattice_direction^,q_dists^)
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