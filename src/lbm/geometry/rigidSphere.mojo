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


def get_rigid_sphere[
    flag_origin:Origin[mut=True],
    FlagLayoutType:TensorLayout,
    //,
    ]
    (
        grid:LBM_Grid,
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

    #Loop through and 

    for nx in range(bounding_box[0][0],bounding_box[0][1]):
        for ny in range(bounding_box[1][0],bounding_box[1][1]):
            for nz in range(bounding_box[2][0],bounding_box[2][1]):
                coord_vec = index_to_coord((nx,ny,nz),grid.dx,grid.origin)
                if inside_boundary(coord_vec,center_vec,radius):
                    flags.store(coord[DType.int32]((nx,ny,nz)),value = Flags.SOLID)
                else:
                    candidate_indices.add((nx,ny,nz))

    indices =  List[Scalar[int_dtype]](length = len(candidate_indices),fill = (-1))
    

    # TODO: we checking inside sphere is cheap so we can count all the links or estimate thats close
    link_owner = List[Scalar[int_dtype]]()
    link_to_solid = List[Scalar[int_dtype]]()
    links_q_dist = List[Scalar[float_dtype]]()

    num_boundary_indices = 0

    temp_link_fluid = List[Scalar[int_dtype]](length = Q,fill = (-1))
    temp_link_solid = List[Scalar[int_dtype]](length = Q,fill = (-1))
    temp_link_q_dist = List[Scalar[float_dtype]](length = Q,fill = (-1))

    for idx in candidate_indices:
        x,y,z = idx
        crd = coord[int_dtype]((x,y,z))
        global_id = flags.layout[linear_idx_type = int_dtype](crd)
        is_fluid_boundary = False
        num_solid = 0
        # We allocate a list of Q to keep it
        for q in range(Q):
            test_direction:InlineArray[Int,3] = [x,y,z]
            
            comptime for i in range(D):
                test_direction[i] += Int(latticeModel.directions[q][i])
            coord_test = index_to_coord((test_direction[0],test_direction[1],test_direction[2]),grid.dx,grid.origin)

            if inside_boundary(coord_test,center_vec,radius) :
                if not is_fluid_boundary:
                    indices[num_boundary_indices] = global_id
                    num_boundary_indices += 1
                    is_fluid_boundary = True
                
                xf = vec3(grid.get_grid_coordinates(x,y,z))
                xs = coord_test

                temp_link_fluid[num_solid] = global_id
                temp_link_solid[num_solid] = Scalar[int_dtype](q)
                temp_link_q_dist[num_solid] = get_q_for_sphere(xf,xs,center_vec,1.)
                num_solid += 1
        # append to links
        
        solid_slices = slice(0,num_solid,1)
        link_owner += temp_link_fluid[solid_slices]
        link_to_solid += temp_link_solid[solid_slices]
        links_q_dist += temp_link_q_dist[solid_slices]
        
    indices.shrink(num_boundary_indices)
    return (indices)^


def get_q_for_sphere[
    float_dtype:DType
    ](
    xf:Vector[float_dtype,3],xs:Vector[float_dtype,3],center:Vector[float_dtype,3],radius:Scalar[float_dtype]
    ) -> Scalar[float_dtype]:
    fluid_to_cent = xf - center
    solid_to_fluid = xs - xf

    a = solid_to_fluid.norm_squared()
    
    b = fluid_to_cent.dot(solid_to_fluid)*2

    c = fluid_to_cent.norm_squared()-radius*radius
    
    t1 = -b + sqrt(b*b - 4*a*c)/(2*a)
    t2 = (-b - sqrt(b*b - 4*a*c))/(2*a)

    return t1 if (t1 >= 0. and t1 <= 1.) else t2

    



    