from src.lbm import LBM_Grid,LatticeModel
from src.lbm.flags import Flags
from std.gpu.host import DeviceContext
from layout import TileTensor,LayoutTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from src.utils import Vector
from std.collections import Set


def get_sphere_boundary_indices[
    flag_origin:Origin[mut=True],
    FlagLayoutType:TensorLayout,
    float_dtype:DType,
    int_dtype:DType,
    nx:Int,ny:Int,nz:Int,
    D:Int,Q:Int,
    tile_size:Int,
    latticeModel:LatticeModel[D,Q,float_dtype,int_dtype],
    //,
    grid:LBM_Grid[latticeModel,nx,ny,nz,tile_size]
    ]
    (   
        flags:TileTensor[DType.uint8,FlagLayoutType,flag_origin],
        center:List[Scalar[float_dtype]],
        radius:Scalar[float_dtype],
        ) raises -> List[Tuple[Int,Int,Int]]:

    # comptime (nx,ny,nz) = (grid.nx,grid.ny,grid.nz)
    
    if len(center) != 3:
        raise Error('centre must be a list of 3 floats got a len of {} instead'.format(len(center)))
    # b:Tuple[Int,Int,Int] = (0,0,0)
    var bounding_box:List[List[Int]] = []
    
    
    for i in range(3):
        if i < D:
            a_min,a_max = center[i] - radius, center[i] + radius
            n_min,n_max = max(0,Int((a_min-grid.origin[i])//grid.dx)), min(grid.shape[i],Int((a_max-grid.origin[i])//grid.dx + 1))
            bounding_box.append([n_min,n_max+1]) # Add 1 as we need to also have search for bounding box of fluid elements
        else:
            bounding_box.append([0,1]) # This means loop is just set to 0 index

    comptime vec3 = Vector[float_dtype,3]
    comptime float = Scalar[float_dtype]
    var center_vec = vec3(center)

    def inside_boundary(coord_vec:vec3,center_vec:vec3,radius:float) -> Bool:
        return ((coord_vec - center_vec)**2).sum() <= radius**2

    # Search Bounding box For candidate NOT In sphere but in Bounding Box
    candidate_indices:Set[Tuple[Int,Int,Int]] = {}
    for nx in range(bounding_box[0][0],bounding_box[0][1]):
        for ny in range(bounding_box[1][0],bounding_box[1][1]):
            for nz in range(bounding_box[2][0],bounding_box[2][1]):
                coord_vec = index_to_coord((nx,ny,nz),grid.dx,grid.origin)
                if inside_boundary(coord_vec,center_vec,radius): # If inside boundary set to 1
                    flags.store(coord[DType.int32]((nx,ny,nz)),value = Flags.SOLID) 
                else: # If Outside add to possible candidate
                    candidate_indices.add((nx,ny,nz))

    indices = List[Tuple[Int,Int,Int]](length = len(candidate_indices),fill = (-1,-1,-1))

    num_boundary_indices = 0
    for idx in candidate_indices:
        x,y,z = idx
        for q in range(Q): # We iterate and check if adjacent index is inside boundary and break if true
            test_direction:InlineArray[Int,3] = [x,y,z]
            comptime for i in range(D):
                test_direction[i] += Int(latticeModel.directions[q][i])
            coord_test = index_to_coord((test_direction[0],test_direction[1],test_direction[2]),grid.dx,grid.origin)
            if inside_boundary(coord_test,center_vec,radius): # If any direction Q touches a solid mark current point as solid
                indices[num_boundary_indices] = (x,y,z)
                num_boundary_indices += 1
                break

    indices.shrink(num_boundary_indices)
    return (indices)^ # Trnsfer Ownership of data


def index_to_coord[float_dtype:DType]
    (
        grid_index:Tuple[Int,Int,Int],
        grid_spacing:Scalar[float_dtype],
        origin:InlineArray[Scalar[float_dtype],3]
    ) -> Vector[float_dtype,3]:
    out = Vector[float_dtype,3](fill =0)
    comptime for i in range(3):
        out[i] = Scalar[float_dtype](grid_index[i])*grid_spacing + origin[i]
    return out
    





def add_sphere[
    float_dtype:DType,
    flag_origin:Origin[mut=True],
    nx:Int,ny:Int,nz:Int,
    D:Int,
    latticeModel:LatticeModel[D,_,float_dtype,...],
    FlagLayoutType:TensorLayout,
    //,
    grid:LBM_Grid[latticeModel,nx,ny,nz,_]]
    (flags:TileTensor[DType.uint8,FlagLayoutType,flag_origin],center:List[Scalar[float_dtype]],radius:Scalar[float_dtype]) raises:
    '''
    Add a sphere into the LBM domain equivalent to a circle in 2D, Sphere/Ball in 3D
    '''
    comptime assert FlagLayoutType.rank == 3
    if len(center) != 3:
        raise Error('centre must be a list of 3 floats got a len of {} instead'.format(len(center)))
    # b:Tuple[Int,Int,Int] = (0,0,0)
    var bounding_box:List[List[Int]] = []
    
    
    for i in range(3):
        if i < D:
            a_min,a_max = center[i] - radius, center[i] + radius
            n_min,n_max = max(0,Int((a_min-grid.origin[i])//grid.dx)), min(grid.shape[i],Int((a_max-grid.origin[i])//grid.dx + 1))
            bounding_box.append([n_min,n_max])
        else:
            bounding_box.append([0,1]) # This means loop is just set to 0 index

    print(bounding_box)
    # bounding_box.append([0,nx])
    # bounding_box.append([0,ny])
    # bounding_box.append([0,nz])
    

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
    float_dtype:DType,
    flag_origin:Origin[mut=True],
    FlagLayoutType:TensorLayout,
    latticeModel:LatticeModel[_,_,float_dtype,...],
    //,
    grid:LBM_Grid[latticeModel,...],
    ]
    (flags:TileTensor[DType.uint8,FlagLayoutType,flag_origin],center:List[Scalar[float_dtype]],radius:Scalar[float_dtype]) raises:
    
    '''
    Alias for Spere for 2D
    '''
    comptime assert float_dtype == grid.latticeModel.float_dtype
    comptime assert grid.latticeModel.D == 2,'Circle only valid for 2D'
    return add_sphere[grid](flags,center,radius)



def add_box[
    float_dtype:DType,
    flag_origin:Origin[mut=True],
    nx:Int,ny:Int,nz:Int,
    D:Int,Q:Int,
    latticeModel:LatticeModel[D,Q,float_dtype,DType.int32],
    tile_size:Int,
    FlagLayoutType:TensorLayout,
    //,
    grid:LBM_Grid[latticeModel,nx,ny,nz,tile_size]]
    (flags:TileTensor[DType.uint8,FlagLayoutType,flag_origin],center:List[Scalar[float_dtype]],box_radius:List[Scalar[float_dtype]]) raises:
    comptime assert FlagLayoutType.rank == 3
    if len(center) != 3:
        raise Error('centre must be a list of 3 floats got a len of {} instead'.format(len(center)))

    if len(box_radius) != 3:
        raise Error('lengths must be a list of 3 floats got a len of {} instead'.format(len(box_radius)))
    # b:Tuple[Int,Int,Int] = (0,0,0)
    var bounding_box:List[List[Int]] = []
    

    for i in range(3):
        if i < D:
            a_min,a_max = center[i] - box_radius[i], center[i] + box_radius[i]
            n_min,n_max = max(0,Int((a_min-grid.origin[i])//grid.dx)), min(grid.shape[i],Int((a_max-grid.origin[i])//grid.dx + 1))
            bounding_box.append([n_min,n_max])
        else:
            bounding_box.append([0,1]) # This means loop is just set to 0 index

    # bounding_box.append([0,nx])
    # bounding_box.append([0,ny])
    # bounding_box.append([0,nz])
    comptime vec3 = Vector[float_dtype,3]
    comptime vec3_bool = Vector[float_dtype,3]
    comptime float = Scalar[float_dtype]
    var center_vec = vec3(center)
    var coord_vec = vec3(fill=0.)
    var box_radius_vec = vec3_bool(box_radius)

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
    if (center-box_radius <= point).all_true() and (point <= (center + box_radius)).all_true():
        return True
    else:
        return False
