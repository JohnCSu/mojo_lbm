from src.lbm import LBM_Grid,LatticeModel
from src.lbm.flags import Flags
from std.gpu.host import DeviceContext
from layout import TileTensor,LayoutTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from src.utils import Vector


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
    (flags:TileTensor[DType.uint8,FlagLayoutType,flag_origin],center:List[Scalar[float_dtype]],radius:Scalar[float_dtype]) raises:
    pass