from std.memory import Pointer 
from layout import TileTensor
from layout.tile_layout import TensorLayout
from .geometry.primatives import add_box,add_sphere,get_sphere_boundary_indices
from .LBM import LBM_Grid

struct BoundaryTracker[
    float_dtype:DType,
    int_dtype:DType,
    nx:Int,ny:Int,nz:Int,
    tile_size:Int,
    D:Int,Q:Int,
    latticeModel:LatticeModel[D,Q,float_dtype,int_dtype],
    //,
    grid:LBM_Grid[latticeModel,nx,ny,nz,tile_size] 
    ]():
    var boundary:Dict[String,List[Tuple[Int,Int,Int]]]
    '''Store The fluid nodes adjacent to the node'''

    def __init__(out self):
        self.boundary = Dict[String,List[Tuple[Int,Int,Int]]]()

    
    def add_sphere[FlagLayoutType:TensorLayout,flag_origin:Origin[mut=True]](
    mut self,
    name:String,
    flags:TileTensor[DType.uint8,FlagLayoutType,flag_origin],
    center:List[Scalar[Self.float_dtype]],
    radius:Scalar[Self.float_dtype],
    ) raises :
        self.boundary[name] = get_sphere_boundary_indices[self.grid](flags,center,radius)
        