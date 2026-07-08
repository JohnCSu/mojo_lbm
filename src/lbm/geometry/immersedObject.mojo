from std.memory import Pointer 
from layout import TileTensor,CoordLike
from layout.tile_layout import TensorLayout,Layout
from .primatives import add_box,add_sphere,get_sphere_boundary_indices
from src.lbm import LBM_Grid,LatticeModel
from layout import TileTensor,row_major,col_major,coord
from src.utils import ContextTileTensor
from std.gpu.host import DeviceContext

struct ImmersedObject[
    float_dtype:DType,
    int_dtype:DType,
    nx:Int,ny:Int,nz:Int,
    tile_size:Int,
    D:Int,Q:Int,
    latticeModel:LatticeModel[D,Q,float_dtype,int_dtype],
    //,
    grid:LBM_Grid[latticeModel,nx,ny,nz,tile_size] 
    ]():
    comptime Int_Scalar = Scalar[Self.int_dtype]

    var boundary:Dict[String,List[Self.Int_Scalar]]
    '''Store The fluid nodes adjacent to the node'''

    def __init__(out self):
        self.boundary = Dict[String,List[Self.Int_Scalar]]()
    
    def add_sphere[FlagLayoutType:TensorLayout,flag_origin:Origin[mut=True]](
    mut self,
    name:String,
    flags:TileTensor[DType.uint8,FlagLayoutType,flag_origin],
    center:List[Scalar[Self.float_dtype]],
    radius:Scalar[Self.float_dtype],
    ) raises :
        self.boundary[name] = get_sphere_boundary_indices[self.grid](flags,center,radius)
    
    def to_ContextTileTensor(
        self,
        name:String,
        deviceContext:DeviceContext
        )
        raises 
        -> ContextTileTensor[Self.int_dtype,type_of( row_major(coord[Self.int_dtype]((1,))) ) ]:

        N = Int(len(self.boundary[name]))
        layout = row_major( coord[Self.int_dtype]((N,) ))
        out = ContextTileTensor[Self.int_dtype](deviceContext,layout)
        out.cpu_buffer().enqueue_copy_from(src = Span(self.boundary[name]))
        return out^ # Must take ownership of ContextTileTensor
    




        
