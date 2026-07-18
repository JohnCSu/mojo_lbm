from std.gpu.host import DeviceContext
from layout import TileTensor,LayoutTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from std.collections import InlineArray
from std.collections import Set,Dict
from src.utils import Vector,ContextTileTensor
from std.utils.numerics import nan,isnan
from .units import UnitSystem

trait GridLike:
    comptime float_dtype:DType
    comptime int_dtype:DType
    comptime D:Int
    comptime Q:Int
    comptime nx:Int
    comptime ny:Int
    comptime nz:Int
    comptime tile_size:Int
    comptime lattice_model:LatticeModel[Self.D,Self.Q,Self.float_dtype,Self.int_dtype]

    @staticmethod
    def datacheck() -> Bool:
        '''
        Method that does checking before hand
        '''
        comptime assert (Self.nx % Self.tile_size == 0 or Self.nx == 1) and (Self.ny % Self.tile_size == 0 or Self.ny == 1) and (Self.nz % Self.tile_size == 0 or Self.nz == 1)
        return True






    