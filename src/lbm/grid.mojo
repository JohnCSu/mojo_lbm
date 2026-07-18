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


struct LBM_Grid[float_dtype_:DType,int_dtype_:DType,D_:Int,Q_:Int,//,
                lattice_model_:LatticeModel[D_,Q_,float_dtype_,int_dtype_],
                nx_:Int,ny_:Int,nz_:Int,
                tile_size_:Int,
                ](ImplicitlyCopyable & GridLike): 
    comptime float_dtype:DType = Self.float_dtype_
    comptime int_dtype:DType = Self.int_dtype_
    comptime D:Int = Self.D_
    comptime Q:Int = Self.Q_
    comptime nx:Int = Self.nx_
    comptime ny:Int = Self.ny_
    comptime nz:Int = Self.nz_
    comptime tile_size:Int = Self.tile_size_
    comptime lattice_model = Self.lattice_model_

    # comptime float_dtype:DType = Self.float_dtype
    comptime Float_Scalar = Scalar[Self.float_dtype]
    comptime __shapes = set_block_shape_and_grid_dim[Self.nx,Self.ny,Self.nz,Self.D,Self.tile_size]()
    comptime BLOCK_SHAPE =  Self.__shapes[0]
    comptime GRID_DIM = Self.__shapes[1]
    comptime THREADS_PER_BLOCK = Self.BLOCK_SHAPE[0]*Self.BLOCK_SHAPE[1]*Self.BLOCK_SHAPE[2]
    comptime n_tiles_x = Self.nx//Self.tile_size
    comptime n_tiles_y = Self.ny//Self.tile_size if Self.D >= 2 else 1
    comptime n_tiles_z = Self.nz//Self.tile_size if Self.D == 3 else 1

    var dx:Self.Float_Scalar
    var domain_size:Tuple[Self.Float_Scalar,Self.Float_Scalar,Self.Float_Scalar]
    var area:Self.Float_Scalar
    var volume:Self.Float_Scalar
    var shape:InlineArray[Int,3]
    var num_points:Int
    var f_field_size: Int
    var vel_field_size: Int
    var bc_field_size:Int
    var origin:InlineArray[Self.Float_Scalar,3]
    def __init__(out self,dx:Self.Float_Scalar,origin:InlineArray[Self.Float_Scalar,3] = [0.,0.,0.]):
        check_model_match_dim[Self.D,Self.nx,Self.ny,Self.nz]()
        self.dx = dx
        self.area = dx**2
        self.volume = dx**3
        self.shape = [Self.nx,Self.ny,Self.nz]
        self.num_points = Self.nx*Self.ny*Self.nz
        self.f_field_size = Self.Q*self.num_points
        self.vel_field_size = Self.D*self.num_points
        self.bc_field_size = (Self.D+1)*self.num_points
        self.domain_size = ( Self.Float_Scalar(Self.nx-1)*dx,Self.Float_Scalar(Self.ny-1)*dx,Self.Float_Scalar(Self.nz-1)*dx)
        self.origin = origin


    def get_grid_coordinates(self,i:Int,j:Int,k:Int) -> InlineArray[Scalar[Self.float_dtype],3]:
        out:InlineArray[Scalar[Self.float_dtype],3] = [0,0,0]
        grid_index = (i,j,k)
        comptime for i in range(3):
            out[i] = Scalar[Self.float_dtype](grid_index[i])*self.dx + self.origin[i]
        return out

    def get_UnitSystem_with_Re(
        self,
        U_phys:Self.Float_Scalar,
        U_lattice:Self.Float_Scalar,
        L_phys:Self.Float_Scalar,
        Re:Self.Float_Scalar,
        density:Self.Float_Scalar = 1.) -> UnitSystem[Self.float_dtype,Self.D]:

        L_lattice = L_phys/self.dx
        kinematic_viscosity = U_phys*L_phys/Re
        return UnitSystem[Self.float_dtype,Self.D](U_phys,U_lattice,L_phys,L_lattice,density,kinematic_viscosity)

    def get_UnitSystem(
        self,
        U_phys:Self.Float_Scalar,
        U_lattice:Self.Float_Scalar,
        L_phys:Self.Float_Scalar,
        kinematic_viscosity:Self.Float_Scalar,
        density:Self.Float_Scalar = 1.) -> UnitSystem[Self.float_dtype,Self.D]:
        L_lattice = L_phys/self.dx
        return UnitSystem[Self.float_dtype,Self.D](U_phys,U_lattice,L_phys,L_lattice,density,kinematic_viscosity)




def set_block_shape_and_grid_dim[nx:Int,ny:Int,nz:Int,D:Int,tile_size:Int]() -> Tuple[Tuple[Int,Int,Int],Tuple[Int,Int,Int]]:
    comptime assert (nx % tile_size == 0 or nx == 1) and (ny % tile_size == 0 or ny == 1) and (nz % tile_size == 0 or nz == 1), 'Tile size must divide nx,ny and nz'
    comptime assert tile_size >= 1
    comptime if tile_size > 1:
        block_shape:Tuple[Int,Int,Int] = (tile_size, tile_size if D >= 2 else 1, tile_size if D == 3 else 1)
        grid_dim:Tuple[Int,Int,Int] = (nx//tile_size, ny//tile_size if D >= 2 else 1, nz//tile_size if D == 3 else 1)

    else:
        if D == 1 :
            g_dim = 256
        elif D == 2:
            g_dim = 16 # 2D Block has 256 Threads
        else:
            g_dim = 8 # 3D block has 512 threads

        def calc_grid_dim(n:Int,g:Int) -> Int:
            return n//g if n % g == 0 else n//g + 1

        block_shape:Tuple[Int,Int,Int] = (g_dim, g_dim if D >= 2 else 1, g_dim if D == 3 else 1)


        grid_dim:Tuple[Int,Int,Int] = (calc_grid_dim(nx,block_shape[0]), calc_grid_dim(ny,block_shape[1]), calc_grid_dim(nz,block_shape[2]))
        
    return block_shape,grid_dim


def check_model_match_dim[D:Int,nx:Int,ny:Int,nz:Int]():
    comptime assert 1 <= D <= 3
    comptime assert nx > 0 and ny > 0 and nz > 0
    comptime grid_D = (1 if nx > 1 else 0) + (1 if ny > 1 else 0) + (1 if nz > 1 else 0)
    comptime assert D == grid_D, 'The given dimension of the LatticeModel does not match that of the dimension of the grid'
    
        



