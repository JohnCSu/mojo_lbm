from std.gpu.host import DeviceContext
from layout import TileTensor,LayoutTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from std.collections import InlineArray
from std.collections import Set,Dict
from src.utils import Vector,ContextTileTensor
from std.utils.numerics import nan,isnan
from .units import UnitSystem
        
struct Grid[float_dtype:DType,int_dtype:DType,D:Int,Q:Int,//,
                latticeModel:LatticeModel[D,Q,float_dtype,int_dtype],
                ](ImplicitlyCopyable): 
    comptime Float_Scalar = Scalar[Self.float_dtype]
    
    var nx:Int
    var ny:Int
    var nz:Int
    var tile_size:Int
    
    var BLOCK_SHAPE:Tuple[Int,Int,Int]
    var GRID_DIM:Tuple[Int,Int,Int] 
    var THREADS_PER_BLOCK:Int 
    var n_tiles_x:Int 
    var n_tiles_y:Int 
    var n_tiles_z:Int 

    var dx:Self.Float_Scalar
    var domain_size:Tuple[Self.Float_Scalar,Self.Float_Scalar,Self.Float_Scalar]
    var area:Self.Float_Scalar
    var volume:Self.Float_Scalar
    var shape:InlineArray[Int,3]
    var num_points:Int
    var origin:InlineArray[Self.Float_Scalar,3]
    def __init__(out self,nx:Int,ny:Int,nz:Int,tile_size:Int,dx:Self.Float_Scalar,origin:InlineArray[Self.Float_Scalar,3] = [0.,0.,0.]):
        self.nx,self.ny,self.nz,self.tile_size = (nx,ny,nz,tile_size)
        self.origin = origin
        self.dx = dx

        # Inferred vars
        self.area = dx**2
        self.volume = dx**3
        self.shape = [self.nx,self.ny,self.nz]
        self.num_points = self.nx*self.ny*self.nz
        self.domain_size = ( Self.Float_Scalar(self.nx-1)*dx,self.Float_Scalar(self.ny-1)*dx,self.Float_Scalar(self.nz-1)*dx)
        
        var __shapes = set_block_shape_and_grid_dim(self.nx,self.ny,self.nz,self.D,self.tile_size)
        
        self.BLOCK_SHAPE =  __shapes[0]
        self.GRID_DIM = __shapes[1]
        self.THREADS_PER_BLOCK = self.BLOCK_SHAPE[0]*self.BLOCK_SHAPE[1]*self.BLOCK_SHAPE[2]

        self.n_tiles_x = self.nx//self.tile_size
        self.n_tiles_y = self.ny//self.tile_size if self.D >= 2 else 1
        self.n_tiles_z = self.nz//self.tile_size if self.D == 3 else 1

    

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

    def lattice_model_matches_grid(self) -> Bool:
        return check_model_match_dim(self.latticeModel.D,self.nx,self.ny,self.nz)

    def tile_size_divides_grid_shape(self) ->Bool:
        return (self.nx % self.tile_size == 0 or self.nx == 1) and (self.ny % self.tile_size == 0 or self.ny == 1) and (self.nz % self.tile_size == 0 or self.nz == 1)


    def datacheck(self)-> Bool:
        '''
        Grid is comptime parameter so you should call this in a comptime assert
        '''
        passed_datacheck = True
        if not self.lattice_model_matches_grid():
            print('The given dimension of the LatticeModel does not match that of the dimension of the grid')
            passed_datacheck =False

        if not self.tile_size_divides_grid_shape():
            print('Tile size must divide nx,ny and nz')
            passed_datacheck =False

        return passed_datacheck 


def set_block_shape_and_grid_dim(nx:Int,ny:Int,nz:Int,D:Int,tile_size:Int) -> Tuple[Tuple[Int,Int,Int],Tuple[Int,Int,Int]]:
    # comptime assert (nx % tile_size == 0 or nx == 1) and (ny % tile_size == 0 or ny == 1) and (nz % tile_size == 0 or nz == 1), 'Tile size must divide nx,ny and nz'
    # comptime assert tile_size >= 1
    if tile_size > 1:
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


def check_model_match_dim(D:Int,nx:Int,ny:Int,nz:Int)->Bool:
    grid_D = (1 if nx > 1 else 0) + (1 if ny > 1 else 0) + (1 if nz > 1 else 0)
    return (1 <= D <= 3) and (nx > 0 and ny > 0 and nz > 0) and D == grid_D
    
    
    # comptime assert , 'The given dimension of the LatticeModel does not match that of the dimension of the grid'
    
        