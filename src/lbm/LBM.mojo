from std.gpu.host import DeviceContext
from layout import TileTensor,LayoutTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from std.math import ceildiv
from std.collections import InlineArray
from std.memory import Pointer
from std.collections import Set,Dict
from src.utils import Vector,ContextTileTensor
from std.utils.numerics import nan,isnan

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
    
        
struct LBM_Grid[float_dtype:DType,int_dtype:DType,D:Int,Q:Int,//,
                latticeModel:LatticeModel[D,Q,float_dtype,int_dtype],
                nx:Int,ny:Int,nz:Int,
                tile_size:Int,
                ](): 
    comptime float_scalar = Scalar[Self.float_dtype]
    comptime __shapes = set_block_shape_and_grid_dim[Self.nx,Self.ny,Self.nz,Self.D,Self.tile_size]()
    comptime BLOCK_SHAPE =  Self.__shapes[0]
    comptime GRID_DIM = Self.__shapes[1]
    comptime THREADS_PER_BLOCK = Self.BLOCK_SHAPE[0]*Self.BLOCK_SHAPE[1]*Self.BLOCK_SHAPE[2]
    comptime n_tiles_x = Self.nx//Self.tile_size
    comptime n_tiles_y = Self.ny//Self.tile_size if Self.D >= 2 else 1
    comptime n_tiles_z = Self.nz//Self.tile_size if Self.D == 3 else 1

    var dx:Self.float_scalar
    var domain_size:Tuple[Self.float_scalar,Self.float_scalar,Self.float_scalar]
    var area:Self.float_scalar
    var volume:Self.float_scalar
    var shape:InlineArray[Int,3]
    var num_points:Int
    var f_field_size: Int
    var vel_field_size: Int
    var bc_field_size:Int
    
    def __init__(out self,dx:Self.float_scalar):
        check_model_match_dim[Self.D,Self.nx,Self.ny,Self.nz]()
        self.dx = dx
        self.area = dx**2
        self.volume = dx**3
        self.shape = [Self.nx,Self.ny,Self.nz]
        self.num_points = Self.nx*Self.ny*Self.nz
        self.f_field_size = Self.Q*self.num_points
        self.vel_field_size = Self.D*self.num_points
        self.bc_field_size = (Self.D+1)*self.num_points
        self.domain_size = ( Self.float_scalar(Self.nx-1)*dx,Self.float_scalar(Self.ny-1)*dx,Self.float_scalar(Self.nz-1)*dx)
             



def set_exterior_walls[float_dtype:DType,
                    flag_origin:Origin[mut=True],
                    bc_origin:Origin[mut=True],
                    nx:Int,ny:Int,nz:Int,
                    D:Int,Q:Int,
                    latticeModel:LatticeModel[D,Q,float_dtype,DType.int32],
                    tile_size:Int,
                    FlagLayoutType:TensorLayout,
                    BCLayoutType:TensorLayout,
                    //,
                    grid:LBM_Grid[latticeModel,nx,ny,nz,tile_size],
                    config:LBM_Config = LBM_Config()
                    ]
                    (flags:TileTensor[DType.uint8,FlagLayoutType,flag_origin],
                            bc:TileTensor[float_dtype,BCLayoutType,bc_origin],
                            side:String,
                            boundary_type:Scalar[DType.uint8],
                            u:List[Scalar[float_dtype]] = [],
                            rho:Scalar[float_dtype] = nan[float_dtype]() ) raises:
    '''
    Apply Boundary conditions to exterior walls
    '''
    comptime assert float_dtype.is_floating_point()
    comptime assert FlagLayoutType.rank == 3 and BCLayoutType.rank == 4
    
    VALID_BOUNDARIES = materialize[config.INCLUDED_BCs]()
    
    axes:Dict[String,Int] = {'X':0,
                    'Y':1,
                    'Z':2,}
    valid_strings:Set[String] = {'-X','+X','-Y','+Y','-Z','+Z'}

    u_is_empty = (len(u) == 0)
    # print(u_is_empty)
    # Error Checking
    if u_is_empty and isnan(rho):
        raise Error('Either velocity or density or both have to be specified. Both cant be left as None')
    
    velocity = [nan[float_dtype]() for _ in range(D)] if u_is_empty else u.copy()
    density:Scalar[float_dtype] = rho

    if (boundary_type ==  SOLID_NODE) and (u_is_empty or isnan(rho)):
        raise Error('For Solid Type you must specify both u and rho')

    if len(velocity) != D:
        raise Error('Input velocity list was of length {} but Grid is {} Dimensional'.format(len(velocity),D))

    if boundary_type not in VALID_BOUNDARIES:
        raise Error('Input Boundary Type was {} but valid boundary types are: {}'.format(boundary_type,VALID_BOUNDARIES))

    if side not in valid_strings:
        raise Error('Side not valid. Input was {} but expects {}'.format(side,valid_strings))
    
    axis = axes[String(side[byte = 1])]
    end_values = [nx,ny,nz]
    
    if side[byte = 0] == '-':
        fixed = 0
    else:
        fixed = end_values[axis] - 1
    if axis == 0: # X-axis, fix x and loop
        x = fixed
        for y in range(ny):
            for z in range(nz):
                flags.store(coord[DType.int32]((x,y,z)),flags.ElementType(boundary_type))
                # flags_lt[fixed,y,z] = flags.ElementType(boundary_type)
                comptime for i in range(D):
                    bc.store(coord[DType.int32]((x,y,z,i)),velocity[i]) 
                bc.store(coord[DType.int32]((x,y,z,D)),density) 
    elif axis == 1:
        y = fixed
        for x in range(nx):
            for z in range(nz):
                flags.store(coord[DType.int32]((x,y,z)),flags.ElementType(boundary_type))
                comptime for i in range(D):
                    bc.store(coord[DType.int32]((x,y,z,i)),velocity[i]) 
                bc.store(coord[DType.int32]((x,y,z,D)),density)
    else: # Loop Z-face
        z = fixed
        for x in range(nx):
            for y in range(ny):
                flags.store(coord[DType.int32]((x,y,z)),flags.ElementType(boundary_type))
                comptime for i in range(D):
                    bc.store(coord[DType.int32]((x,y,z,i)),velocity[i]) 
                bc.store(coord[DType.int32]((x,y,z,D)),density)






    


