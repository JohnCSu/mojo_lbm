from std.gpu.host import DeviceContext
from layout import TileTensor,LayoutTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from std.collections import InlineArray
from std.collections import Set,Dict
from src.utils import Vector,ContextTileTensor
from std.utils.numerics import nan,isnan
from .units import UnitSystem
        
struct LBM_Grid[float_dtype:DType,int_dtype:DType,D:Int,Q:Int,//,
                latticeModel:LatticeModel[D,Q,float_dtype,int_dtype],
                nx:Int,ny:Int,nz:Int,
                tile_size:Int,
                ](ImplicitlyCopyable): 
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
    
        



def set_exterior_walls[float_dtype:DType,
                    flag_origin:Origin[mut=True],
                    bc_origin:Origin[mut=True],
                    nx:Int,ny:Int,nz:Int,
                    D:Int,Q:Int,
                    latticeModel:LatticeModel[D,Q,float_dtype,DType.int32],
                    FlagLayoutType:TensorLayout,
                    BCLayoutType:TensorLayout,
                    //,
                    grid:LBM_Grid[latticeModel,nx,ny,nz,_],
                    config:LBM_Config = LBM_Config()
                    ]
                    (flags:TileTensor[DType.uint8,FlagLayoutType,flag_origin],
                            bc:TileTensor[float_dtype,BCLayoutType,bc_origin],
                            side:String,
                            boundary_type:Scalar[DType.uint8],
                            u:List[Scalar[float_dtype]] = [],
                            rho:Scalar[float_dtype] = nan[float_dtype](),
                            unitSystem:Optional[UnitSystem[float_dtype,D]] = None) raises:
    '''
    Apply Boundary conditions to exterior walls to flags TileTensor. Default assumes boundary values given
    are in lattice units unless a unitSystem is passed in.

    Parameters:
        grid: LBM_Grid struct that defines the domain for LBM.
        config: LBM_Config struct that contain a set of valid boundaries flags is allowed to take

    Args:
        flags: TileTensor of uint8 that describe what each node in the lattice is (e.g. 0 - Fluid).
        bc: TileTensor that stores the velocity and density for each node.
        side: String that sets which exterior wall to apply the BC to.
        boundary_type: Uint8 value to set the flags at the target wall to.
        u: List of floats to set the velocity. Default empty list implies the velocity is free.
        rho: Float Scalar to set the density. If Nan then implies density is a free variable.
        unitSystem: Optional System of units to use. If passed in, it is assume that the values u and rho
            are in physical units.

    Raises:
        - both u and rho are not specified.
        - velocity list length does not match that of the grid dimension.
        - boundary type is not a boundary type in LBM_Config parameter
        - side is not a valid string
        
    '''
    comptime assert float_dtype.is_floating_point()
    comptime assert FlagLayoutType.rank == 3 and BCLayoutType.rank == 4
    
    VALID_BOUNDARIES = materialize[config.INCLUDED_BCs]()
    
    axes:Dict[String,Int] = {'X':0,
                    'Y':1,
                    'Z':2,}
    valid_strings:Set[String] = {'-X','+X','-Y','+Y','-Z','+Z'}

    u_is_empty = (len(u) == 0)
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
    
    if unitSystem: # if not None then implies bc give are not in 
        density *=unitSystem.value().density.C_phys_to_lat()
        velocity = [unitSystem.value().U.C_phys_to_lat()*u for u in velocity]

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



def set_exterior_walls_with_func[
                    float_dtype:DType,
                    flag_origin:Origin[mut=True],
                    bc_origin:Origin[mut=True],
                    nx:Int,ny:Int,nz:Int,
                    D:Int,Q:Int,
                    latticeModel:LatticeModel[D,Q,float_dtype,DType.int32],
                    FlagLayoutType:TensorLayout,
                    BCLayoutType:TensorLayout,
                    //,
                    grid:LBM_Grid[latticeModel,nx,ny,nz,_],
                    config:LBM_Config = LBM_Config(),
                    *,
                    u: def[float_dtype:DType,D:Int]
                        (Scalar[float_dtype],Scalar[float_dtype],Scalar[float_dtype],mut InlineArray[Scalar[float_dtype],D]) 
                        capturing
                    # u: Optional[def[float_dtype:DType,D:Int]
                    #     (Scalar[float_dtype],Scalar[float_dtype],Scalar[float_dtype],mut InlineArray[Scalar[float_dtype],D]) 
                    #     capturing] = None
                    ]
                    (flags:TileTensor[DType.uint8,FlagLayoutType,flag_origin],
                            bc:TileTensor[float_dtype,BCLayoutType,bc_origin],
                            side:String,
                            boundary_type:Scalar[DType.uint8],
                            unitSystem:Optional[UnitSystem[float_dtype,D]],
                            rho:Scalar[float_dtype] = nan[float_dtype](),
                            ) raises:
    '''
    Apply Boundary conditions to exterior walls to flags TileTensor. Default assumes boundary values given
    are in lattice units unless a unitSystem is passed in.

    Parameters:
        grid: LBM_Grid struct that defines the domain for LBM.
        config: LBM_Config struct that contain a set of valid boundaries flags is allowed to take

    Args:
        flags: TileTensor of uint8 that describe what each node in the lattice is (e.g. 0 - Fluid).
        bc: TileTensor that stores the velocity and density for each node.
        side: String that sets which exterior wall to apply the BC to.
        boundary_type: Uint8 value to set the flags at the target wall to.
        u: List of floats to set the velocity. Default empty list implies the velocity is free.
        rho: Float Scalar to set the density. If Nan then implies density is a free variable.
        unitSystem: Optional System of units to use. If passed in, it is assume that the values u and rho
            are in physical units.

    Raises:
        - both u and rho are not specified.
        - velocity list length does not match that of the grid dimension.
        - boundary type is not a boundary type in LBM_Config parameter
        - side is not a valid string
        
    '''
    comptime assert float_dtype.is_floating_point()
    comptime assert FlagLayoutType.rank == 3 and BCLayoutType.rank == 4
    # comptime assert u is not None
    comptime u_func = u
    VALID_BOUNDARIES = materialize[config.INCLUDED_BCs]()
    
    axes:Dict[String,Int] = {'X':0,
                    'Y':1,
                    'Z':2,}
    valid_strings:Set[String] = {'-X','+X','-Y','+Y','-Z','+Z'}

    density:Scalar[float_dtype] = rho

    if (boundary_type ==  SOLID_NODE) and (isnan(rho)):
        raise Error('For Solid Type you must specify both u and rho')

    if boundary_type not in VALID_BOUNDARIES:
        raise Error('Input Boundary Type was {} but valid boundary types are: {}'.format(boundary_type,VALID_BOUNDARIES))

    if side not in valid_strings:
        raise Error('Side not valid. Input was {} but expects {}'.format(side,valid_strings))
    
    if unitSystem: # if not None then implies bc give are not in 
        density *=unitSystem.value().density.C_phys_to_lat()
        # velocity = [unitSystem.value().U.C_phys_to_lat()*u for u in velocity]

    axis = axes[String(side[byte = 1])]
    end_values = [nx,ny,nz]
    
    if side[byte = 0] == '-':
        fixed = 0
    else:
        fixed = end_values[axis] - 1

    conversion_factor = unitSystem.value().U.C_phys_to_lat() if unitSystem else 1.
    if axis == 0: # X-axis, fix x and loop
        x = fixed
        for y in range(ny):
            for z in range(nz):
                flags.store(coord[DType.int32]((x,y,z)),flags.ElementType(boundary_type))
                velocity = InlineArray[Scalar[float_dtype],D](fill =0)
                grid_coords =  grid.get_grid_coordinates(x,y,z)
                u_func(grid_coords[0],grid_coords[1],grid_coords[2],velocity)
                comptime for i in range(D):
                    bc.store(coord[DType.int32]((x,y,z,i)),velocity[i]*(conversion_factor) ) 
                bc.store(coord[DType.int32]((x,y,z,D)),density) 
    elif axis == 1:
        y = fixed
        for x in range(nx):
            for z in range(nz):
                flags.store(coord[DType.int32]((x,y,z)),flags.ElementType(boundary_type))
                velocity = InlineArray[Scalar[float_dtype],D](fill =0)
                grid_coords =  grid.get_grid_coordinates(x,y,z)
                u_func(grid_coords[0],grid_coords[1],grid_coords[2],velocity)
                comptime for i in range(D):
                    bc.store(coord[DType.int32]((x,y,z,i)),velocity[i]*(conversion_factor) ) 
                bc.store(coord[DType.int32]((x,y,z,D)),density) 
    else: # Loop Z-face
        z = fixed
        for x in range(nx):
            for y in range(ny):
                flags.store(coord[DType.int32]((x,y,z)),flags.ElementType(boundary_type))
                velocity = InlineArray[Scalar[float_dtype],D](fill =0)
                grid_coords =  grid.get_grid_coordinates(x,y,z)
                u_func(grid_coords[0],grid_coords[1],grid_coords[2],velocity)
                comptime for i in range(D):
                    bc.store(coord[DType.int32]((x,y,z,i)),velocity[i]*(conversion_factor) ) 
                bc.store(coord[DType.int32]((x,y,z,D)),density) 




    


