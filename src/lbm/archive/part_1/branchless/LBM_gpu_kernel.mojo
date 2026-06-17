from std.gpu import block_dim,block_idx,thread_idx,barrier
from layout import TileTensor,LayoutTensor
from layout.tile_tensor import stack_allocation
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from std.gpu.memory import AddressSpace
from src.lbm.lattice_models import LatticeModel
from src.lbm.LBM import LBM_Grid
from src.lbm.flags import SOLID_NODE,FLUID_NODE
from src.utils import Vector,ContextTileTensor


def LBM_kernel[ float_dtype:DType,D:Int,Q:Int,
                lattice_model:LatticeModel[D,Q,float_dtype,DType.int32],
                nx:Int,ny:Int,nz:Int,
                tile_size:Int,
                //,
                grid: LBM_Grid[lattice_model,nx,ny,nz,tile_size], 
                Flayout:Layout[...] where Flayout.rank == 4,  
                BClayout:Layout[...] where BClayout.rank == 4,
                Flaglayout:Layout[...] where Flaglayout.rank == 3,
                *,
                reorder_threads:Bool = True
                ]
                ( 
                f_out:TileTensor[float_dtype,type_of(Flayout),MutAnyOrigin],
                f_in:TileTensor[float_dtype,type_of(Flayout),MutAnyOrigin],
                bc:TileTensor[float_dtype,type_of(BClayout),MutAnyOrigin],
                flags:TileTensor[DType.uint8,type_of(Flaglayout),MutAnyOrigin],
                inv_tau:Scalar[float_dtype]
                ):
    '''
    From part_1/tiled. Default layout should be Col_major Tile and ORw major tiler. Uses Compile time unrolling for last 2 for loops. Main Stream/BC comptime looping does not work.
    ''' 
    # Convience Variable Names and constants
    comptime weights = lattice_model.weights
    comptime float_directions = lattice_model.float_directions
    comptime directions = lattice_model.directions
    comptime opposite_index = lattice_model.opposite_indices
    comptime grid_shape = Vector[DType.int32,3](Int32(nx),Int32(ny),Int32(nz))

    # Conversion to Layout Tensor as tiletensor.to_layout_tensor() does not support nested
    comptime f_as_lt = LayoutTensor[float_dtype,Flayout.to_layout(),MutAnyOrigin]
    comptime flag_as_lt = LayoutTensor[DType.uint8,Flaglayout.to_layout(),MutAnyOrigin]
    comptime bc_as_lt = LayoutTensor[float_dtype,BClayout.to_layout(),MutAnyOrigin]
    comptime assert f_as_lt.element_size == 1 and flag_as_lt.element_size == 1 and bc_as_lt.element_size == 1
    
    f_in_lt = f_as_lt(f_in.ptr)
    f_out_lt = f_as_lt(f_out.ptr)
    flags_lt = flag_as_lt(flags.ptr)
    bc_lt = bc_as_lt(bc.ptr)
    comptime if reorder_threads:
        comptime if D==1: # Indexing based on Dimension of Grid
            x = block_dim.x * block_idx.x + thread_idx.x
            y,z = 0,0
        elif D == 2:
            x = block_dim.y * block_idx.y + thread_idx.y
            y = block_dim.x * block_idx.x + thread_idx.x
            z = 0
        else:
            x = block_dim.z * block_idx.z + thread_idx.z
            y = block_dim.y * block_idx.y + thread_idx.y
            z = block_dim.x * block_idx.x + thread_idx.x
    else:
        x = block_dim.x * block_idx.x + thread_idx.x
        y = block_dim.y * block_idx.y + thread_idx.y
        z = block_dim.z * block_idx.z + thread_idx.z
        
    index = Vector[DType.int32,3](Int32(x),Int32(y),Int32(z))

    # Main Compute
    if (index[0] < grid_shape[0]) and (index[1] < grid_shape[1]) and (index[2] < grid_shape[2]): # Basic Guard
        var f_new = Vector[float_dtype,Q](fill = 0.)
        var velocity = Vector[float_dtype,D]()
        comptime for q in range(Q):
            f_opp = f_in_lt.load_scalar(opposite_index[q],x,y,z) # Need this as  Element Type is a Simd Vec of size 1
            direction = directions[q]
            
            pull_index = get_adjacent_idx[D,-1](index,grid_shape,direction) # Pulling Scheme
            pulled_f = f_in_lt.load_scalar(q,pull_index[0],pull_index[1],pull_index[2])            
            pulled_flag = flags_lt.load_scalar(pull_index[0],pull_index[1],pull_index[2])

            if pulled_flag == FLUID_NODE: # Stream
                f_new[q] = pulled_f
            elif pulled_flag == SOLID_NODE: # BounceBack with moving wall BC put together (2nd term is 0 if stationary wall)
                comptime for ii in range(D):
                    velocity[ii] = bc_lt.load_scalar(pull_index[0],pull_index[1],pull_index[2],ii)
                rho = bc_lt.load_scalar(pull_index[0],pull_index[1],pull_index[2],D)
                f_new[q] = f_opp + 2.*3.*weights[q]*rho*(float_directions[q].dot(velocity))
                
        # Get Velocity and Density
        velocity.fill(0)
        rho = 0
        comptime for q in range(Q):
            rho += f_new[q]
            velocity += f_new[q]*float_directions[q]
        velocity /= rho
        # Collision Term
        comptime for q in range(Q):
            f_eq = SRT(weights[q],rho,velocity,float_directions[q])            
            f_out_lt[q,x,y,z] = f_new[q] -  inv_tau*(f_new[q]- f_eq)



def get_adjacent_idx[D:Int,shift:Int32 = 1](index:Vector[DType.int32,3],grid_shape:Vector[DType.int32,3],direction:Vector[DType.int32,D],) -> Vector[DType.int32,3]:
    comptime assert D <= 3 
    adj_index = Vector[DType.int32,3]()
    comptime for d in range(D):
        adj_index[d] = (index[d] + shift*direction[d]) % grid_shape[d]
    return adj_index


def SRT[dtype:DType,D:Int,//](weight:Scalar[dtype],density:Scalar[dtype],velocity:Vector[dtype,D],direction:Vector[dtype,D]) -> Scalar[dtype]:
    comptime assert dtype.is_floating_point(), 'DType to BGK_collision term should be Float point like' # Weied using where statement cause compile error?
    ei_dot_u = velocity.dot(direction)
    return weight*density*(1 + 3.*ei_dot_u + 4.5*ei_dot_u*ei_dot_u - 1.5*velocity.dot(velocity))

