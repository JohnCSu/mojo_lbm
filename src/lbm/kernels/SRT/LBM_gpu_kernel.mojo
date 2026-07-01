from std.gpu import block_dim,block_idx,thread_idx,barrier
from layout import TileTensor,LayoutTensor,coord
from layout.tile_tensor import stack_allocation
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from std.gpu.memory import AddressSpace
from src.lbm import LBM_Grid,LBM_Config,LatticeModel
from src.lbm.flags import SOLID_NODE,FLUID_NODE,Flags
from src.utils import Vector,ContextTileTensor
from .load_and_store import load_f,store_f
from std.utils.numerics import nan,isnan

def LBM_kernel[ float_dtype:DType,D:Int,Q:Int,
                lattice_model:LatticeModel[D,Q,float_dtype,DType.int32],
                nx:Int,ny:Int,nz:Int,tile_size:Int,
                //,
                grid: LBM_Grid[lattice_model,nx,ny,nz,tile_size],
                Flayout:Layout[...],
                BClayout:Layout[...],
                Flaglayout:Layout[...],
                config:LBM_Config = LBM_Config(),
                *,
                f_dtype:DType = config.f_dtype.value() if config.f_dtype is not None else float_dtype
                ]
                (
                f_out:TileTensor[f_dtype,type_of(Flayout),MutAnyOrigin],
                f_in:TileTensor[f_dtype,type_of(Flayout),ImmutAnyOrigin],
                bc:TileTensor[float_dtype,type_of(BClayout),ImmutAnyOrigin],
                flags:TileTensor[DType.uint8,type_of(Flaglayout),ImmutAnyOrigin],
                tau:Scalar[float_dtype]
                )
                where tile_size >= 1 and Flayout.rank == 4 and BClayout.rank == 4 and Flaglayout.rank == 3:
    '''
    Base LBM to also handle 3D and non_square Grids. Key assumption is that block dim == tile-size 
    i.e. grid can be non-square but block is squre (same block dim in each x,y,z).
    ''' 
    # Convience Variable Names and constants
    comptime weights = lattice_model.weights
    comptime float_directions = lattice_model.float_directions
    comptime directions = lattice_model.directions
    comptime opposite_index = lattice_model.opposite_indices
    comptime grid_shape:InlineArray[Int,3] = [nx,ny,nz]
    comptime load_f_from_xyzq = load_f[float_dtype,config.use_float16c]

    # Comptime asserts
    comptime assert not directions[0].all_true(), 'The first direction for the lattice model should be all 0s i.e directions[0]=[0,0,0]'


    block_x,block_dim_x = block_idx.x,block_dim.x
    block_y,block_dim_y = block_idx.y,block_dim.y
    block_z,block_dim_z = block_idx.z,block_dim.z

    local_x = thread_idx.x
    local_y = thread_idx.y
    local_z = thread_idx.z
    
    x = block_x*block_dim_x + local_x
    y = block_y*block_dim_y + local_y
    z = block_z*block_dim_z + local_z
    
    var index:InlineArray[Int,3] = [x,y,z]    
    var pull_flags = InlineArray[UInt8,Q](uninitialized = True)
    var pull_indices = InlineArray[InlineArray[Int,3],Q](uninitialized = True)
    # Main Compute
    if (index[0] < grid_shape[0]) and (index[1] < grid_shape[1]) and (index[2] < grid_shape[2]): # Basic Guard
        var f_new = Vector[float_dtype,Q](fill = 0.)
        var velocity = Vector[float_dtype,D](uninitialized = True)
        var rho:Scalar[float_dtype] = 0
        
        # Streaming Step
        comptime for q in range(Q):
            direction = directions[q]
            pull_indices[q] = get_adjacent_idx[D,-1](index,grid_shape,direction) # Pulling Scheme
            pull_flags[q] = flags.load(coord[DType.uint32]((pull_indices[q][0],pull_indices[q][1],pull_indices[q][2])))[0]
            f_new[q] =  load_f_from_xyzq(f_in,pull_indices[q],q)
        
        # Bounce Back
        comptime for q in range(Q):
            if pull_flags[q] == SOLID_NODE:
                opp_q = Int(opposite_index[q]) 
                comptime for ii in range(D):
                    velocity[ii] = bc.load(coord[DType.uint32]((pull_indices[q][0],pull_indices[q][1],pull_indices[q][2],ii)))[0]
                rho = bc.load(coord[DType.uint32]((pull_indices[q][0],pull_indices[q][1],pull_indices[q][2],D)))[0]
                # Bounceback is always included
                f_new[q] = load_f_from_xyzq(f_in,index,opp_q) + 2.*3.*weights[q]*rho*(float_directions[q].dot(velocity))
        
        # Equilibrium BC
        comptime if Flags.EQUILIBRIUM in config.INCLUDED_BCs:
            current_flag = pull_flags[0] # comptime assert gurantees this is the flag for the current node
            if current_flag  == Flags.EQUILIBRIUM: 
                comptime for ii in range(D):
                    velocity[ii] = bc.load(coord[DType.uint32]((x,y,z,ii)))[0]
                rho = bc.load(coord[DType.uint32]((x,y,z,D)))[0]

                rho_local,u_l = get_eq_density_and_velocity[config.DDF_shift](f_new,float_directions,weights,index,pull_indices)
                
                u_local = u_l if isnan(velocity[0]) else velocity # nan means the vel is free
                rho_local = rho_local if isnan(rho) else rho # Nan means density is free
                u_dot_u = u_local.dot(u_local)
                comptime for q in range(Q):
                    f_eq = SRT[config.DDF_shift](weights[q],rho_local,u_local,u_dot_u,float_directions[q])
                    f_new[q] = f_eq 

        # Get Velocity and Density
        velocity.fill(0)
        rho = 0
        comptime for q in range(Q):    
            rho += f_new[q]
            velocity += f_new[q]*float_directions[q]
        comptime if config.DDF_shift:
            rho += 1
        velocity /= rho

        # Collision Term
        u_dot_u = velocity.dot(velocity)
        comptime for q in range(Q):
            f_eq = SRT[config.DDF_shift](weights[q],rho,velocity,u_dot_u,float_directions[q])
            store_f[config.use_float16c](f_out,(f_new[q] -  (f_new[q]- f_eq)/tau ),index,q)



@always_inline
def get_eq_density_and_velocity[
    float_dtype:DType,D:Int,Q:Int,//,
    DDF_shift:Bool = False]
    (
        f_vec:Vector[float_dtype,Q],
        float_directions:InlineArray[Vector[float_dtype,D],Q],
        weights:Vector[float_dtype,Q],
        index:InlineArray[Int,3],
        pull_indices:InlineArray[InlineArray[Int,3],Q]) 
    -> Tuple[Scalar[float_dtype],Vector[float_dtype,D]]:

    var velocity = Vector[float_dtype,D](fill = 0.)
    var rho:Scalar[float_dtype] = 0

    comptime for q in range(Q):
        comptime if DDF_shift:
            rest_f:Scalar[float_dtype] = 0.
        else:
            rest_f = weights[q]
            
        is_oob = False
        comptime for i in range(3):
            # So if any of the indices wraps is_oob is always trye
            is_oob = ((abs(pull_indices[q][i] - index[i]) > 1) or is_oob) 

        # We set unknown fs (i.e from out of bounds/wrapped around fs) to rest value
        fq = rest_f if is_oob else f_vec[q]
        rho += fq
        velocity += fq*float_directions[q]

    comptime if DDF_shift:
        rho += 1
    velocity /= rho
    return rho,velocity


@always_inline
def get_adjacent_idx[D:Int,shift:Int = 1](index:InlineArray[Int,3],grid_shape:InlineArray[Int,3],direction:Vector[DType.int32,D],) -> InlineArray[Int,3]:
    comptime assert D <= 3 
    adj_index = InlineArray[Int,3](fill = 0 )
    comptime for d in range(D):
        adj_index[d] = (index[d] + shift*Int(direction[d])) % grid_shape[d]
    return adj_index

@always_inline
def SRT[dtype:DType,D:Int,//,DDF_shift:Bool = False](weight:Scalar[dtype],density:Scalar[dtype],velocity:Vector[dtype,D],u_dot_u:Scalar[dtype],direction:Vector[dtype,D]) -> Scalar[dtype]:
    comptime assert dtype.is_floating_point(), 'DType to BGK_collision term should be Float point like' # Weied using where statement cause compile error?
    ei_dot_u = velocity.dot(direction)
    comptime if DDF_shift:
        return weight*density*(3.*ei_dot_u + 4.5*ei_dot_u*ei_dot_u - 1.5*u_dot_u) +weight*(density - 1)
    else:
        return weight*density*(1 + 3.*ei_dot_u + 4.5*ei_dot_u*ei_dot_u - 1.5*u_dot_u)
