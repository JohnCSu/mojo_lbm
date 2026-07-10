from layout import TileTensor,LayoutTensor,coord
from layout.tile_tensor import stack_allocation
from layout.tile_layout import Layout,row_major,Coord,TensorLayout

from std.gpu import block_dim,block_idx,thread_idx,barrier
from std.gpu.memory import AddressSpace
from std.utils.numerics import nan,isnan
from std.math import sqrt

from src.lbm import LBM_Grid,LBM_Config,LatticeModel
from src.lbm.constants import SOLID_NODE,FLUID_NODE,Flags,cs_squared
from src.lbm.kernels.utils.index import get_adjacent_idx
from src.lbm.kernels.utils.load_and_store import load_f,store_f

from src.utils import Vector,ContextTileTensor
from .moment import get_density,get_velocity,get_strain_rate_tensor,get_second_velocity_moment,get_density_and_velocity_for_eq_BC
from .turbulence import get_Smagorinsky_LES_tau

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
                tau:Scalar[float_dtype],
                )
                where tile_size >= 1 and Flayout.rank == 4 and BClayout.rank == 4 and Flaglayout.rank == 3:
    '''
    Base LBM to also handle 3D and non_square Grids. Key assumption is that block dim == tile-size 
    i.e. grid can be non-square but block is squre (same block dim in each x,y,z).
    ''' 
    # Convience Variable Names and constants
    comptime assert f_out.flat_rank == 8
    comptime weights = lattice_model.weights
    comptime float_directions = lattice_model.float_directions
    comptime directions = lattice_model.directions
    comptime opposite_index = lattice_model.opposite_indices
    comptime grid_shape:InlineArray[Int,3] = [nx,ny,nz]
    comptime non_temporal = True
    comptime load_f_from_xyzq = load_f[float_dtype,config.use_float16c,non_temporal]
    comptime stress_indices = lattice_model.stress_indices
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
                rho_local,u_l = get_density_and_velocity_for_eq_BC[config.DDF_shift](f_new,float_directions,weights,index,pull_indices)
                u_local = u_l if isnan(velocity[0]) else velocity # nan means the vel is free
                rho_local = rho_local if isnan(rho) else rho # Nan means density is free
                u_dot_u = u_local.dot(u_local)
                comptime for q in range(Q):
                    f_new[q] = f_eq[config.DDF_shift](weights[q],rho_local,u_local,u_dot_u,float_directions[q])

        # Get Velocity and Density
        rho = get_density[config.DDF_shift](f_new)
        velocity = get_velocity[float_directions](f_new,rho)
        tau_local = tau # Create a local variable if we need to modify tau with LES,KBC EELBM etc

        # LES
        comptime if config.second_moment:
            second_moment = get_second_velocity_moment[stress_indices,float_directions,config.DDF_shift](f_new)
            strain_rate = get_strain_rate_tensor[stress_indices,config.DDF_shift](second_moment,velocity,rho,tau)
            comptime if config.LES:
                comptime Cs = 0.1
                tau_eddy = get_Smagorinsky_LES_tau[stress_indices](strain_rate,Cs)
                tau_local += tau_eddy
            
        # Collision Term
        u_dot_u = velocity.dot(velocity)
        inv_tau = 1./tau_local # This is faster by 0.4 ms on the 256^3 benchmark
        comptime for q in range(Q):
            f_eq_q = f_eq[config.DDF_shift](weights[q],rho,velocity,u_dot_u,float_directions[q])
            store_f[config.use_float16c,non_temporal](f_out,(f_new[q] -  inv_tau*(f_new[q]- f_eq_q) ),index,q)


@always_inline
def f_eq[dtype:DType,D:Int,//,DDF_shift:Bool = False](weight:Scalar[dtype],density:Scalar[dtype],velocity:Vector[dtype,D],u_dot_u:Scalar[dtype],direction:Vector[dtype,D]) -> Scalar[dtype]:
    comptime assert dtype.is_floating_point(), 'DType to BGK_collision term should be Float point like' # Weied using where statement cause compile error?
    ei_dot_u = velocity.dot(direction)
    comptime if DDF_shift:
        return weight*density*(3.*ei_dot_u + 4.5*ei_dot_u*ei_dot_u - 1.5*u_dot_u) +weight*(density - 1)
    else:
        return weight*density*(1 + 3.*ei_dot_u + 4.5*ei_dot_u*ei_dot_u - 1.5*u_dot_u)
