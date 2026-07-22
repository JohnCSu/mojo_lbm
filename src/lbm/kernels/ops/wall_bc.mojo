from src.utils import Vector
from layout import TileTensor,coord
from src.lbm.kernels.utils.index import get_adjacent_idx
from src.lbm.kernels.utils.load_and_store import load_f
from src.lbm.kernels.utils.moment import get_density_and_velocity_for_eq_BC
from src.lbm.kernels.utils.equilibrium import get_f_eq_vec
from std.utils.numerics import nan,isnan
from std.math import sqrt

def wall_bc[
    float_dtype:DType,int_dtype:DType,f_dtype:DType,D:Int,Q:Int,//,
    include_bounceback:Bool,
    directions:InlineArray[Vector[int_dtype, D], Q],
    opposite_indices:InlineArray[Scalar[int_dtype], Q],
    weights:Vector[float_dtype,Q],
    use_float16c:Bool,
    *,
    start_idx:Int = 1,
    non_temporal:Bool = False
    ]
    (
    mut f_vec:Vector[float_dtype,Q],
    mut pull_flags:InlineArray[UInt8,Q],
    f:TileTensor[f_dtype,...,address_space = AddressSpace.GENERIC],
    flags:TileTensor[DType.uint8,...],
    bc:TileTensor[float_dtype,...],
    index:InlineArray[Int,3],
    grid_shape:InlineArray[Int,3],
    ): 

    comptime assert bc.rank == 4 and flags.rank == 3 and f.rank == 4
    comptime load_f_from_xyzq = load_f[float_dtype,use_float16c,non_temporal]

    var velocity = Vector[float_dtype,D](uninitialized = True)
    var rho:Scalar[float_dtype]

    comptime for q in range(start_idx,Q):
        comptime direction = directions[q]
        pull_index = get_adjacent_idx[shift = -1](index,grid_shape,direction) # Pulling Scheme
        pull_flags[q] = flags.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2])))[0]
        if pull_flags[q] == Flags.SOLID:
            comptime float_direction = directions[q].cast_to[float_dtype]()
            comptime weight = weights[q]
            comptime for ii in range(D):
                velocity[ii] = bc.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2],ii)))[0]
            rho = bc.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2],D)))[0]

            comptime if include_bounceback:
                 # Double Buffer we need to do bounceback 
                comptime opp_q = Int(opposite_indices[q])
                f_vec[q] = load_f_from_xyzq(f,index,opp_q) + 2.*3.*weights[q]*rho*(float_direction.dot(velocity))
            else:
                f_vec[q] += 2.*3.*weight*rho*(float_direction.dot(velocity))


@always_inline
def equilibrium_bc[
    float_dtype:DType,int_dtype:DType,D:Int,Q:Int,//,
    directions:InlineArray[Vector[int_dtype, D], Q],
    weights:Vector[float_dtype,Q],
    DDF_shift:Bool,
    ]
    (
    mut f_vec:Vector[float_dtype,Q],
    mut pull_flags:InlineArray[UInt8,Q],
    bc:TileTensor[float_dtype,...],
    index:InlineArray[Int,3],
    grid_shape:InlineArray[Int,3],
    ):
    current_flag = pull_flags[0] # comptime assert gurantees this is the flag for the current node
    if current_flag  == Flags.EQUILIBRIUM:
        var velocity = Vector[float_dtype,D](uninitialized = True)
        var rho:Scalar[float_dtype] = 0
        comptime for ii in range(D):
            velocity[ii] = bc.load(coord[DType.uint32]((index[0],index[1],index[2],ii)))[0]
        rho = bc.load(coord[DType.uint32]((index[0],index[1],index[2],D)))[0]
        
        rho_local,u_l = get_density_and_velocity_for_eq_BC[directions,DDF_shift](f_vec,weights,index,grid_shape)
        
        u_local = u_l if isnan(velocity[0]) else velocity # nan means the vel is free
        rho_local = rho_local if isnan(rho) else rho # Nan means density is free

        f_vec = get_f_eq_vec[directions,weights,DDF_shift](f_vec,rho_local,u_local)
