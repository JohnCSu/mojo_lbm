from std.gpu import block_dim,block_idx,thread_idx
from layout import TileTensor
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from LBM import LatticeModel,LBM_Grid
from vector import Vector
from flags import SOLID_NODE,FLUID_NODE

def LBM_kernel[ float_dtype:DType,D:Int,Q:Int,
                lattice_model:LatticeModel[D,Q,float_dtype,DType.int32],
                nx:Int,ny:Int,nz:Int,
                FlayoutType:TensorLayout,BClayoutType:TensorLayout,FlaglayoutType:TensorLayout,
                //,
                grid: LBM_Grid[lattice_model,nx,ny,nz], 
                Flayout:FlayoutType,
                BClayout:BClayoutType,
                Flaglayout:FlaglayoutType,
                ]
                ( 
                f_out:TileTensor[float_dtype,FlayoutType,MutAnyOrigin],
                f_in:TileTensor[float_dtype,FlayoutType,MutAnyOrigin],
                bc:TileTensor[float_dtype,BClayoutType,MutAnyOrigin],
                flags:TileTensor[DType.uint8,FlaglayoutType,MutAnyOrigin],
                inv_tau:Scalar[float_dtype]
                ):
    
    comptime assert f_in.flat_rank == 4 and f_in.flat_rank == f_out.flat_rank and f_in.static_shape[0] == Q 
    comptime assert bc.flat_rank == 4
    comptime assert flags.flat_rank == 3
    comptime grid_shape = Vector[DType.int32,3](Int32(nx),Int32(ny),Int32(nz))

    x = block_dim.x * block_idx.x + thread_idx.x
    y = block_dim.y * block_idx.y + thread_idx.y
    z = block_dim.z * block_idx.z + thread_idx.z
    index = Vector[DType.int32,3](Int32(x),Int32(y),Int32(z))
    
    if index[0] < grid_shape[0] and index[1] < grid_shape[1] and index[2] < grid_shape[2]: # Basic Guard
        # print(index[0],index[1],index[2])
        var f_new = Vector[float_dtype,D](fill = 0.)
        var velocity = Vector[float_dtype,D]()
        # print(flags[x,y,z])
        for q in range(Q):
            f_opp = f_in[lattice_model.opposite_indices[q],x,y,z]
            direction = lattice_model.directions[q]
            
            pull_index = get_adjacent_idx[D,-1](index,grid_shape,direction) # Pulling Scheme
            pulled_f = f_in[q,pull_index[0],pull_index[1],pull_index[2]]
            
            comptime for ii in range(D):
                velocity[ii] = bc[pull_index[0],pull_index[1],pull_index[2],ii]
            rho = bc[pull_index[0],pull_index[1],pull_index[2],D]


            pulled_flag = flags[pull_index[0],pull_index[1],pull_index[2]]
            
            if pulled_flag == 0: # Stream
                f_new[q] = pulled_f
            elif pulled_flag == 1: # BounceBack with moving wall BC put together (2nd term is 0 if stationary wall)
                f_new[q] = f_opp + 2.*3.*lattice_model.weights[q]*rho*(lattice_model.float_directions[q].dot(velocity))
                # print(rho,lattice_model.weights[q],lattice_model.float_directions[q][0],lattice_model.float_directions[q][1])
                # print(velocity[0],velocity[1],2.*3.*lattice_model.weights[q]*rho*(lattice_model.float_directions[q].dot(velocity)))
        # Get Velocity and Density
        velocity.fill(0)
        rho = 0
        for q in range(Q):
            rho += f_new[q]
            velocity += f_new[q]*lattice_model.float_directions[q]
        velocity /= rho

        # Collision Term
        for q in range(Q):
            f_eq = BGK_Collision(lattice_model.weights[q],rho,velocity,lattice_model.float_directions[q])
            
            f_new[q] += inv_tau*(f_eq - f_new[q])

        # Write to data
        for q in range(Q):
            f_out[q,x,y,z] = f_new[q]


def get_adjacent_idx[D:Int,shift:Int32 = 1](index:Vector[DType.int32,3],grid_shape:Vector[DType.int32,3],direction:Vector[DType.int32,D],) -> Vector[DType.int32,3]:
    comptime assert D <= 3 
    adj_index = Vector[DType.int32,3]()
    comptime for d in range(D):
        adj_index[d] = (index[d] + shift*direction[d]) % grid_shape[d]
    return adj_index


def BGK_Collision[dtype:DType,D:Int,//](weight:Scalar[dtype],density:Scalar[dtype],velocity:Vector[dtype,D],direction:Vector[dtype,D]) -> Scalar[dtype]:
    comptime assert dtype.is_floating_point(), 'DType to BGK_collision term should be Float point like' # Weied using where statement cause compile error?
    ei_dot_u = velocity.dot(direction)
    return weight*density*(1 + 3.*ei_dot_u + 4.5*ei_dot_u*ei_dot_u +- 1.5*velocity.dot(velocity))


def calculate_rho_and_velocity[  float_dtype:DType,D:Int,Q:Int,
                                lattice_model:LatticeModel[D,Q,float_dtype,DType.int32],
                                nx:Int,ny:Int,nz:Int,
                                FlayoutType:TensorLayout,RhoLayoutType:TensorLayout,VelocityLayoutType:TensorLayout,
                                //,
                                grid: LBM_Grid[lattice_model,nx,ny,nz], 
                                Flayout:FlayoutType,
                                Rholayout:RhoLayoutType,
                                Velocitylayout:VelocityLayoutType,
                                ]
                                (
                                    f:TileTensor[float_dtype,FlayoutType,MutAnyOrigin],
                                    density:TileTensor[float_dtype,RhoLayoutType,MutAnyOrigin],
                                    velocity:TileTensor[float_dtype,VelocityLayoutType,MutAnyOrigin],
                                ):
    
    comptime assert f.flat_rank == 4
    comptime assert density.flat_rank == 3
    comptime assert velocity.flat_rank == 4
    comptime grid_shape = Vector[DType.int32,3](Int32(nx),Int32(ny),Int32(nz))

    x = block_dim.x * block_idx.x + thread_idx.x
    y = block_dim.y * block_idx.y + thread_idx.y
    z = block_dim.z * block_idx.z + thread_idx.z
    index = Vector[DType.int32,3](Int32(x),Int32(y),Int32(z))

    if index[0] < grid_shape[0] and index[1] < grid_shape[1] and index[2] < grid_shape[2]: # Basic Guard
        var u = Vector[float_dtype,D](fill = 0.)
        var rho = Scalar[float_dtype](0)
        for q in range(Q):
            rho += f[q,x,y,z]
            u += f[q,x,y,z]*lattice_model.float_directions[q]
        u /= rho

        density[x,y,z] = rho
        comptime for i in range(D):
            velocity[x,y,z,i] = u[i]
