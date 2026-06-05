from std.gpu import block_dim,block_idx,thread_idx
from layout import TileTensor,LayoutTensor
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from .LBM import LatticeModel,LBM_Grid
from src.utils import Vector,ContextTileTensor


def calculate_rho_and_velocity[ float_dtype:DType,D:Int,Q:Int,
                                lattice_model:LatticeModel[D,Q,float_dtype,DType.int32],
                                nx:Int,ny:Int,nz:Int,
                                //,
                                grid: LBM_Grid[lattice_model,nx,ny,nz], 
                                Flayout:Layout[...] where Flayout.rank == 4,
                                RhoLayout:Layout[...] where RhoLayout.rank == 3,
                                VelocityLayout:Layout[...] where VelocityLayout.rank == 4,
                                ]
                                (
                                    f:TileTensor[float_dtype,type_of(Flayout),MutAnyOrigin],
                                    density:TileTensor[float_dtype,type_of(RhoLayout),MutAnyOrigin],
                                    velocity:TileTensor[float_dtype,type_of(VelocityLayout),MutAnyOrigin],
                                ):
    # Run on GPU
    '''
    Compute the Velocity and Density from f dist. Converts to layout tensor to allow layout independent assignment. This should be run on the gpu
    '''
    
    comptime grid_shape = Vector[DType.int32,3](Int32(nx),Int32(ny),Int32(nz))
    comptime f_as_lt = LayoutTensor[float_dtype,Flayout.to_layout(),MutAnyOrigin]
    comptime vel_as_lt = LayoutTensor[float_dtype,VelocityLayout.to_layout(),MutAnyOrigin]
    comptime rho_as_lt = LayoutTensor[float_dtype,RhoLayout.to_layout(),MutAnyOrigin]
    f_lt = f_as_lt(f.ptr)
    velocity_lt = vel_as_lt(velocity.ptr)
    density_lt = rho_as_lt(density.ptr)

    x = block_dim.x * block_idx.x + thread_idx.x
    y = block_dim.y * block_idx.y + thread_idx.y
    z = block_dim.z * block_idx.z + thread_idx.z
    index = Vector[DType.int32,3](Int32(x),Int32(y),Int32(z))

    if index[0] < grid_shape[0] and index[1] < grid_shape[1] and index[2] < grid_shape[2]: # Basic Guard
        var u = Vector[float_dtype,D](fill = 0.)
        var rho = Scalar[float_dtype](0)
        for q in range(Q):
            rho += f_lt[q,x,y,z][0]
            u += f_lt[q,x,y,z][0]*lattice_model.float_directions[q]
        u /= rho

        density_lt[x,y,z] = rho
        comptime for i in range(D):
            velocity_lt[i,x,y,z] = u[i]
