from std.gpu import block_dim,block_idx,thread_idx
from layout import TileTensor,LayoutTensor,coord
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from .LBM import LBM_Grid
from .lattice_models import LatticeModel
from src.utils import Vector,ContextTileTensor


def copy_4D_to_rowMajor_layout[  
                        float_dtype:DType,
                        src_Origin:Origin[mut=True],
                        dest_Origin:Origin[mut=True],
                        //,  
                        srclayoutType:TensorLayout,
                        destlayoutType:TensorLayout,
                        ]
                        (src_tensor:TileTensor[float_dtype,srclayoutType,src_Origin],dest_tensor:TileTensor[float_dtype,destlayoutType,dest_Origin]):
                        
                        comptime assert srclayoutType.rank == destlayoutType.rank
                        comptime assert destlayoutType.flat_rank == destlayoutType.rank # Row or Col Major
                        comptime assert destlayoutType.rank == 4
                        comptime nx,ny,nz,D = (dest_tensor.static_shape[0],dest_tensor.static_shape[1],dest_tensor.static_shape[2],dest_tensor.static_shape[3])

                        for x in range(nx):
                            for y in range(ny):
                                for z in range(nz):
                                    for d in range(D):
                                        idx = coord[DType.int32]((x,y,z,d))
                                        value = src_tensor.load(idx)[0]
                                        dest_tensor.store(idx,value)





def calculate_rho_and_velocity[ float_dtype:DType,D:Int,Q:Int,
                                lattice_model:LatticeModel[D,Q,float_dtype,DType.int32],
                                nx:Int,ny:Int,nz:Int,tile_size:Int,
                                //,
                                grid: LBM_Grid[lattice_model,nx,ny,nz,tile_size], 
                                Flayout:Layout[...] ,
                                RhoLayout:Layout[...],
                                VelocityLayout:Layout[...] ,
                                ]
                                (
                                    f:TileTensor[float_dtype,type_of(Flayout),MutAnyOrigin],
                                    density:TileTensor[float_dtype,type_of(RhoLayout),MutAnyOrigin],
                                    velocity:TileTensor[float_dtype,type_of(VelocityLayout),MutAnyOrigin],
                                
                                )
                                where VelocityLayout.rank == 4 and RhoLayout.rank == 3 and Flayout.rank == 4:
                                
    # Run on GPU
    '''
    Compute the Velocity and Density from f dist. Converts to layout tensor to allow layout independent assignment. This should be run on the gpu
    '''
    
    comptime grid_shape = Vector[DType.int32,3](Int32(nx),Int32(ny),Int32(nz))
    comptime f_as_lt = LayoutTensor[float_dtype,Flayout.to_layout(),MutAnyOrigin]
    comptime vel_as_lt = LayoutTensor[float_dtype,VelocityLayout.to_layout(),MutAnyOrigin]
    comptime rho_as_lt = LayoutTensor[float_dtype,RhoLayout.to_layout(),MutAnyOrigin]

    comptime f_is_first_index = (f_as_lt.shape[0]() == Q and f_as_lt.shape[1]() == nx and f_as_lt.shape[2]() == ny and f_as_lt.shape[3]() == nz)
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
            comptime if f_is_first_index:
                rho += f_lt[q,x,y,z][0]
                u += f_lt[q,x,y,z][0]*lattice_model.float_directions[q]
            else:
                rho += f_lt[x,y,z,q][0]
                u += f_lt[x,y,z,q][0]*lattice_model.float_directions[q]
        u /= rho

        density_lt[x,y,z] = rho
        comptime for i in range(D):

            velocity_lt[i,x,y,z] = u[i]


