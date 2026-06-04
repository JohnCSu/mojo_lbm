from std.gpu import block_dim,block_idx,thread_idx
from layout import TileTensor
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from .LBM import LatticeModel,LBM_Grid
from src.utils import Vector,ContextTileTensor

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
    comptime assert velocity.flat_rank == 4 and Velocitylayout.static_shape[0] == D
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
            velocity[i,x,y,z] = u[i]
