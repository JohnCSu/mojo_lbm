from src.utils import Vector

struct LatticeModel[D:Int,Q:Int,float_dtype:DType,int_dtype:DType](ImplicitlyCopyable):
    comptime int_vector = Vector[Self.int_dtype,Self.D]
    comptime float_vector = Vector[Self.float_dtype,Self.D]
    comptime dimension = Self.Q
    comptime int_scalar = Scalar[Self.int_dtype]
    comptime float_scalar = Scalar[Self.float_dtype]

    var directions:InlineArray[Self.int_vector,Self.Q]
    var float_directions:InlineArray[Self.float_vector,Self.Q]

    var weights:Vector[Self.float_dtype,Self.Q]
    var opposite_indices:InlineArray[Self.int_scalar,Self.Q]

    def __init__(out self,directions:InlineArray[Self.int_vector,Self.Q],float_directions:InlineArray[Self.float_vector,Self.Q],weights:Vector[Self.float_dtype,Self.Q]):
        self.directions = directions
        self.weights = weights
        self.opposite_indices = InlineArray[self.int_scalar,Self.Q](fill = 0)
        self.float_directions = float_directions
        self._get_opposite_indices()
        
    def _get_opposite_indices(mut self):
        for i in range(Self.Q): # Cant be bothered making an effecient algorithim to search opposite
            opp_direction = self.directions[i].copy()
            for j in range(Self.D):
                opp_direction[j] = opp_direction[j]*(-1)
            for k in range(Self.Q):
                if opp_direction == self.directions[k]:
                    self.opposite_indices[i] = self.int_scalar(k)
                    break


def get_D3Q19[float_dtype:DType = DType.float32,int_dtype:DType = DType.int32]() -> LatticeModel[2,9,float_dtype,int_dtype]:  
    comptime D = 2
    comptime Q = 9
    comptime int_vector = Vector[int_dtype,D]
    comptime float_vector = Vector[float_dtype,D]
    
    directions_list:List[List[Scalar[int_dtype]]]  =  
                                      [
                                        # Center (1)
                                        [ 0,  0,  0],
                                        # Faces (6)
                                        [ 1,  0,  0], [-1,  0,  0], 
                                        [ 0,  1,  0], [ 0, -1,  0], 
                                        [ 0,  0,  1], [ 0,  0, -1],
                                        # Edges (12)
                                        [ 1,  1,  0], [-1, -1,  0], [ 1, -1,  0], [-1,  1,  0],
                                        [ 1,  0,  1], [-1,  0, -1], [ 1,  0, -1], [-1,  0,  1],
                                        [ 0,  1,  1], [ 0, -1, -1], [ 0,  1, -1], [ 0, -1,  1]
                                    ]
    float_directions = InlineArray[float_vector,Q](uninitialized = True)
    for i in range(Q):
        float_directions[i].fill_and_cast_from_list(directions_list[i])

    directions = InlineArray[int_vector,Q](uninitialized = True)
    for i in range(Q):
        directions[i].fill_and_cast_from_list(directions_list[i])

    weights =  Vector[float_dtype,Q](
                                    4./9.,                          # 0: Center
                                    1./9., 1./9., 1./9., 1./9.,           # 1-4: Axis
                                    1./36., 1/36., 1./36., 1./36.        # 5-8: Diagonal
                                    )

    return LatticeModel[D,Q,float_dtype,int_dtype](directions,float_directions,weights)
    


def get_D2Q9[float_dtype:DType = DType.float32,int_dtype:DType = DType.int32]() -> LatticeModel[2,9,float_dtype,int_dtype]:  
    comptime D = 2
    comptime Q = 9
    comptime int_vector = Vector[int_dtype,D]
    comptime float_vector = Vector[float_dtype,D]
    
    float_directions_list:List[List[Scalar[float_dtype]]]  =  
                                        [
                                        [ 0,  0], # 0: Center (rest)
                                        [ 1,  0], # 1: East
                                        [ 0,  1], # 2: North
                                        [-1,  0], # 3: West
                                        [ 0, -1], # 4: South
                                        [ 1,  1], # 5: North-East
                                        [-1,  1], # 6: North-West
                                        [-1, -1], # 7: South-West
                                        [ 1, -1]  # 8: South-East
                                        ]
    float_directions = InlineArray[float_vector,Q](uninitialized = True)
    for i in range(Q):
        float_directions[i].fill(float_directions_list[i])
    
    directions_list:List[List[Scalar[int_dtype]]] =
                                    [
                                        [ 0,  0], # 0: Center (rest)
                                        [ 1,  0], # 1: East
                                        [ 0,  1], # 2: North
                                        [-1,  0], # 3: West
                                        [ 0, -1], # 4: South
                                        [ 1,  1], # 5: North-East
                                        [-1,  1], # 6: North-West
                                        [-1, -1], # 7: South-West
                                        [ 1, -1]  # 8: South-East
                                    ]

    directions = InlineArray[int_vector,Q](uninitialized = True)
    for i in range(Q):
        directions[i].fill(directions_list[i])

    weights =  Vector[float_dtype,Q](
                                    4./9.,                          # 0: Center
                                    1./9., 1./9., 1./9., 1./9.,           # 1-4: Axis
                                    1./36., 1/36., 1./36., 1./36.        # 5-8: Diagonal
                                    )

    return LatticeModel[D,Q,float_dtype,int_dtype](directions,float_directions,weights)
    
