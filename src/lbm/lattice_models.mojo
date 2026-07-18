"""Defines `LatticeModel` and constructors for the D2Q9, D3Q19, and D3Q27 lattices.

A `LatticeModel` packages the discrete velocities, weights, opposite-index
map, and stress-tensor index pairs for a given lattice. The constructor
functions build the standard lattice models with the requested float and
integer dtypes.
"""
from src.utils import Vector


struct LatticeModel[D: Int, Q: Int, float_dtype: DType, int_dtype: DType](
    ImplicitlyCopyable
):
    """Stores the discrete velocities and weights for an LBM lattice.

    Holds the integer and float-valued velocity directions, the quadrature
    weights, the opposite-direction index map, and the symmetric stress-index
    pairs for the configured dimension.

    Parameters:
        D: The spatial dimension of the lattice.
        Q: The number of discrete velocities per node.
        float_dtype: The `DType` used for float-valued directions and weights.
        int_dtype: The `DType` used for integer-valued directions and indices.
    """

    comptime int_vector = Vector[Self.int_dtype, Self.D]
    comptime float_vector = Vector[Self.float_dtype, Self.D]
    comptime dimension = Self.Q
    comptime int_scalar = Scalar[Self.int_dtype]
    comptime float_scalar = Scalar[Self.float_dtype]
    comptime n_stress_components = Self.D * (Self.D + 1) // 2
    var directions: InlineArray[Self.int_vector, Self.Q]
    """The integer-valued discrete velocity directions."""
    var float_directions: InlineArray[Self.float_vector, Self.Q]
    """The float-valued discrete velocity directions."""
    var stress_indices: InlineArray[
        InlineArray[Self.int_scalar, 2], Self.n_stress_components
    ]
    """The symmetric stress-tensor index pairs for the dimension."""
    var weights: Vector[Self.float_dtype, Self.Q]
    """The quadrature weights for each discrete velocity."""
    var opposite_indices: InlineArray[Self.int_scalar, Self.Q]
    """The index of the opposite direction for each discrete velocity."""

    def __init__(
        out self,
        directions: InlineArray[Self.int_vector, Self.Q],
        float_directions: InlineArray[Self.float_vector, Self.Q],
        weights: Vector[Self.float_dtype, Self.Q],
    ):
        """Constructs a `LatticeModel` from its directions and weights.

        Computes the opposite-direction index map and the symmetric
        stress-index pairs at construction time.

        Args:
            directions: The integer-valued discrete velocity directions.
            float_directions: The float-valued discrete velocity directions.
            weights: The quadrature weights for each direction.
        """
        self.directions = directions
        self.weights = weights
        self.opposite_indices = InlineArray[self.int_scalar, Self.Q](fill=0)
        self.float_directions = float_directions
        self.stress_indices = get_stress_indices[Self.D, self.int_dtype]()

        self._get_opposite_indices()

    def _get_opposite_indices(mut self):
        """Populates `opposite_indices` by searching for each direction's negation.

        For each direction `i`, finds the index `k` such that
        `directions[k] == -directions[i]`.
        """
        for i in range(
            Self.Q
        ):  # Cant be bothered making an effecient algorithim to search opposite
            opp_direction = self.directions[i].copy()
            for j in range(Self.D):
                opp_direction[j] = opp_direction[j] * (-1)
            for k in range(Self.Q):
                if (self.directions[k] == opp_direction).all_true():
                    self.opposite_indices[i] = self.int_scalar(k)
                    break


def get_D3Q27[
    float_dtype: DType = DType.float32, int_dtype: DType = DType.int32
]() -> LatticeModel[3, 27, float_dtype, int_dtype]:
    """Returns a D3Q27 `LatticeModel` with the requested dtypes.

    Parameters:
        float_dtype: The `DType` for float directions and weights (defaults
            to `DType.float32`).
        int_dtype: The `DType` for integer directions and indices (defaults
            to `DType.int32`).

    Returns:
        A populated `LatticeModel[3, 27, float_dtype, int_dtype]`.
    """
    comptime D = 3
    comptime Q = 27
    comptime int_vector = Vector[int_dtype, D]
    comptime float_vector = Vector[float_dtype, D]

    directions_list: List[List[Scalar[int_dtype]]] = [
        # Center (1)
        [0, 0, 0],
        # Faces (6)
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        # Edges (12)
        [1, 1, 0],
        [-1, -1, 0],
        [1, -1, 0],
        [-1, 1, 0],
        [1, 0, 1],
        [-1, 0, -1],
        [1, 0, -1],
        [-1, 0, 1],
        [0, 1, 1],
        [0, -1, -1],
        [0, 1, -1],
        [0, -1, 1],
        # Corners (8)
        [1, 1, 1],
        [-1, -1, -1],
        [1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
    ]
    float_directions = InlineArray[float_vector, Q](uninitialized=True)
    for i in range(Q):
        float_directions[i].fill_and_cast_from_list(directions_list[i])

    directions = InlineArray[int_vector, Q](uninitialized=True)
    for i in range(Q):
        directions[i].fill_and_cast_from_list(directions_list[i])

    weights = Vector[float_dtype, Q](
        # Center
        8 / 27.0,
        # Faces
        2 / 27.0,
        2 / 27.0,
        2 / 27.0,
        2 / 27.0,
        2 / 27.0,
        2 / 27.0,
        # Edges
        1 / 54.0,
        1 / 54.0,
        1 / 54.0,
        1 / 54.0,
        1 / 54.0,
        1 / 54.0,
        1 / 54.0,
        1 / 54.0,
        1 / 54.0,
        1 / 54.0,
        1 / 54.0,
        1 / 54.0,
        # Corners
        1 / 216.0,
        1 / 216.0,
        1 / 216.0,
        1 / 216.0,
        1 / 216.0,
        1 / 216.0,
        1 / 216.0,
        1 / 216.0,
    )

    return LatticeModel[D, Q, float_dtype, int_dtype](
        directions, float_directions, weights
    )


def get_D3Q19[
    float_dtype: DType = DType.float32, int_dtype: DType = DType.int32
]() -> LatticeModel[3, 19, float_dtype, int_dtype]:
    """Returns a D3Q19 `LatticeModel` with the requested dtypes.

    Parameters:
        float_dtype: The `DType` for float directions and weights (defaults
            to `DType.float32`).
        int_dtype: The `DType` for integer directions and indices (defaults
            to `DType.int32`).

    Returns:
        A populated `LatticeModel[3, 19, float_dtype, int_dtype]`.
    """
    comptime D = 3
    comptime Q = 19
    comptime int_vector = Vector[int_dtype, D]
    comptime float_vector = Vector[float_dtype, D]

    directions_list: List[List[Scalar[int_dtype]]] = [
        # Center (1)
        [0, 0, 0],
        # Faces (6)
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        # Edges (12)
        [1, 1, 0],
        [-1, -1, 0],
        [1, -1, 0],
        [-1, 1, 0],
        [1, 0, 1],
        [-1, 0, -1],
        [1, 0, -1],
        [-1, 0, 1],
        [0, 1, 1],
        [0, -1, -1],
        [0, 1, -1],
        [0, -1, 1],
    ]
    float_directions = InlineArray[float_vector, Q](uninitialized=True)
    for i in range(Q):
        float_directions[i].fill_and_cast_from_list(directions_list[i])

    directions = InlineArray[int_vector, Q](uninitialized=True)
    for i in range(Q):
        directions[i].fill_and_cast_from_list(directions_list[i])

    weights = Vector[float_dtype, Q](
        # Center
        1.0 / 3,
        # Faces
        1.0 / 18.0,
        1.0 / 18.0,
        1 / 18.0,
        1.0 / 18.0,
        1.0 / 18.0,
        1.0 / 18.0,
        # Edges
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
    )

    return LatticeModel[D, Q, float_dtype, int_dtype](
        directions, float_directions, weights
    )


def get_D2Q9[
    float_dtype: DType = DType.float32, int_dtype: DType = DType.int32
]() -> LatticeModel[2, 9, float_dtype, int_dtype]:
    """Returns a D2Q9 `LatticeModel` with the requested dtypes.

    Parameters:
        float_dtype: The `DType` for float directions and weights (defaults
            to `DType.float32`).
        int_dtype: The `DType` for integer directions and indices (defaults
            to `DType.int32`).

    Returns:
        A populated `LatticeModel[2, 9, float_dtype, int_dtype]`.
    """
    comptime D = 2
    comptime Q = 9
    comptime int_vector = Vector[int_dtype, D]
    comptime float_vector = Vector[float_dtype, D]

    float_directions_list: List[List[Scalar[float_dtype]]] = [
        [0, 0],  # 0: Center (rest)
        [1, 0],  # 1: East
        [0, 1],  # 2: North
        [-1, 0],  # 3: West
        [0, -1],  # 4: South
        [1, 1],  # 5: North-East
        [-1, 1],  # 6: North-West
        [-1, -1],  # 7: South-West
        [1, -1],  # 8: South-East
    ]
    float_directions = InlineArray[float_vector, Q](uninitialized=True)
    for i in range(Q):
        float_directions[i].fill(float_directions_list[i])

    directions_list: List[List[Scalar[int_dtype]]] = [
        [0, 0],  # 0: Center (rest)
        [1, 0],  # 1: East
        [0, 1],  # 2: North
        [-1, 0],  # 3: West
        [0, -1],  # 4: South
        [1, 1],  # 5: North-East
        [-1, 1],  # 6: North-West
        [-1, -1],  # 7: South-West
        [1, -1],  # 8: South-East
    ]

    directions = InlineArray[int_vector, Q](uninitialized=True)
    for i in range(Q):
        directions[i].fill(directions_list[i])

    weights = Vector[float_dtype, Q](
        4.0 / 9.0,  # 0: Center
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,  # 1-4: Axis
        1.0 / 36.0,
        1 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,  # 5-8: Diagonal
    )

    return LatticeModel[D, Q, float_dtype, int_dtype](
        directions, float_directions, weights
    )


def get_stress_indices[
    D: Int, dtype: DType
]() -> InlineArray[InlineArray[Scalar[dtype], 2], (D * (D + 1)) // 2]:
    """Returns the symmetric stress-tensor index pairs for a given dimension.

    For `D == 2` returns `[(0,0), (0,1), (1,1)]`, and for `D == 3` returns
    `[(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]`, exploiting the symmetry
    of the stress tensor.

    Parameters:
        D: The spatial dimension. Constrained to 1, 2, or 3.
        dtype: The `DType` of the returned index scalars.

    Returns:
        An `InlineArray` of `[(alpha, beta), ...]` index pairs.
    """
    comptime n = (D * (D + 1)) // 2
    comptime assert D == 1 or D == 2 or D == 3
    comptime int_scalar = Scalar[dtype]
    comptime if D == 1:
        stress_indices: InlineArray[InlineArray[int_scalar, 2], n] = [[0, 0]]
        return stress_indices
    elif D == 2:
        stress_indices: InlineArray[InlineArray[int_scalar, 2], n] = [
            [0, 0],
            [0, 1],
            [1, 1],
        ]
        return stress_indices
    elif D == 3:
        stress_indices: InlineArray[InlineArray[int_scalar, 2], n] = [
            [0, 0],  # xx
            [0, 1],  # xy
            [0, 2],  # xz
            [1, 1],  # yy
            [1, 2],  # yz
            [2, 2],  # zz
        ]
        return stress_indices

    else:  # This is needed to make mojo happy. Cant happen with comptime asserts but in case of fallback set everything to -1
        stress_indices = InlineArray[InlineArray[int_scalar, 2], n](
            fill=[-1, -1]
        )
        return stress_indices
