"""Defines `ContextCSR`, a host/device container for sparse CSR matrices.

The struct stores the row-offsets and column-indices of a compressed sparse
row matrix as `ContextTileTensor` buffers tied to a `DeviceContext`. The
implementation is a work in progress.
"""
from .contextTileTensor import ContextTileTensor
from std.gpu.host import DeviceContext
from std.gpu import HostBuffer, DeviceBuffer
from layout import TileTensor, row_major, col_major, coord, Coord
from layout.tile_layout import Layout
from std.builtin.variadics import TypeList


struct ContextCSR[int_dtype: DType = DType.int32]():
    """Stores a compressed sparse row (CSR) matrix across host and device buffers.

    Holds row-offsets and column-indices as paired `ContextTileTensor` views
    so the CSR data can be moved between CPU and GPU with the same sync
    semantics as the rest of the solver.

    Parameters:
        int_dtype: The integer `DType` used for row-offsets and
            column-indices (defaults to `DType.int32`).
    """

    # comptime dim = len(Self.ints)
    # var rows
    # var

    # comptime
    comptime dim = 3
    comptime RowMajor1D_Type = type_of(
        row_major(coord[DType.int32]((3,)))
    )  # the 3 is a dummy variable to capture the runtime type
    var deviceContext: DeviceContext
    var shape: Tuple[Int, Int]
    # var rows:Int
    # var cols:Int
    var row_offsets: ContextTileTensor[Self.int_dtype, Self.RowMajor1D_Type]
    """The CSR row-offsets buffer (length `n_rows + 1`)."""
    var col_indices: ContextTileTensor[Self.int_dtype, Self.RowMajor1D_Type]
    """The CSR column-indices buffer (length `nnz`)."""
    var nnz: Int
    """The number of stored non-zero entries."""

    def __init__(
        out self,
        ctx: DeviceContext,
        shape: Tuple[Int, Int],
        mut indices: List[Tuple[Int, Int]],
    ) raises:
        """Builds a CSR matrix from a list of `(row, col)` index pairs.

        Sorts the input indices by row then column, builds the row-offset
        buffer, and copies the column indices into the column-indices buffer.

        Args:
            ctx: The device context that owns the buffers.
            shape: The `(n_rows, n_cols)` shape of the dense matrix.
            indices: The list of `(row, col)` coordinate pairs to store.
        """
        self.deviceContext = ctx

        self.shape = shape
        n_rows, n_cols = shape
        self.nnz = len(indices)

        self.row_offsets = ContextTileTensor[Self.int_dtype](
            ctx, layout=row_major(coord[DType.int32]((n_rows + 1,)))
        )
        self.col_indices = ContextTileTensor[Self.int_dtype](
            ctx, layout=row_major(coord[DType.int32]((self.nnz,)))
        )

        # Sort the rows
        def cmp(src: Tuple[Int, Int], other: Tuple[Int, Int]) capturing -> Bool:
            return src[0] < src[0] if src[0] != other[0] else src[1] < other[1]

        sort[cmp_fn=cmp, stable=True](Span(indices))

        self.row_offsets.cpu()[0] = 0

        current_row = 0
        offset_count = 0
        row_inc = 1
        for i, (row, col) in enumerate(indices):
            self.col_indices.cpu()[i] = Scalar[Self.int_dtype](col)
            offset_count += 1
            if row != current_row:
                self.row_offsets.cpu()[row_inc] = Scalar[Self.int_dtype](
                    offset_count
                )
                row_inc += 1
                offset_count = 0
