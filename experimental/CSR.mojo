"""Defines the `CSRLike` trait plus `CSR` and `BitmaskCSR` sparse-matrix structs.

`CSR` is the canonical compressed sparse row format with explicit
row-offsets and column-index buffers. `BitmaskCSR` is a packed variant for
matrices with at most 32 columns: a single `UInt32` per row stores the set of
non-zero columns as a bitset. Both own their buffers as 1D column-major
`ContextTileTensor`s tied to a `DeviceContext`, and both conform to the
`CSRLike` trait that declares the shared `nnz`, `values_tensor`, and
`row_vector_tensor` API.
"""
from src.utils import ContextTileTensor
from std.gpu.host import DeviceContext
from layout import col_major, coord, Coord
from layout.tile_layout import Layout, TensorLayout


def _is_integer_dtype(d: DType) -> Bool:
    """Returns `True` when `d` is an integer `DType` (signed or unsigned)."""
    return d.is_integral()


def _validate_and_sort_pairs[D: DType](
    row_indices: Span[Scalar[D], mut=..., origin=...],
    col_indices: Span[Scalar[D], mut=..., origin=...],
    shape: Tuple[Int, Int],
    mut argsort: List[Int],
) raises -> List[Tuple[Int, Int]]:
    """Validates a pair of index spans and returns them sorted by `(row, col)`.

    Asserts at compile time that the element dtype is an integer dtype, raises
    if the two spans differ in length, raises on out-of-bounds indices, and
    raises on duplicate `(row, col)` pairs.

    Args:
        row_indices: Span of row indices in original order.
        col_indices: Span of column indices in original order.
        shape: The `(n_rows, n_cols)` dense matrix shape for bounds checking.
        argsort: Pre-initialized empty list that is appended to in place so
            that `argsort[k]` ends up holding the original position of the
            pair now at sorted position `k`.

    Returns:
        The `(row, col)` pairs sorted lexicographically by `(row, col)`.
    """
    comptime assert _is_integer_dtype(D), "CSR index dtype must be an integer dtype"
    if len(row_indices) != len(col_indices):
        raise Error(
            "CSR: row and col index spans must be the same size (got "
            + String(len(row_indices)) + " and " + String(len(col_indices)) + ")"
        )
    n_rows, n_cols = shape
    triples = List[Tuple[Int, Int, Int]]()
    for k in range(len(row_indices)):
        var r = Int(row_indices[k])
        var c = Int(col_indices[k])
        if r < 0 or r >= n_rows:
            raise Error("CSR: row index " + String(r) + " out of bounds for " + String(n_rows) + " rows")
        if c < 0 or c >= n_cols:
            raise Error("CSR: col index " + String(c) + " out of bounds for " + String(n_cols) + " cols")
        triples.append((r, c, k))

    def cmp(a: Tuple[Int, Int, Int], b: Tuple[Int, Int, Int]) capturing -> Bool:
        if a[0] != b[0]:
            return a[0] < b[0]
        return a[1] < b[1]

    sort[cmp_fn=cmp, stable=True](Span(triples))

    for k in range(1, len(triples)):
        if triples[k][0] == triples[k - 1][0] and triples[k][1] == triples[k - 1][1]:
            raise Error(
                "CSR: duplicate (" + String(triples[k][0]) + ", " + String(triples[k][1]) + ") index pairs are not allowed"
            )

    var pairs = List[Tuple[Int, Int]]()
    for k in range(len(triples)):
        var r, c, orig = triples[k]
        pairs.append((r, c))
        argsort.append(orig)
    return pairs^


def _fill_csr_buffers[D: DType, L: TensorLayout](
    mut row_offsets: ContextTileTensor[D, L],
    mut col_indices: ContextTileTensor[D, L],
    n_rows: Int,
    pairs: List[Tuple[Int, Int]],
) raises:
    """Fills the row-offset and column-index buffers from sorted index pairs.

    Walks the pairs once (relying on the `(row, col)` sort order) so each row
    boundary is `k` — the count of pairs seen so far — and fills any gap or
    trailing rows whose non-zero count is zero with the same boundary value,
    eliminating the per-row count buffer and the prefix-sum pass.
    """
    var offset_view = row_offsets.cpu()
    var col_view = col_indices.cpu()
    comptime assert offset_view.flat_rank == 1, "expected 1D row-offset tensor"
    comptime assert col_view.flat_rank == 1, "expected 1D column-index tensor"

    offset_view[0] = Scalar[D](0)
    var current_row = 0
    for k in range(len(pairs)):
        var row, _ = pairs[k]
        col_view[k] = Scalar[D](pairs[k][1])
        while current_row < row:
            current_row += 1
            offset_view[current_row] = Scalar[D](k)

    var nnz = len(pairs)
    for i in range(current_row + 1, n_rows + 1):
        offset_view[i] = Scalar[D](nnz)


def _fill_bitmask_csr_buffers[D: DType, L: TensorLayout, ML: TensorLayout](
    mut row_offsets: ContextTileTensor[D, L],
    mut row_masks: ContextTileTensor[DType.uint32, ML],
    n_rows: Int,
    pairs: List[Tuple[Int, Int]],
) raises:
    """Fills the row-offset and row-mask buffers of a `BitmaskCSR` from sorted pairs.

    Same single-pass scheme as `_fill_csr_buffers` (sorted by `(row, col)`):
    `row_offsets[r]` records the position of the first pair whose row is `r`,
    so empty rows share the next row's start offset; and `row_masks[r]`
    accumulates `1 << col` for every pair on row `r`. The popcount of
    `row_masks[r]` therefore equals `row_offsets[r+1] - row_offsets[r]`.
    """
    var offset_view = row_offsets.cpu()
    var mask_view = row_masks.cpu()
    comptime assert offset_view.flat_rank == 1, "expected 1D row-offset tensor"
    comptime assert mask_view.flat_rank == 1, "expected 1D row-mask tensor"

    for i in range(n_rows):
        mask_view[i] = UInt32(0)

    offset_view[0] = Scalar[D](0)
    var current_row = 0
    for k in range(len(pairs)):
        var row, col = pairs[k]
        mask_view[row] = mask_view[row] | UInt32(UInt32(1) << UInt32(col))
        while current_row < row:
            current_row += 1
            offset_view[current_row] = Scalar[D](k)

    var nnz = len(pairs)
    for i in range(current_row + 1, n_rows + 1):
        offset_view[i] = Scalar[D](nnz)


def _make_values_tensor[V: DType, L: TensorLayout](
    ctx: DeviceContext,
    layout: L,
    nnz: Int,
    argsort: List[Int],
    values: Span[Scalar[V], mut=..., origin=...],
    sort: Bool,
) raises -> ContextTileTensor[V, L]:
    """Builds an owning non-zero-value tensor of the given `layout` type.

    Shared by `CSR.values_tensor` and `BitmaskCSR.values_tensor`. Checks the
    span length against `nnz`, then either permutes via `argsort` (`sort=True`)
    or copies in place (`sort=False`).
    """
    if len(values) != nnz:
        raise Error(
            "CSR: values span length (" + String(len(values))
            + ") must match nnz (" + String(nnz) + ")"
        )
    var tensor = ContextTileTensor[V, L](ctx, layout=layout)
    var view = tensor.cpu()
    comptime assert view.flat_rank == 1, "expected 1D values tensor"
    if sort:
        for i in range(nnz):
            view[i] = values[argsort[i]]
    else:
        for i in range(nnz):
            view[i] = values[i]
    return tensor^


def _make_row_vector_tensor[V: DType, L: TensorLayout](
    ctx: DeviceContext,
    layout: L,
    n_rows: Int,
    values: Span[Scalar[V], mut=..., origin=...],
) raises -> ContextTileTensor[V, L]:
    """Builds an owning per-row tensor of length `n_rows` and the given layout.

    Shared by `CSR.row_vector_tensor` and `BitmaskCSR.row_vector_tensor`.
    """
    if len(values) != n_rows:
        raise Error(
            "CSR: row-vector span length (" + String(len(values))
            + ") must match n_rows (" + String(n_rows) + ")"
        )
    var tensor = ContextTileTensor[V, L](ctx, layout=layout)
    var view = tensor.cpu()
    comptime assert view.flat_rank == 1, "expected 1D row-vector tensor"
    for i in range(n_rows):
        view[i] = values[i]
    return tensor^


trait CSRLike:
    """Declares the shared interface for CSR-format sparse matrices.

    Conforming types own a `DeviceContext`, store their buffers as 1D
    column-major `ContextTileTensor`s over their `int_dtype`, expose the
    `(n_rows, n_cols)` dense shape and non-zero count, and produce owning
    tensors of non-zero values (`values_tensor`, with optional argsort-based
    permutation) and per-row data (`row_vector_tensor`).
    """
    comptime int_dtype: DType
    """The integer `DType` of the row/column index and offset storage."""
    comptime ColMajor1D_Type: TensorLayout
    """The compile-time type of the 1D column-major runtime `Int32` layout."""

    def __len__(self) -> Int:
        ...

    def n_rows(self) -> Int:
        ...

    def n_cols(self) -> Int:
        ...

    def values_tensor[V: DType](
        self,
        values: Span[Scalar[V], mut=..., origin=...],
        sort: Bool,
    ) raises -> ContextTileTensor[V, Self.ColMajor1D_Type]:
        ...

    def row_vector_tensor[V: DType](
        self,
        values: Span[Scalar[V], mut=..., origin=...],
    ) raises -> ContextTileTensor[V, Self.ColMajor1D_Type]:
        ...


struct CSR[int_dtype_: DType = DType.int32](CSRLike & Movable):
    """Stores a compressed sparse row (CSR) matrix in `ContextTileTensor` buffers.

    Both the row-offsets and column-indices arrays are held as 1D column-major
    `ContextTileTensor`s whose runtime extent is fixed at construction time, so
    the sparse data follows the same host/device copy semantics as the rest of
    the solver.

    Parameters:
        int_dtype_: The integer `DType` used for the row-offset and
            column-index buffers (defaults to `DType.int32`). Exposed to
            conforming code via the `Self.int_dtype` trait slot.
    """
    comptime int_dtype = Self.int_dtype_
    """Trait slot binding — see `CSRLike.int_dtype`."""
    comptime ColMajor1D_Type = type_of(col_major(coord[DType.int32]((1,))))
    """The compile-time type of a 1D column-major runtime `Int32` layout."""

    var deviceContext: DeviceContext
    var shape: Tuple[Int, Int]
    var row_offsets: ContextTileTensor[Self.int_dtype, Self.ColMajor1D_Type]
    """The CSR row-offsets buffer (length `n_rows + 1`)."""
    var col_indices: ContextTileTensor[Self.int_dtype, Self.ColMajor1D_Type]
    """The CSR column-indices buffer (length `nnz`)."""
    var nnz: Int
    var argsort: List[Int]

    def __init__(
        out self,
        ctx: DeviceContext,
        shape: Tuple[Int, Int],
        row_indices: Span[Scalar[Self.int_dtype], mut=..., origin=...],
        col_indices: Span[Scalar[Self.int_dtype], mut=..., origin=...],
    ) raises:
        """Builds a CSR matrix from two `Span`s of `(row, col)` index scalars.

        Both spans must contain scalars of an integer dtype, must be the same
        length, and must not contain duplicate `(row, col)` pairs. The
        argsort permutation mapping sorted positions back to original positions
        is stored on the struct so callers can re-permute value arrays supplied
        in the original order via `values_tensor(..., sort=True)`.
        """
        comptime assert _is_integer_dtype(Self.int_dtype), "CSR int_dtype must be an integer dtype"
        self.deviceContext = ctx
        self.shape = shape
        self.argsort = List[Int]()
        var pairs = _validate_and_sort_pairs[Self.int_dtype](
            row_indices, col_indices, shape, self.argsort
        )
        self.nnz = len(pairs)
        n_rows, _ = shape
        self.row_offsets = ContextTileTensor[Self.int_dtype, Self.ColMajor1D_Type](
            ctx, layout=col_major(coord[DType.int32]((n_rows + 1,))),
        )
        self.col_indices = ContextTileTensor[Self.int_dtype, Self.ColMajor1D_Type](
            ctx, layout=col_major(coord[DType.int32]((self.nnz,))),
        )
        _fill_csr_buffers[Self.int_dtype, Self.ColMajor1D_Type](
            self.row_offsets, self.col_indices, n_rows, pairs
        )

    def __len__(self) -> Int:
        """Returns the number of stored non-zero entries."""
        return self.nnz

    def n_rows(self) -> Int:
        """Returns the number of rows in the dense matrix."""
        return self.shape[0]

    def n_cols(self) -> Int:
        """Returns the number of columns in the dense matrix."""
        return self.shape[1]

    def values_tensor[V: DType](
        self,
        values: Span[Scalar[V], mut=..., origin=...],
        sort: Bool,
    ) raises -> ContextTileTensor[V, Self.ColMajor1D_Type]:
        """Returns an owning 1D column-major tensor of the non-zero values.

        Args:
            values: Span of per-non-zero values, in either sorted CSR order
                (`sort=False`) or in the original pre-sort order supplied to
                `__init__` (`sort=True`).
            sort: When `True`, permutes `values` with `self.argsort` so each
                sorted position `k` reads `values[argsort[k]]`. When `False`,
                copies `values` element-by-element in CSR order.
        """
        var layout = col_major(coord[DType.int32]((self.nnz,)))
        return _make_values_tensor[V, Self.ColMajor1D_Type](
            self.deviceContext, layout, self.nnz, self.argsort, values, sort
        )

    def row_vector_tensor[V: DType](
        self,
        values: Span[Scalar[V], mut=..., origin=...],
    ) raises -> ContextTileTensor[V, Self.ColMajor1D_Type]:
        """Returns an owning 1D column-major tensor of per-row data.

        Checks that the span length matches the number of rows.
        """
        var layout = col_major(coord[DType.int32]((self.n_rows(),)))
        return _make_row_vector_tensor[V, Self.ColMajor1D_Type](
            self.deviceContext, layout, self.n_rows(), values
        )

    def to_bitmask_csr(mut self) raises -> BitmaskCSR[Self.int_dtype]:
        """Converts to a `BitmaskCSR` with the same non-zero pattern.

        Raises:
            Error: If `self.n_cols() > 32`, since the bitmask format supports
                at most 32 columns.
        """
        n_rows, n_cols = self.shape
        if n_cols > 32:
            raise Error(
                "CSR: cannot convert to BitmaskCSR with n_cols=" + String(n_cols)
                + " (max 32)"
            )
        var rows = List[Scalar[Self.int_dtype]]()
        var cols = List[Scalar[Self.int_dtype]]()
        var ro = self.row_offsets.cpu()
        comptime assert ro.flat_rank == 1
        var ci = self.col_indices.cpu()
        comptime assert ci.flat_rank == 1
        for r in range(n_rows):
            for k in range(Int(ro[r]), Int(ro[r + 1])):
                rows.append(Scalar[Self.int_dtype](r))
                cols.append(ci[k])
        return BitmaskCSR[Self.int_dtype](
            self.deviceContext, self.shape, Span(rows), Span(cols)
        )


struct BitmaskCSR[int_dtype_: DType = DType.int32](CSRLike & Movable):
    """Stores a packed CSR matrix for sparse matrices with at most 32 columns.

    Keeps standard CSR `row_offsets` (length `n_rows + 1`, cumulative non-zero
    count) so rows are addressable in `O(1)` exactly like a regular CSR, but
    replaces the explicit `col_indices` buffer with one `UInt32` bitset per
    row (`row_masks`, length `n_rows`) encoding the set of non-zero columns.
    The `k`-th set bit of `row_masks[r]` (enumerated in ascending column
    order) corresponds to the `row_offsets[r] + k`-th flattened non-zero
    entry, so `popcount(row_masks[r]) == row_offsets[r+1] - row_offsets[r]`.

    Requires `n_cols <= 32` (enforced at construction).

    Parameters:
        int_dtype_: The integer `DType` of the row-offset storage and the
            row/column index values supplied to `__init__` (defaults to
            `DType.int32`). The mask buffer is always `DType.uint32`.
    """
    comptime int_dtype = Self.int_dtype_
    comptime ColMajor1D_Type = type_of(col_major(coord[DType.int32]((1,))))
    comptime MaskDType = DType.uint32
    """The `DType` of each row's bitset (always `DType.uint32`)."""
    comptime MaxCols = 32
    """The column bound for the bitmask format."""

    var deviceContext: DeviceContext
    var shape: Tuple[Int, Int]
    var row_offsets: ContextTileTensor[Self.int_dtype, Self.ColMajor1D_Type]
    """Standard CSR row offsets (length `n_rows + 1`). `row_offsets[r]` is
    the flattened position of the first non-zero in row `r`; the count of
    non-zeros in row `r` is `row_offsets[r+1] - row_offsets[r]`."""
    var row_masks: ContextTileTensor[Self.MaskDType, Self.ColMajor1D_Type]
    """The per-row non-zero-column bitsets (length `n_rows`). Bit `c` of
    `row_masks[r]` is set iff column `c` is non-zero in row `r`."""
    var nnz: Int
    var argsort: List[Int]

    def __init__(
        out self,
        ctx: DeviceContext,
        shape: Tuple[Int, Int],
        row_indices: Span[Scalar[Self.int_dtype], mut=..., origin=...],
        col_indices: Span[Scalar[Self.int_dtype], mut=..., origin=...],
    ) raises:
        """Builds a packed CSR matrix from two `Span`s of `(row, col)` index scalars.

        Raises:
            Error: If `shape[1] > 32`, if the two spans differ in length, on
                out-of-bounds indices, or on duplicate `(row, col)` pairs.
        """
        comptime assert _is_integer_dtype(Self.int_dtype), "BitmaskCSR int_dtype must be an integer dtype"
        _, n_cols = shape
        if n_cols > 32:
            raise Error(
                "BitmaskCSR: n_cols=" + String(n_cols) + " exceeds the max of 32"
            )
        self.deviceContext = ctx
        self.shape = shape
        self.argsort = List[Int]()
        var pairs = _validate_and_sort_pairs[Self.int_dtype](
            row_indices, col_indices, shape, self.argsort
        )
        self.nnz = len(pairs)
        n_rows, _ = shape
        self.row_offsets = ContextTileTensor[Self.int_dtype, Self.ColMajor1D_Type](
            ctx, layout=col_major(coord[DType.int32]((n_rows + 1,))),
        )
        self.row_masks = ContextTileTensor[Self.MaskDType, Self.ColMajor1D_Type](
            ctx, layout=col_major(coord[DType.int32]((n_rows,))),
        )
        _fill_bitmask_csr_buffers[Self.int_dtype, Self.ColMajor1D_Type, Self.ColMajor1D_Type](
            self.row_offsets, self.row_masks, n_rows, pairs
        )

    def __len__(self) -> Int:
        """Returns the number of stored non-zero entries."""
        return self.nnz

    def n_rows(self) -> Int:
        """Returns the number of rows in the dense matrix."""
        return self.shape[0]

    def n_cols(self) -> Int:
        """Returns the number of columns in the dense matrix."""
        return self.shape[1]

    def values_tensor[V: DType](
        self,
        values: Span[Scalar[V], mut=..., origin=...],
        sort: Bool,
    ) raises -> ContextTileTensor[V, Self.ColMajor1D_Type]:
        """Returns an owning 1D column-major tensor of the non-zero values.

        The flattened non-zero order matches the `(row, col)` sorted order
        constructed from the bitset iteration order — the same convention as
        `CSR.values_tensor`, so `sort=True` permutes with `self.argsort` and
        `sort=False` assumes the values already follow CSR order.
        """
        var layout = col_major(coord[DType.int32]((self.nnz,)))
        return _make_values_tensor[V, Self.ColMajor1D_Type](
            self.deviceContext, layout, self.nnz, self.argsort, values, sort
        )

    def row_vector_tensor[V: DType](
        self,
        values: Span[Scalar[V], mut=..., origin=...],
    ) raises -> ContextTileTensor[V, Self.ColMajor1D_Type]:
        """Returns an owning 1D column-major tensor of per-row data.

        Checks that the span length matches the number of rows.
        """
        var layout = col_major(coord[DType.int32]((self.n_rows(),)))
        return _make_row_vector_tensor[V, Self.ColMajor1D_Type](
            self.deviceContext, layout, self.n_rows(), values
        )

    def row_offset(mut self, r: Int) raises -> Int:
        """Returns `row_offsets[r]`, the flattened start index of row `r`."""
        var view = self.row_offsets.cpu()
        comptime assert view.flat_rank == 1, "expected 1D row-offset tensor"
        return Int(view[r])

    def row_mask(mut self, r: Int) raises -> UInt32:
        """Returns the `UInt32` bitset of non-zero columns on row `r`."""
        var view = self.row_masks.cpu()
        comptime assert view.flat_rank == 1, "expected 1D row-mask tensor"
        return UInt32(view[r])

    def is_set(mut self, r: Int, c: Int) raises -> Bool:
        """Returns `True` when `(r, c)` is a stored non-zero."""
        if r < 0 or r >= self.n_rows():
            raise Error("BitmaskCSR: row out of bounds")
        if c < 0 or c >= self.n_cols():
            raise Error("BitmaskCSR: col out of bounds")
        var mask = self.row_mask(r)
        return (mask & UInt32(UInt32(1) << UInt32(c))) != 0

    def to_csr(mut self) raises -> CSR[Self.int_dtype]:
        """Converts to a standard `CSR` with the same non-zero pattern.

        Walks `row_masks` row-by-row, enumerating set columns in ascending
        order so the reconstructed `(row, col)` pairs are already sorted and
        line up with `row_offsets` exactly.
        """
        n_rows, n_cols = self.shape
        var rows = List[Scalar[Self.int_dtype]]()
        var cols = List[Scalar[Self.int_dtype]]()
        var view = self.row_masks.cpu()
        comptime assert view.flat_rank == 1
        for r in range(n_rows):
            var mask = UInt32(view[r])
            for c in range(n_cols):
                if (mask & UInt32(UInt32(1) << UInt32(c))) != 0:
                    rows.append(Scalar[Self.int_dtype](r))
                    cols.append(Scalar[Self.int_dtype](c))
        return CSR[Self.int_dtype](
            self.deviceContext, self.shape, Span(rows), Span(cols)
        )