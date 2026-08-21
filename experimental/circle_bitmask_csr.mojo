"""Example: marking a circle in a 16x16 grid and packing its fluid-boundary
links into a `BitmaskCSR` over only the boundary nodes (~40 rows).

Builds a 16x16 D2Q9 flag grid, marks a centered circle of radius 4 lattice
units as `SOLID`, tags the surrounding fluid boundary nodes with the value
`2`, and collects the `(boundary_row, q)` link pairs into a
`BitmaskCSR` whose rows are the packed boundary indices `0..n_boundary-1`
(so the CSR has roughly 40 rows, not 256). A companion
`boundary_global_ids` array maps CSR row `k` to the grid's linear memory
index of the k-th boundary node.

The example prints the flag grid ASCII, the per-row bitmask (with set-bit q
indices), the same data viewed as a standard CSR (row_offsets / col_indices),
the per-link `q_dist` values (uploaded via `csr.values_tensor(Span(q_dists),
sort=True)` so they line up with the CSR sorted set-bit traversal), and the
per-row `boundary_global_ids` array uploaded via `csr.row_vector_tensor`.
"""
from max.gpu.host import DeviceContext
from layout import coord
from std.collections import InlineArray, Set
from src.lbm import LBM_Grid, get_D2Q9, Flags
from src.utils import ContextTileTensor
from src.lbm.geometry.rigidSphere import get_rigid_sphere


comptime D2Q9 = get_D2Q9()
comptime grid = LBM_Grid[D2Q9, 16, 16, 1, 1](
    Scalar[DType.float32](1.0), [0.0, 0.0, 0.0]
)
comptime nx = grid.nx
comptime ny = grid.ny
comptime Q = grid.Q


def main() raises:
    var ctx = DeviceContext()

    # Build the flag tensor and zero it to FLUID
    var flags_ctt = ContextTileTensor[DType.uint8](
        ctx, grid.layouts.flag_layout, fill=Flags.FLUID
    )
    var flags = flags_ctt.cpu()

    # Center circle at (8, 8), radius 4 lattice units (physical radius = 4.0)
    var center = List[Scalar[DType.float32]]()
    center.append(8.0); center.append(8.0); center.append(0.0)
    var radius = Scalar[DType.float32](3.9)

    # Per-link pre-sort raw data (insertion order) for upload
    var boundary_global_ids = List[Scalar[DType.int32]]()
    var row_indices = List[Scalar[DType.int32]]()
    var col_indices = List[Scalar[DType.int32]]()
    var q_dists = List[Scalar[DType.float32]]()

    var csr = get_rigid_sphere[grid](
        ctx, flags, center, radius,
        boundary_global_ids, row_indices, col_indices, q_dists,
    )

    print("=== BitmaskCSR over boundary nodes only ===")
    print("n_boundary_rows:", csr.n_rows(), "  (grid", grid.num_points, ")  ncols(Q)=", csr.n_cols(), "  nnz=", csr.nnz, sep="")
    print("argsort:", csr.argsort)

    print("=== flag grid (16x16) ===")
    print("  '.'=FLUID  '#'=SOLID  'B'=FLUID_BOUNDARY (value 2)")
    for y in range(ny - 1, -1, -1):
        var row = String("")
        for x in range(nx):
            var f = Int(flags.load(coord[DType.int32]((x, y, 0))))
            var ch = "0 " if f == 0 else ("# " if f == 1 else "B ")
            row += ch
        print(row)

    # Per-row bitmask table: CSR row k, global_id, (x,y), mask, set-bit q indices
    print("=== bitmask CSR rows ===")
    print("row |  global_id  (x,y)  | mask(dec)  binary[Q-1..0]  q-indices")
    for k in range(csr.n_rows()):
        var mask = csr.row_mask(k)
        var gid = Int(boundary_global_ids[k])
        var x = gid % 16
        var y = gid // 16
        var bin = String("")
        for q in range(Q - 1, -1, -1):
            bin += "1" if (mask & UInt32(UInt32(1) << UInt32(q))) != 0 else "0"
        var qs = List[Int]()
        for q in range(Q):
            if (mask & UInt32(UInt32(1) << UInt32(q))) != 0:
                qs.append(q)
        print("  ", k, "  ", gid, "  (", x, ",", y, ")  ", Int(mask), "  ", bin, "  ", qs)

    # Convert to a regular CSR and print row_offsets + col_indices
    print("=== BitmaskCSR's own row_offsets (standard CSR offsets) ===")
    print("row_offsets:", [csr.row_offset(i) for i in range(csr.n_rows() + 1)])
    # Verify popcount(row_masks[r]) == row_offsets[r+1] - row_offsets[r]
    var popcount_ok = True
    for r in range(csr.n_rows()):
        var mask = csr.row_mask(r)
        var pc = 0
        for q in range(Q):
            if (mask & UInt32(UInt32(1) << UInt32(q))) != 0:
                pc += 1
        if pc != csr.row_offset(r + 1) - csr.row_offset(r):
            popcount_ok = False
    print("popcount(row_masks[r]) == row_offsets[r+1]-row_offsets[r] :", popcount_ok)
    assert(popcount_ok)

    print("=== via BitmaskCSR.to_csr() ===")
    var reg = csr.to_csr()
    var ro = reg.row_offsets.cpu()
    comptime assert ro.flat_rank == 1
    var ci = reg.col_indices.cpu()
    comptime assert ci.flat_rank == 1
    print("reg.nnz:", reg.nnz)
    print("row_offsets:", [Int(ro[i]) for i in range(reg.n_rows() + 1)])
    print("col_indices:", [Int(ci[i]) for i in range(reg.nnz)])
    # The reconstructed CSR's row_offsets should match the BitmaskCSR's exactly
    for i in range(reg.n_rows() + 1):
        assert(Int(ro[i]) == csr.row_offset(i))
    print("row_offsets match between BitmaskCSR and CSR: OK")

    # q_dist tensor — permuted through csr.argsort via sort=True so the values
    # line up with the sorted set-bit iteration order of the CSR.
    print("=== q_dist tensor (csr.values_tensor(Span(q_dists), sort=True)) ===")
    var qt = csr.values_tensor(Span(q_dists), sort=True)
    var qv = qt.cpu()
    comptime assert qv.flat_rank == 1
    print("size:", qt.size())
    var any_nan = False
    for k in range(qt.size()):
        var q = Float64(qv[k])
        if q != q:  # NaN check
            any_nan = True
        print("  row's k-th set-bit link k=", k, "  q_dist=", q)
    if any_nan:
        print("ERROR: NaN detected in q_dist tensor")
    else:
        print("OK: no NaNs")

    # boundary_global_ids as a per-row owning tensor via row_vector_tensor
    print("=== boundary_global_ids tensor ===")
    var gt = csr.row_vector_tensor(Span(boundary_global_ids))
    var gv = gt.cpu()
    comptime assert gv.flat_rank == 1
    print("size:", gt.size(), "  (should equal n_boundary_rows:", csr.n_rows(), ")")
    print("values:", [Int(gv[k]) for k in range(csr.n_rows())])

    # Sanity: total set bits across rows == nnz
    var total_bits = 0
    for r in range(csr.n_rows()):
        var mask = csr.row_mask(r)
        for q in range(Q):
            if (mask & UInt32(UInt32(1) << UInt32(q))) != 0:
                total_bits += 1
    print("=== sanity ===")
    print("total set bits across rows =", total_bits, "  csr.nnz =", csr.nnz)
    assert(total_bits == csr.nnz)
    assert(not any_nan)
    assert(popcount_ok)
    print("ALL OK")