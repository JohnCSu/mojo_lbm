# Mojo 1.0.0 Transition Progress

> Branch: `mojo-1.0-transition` (HEAD `3597f3a` + 5 user commits `129164d`..`88e5ab1` on top of `main:cac411f`)
> Tool: `pixi run precompile_test` → `mojo precompile src -o src.mojoc --disable-warnings` (warnings ignored per `transition.md:5`)
> Current: **56 total / 22 core (`!/archive/`) / 34 archive** – down from **331 total / ~74 core** at `ac1d64f` (phase 1-3) and **331 total** initial.

## Goals (transition.md)

1. Ignore warnings, fix errors only.
2. Reduce `precompile_test` errors bottom-up, slowly.
3. Map dependency tree, start with leaves (std/max/python only).
4. Create branch, incremental commits, PR to `main` when green.

## Dependency Tree (src, non-archive, bottom-up)

```
Tier 0 leaves (no src deps):
  src/utils/vector.mojo, custom_fp.mojo, runtimeLayouts.mojo, contextTileTensor.mojo, contextCSR.mojo
  src/lbm/constants.mojo, units.mojo, runtimeParams.mojo, layoutstruct.mojo
  src/lbm/kernels/utils/finite_difference.mojo, shared_tile.mojo

Tier 1 (Tier0 only):
  src/lbm/lattice.mojo → Vector
  src/lbm/config.mojo → custom_fp, constants
  src/lbm/kernels/utils/checks.mojo, equilibrium.mojo, index.mojo, moment.mojo → Vector, constants

Tier 2:
  src/lbm/grid.mojo → Vector, TiledLayouts
  src/lbm/geometry/primatives.mojo, rigidSphere.mojo, rigidstationary.mojo → LBM_Grid, index
  src/lbm/kernels/utils/load_and_store.mojo → LBM_Grid, Vector, Float16C
  src/lbm/kernels/ops/*, output/Q_criterion.mojo, velocity.mojo, visualization/_python_importer.mojo
  src/lbm/preprocess/boundary_condition.mojo, outputrequest.mojo, solver.mojo, assembly.mojo

Tier 3:
  src/lbm/geometry/interpolated_BB.mojo, nodewise_interpolated_BB.mojo → index, load_and_store, Grid, runtimeLayouts
  src/lbm/kernels/double_buffer.mojo, esoteric_pull.mojo → Grid, checks, load_and_store, steps
  src/lbm/kernels/ops/boundary_condition.mojo, load_and_store.mojo → index, load_and_store, equilibrium
  src/lbm/kernels/steps/collide.mojo → index, load_and_store, moment, equilibrium, turbulence, ops

Tier 4 roots:
  src/lbm/kernels/steps/load_f.mojo, store_f.mojo, stream.mojo → ops/load_and_store
  src/lbm/preprocess/initial_condition.mojo → GridLike, equilibrium, load_and_store
  src/lbm/kernels/benchmark/* → Grid, preprocess, kernels

Tier 5 archive (56 files, defer): src/lbm/archive/***
```

## Error Taxonomy (from precompile)

| # | Bucket | Count (initial 331) | Current core 26 | Fix |
|---|--------|---------------------|-----------------|-----|
| A | `coord` → `dyn_coord` (`parameter values: Int vs DType`) | 63 | 0 | `from std.utils.coord import dyn_coord`, `coord[` → `dyn_coord[` |
| D | `std.gpu` `HostBuffer`/`DeviceBuffer`/`barrier` moved | 27 | 0 | `max.gpu.host: DeviceContext, DeviceBuffer, HostBuffer`, `max.gpu.sync: barrier` |
| F | `Int`→`Int32` `get_adjacent_idx` | 12 | 0 | `Int(direction[d])` → `Int32` (archive) |
| G | `InlineArray` literal `[[0,0]]` | 5 | 0 | `uninitialized=True` + `stress_indices[0]=[0,0]` |
| B | `cannot materialize comptime Array` (`not ImplicitlyCopyable`) | ~76 | ~6 | `materialize[array]()` + move array params `[]`→`()` |
| C | `Array cannot be implicitly copied` (`return`/`assign`) | ~30 | ~6 | `return arr^` / `arr.copy()` per `transition.md:40-41` |
| E | `iter_custom` → `bencher_iter_custom` | 21 | 4 | `max.benchmark: bencher_iter_custom` + unified `{imm}` |
| H | unknown `LBM_Config`/`GridLike`/`SOLID_NODE` | 12 | 0 | `LBM_Config` default `lbm_method=DOUBLE_BUFFER`, add `Flags`/`SOLID_NODE` imports |
| L | TileTensor origin `DType` mismatch | 6 | 0 | (deferred) |

## What Has Been Done

### Phase 1 – Tier 0 leaves (commit `ac1d64f`)
- `vector.mojo:9,18,22,257` – keep `ImplicitlyCopyable` via explicit `__init__(out self, *, copy: Self): self.data = copy.data.copy()` / `deinit move`, `unsafe_ptr` `ElementType` → `Scalar[dtype]`, `coord` → `Coord` from `std.utils.coord`.
- `runtimeLayouts.mojo:9` – `coord` → `dyn_coord` (`row_major(dyn_coord[int_dtype]((1,)))` etc).
- `contextTileTensor.mojo:8`, `contextCSR.mojo:9,28` – `std.gpu: HostBuffer, DeviceBuffer` → `max.gpu.host`, `coord` → `dyn_coord` (`row_major(dyn_coord[DType.int32]((3,)))`), `Coord` import.
- `constants.mojo` – `Set` `std.` prefix verified.

### Phase 2 – Tier 1 core
- `lattice.mojo:11,44,62,77,353` – explicit copy/move for `Lattice` (`directions/stress_indices/weights/opposite_indices/float_directions`), `get_stress_indices` `[[0,0]]` → `uninitialized`+`stress_indices[0]=[0,0]` + `return ^`, `self.directions = directions.copy()`.
- `index.mojo:44,68,121` – `get_adjacent_idx` `Int`→`Int32`→`Int` revert, `return ^`, `dyn_coord` import, `get_rank4_coord` `index[3]`→`index[2]`, `Coord` import.
- `moment.mojo:183` – stray `[` → `def get_strain_rate_tensor_norm_squared[` same line.
- `config.mojo:50` – `LBM_Config[lbm_method=DOUBLE_BUFFER]` default to allow bare `LBM_Config`.
- `grid.mojo:35,121,160` – `LBM_Grid` copy/move (`origin.copy()`), `get_grid_coordinates` `return out^`, `self.origin = origin.copy()`, kept `GridLike` trait `comptime`.
- `checks.mojo:9`, `finite_difference.mojo:1`, `output/*.mojo` – missing `Flags`/`SOLID_NODE`/`LBM_method` imports.

### Phase 3 – Tier 2 mid (commit `ac1d64f` + user commits `129164d`..`88e5ab1`)
- `primatives`/`rigidSphere` `grid.origin[i]`/`grid.shape[i]` → `materialize[grid.origin]()[i]` (`src/lbm/geometry/primatives.mojo:66`), `latticeModel.directions[q][i]` → `var latticeModel = materialize[grid.lattice]()` + `latticeModel.directions[q][i]` (user `0f1cbf0` with `var directions = materialize[grid.lattice.directions]()`).
- `visualization/_python_importer.mojo:57` `grid.origin[0]` → `materialize[grid.origin]()[0]`.
- `double_buffer.mojo:70,75` `esoteric_pull.mojo:70` – `var directions = materialize[lattice.directions]()` + `comptime assert not lattice.directions[0].all_true()` (fix `directions[0].all_true()` on `var`).
- `finite_difference.mojo:16` `dx` `[]`→`()` (`def get_velocity_gradient[...,//](..., dx:Scalar[float_dtype]=1.)`) and calls `src/lbm/output/Q_criterion.mojo:69` `get_velocity_gradient[1](...)` → `get_velocity_gradient(..., dx)` per `129164d`.
- `outputrequest.mojo:243` `var grid_shape = materialize[Self.grid.shape]()` (`cfa787c`).

### Phase 4 – Tier 3-4 (commits `dd8e249` + `1cc2bcc` + `9f0fe66`)
- `moment.mojo:45,80,108,176` – `directions`/`stress_indices` `[]`→`()` (`def get_velocity[...](f_vec:..., directions:InlineArray)` etc) + `var` for `float_direction/Qiab/Q_neq/ss`, `return ^`, calls `src/lbm/kernels/steps/collide.mojo:51` `get_velocity[directions](f_vec,rho)` → `get_velocity(f_vec,rho, directions)` etc for `get_Qiab`, `get_non_eq_second_order_moment`, `get_strain_rate_tensor_norm_squared`.
- `equilibrium.mojo:10,41,80` – `f_eq` `DDF_shift` `[]`→`()` `def f_eq[dtype:DType,D:Int](..., DDF_shift:Bool)` + `if` not `comptime if`; `get_f_eq_vec`/`get_f_noneq_vec` `directions,weights,DDF_shift` `[]`→`()` + `var u_dot_u`, `return ^`; calls `src/lbm/kernels/ops/boundary_condition.mojo:137` etc updated.
- `load_and_store.mojo:90,164,225,270` – `esoteric_pull_*` `is_even_time_step, use_float16c, non_temporal` `[]`→`()` and `directions` `[]`→`()` for `esoteric_pull_load_single_f` etc, `comptime assert opposite_indices_are_adjacent(directions)` kept `lattice.directions` comptime, `index_to_load` `var ... = index.copy() if ... else pull_index.copy()` per `transition.md:40`.
- `boundary_condition`/`collide`/`KBC` `SRT`/`TRT`/`RLBM` `DDF_shift` `[]`→`()` (`src/lbm/kernels/ops/collisions/collision.mojo:13` `def SRT[..., DDF_shift:Bool]` → `def SRT[...](..., DDF_shift:Bool)`), `collide` `stress_indices` `comptime`→`var stress_indices = materialize[lattice.stress_indices]()` (`src/lbm/kernels/steps/collide.mojo:47`), calls updated to `SRT(f_vec,velocity,rho,tau, directions,weights,config.DDF_shift)` style.
- `finite_difference` `get_adj_finite_difference` `dx` `[]`→`()` (`src/lbm/kernels/utils/finite_difference.mojo:99` `def get_adj_finite_difference[float_dtype:DType, side:StaticString](..., dx:Scalar[float_dtype])`) + calls `src/lbm/kernels/utils/finite_difference.mojo:79` `get_adj_finite_difference[float_dtype,'left'](..., dx)`.
- `initial_condition.mojo:162,175,183` – `comptime float_direction` → `var float_direction`, `fi_neq[directions]` `[]`→`()` (`def fi_neq[...,](..., directions:InlineArray)`), call `fi_neq(..., directions)`, `esoteric_pull_store_f_vec` `directions` `[]`→`()`.

## What Needs To Be Done (22 core + 34 archive = 56 total)

**Core 22** (`pixi run mojo precompile src -o /tmp/src.mojoc --disable-warnings` `grep -v /archive/`):

- `initial_condition.mojo:162,175,223` – `fi_neq`/`esoteric_pull_store_f_vec` `directions` still `materialize` mismatch (`Array[Vector[int_dtype,D],Q]`), `esoteric_pull_store_f_vec` call `lattice.directions` comptime vs `var` runtime.
- `rigidstationary.mojo:169` – `Array[Scalar[...],D]` `return` without `^` (`src/lbm/geometry/rigidstationary.mojo:169`).
- `benchmark` 4 errs – `b.iter_custom[run_kernel](ctx)` (`src/lbm/kernels/benchmark/double_buffer.mojo:112`, `esoteric.mojo:113`) → `bencher_iter_custom(b, run_kernel, ctx)` + `def run_kernel(ctx: DeviceContext) raises {imm}:` + `from max.benchmark import bencher_iter_custom` (user did for `double_buffer` but `benchmark.mojo` still `LBM_method` unknown at `src/lbm/kernels/benchmark/benchmark.mojo:19`).
- `boundary_condition.mojo:137` – `get_f_eq_vec` `directions` `Array[Vector[int_dtype,D],Q]` `materialize` (needs `directions` as `()` runtime, call `get_f_eq_vec(..., directions,weights,DDF_shift)`).
- `load_and_store.mojo:256,263` (6 errs) – `Array[Int,3]` `index_to_load`/`pull_index`/`push_index` `if` ternary without `.copy()`/`^` (`src/lbm/kernels/ops/load_and_store.mojo:256` `var index_to_load = index.copy() if is_pos_q else pull_index.copy()` already, but still `pull_flags` etc).
- `finite_difference` already fixed, but `collision.mojo:41,74,187` – `SRT`/`TRT`/`RLBM` `comptime direction = directions[q].cast_to` where `directions` is now `var` runtime but `comptime direction =` tries `comptime` assign of runtime (`src/lbm/kernels/ops/collisions/collision.mojo:41`).
- `collide.mojo:68,76,78,83,87` – `stress_indices` `Array[Array[Scalar[int_dtype],2],n_stress]` `comptime` vs `var`, `SRT`/`TRT`/`KBC`/`RLBM` `directions,weights` still `[]` in some paths, `get_f_noneq_vec` inference `float_dtype` (`src/lbm/kernels/steps/collide.mojo:67`).
- `Q_criterion.mojo:72,209,211` – `get_velocity_gradient` `TileTensor[DType.int]` vs `TileTensor[float_dtype]` (`src/lbm/output/Q_criterion.mojo:72`), `get_f_noneq_vec` `float_dtype` inference.
- `outputrequest.mojo:246,275` – `OutputRequest` `Self.materialize` (already `var grid_shape = materialize[Self.grid.shape]()` per `cfa787c`, but still `Self.materialize` in some path).
- `KBC_.mojo:224` – `get_f_eq_vec` `directions` `Array[Vector[int_dtype,D],Q]`.

**Archive 34** (all `src/lbm/archive/**`): same buckets A/C/E/F – `Int→Int32` (`src/lbm/archive/*/LBM_gpu_kernel.mojo:100` `Int(opposite_index[q])`), `coord→dyn_coord`, `materialize` for `directions[q]` (`src/lbm/archive/*/LBM_gpu_kernel.mojo:61` `direction = directions[q]`), `barrier` (`src/lbm/archive/*/LBM_gpu_kernel.mojo:1` `from std.gpu import barrier`), `iter_custom` (`src/lbm/archive/*/benchmark.mojo:70` `b.iter_custom[run_kernel](ctx)`). Can be bulk `rg` or excluded from `precompile_test` (move `archive` out of `src` or add `comptime if False` gate per `transition_plan.md` Tier 5).

## Steps To Execute

1. **Fix remaining core array returns** per `transition.md:40-41`: add `^`/`copy()` to all `return InlineArray`/`Vector`/`Array[Scalar]` at `rigidstationary.mojo:169`, `load_and_store.mojo:256`, `outputrequest.mojo:246`, `moment.mojo:141` etc – minimise: `return arr^` not new `var`.
2. **Finish `[]`→`()` migration** for `SRT`/`TRT`/`KBC`/`RLBM`/`get_kbc_Qiab`/`moving_wall_bc`/`equilibrium_bc`/`set_adjacent_flags`/`double_buffer_pull_load_f_vec` etc: move `directions,weights,stress_indices,opposite_indices` from `[]` (with `//`) to `()` and update `collide.mojo:76,83,87` `SRT(f_vec,velocity,rho,tau, directions,weights,config.DDF_shift)` etc, `apply_boundary_condition.mojo:38` `moving_wall_bc[directions,...]` → `moving_wall_bc(..., directions,...)`, `finite_difference` already done.
3. **Fix `collision.mojo:41` `comptime direction = directions[q]`** where `directions` is now `var` → `var direction = directions[q].cast_to`.
4. **Fix `benchmark` `iter_custom`** in `src/lbm/kernels/benchmark/*.mojo` and `src/lbm/archive/*/benchmark.mojo` → `bencher_iter_custom` per `mojo-gpu-fundamentals:584` + `closure_migration` `{imm}` (user already did for `double_buffer.mojo:112` but `benchmark.mojo:19` still `LBM_method` unknown).
5. **Fix `initial_condition`/`Q_criterion` `get_f_noneq_vec` inference** – make `post_collision` keyword-only per `88f6e5d` (`def get_f_noneq_vec[..., post_collision:Bool,]` with `//`) and call `get_f_noneq_vec[post_collision=False](..., directions,weights,config.DDF_shift)` with explicit `float_dtype` if needed, or pass `directions` as `materialize[lattice.directions]()` for `fi_neq`.
6. **Verify**: `pixi run precompile_test 2>&1 | grep -c error:` →0, `pixi run precompile_test 2>&1 | grep "core" -v archive` →0. Then `git add -A && git commit -m "phase 5: fix remaining core"` and PR `mojo-1.0-transition` → `main` per `transition.md:5`.

*Last verified: `pixi run mojo precompile src -o /tmp/src.mojoc --disable-warnings` 56 total / 22 core on `3597f3a`.*
