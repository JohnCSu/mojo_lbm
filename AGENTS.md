# AGENTS.md

GPU lattice-Boltzmann fluid solver in Mojo (single SRT kernel, D2Q9/D3Q19/D3Q27), using only the Mojo stdlib + `max.gpu`. Pixi-managed, linux-64 only, needs an NVIDIA/AMD/Apple GPU.

## Commands

All commands run through pixi (do not invoke `mojo`/`python` directly):

```bash
pixi run lbm <path/to/script.mojo>   # precompile src/ into src.mojoc next to the script, run it, delete the artifact
pixi run compiletest                  # mojo package src/ -> tests/compiletest/src.mojopkg, run every *.mojo in tests/compiletest/, report pass/fail, clean up
pixi run precompile_test              # mojo precompile src -o src.mojoc --disable-warnings
pixi run capture_precompile_stdout    # collect all precompile warnings/errors into mojo_warnings.txt (gitignored) + /tmp/mojo_full_output.txt
```

- There are no unit tests yet (`tests/unit/` is empty). Verification = `pixi run compiletest` (each script is a full GPU simulation) and clean precompile via `capture_precompile_stdout`. Compiletests require a GPU.
- Unit tests (when added) live in `tests/unit/` and must mirror the `src/` tree: `src/lbm/kernels/utils/index.mojo` -> `tests/unit/lbm/kernels/utils/test_index.mojo` (test files prefixed `test_`). Since modules can move, re-check `src/` and restructure `tests/unit/` to match before adding tests; `tests/unit/__init__.mojo` files mirror `src/` so imports like `from src.lbm...` resolve identically.
- Scratch files are gitignored: `test*.mojo`, `debug*.mojo`, `*.mojoc`, `*.mojopkg`, `mojo_warnings.txt`. Put experiments in files with these names, not in `src/`.
- `dev_tools/replace.sh <search> <replace>` does a prompted whole-word rename across git-tracked files (`-y` skips the prompt).
- Mojo skills (`.agents/skills/`, tracked in `skills-lock.json`) require Node/npx, which only exists in the `dev` pixi environment. If skills are missing, restore them with `pixi run -e dev npx skills experimental_install` (add/update: `pixi run -e dev npx skills add modular/skills`), then re-run `pixi run -e dev npx skills list` to confirm.

## Imports / running scripts

- Package root is `src/`; imports are `from src.lbm import ...` and `from src.utils import ...` (the `src.` prefix is intentional).
- Scripts are run with `mojo -I .` from their own directory after precompiling `src/` beside them — that is what `pixi run lbm` automates. Don't try `mojo run` from the repo root.
- Examples: `examples/*.mojo` (entry-point sims). Benchmarks: `benchmarks/` (LDC, Cylinder2D). `src/lbm/archive/` is historical code — don't extend it.

## Compile errors / warnings workflow

Order of operations (see `dev_tools/docs/mojo_warnings_and_errors.md`):
1. Fix all `error:` first, then warnings; docs warnings last.
2. Warnings are tracked: run `pixi run capture_precompile_stdout` after changes and fix new entries in `mojo_warnings.txt`.
- Implicit-var warning pattern: declare the variable before an `if`/`for` scope, assign inside it. For tuple unpacking of movable types use `ref a, b, c = x`; implicitly-copyable tuples can use plain `var a, b, c = x`.
- `mojo format` is the canonical formatter (80-column limit).

## Style (repo-specific, from dev_tools/docs/style.md)

- Functions are either in-place (mutate args, no return) or functional (return a value) — never both. Exception: struct methods may mutate `self` and return.
- Mutable args first, immutable after; at most 3 mutated args (kernels/CPU entry points exempt).
- Naming: `get_` prefix for value-returning helpers, `set_` for mutating helpers; snake_case for functions, CamelCase for structs/traits.
- Docstrings follow `dev_tools/docs/docstring-style-guide.md`; when updating a docstring, append `last modified by: <model name> on <YYYY/MM/DD>` at the bottom of the docstring. Overide the last AI to modify it if present.

## Kernel layering (src/lbm/kernels/ReadMe.md)

Strict layering: `utils/ -> ops/ -> steps/` (steps are what the kernel calls).
- `utils/`: fully generic, functional (always returns), no dependence on `grid`/`config` — pass needed values (e.g. `Q: Int`, not `grid.Q`) as explicit parameters. `@always_inline` expected.
- `ops/`: LBM-method-specific compositions of utils; may mutate, prefer no return value; still no `grid`/`config` dependence.
- `steps/`: resolve parameters and call ops; take `grid`/`config`; stay LBM-method-independent.

## Conventions

- Everything simulation-side is comptime-parameterized: grid dims, dtypes, lattice (`comptime float_dtype = DType.float32` etc.). New features should be comptime generic, not runtime branches.
- All indexing is `(x, y, z, q)` regardless of underlying layout; kernels must stay layout-independent (row/col major, tiled).
- `ContextTileTensor` (src/utils) syncs host/device buffers lazily — `.cpu()`/`.gpu()` copies only when switching devices.
- Git commit messages are short lowercase one-liners (see `git log`).
