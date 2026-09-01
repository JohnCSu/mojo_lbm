#!/usr/bin/env python3
"""Captures all warnings/errors from `pixi run mojo precompile src -o src.mojoc` into a text file."""
import subprocess
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUT = pathlib.Path("/tmp/mojo_warnings.txt")
# Also keep a copy in repo for reference
REPO_OUTPUT = REPO_ROOT / "mojo_warnings.txt"
MOJOC_OUTPUT = REPO_ROOT / "src.mojoc"

cmd = ["pixi", "run", "mojo", "precompile", "src", "-o", str(MOJOC_OUTPUT)]

print(f"Running: {' '.join(cmd)} (cwd={REPO_ROOT})")
result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
combined = result.stdout + "\n" + result.stderr

# Extract warning and error lines
warning_lines = [l for l in combined.splitlines() if "warning:" in l or "error:" in l]
print(f"Total warnings/errors: {len(warning_lines)}")
print(f"Total output lines: {len(combined.splitlines())}")

OUTPUT.write_text("\n".join(warning_lines) + "\n" if warning_lines else "")
REPO_OUTPUT.write_text("\n".join(warning_lines) + "\n" if warning_lines else "")

# Also dump full output for debugging
pathlib.Path("/tmp/mojo_full_output.txt").write_text(combined)

print(f"Wrote {len(warning_lines)} warnings/errors to {OUTPUT} and {REPO_OUTPUT}")
print(f"Full output to /tmp/mojo_full_output.txt")
# Print first 20 warnings/errors for quick check
for w in warning_lines[:20]:
    print(w)

# Clean up precompiled output in root directory after capturing
if MOJOC_OUTPUT.exists():
    MOJOC_OUTPUT.unlink()
    print(f"Deleted {MOJOC_OUTPUT}")
