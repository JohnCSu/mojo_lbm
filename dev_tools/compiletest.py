import os
import sys
import subprocess
from pathlib import Path


def main():
    project_root = Path.cwd().resolve()
    compiletest_dir = project_root / "compiletest"
    src_dir = project_root / "src"
    output_package = compiletest_dir / "src.mojopkg"

    if not compiletest_dir.is_dir():
        print("❌ compiletest/ directory not found in project root", file=sys.stderr)
        sys.exit(1)

    mojo_files = sorted(compiletest_dir.glob("*.mojo"))
    if not mojo_files:
        print("❌ No .mojo files found in compiletest/", file=sys.stderr)
        sys.exit(1)

    # Step 1: Build the Mojo package once into compiletest/
    print(f"🔨 Building Mojo package into {compiletest_dir}...")
    build_cmd = ["mojo", "package", str(src_dir), "-o", str(output_package)]
    result = subprocess.run(build_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Mojo Build Failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)

    # Step 2: Switch working directory to compiletest/ so `mojo -I .` resolves
    print(f"📂 Switching working directory to: {compiletest_dir}")
    os.chdir(compiletest_dir)

    # Step 3: Run each .mojo script in turn
    passed: list[str] = []
    failed: list[tuple[str, int]] = []

    try:
        for mojo_file in mojo_files:
            name = mojo_file.name
            print(f"\n🚀 Running {name}...")
            run_cmd = ["mojo", "-I", ".", name]
            try:
                run_result = subprocess.run(run_cmd)
                if run_result.returncode == 0:
                    print(f"✅ {name} passed")
                    passed.append(name)
                else:
                    print(f"❌ {name} failed (exit code {run_result.returncode})",
                          file=sys.stderr)
                    failed.append((name, run_result.returncode))
            except Exception as e:
                print(f"❌ Failed to run {name}: {e}", file=sys.stderr)
                failed.append((name, -1))
    finally:
        # Step 4: Clean up the generated Mojo package after all runs
        if output_package.exists():
            try:
                output_package.unlink()
                print(f"\n🧹 Deleted {output_package}")
            except OSError as e:
                print(f"⚠️ Could not delete {output_package}: {e}",
                      file=sys.stderr)

    # Step 5: Summary
    print("\n" + "=" * 60)
    print(f"Total: {len(mojo_files)} | Passed: {len(passed)} | "
          f"Failed: {len(failed)}")
    if passed:
        print("\nPassed files:")
        for n in passed:
            print(f"  ✅ {n}")
    if failed:
        print("\nFailed files:")
        for n, code in failed:
            print(f"  ❌ {n} (exit {code})")
        sys.exit(1)


if __name__ == "__main__":
    main()
