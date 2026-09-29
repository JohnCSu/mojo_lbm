import os
import sys
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("❌ Error: Please provide a script to run", file=sys.stderr)
        print("Usage: pixi run lbm <script.mojo> [-D NAME=VALUE ...] [--script args ...]",
              file=sys.stderr)
        print("  -D NAME=VALUE         compile-time defines (e.g. -D N=32 for the grid size)",
              file=sys.stderr)
        print("  --iters N | --iters=N max iterations (runtime, default set per script)",
              file=sys.stderr)
        print("  --plot / --no-plot    toggle plotting (plotting scripts only, default on)",
              file=sys.stderr)
        print("  anything else         passed to the script", file=sys.stderr)
        sys.exit(1)

    # 1. Save absolute paths before we change directories
    project_root = Path.cwd().resolve()
    script_path = Path(sys.argv[1]).resolve()
    target_dir = script_path.parent

    # Split extra CLI args: -D defines go to the mojo compiler, everything
    # else is forwarded to the script (readable via sys.argv in Mojo)
    defines: list[str] = []
    script_args: list[str] = []
    extra = sys.argv[2:]
    i = 0
    while i < len(extra):
        arg = extra[i]
        if arg == "-D":
            if i + 1 >= len(extra):
                print("❌ Error: -D requires a value (e.g. -D N=32)", file=sys.stderr)
                sys.exit(1)
            defines.extend(["-D", extra[i + 1]])
            i += 2
        elif arg.startswith("-D"):
            defines.append(arg)
            i += 1
        else:
            script_args.append(arg)
            i += 1

    # Source directory is at the root of your project
    src_dir = project_root / "src"
    # output_package = target_dir / "src.mojopkg"
    output_package = target_dir / "src.mojoc"

    # 2. Step 1: Build the Mojo Package (using absolute paths to be safe)
    print(f"🔨 Building Mojo package into {target_dir}...")
    # build_cmd = ["mojo", "package", str(src_dir), "-o", str(output_package)]
    build_cmd = ["mojo", "precompile", str(src_dir), "-o", str(output_package)]
    result = subprocess.run(build_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Mojo Build Failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)

    # 3. Change the working directory to the examples/ folder
    print(f"📂 Switching working directory to: {target_dir}")
    os.chdir(target_dir)

    # 4. Step 2: Run Mojo Script from within that folder
    # Since we are inside the target_dir, we run the filename directly
    print(f"🚀 Running {script_path.name} from inside {target_dir.name}/...")
    if defines:
        print(f"   compile-time defines: {' '.join(defines)}")
    if script_args:
        print(f"   script args: {' '.join(script_args)}")
    run_cmd = ["mojo", *defines, "-I", ".", script_path.name, *script_args]
    
    try:
        final_run = subprocess.run(run_cmd)
        exit_code = final_run.returncode
    except Exception as e:
        print(f"❌ Failed to run Mojo script: {e}", file=sys.stderr)
        exit_code = 1
    finally:
        # 5. Clean up the generated Mojo package after the run
        if output_package.exists():
            try:
                output_package.unlink()
                print(f"🧹 Deleted {output_package}")
            except OSError as e:
                print(f"⚠️ Could not delete {output_package}: {e}", file=sys.stderr)

    sys.exit(exit_code)

if __name__ == "__main__":
    main()