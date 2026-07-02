import os
import sys
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("❌ Error: Please provide a script to run", file=sys.stderr)
        sys.exit(1)

    # 1. Save absolute paths before we change directories
    project_root = Path.cwd().resolve()
    script_path = Path(sys.argv[1]).resolve()
    target_dir = script_path.parent
    
    # Source directory is at the root of your project
    src_dir = project_root / "src"
    output_package = target_dir / "src.mojopkg"

    # 2. Step 1: Build the Mojo Package (using absolute paths to be safe)
    print(f"🔨 Building Mojo package into {target_dir}...")
    build_cmd = ["mojo", "package", str(src_dir), "-o", str(output_package)]
    
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
    run_cmd = ["mojo", "-I", ".", script_path.name]
    
    final_run = subprocess.run(run_cmd)
    sys.exit(final_run.returncode)

if __name__ == "__main__":
    main()