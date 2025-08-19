#!/usr/bin/env python3
"""Standalone script to generate protobuf files for development."""

import re
import subprocess
import sys
from pathlib import Path


def fix_protobuf_imports(proto_dir: Path):
    """Fix import statements in generated protobuf files to use relative imports."""
    print("Fixing protobuf imports...")

    # Find all generated _pb2.py files
    pb2_files = list(proto_dir.glob("*_pb2.py"))

    for pb2_file in pb2_files:
        print(f"Processing {pb2_file.name}...")

        # Read the file content
        content = pb2_file.read_text()

        # Fix imports of other protobuf modules
        # Replace "import common_pb2 as" with "from . import common_pb2 as"
        content = re.sub(
            r"^import (\w+_pb2) as (\w+)$",
            r"from . import \1 as \2",
            content,
            flags=re.MULTILINE,
        )

        # Replace standalone "import module_pb2" with "from . import module_pb2"
        content = re.sub(
            r"^import (\w+_pb2)$", r"from . import \1", content, flags=re.MULTILINE
        )

        # Write the fixed content back
        pb2_file.write_text(content)

    print(f"Fixed imports in {len(pb2_files)} files")


def generate_proto_for_directory(proto_dir: Path):
    """Generate protobuf files for a specific directory."""
    if not proto_dir.exists():
        print(f"Proto directory not found: {proto_dir}")
        return False

    print(f"Generating protobuf files in: {proto_dir}")

    # Find all .proto files
    proto_files = list(proto_dir.glob("*.proto"))

    if not proto_files:
        print(f"No .proto files found in {proto_dir}")
        return False

    # Generate Python files from proto definitions
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={proto_dir}",
        f"--python_out={proto_dir}",
        f"--grpc_python_out={proto_dir}",
    ]
    cmd.extend(str(f) for f in proto_files)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Protobuf files generated successfully in {proto_dir}")

        # Fix import statements in generated files
        fix_protobuf_imports(proto_dir)

        print(f"Generated {len(proto_files)} proto files in: {proto_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating protobuf files in {proto_dir}: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def main():
    """Generate protobuf files."""
    root = Path(__file__).parent.parent.parent
    
    # List of proto directories to process
    proto_directories = [
        root / "genteval" / "proto",
        root / "genteval" / "compressors" / "simple_gent" / "proto",
    ]

    success_count = 0
    total_count = 0

    for proto_dir in proto_directories:
        total_count += 1
        if generate_proto_for_directory(proto_dir):
            success_count += 1

    print(f"\nCompleted: {success_count}/{total_count} directories processed successfully")
    
    if success_count == 0:
        print("No protobuf files were generated")
        sys.exit(1)


if __name__ == "__main__":
    main()
