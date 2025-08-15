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


def main():
    """Generate protobuf files."""
    root = Path(__file__).parent.parent.parent
    proto_dir = root / "genteval" / "proto"

    if not proto_dir.exists():
        print(f"Proto directory not found: {proto_dir}")
        sys.exit(1)

    print("Generating protobuf files...")

    # Find all .proto files
    proto_files = list(proto_dir.glob("*.proto"))

    if not proto_files:
        print("No .proto files found")
        return

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
        print("Protobuf files generated successfully")

        # Fix import statements in generated files
        fix_protobuf_imports(proto_dir)

        print(f"Generated files in: {proto_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating protobuf files: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()
