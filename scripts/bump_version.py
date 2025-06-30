#!/usr/bin/env python3
"""
Version bumping script for MOSAIC project.
Bumps version in both pyproject.toml and src/mosaic/__init__.py
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Tuple


def parse_version(version_str: str) -> Tuple[int, int, int]:
    """Parse a semantic version string into major, minor, patch components."""
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)$', version_str.strip())
    if not match:
        raise ValueError(f"Invalid version format: {version_str}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def format_version(major: int, minor: int, patch: int) -> str:
    """Format version components into a semantic version string."""
    return f"{major}.{minor}.{patch}"


def bump_version(version_str: str, bump_type: str) -> str:
    """Bump version according to semantic versioning rules."""
    major, minor, patch = parse_version(version_str)
    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "patch":
        patch += 1
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")
    return format_version(major, minor, patch)


def update_pyproject_toml(file_path: Path, new_version: str) -> None:
    """Update version in pyproject.toml file."""
    content = file_path.read_text()
    # Find and replace version line
    pattern = r'^version = "[^"]*"'
    replacement = f'version = "{new_version}"'
    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    if new_content == content:
        raise ValueError("Version line not found in pyproject.toml")
    file_path.write_text(new_content)


def update_init_py(file_path: Path, new_version: str) -> None:
    """Update version in __init__.py file."""
    content = file_path.read_text()
    # Find and replace __version__ line
    pattern = r'^__version__ = "[^"]*"'
    replacement = f'__version__ = "{new_version}"'
    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    if new_content == content:
        raise ValueError("__version__ line not found in __init__.py")
    file_path.write_text(new_content)


def get_current_version(pyproject_path: Path) -> str:
    """Get current version from pyproject.toml."""
    content = pyproject_path.read_text()
    match = re.search(r'^version = "([^"]*)"', content, re.MULTILINE)
    if not match:
        raise ValueError("Version not found in pyproject.toml")
    return match.group(1)


def main():
    parser = argparse.ArgumentParser(description="Bump version in MOSAIC project")
    parser.add_argument(
        "bump_type",
        choices=["major", "minor", "patch"],
        help="Type of version bump to perform"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()
    # Define file paths
    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"
    init_py_path = project_root / "src" / "mosaic" / "__init__.py"
    # Verify files exist
    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found", file=sys.stderr)
        sys.exit(1)
    if not init_py_path.exists():
        print(f"Error: {init_py_path} not found", file=sys.stderr)
        sys.exit(1)
    try:
        # Get current version
        current_version = get_current_version(pyproject_path)
        print(f"Current version: {current_version}")
        # Calculate new version
        new_version = bump_version(current_version, args.bump_type)
        print(f"New version: {new_version}")
        if args.dry_run:
            print("Dry run - no changes made")
            return
        # Update files
        update_pyproject_toml(pyproject_path, new_version)
        update_init_py(init_py_path, new_version)
        print(f"Successfully bumped version from {current_version} to {new_version}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
