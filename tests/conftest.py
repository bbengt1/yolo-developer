"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest  # noqa: F401


def get_uv_command() -> list[str]:
    """Get the uv command, using full path if not in PATH.

    Returns:
        List containing the uv command path, suitable for subprocess calls.
    """
    uv_path = shutil.which("uv")
    if uv_path:
        return [uv_path]
    # Try common installation locations
    for path in [
        Path.home() / ".local" / "bin" / "uv",
        Path.home() / ".cargo" / "bin" / "uv",
        Path("/usr/local/bin/uv"),
    ]:
        if path.exists():
            return [str(path)]
    return ["uv"]  # Fall back to PATH lookup


# Shared constants for uv command usage in tests
UV_CMD = get_uv_command()
UV_AVAILABLE = shutil.which(UV_CMD[0]) is not None or Path(UV_CMD[0]).exists()
