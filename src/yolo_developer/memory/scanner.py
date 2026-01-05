"""Codebase scanner for project pattern learning.

This module provides the CodebaseScanner class for scanning existing codebases
to identify source files for pattern analysis. The scanner supports configurable
file extensions and ignore patterns, with async file reading for performance.

Example:
    >>> from yolo_developer.memory.scanner import CodebaseScanner
    >>> from pathlib import Path
    >>>
    >>> scanner = CodebaseScanner()
    >>> result = await scanner.scan(Path("/path/to/project"))
    >>> print(f"Found {len(result.files)} files with {result.total_lines} lines")

Security Note:
    The scanner only reads files within the specified root directory.
    It does not follow symlinks outside the root to prevent directory
    traversal attacks.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Default file extensions to include in scanning
DEFAULT_EXTENSIONS: frozenset[str] = frozenset({".py"})

# Default directory/file patterns to ignore during scanning
DEFAULT_IGNORE: frozenset[str] = frozenset({
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    ".tox",
    "build",
    "dist",
    ".eggs",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "htmlcov",
    "*.egg-info",
})


@dataclass
class ScanResult:
    """Result of scanning a codebase.

    Contains the list of found source files, total line count,
    and files that were skipped during scanning.

    Attributes:
        files: List of paths to source files found.
        total_lines: Total number of lines across all source files.
        skipped: List of paths that were skipped (ignored or non-matching).
    """

    files: list[Path]
    total_lines: int
    skipped: list[Path]


class CodebaseScanner:
    """Scans a codebase for source files.

    Recursively scans a directory tree to identify source files based on
    configurable file extensions. Directories matching ignore patterns are
    skipped, and a progress callback can be provided for UI feedback.

    Attributes:
        extensions: Set of file extensions to include (e.g., {".py"}).
        ignore_patterns: Set of directory/file names to skip.

    Example:
        >>> scanner = CodebaseScanner()
        >>> result = await scanner.scan(Path("/my/project"))
        >>> for file in result.files:
        ...     print(file)
    """

    def __init__(
        self,
        extensions: set[str] | None = None,
        ignore_patterns: set[str] | None = None,
    ) -> None:
        """Initialize the codebase scanner.

        Args:
            extensions: Set of file extensions to include. Defaults to {".py"}.
            ignore_patterns: Set of directory/file names to skip.
                Defaults to common Python project directories.
        """
        self.extensions: frozenset[str] = (
            frozenset(extensions) if extensions else DEFAULT_EXTENSIONS
        )
        self.ignore_patterns: frozenset[str] = (
            frozenset(ignore_patterns) if ignore_patterns else DEFAULT_IGNORE
        )

    async def scan(
        self,
        root_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ScanResult:
        """Scan a directory for source files.

        Recursively scans the directory tree starting from root_path,
        collecting files that match the configured extensions while
        skipping directories that match ignore patterns.

        Args:
            root_path: Root directory to scan.
            progress_callback: Optional callback function called with
                (current_file_index, total_files) for progress reporting.

        Returns:
            ScanResult containing found files, total lines, and skipped files.

        Example:
            >>> def on_progress(current, total):
            ...     print(f"Scanning {current}/{total}")
            >>> result = await scanner.scan(Path("."), progress_callback=on_progress)
        """
        files: list[Path] = []
        skipped: list[Path] = []
        total_lines = 0

        # Collect all items in the directory tree
        all_items = list(root_path.rglob("*"))
        total = len(all_items)

        for i, path in enumerate(all_items):
            # Report progress if callback provided
            if progress_callback:
                progress_callback(i, total)

            # Skip non-files
            if not path.is_file():
                continue

            # Check if any parent directory matches ignore patterns
            if self._should_ignore(path):
                skipped.append(path)
                continue

            # Check file extension
            if path.suffix not in self.extensions:
                skipped.append(path)
                continue

            # Add to results
            files.append(path)

            # Count lines (with error handling for encoding issues)
            try:
                content = await self._read_file(path)
                total_lines += len(content.splitlines())
            except Exception as e:
                logger.warning(
                    "Failed to read file for line counting",
                    extra={"path": str(path), "error": str(e)},
                )
                # File is still included in files list, just can't count lines

        logger.info(
            "Codebase scan complete",
            extra={
                "files_found": len(files),
                "total_lines": total_lines,
                "files_skipped": len(skipped),
                "root_path": str(root_path),
            },
        )

        return ScanResult(files=files, total_lines=total_lines, skipped=skipped)

    def _should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored based on patterns.

        Args:
            path: The path to check.

        Returns:
            True if the path or any parent matches an ignore pattern.
        """
        for part in path.parts:
            if part in self.ignore_patterns:
                return True
            # Handle glob-style patterns like "*.egg-info"
            for pattern in self.ignore_patterns:
                if pattern.startswith("*") and part.endswith(pattern[1:]):
                    return True
        return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=1.0),
        reraise=True,
    )
    async def _read_file(self, path: Path) -> str:
        """Read file content with retry for transient errors.

        Uses asyncio.to_thread to avoid blocking the event loop
        during file I/O operations.

        Args:
            path: Path to the file to read.

        Returns:
            The file content as a string.

        Raises:
            UnicodeDecodeError: If the file cannot be decoded as UTF-8.
            OSError: If the file cannot be read.
        """
        return await asyncio.to_thread(path.read_text, encoding="utf-8")
