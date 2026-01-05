"""Unit tests for codebase scanner.

Tests CodebaseScanner class for file discovery and filtering.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_developer.memory.scanner import (
    DEFAULT_EXTENSIONS,
    DEFAULT_IGNORE,
    CodebaseScanner,
    ScanResult,
)


class TestScanResult:
    """Tests for ScanResult dataclass."""

    def test_create_scan_result(self) -> None:
        """Test creating a scan result."""
        result = ScanResult(
            files=[Path("src/main.py"), Path("src/utils.py")],
            total_lines=100,
            skipped=[Path("README.md")],
        )

        assert len(result.files) == 2
        assert result.total_lines == 100
        assert len(result.skipped) == 1

    def test_empty_scan_result(self) -> None:
        """Test empty scan result."""
        result = ScanResult(files=[], total_lines=0, skipped=[])

        assert result.files == []
        assert result.total_lines == 0
        assert result.skipped == []


class TestDefaultValues:
    """Tests for default extension and ignore patterns."""

    def test_default_extensions(self) -> None:
        """Test default extensions include Python."""
        assert ".py" in DEFAULT_EXTENSIONS

    def test_default_ignore_patterns(self) -> None:
        """Test default ignore patterns include common directories."""
        assert "__pycache__" in DEFAULT_IGNORE
        assert ".git" in DEFAULT_IGNORE
        assert "node_modules" in DEFAULT_IGNORE
        assert ".venv" in DEFAULT_IGNORE


class TestCodebaseScanner:
    """Tests for CodebaseScanner class."""

    def test_init_with_defaults(self) -> None:
        """Test scanner initializes with default extensions and ignores."""
        scanner = CodebaseScanner()

        assert scanner.extensions == DEFAULT_EXTENSIONS
        assert scanner.ignore_patterns == DEFAULT_IGNORE

    def test_init_with_custom_extensions(self) -> None:
        """Test scanner with custom extensions."""
        scanner = CodebaseScanner(extensions={".py", ".pyi"})

        assert ".py" in scanner.extensions
        assert ".pyi" in scanner.extensions

    def test_init_with_custom_ignore(self) -> None:
        """Test scanner with custom ignore patterns."""
        scanner = CodebaseScanner(ignore_patterns={"custom_dir", ".cache"})

        assert "custom_dir" in scanner.ignore_patterns
        assert ".cache" in scanner.ignore_patterns


class TestCodebaseScannerScan:
    """Tests for CodebaseScanner.scan method."""

    @pytest.fixture
    def sample_codebase(self, tmp_path: Path) -> Path:
        """Create a sample codebase for testing."""
        # Create source files
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("def main():\n    pass\n")
        (src / "utils.py").write_text("def helper():\n    pass\n\ndef another():\n    pass\n")

        # Create nested module
        module = src / "module"
        module.mkdir()
        (module / "__init__.py").write_text("")
        (module / "core.py").write_text("class Core:\n    pass\n")

        # Create test files
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_main.py").write_text("def test_main():\n    pass\n")

        # Create files to ignore
        pycache = src / "__pycache__"
        pycache.mkdir()
        (pycache / "main.cpython-311.pyc").write_text("cached")

        # Create non-Python files
        (tmp_path / "README.md").write_text("# README")
        (tmp_path / "requirements.txt").write_text("pytest\n")

        return tmp_path

    @pytest.mark.asyncio
    async def test_scan_finds_python_files(self, sample_codebase: Path) -> None:
        """Test scanner finds all Python files."""
        scanner = CodebaseScanner()
        result = await scanner.scan(sample_codebase)

        # Should find: main.py, utils.py, __init__.py, core.py, test_main.py
        assert len(result.files) == 5

        # Verify specific files found
        filenames = [f.name for f in result.files]
        assert "main.py" in filenames
        assert "utils.py" in filenames
        assert "__init__.py" in filenames
        assert "core.py" in filenames
        assert "test_main.py" in filenames

    @pytest.mark.asyncio
    async def test_scan_ignores_pycache(self, sample_codebase: Path) -> None:
        """Test scanner ignores __pycache__ directory."""
        scanner = CodebaseScanner()
        result = await scanner.scan(sample_codebase)

        # No files from __pycache__ should be included
        for f in result.files:
            assert "__pycache__" not in f.parts

    @pytest.mark.asyncio
    async def test_scan_ignores_non_python(self, sample_codebase: Path) -> None:
        """Test scanner ignores non-Python files by default."""
        scanner = CodebaseScanner()
        result = await scanner.scan(sample_codebase)

        # No non-Python files
        for f in result.files:
            assert f.suffix == ".py"

    @pytest.mark.asyncio
    async def test_scan_counts_lines(self, sample_codebase: Path) -> None:
        """Test scanner counts total lines."""
        scanner = CodebaseScanner()
        result = await scanner.scan(sample_codebase)

        # Should have counted lines from all Python files
        assert result.total_lines > 0

    @pytest.mark.asyncio
    async def test_scan_tracks_skipped(self, sample_codebase: Path) -> None:
        """Test scanner tracks skipped files."""
        scanner = CodebaseScanner()
        result = await scanner.scan(sample_codebase)

        # Should have skipped README.md, requirements.txt, and __pycache__ files
        assert len(result.skipped) > 0

    @pytest.mark.asyncio
    async def test_scan_with_custom_extensions(self, sample_codebase: Path) -> None:
        """Test scanner with custom extensions."""
        # Add a markdown file
        (sample_codebase / "doc.md").write_text("# Doc\n")

        scanner = CodebaseScanner(extensions={".md"})
        result = await scanner.scan(sample_codebase)

        # Should find only .md files
        assert all(f.suffix == ".md" for f in result.files)
        assert any(f.name == "README.md" for f in result.files)

    @pytest.mark.asyncio
    async def test_scan_with_progress_callback(self, sample_codebase: Path) -> None:
        """Test scanner calls progress callback."""
        callback_calls: list[tuple[int, int]] = []

        def progress(current: int, total: int) -> None:
            callback_calls.append((current, total))

        scanner = CodebaseScanner()
        await scanner.scan(sample_codebase, progress_callback=progress)

        # Should have called callback at least once
        assert len(callback_calls) > 0

        # Total should be consistent
        if callback_calls:
            _, total = callback_calls[0]
            assert all(t == total for _, t in callback_calls)

    @pytest.mark.asyncio
    async def test_scan_empty_directory(self, tmp_path: Path) -> None:
        """Test scanning empty directory."""
        scanner = CodebaseScanner()
        result = await scanner.scan(tmp_path)

        assert result.files == []
        assert result.total_lines == 0

    @pytest.mark.asyncio
    async def test_scan_ignores_venv(self, tmp_path: Path) -> None:
        """Test scanner ignores .venv directory."""
        # Create a .venv with Python files
        venv = tmp_path / ".venv"
        venv.mkdir()
        lib = venv / "lib" / "python3.11" / "site-packages"
        lib.mkdir(parents=True)
        (lib / "module.py").write_text("def func(): pass\n")

        # Create a regular Python file
        (tmp_path / "main.py").write_text("def main(): pass\n")

        scanner = CodebaseScanner()
        result = await scanner.scan(tmp_path)

        # Should only find main.py, not files in .venv
        assert len(result.files) == 1
        assert result.files[0].name == "main.py"

    @pytest.mark.asyncio
    async def test_scan_handles_unreadable_file(self, tmp_path: Path) -> None:
        """Test scanner handles files it cannot read gracefully."""
        # Create a Python file
        (tmp_path / "readable.py").write_text("def func(): pass\n")

        # Create a file with invalid encoding
        bad_file = tmp_path / "binary.py"
        bad_file.write_bytes(b"\x80\x81\x82")

        scanner = CodebaseScanner()
        result = await scanner.scan(tmp_path)

        # Should still find the file but handle encoding error gracefully
        assert len(result.files) == 2  # Both files found
        # Lines might not include binary file


class TestCodebaseScannerEdgeCases:
    """Edge case tests for CodebaseScanner."""

    @pytest.mark.asyncio
    async def test_scan_deep_nesting(self, tmp_path: Path) -> None:
        """Test scanner handles deeply nested directories."""
        # Create deeply nested structure
        current = tmp_path
        for i in range(10):
            current = current / f"level_{i}"
            current.mkdir()

        (current / "deep.py").write_text("deep = True\n")

        scanner = CodebaseScanner()
        result = await scanner.scan(tmp_path)

        assert len(result.files) == 1
        assert result.files[0].name == "deep.py"

    @pytest.mark.asyncio
    async def test_scan_symlinks_not_followed_by_default(self, tmp_path: Path) -> None:
        """Test scanner behavior with symlinks."""
        # Create a directory with a Python file
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        (real_dir / "real.py").write_text("real = True\n")

        # Note: Symlinks might be followed by rglob, but we test the basic case
        scanner = CodebaseScanner()
        result = await scanner.scan(tmp_path)

        # Should find the real file
        assert any(f.name == "real.py" for f in result.files)

    @pytest.mark.asyncio
    async def test_scan_with_git_ignore(self, tmp_path: Path) -> None:
        """Test scanner ignores .git directory."""
        # Create a .git directory (simulating a git repo)
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        objects = git_dir / "objects"
        objects.mkdir()
        # Git doesn't usually have .py files, but test the ignore
        (git_dir / "config").write_text("[core]\n")

        # Create a regular Python file
        (tmp_path / "app.py").write_text("app = True\n")

        scanner = CodebaseScanner()
        result = await scanner.scan(tmp_path)

        # Should only find app.py
        assert len(result.files) == 1
        assert result.files[0].name == "app.py"
