"""Unit tests for the init command."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from yolo_developer.cli.commands.init import (
    PYPROJECT_TEMPLATE,
    README_TEMPLATE,
    create_directory_structure,
    create_py_typed,
    create_pyproject_toml,
    create_readme,
    init_command,
    run_uv_sync,
    validate_python_version,
)


class TestPyprojectTemplate:
    """Tests for pyproject.toml template validation."""

    def test_template_is_valid_toml(self) -> None:
        """Test that the pyproject.toml template produces valid TOML."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        # Format the template with test values
        content = PYPROJECT_TEMPLATE.format(
            name="test-project",
            author="Test Author",
            email="test@example.com",
        )

        # Should parse without error
        parsed = tomllib.loads(content)
        assert parsed is not None

    def test_template_has_project_section(self) -> None:
        """Test that template has [project] section."""
        assert "[project]" in PYPROJECT_TEMPLATE

    def test_template_has_build_system(self) -> None:
        """Test that template has [build-system] section."""
        assert "[build-system]" in PYPROJECT_TEMPLATE

    def test_template_has_project_scripts(self) -> None:
        """Test that template has [project.scripts] section."""
        assert "[project.scripts]" in PYPROJECT_TEMPLATE


class TestRequiredDependencies:
    """Tests for required dependencies in template."""

    REQUIRED_DEPS: ClassVar[list[str]] = [
        "langgraph>=1.0.5",
        "langchain-core",
        "langchain-anthropic",
        "langchain-openai",
        "chromadb>=1.2.0",
        "typer",
        "rich",
        "pydantic>=2.0.0",
        "pydantic-settings",
        "litellm",
        "tenacity",
        "structlog",
        "pyyaml",
        "python-dotenv",
        "fastmcp>=2.0.0",
    ]

    def test_all_required_dependencies_present(self) -> None:
        """Test that all required dependencies are in the template."""
        for dep in self.REQUIRED_DEPS:
            assert dep in PYPROJECT_TEMPLATE, f"Missing dependency: {dep}"

    def test_dev_dependencies_section_exists(self) -> None:
        """Test that [project.optional-dependencies] section exists."""
        assert "[project.optional-dependencies]" in PYPROJECT_TEMPLATE

    def test_pytest_in_dev_dependencies(self) -> None:
        """Test that pytest is in dev dependencies."""
        assert '"pytest"' in PYPROJECT_TEMPLATE or "'pytest'" in PYPROJECT_TEMPLATE


class TestPEP621Metadata:
    """Tests for PEP 621 metadata compliance."""

    def test_has_name_placeholder(self) -> None:
        """Test that name field has placeholder."""
        assert 'name = "{name}"' in PYPROJECT_TEMPLATE

    def test_has_version(self) -> None:
        """Test that version field is present."""
        assert 'version = "0.1.0"' in PYPROJECT_TEMPLATE

    def test_has_description(self) -> None:
        """Test that description field is present."""
        assert 'description = "Autonomous multi-agent AI development system' in PYPROJECT_TEMPLATE

    def test_has_requires_python(self) -> None:
        """Test that requires-python field is present."""
        assert 'requires-python = ">=3.10"' in PYPROJECT_TEMPLATE

    def test_has_license(self) -> None:
        """Test that license field is present."""
        assert "license" in PYPROJECT_TEMPLATE
        assert "MIT" in PYPROJECT_TEMPLATE

    def test_has_authors(self) -> None:
        """Test that authors field has placeholders."""
        assert "authors" in PYPROJECT_TEMPLATE
        assert "{author}" in PYPROJECT_TEMPLATE
        assert "{email}" in PYPROJECT_TEMPLATE


class TestEntryPoint:
    """Tests for entry point configuration."""

    def test_yolo_entry_point_defined(self) -> None:
        """Test that yolo entry point is defined."""
        assert 'yolo = "yolo_developer.cli.main:app"' in PYPROJECT_TEMPLATE

    def test_hatchling_build_backend(self) -> None:
        """Test that hatchling is the build backend."""
        assert 'build-backend = "hatchling.build"' in PYPROJECT_TEMPLATE


class TestValidatePythonVersion:
    """Tests for Python version validation."""

    def test_valid_python_310(self) -> None:
        """Test that Python 3.10 is valid."""
        with patch.object(sys, "version_info", (3, 10, 0)):
            assert validate_python_version() is True

    def test_valid_python_311(self) -> None:
        """Test that Python 3.11 is valid."""
        with patch.object(sys, "version_info", (3, 11, 0)):
            assert validate_python_version() is True

    def test_valid_python_312(self) -> None:
        """Test that Python 3.12 is valid."""
        with patch.object(sys, "version_info", (3, 12, 0)):
            assert validate_python_version() is True

    def test_invalid_python_39(self) -> None:
        """Test that Python 3.9 is invalid."""
        with patch.object(sys, "version_info", (3, 9, 0)):
            assert validate_python_version() is False

    def test_invalid_python_2(self) -> None:
        """Test that Python 2.x is invalid."""
        with patch.object(sys, "version_info", (2, 7, 0)):
            assert validate_python_version() is False


class TestCreateDirectoryStructure:
    """Tests for directory structure creation."""

    def test_creates_src_directory(self) -> None:
        """Test that src directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            create_directory_structure(project_path)
            assert (project_path / "src" / "yolo_developer").exists()

    def test_creates_cli_directory(self) -> None:
        """Test that CLI directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            create_directory_structure(project_path)
            assert (project_path / "src" / "yolo_developer" / "cli").exists()

    def test_creates_tests_directory(self) -> None:
        """Test that tests directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            create_directory_structure(project_path)
            assert (project_path / "tests" / "unit").exists()

    def test_creates_init_files(self) -> None:
        """Test that __init__.py files are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            create_directory_structure(project_path)
            assert (project_path / "src" / "yolo_developer" / "__init__.py").exists()


class TestCreatePyprojectToml:
    """Tests for pyproject.toml creation."""

    def test_creates_file(self) -> None:
        """Test that pyproject.toml file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            result = create_pyproject_toml(
                project_path, "test-project", "Test Author", "test@example.com"
            )
            assert result.exists()
            assert result.name == "pyproject.toml"

    def test_file_contains_project_name(self) -> None:
        """Test that file contains the project name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            create_pyproject_toml(
                project_path, "my-cool-project", "Test Author", "test@example.com"
            )
            content = (project_path / "pyproject.toml").read_text()
            assert 'name = "my-cool-project"' in content

    def test_file_contains_author_info(self) -> None:
        """Test that file contains author information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            create_pyproject_toml(project_path, "test-project", "John Doe", "john@example.com")
            content = (project_path / "pyproject.toml").read_text()
            assert "John Doe" in content
            assert "john@example.com" in content


class TestCreateReadme:
    """Tests for README.md creation."""

    def test_creates_file(self) -> None:
        """Test that README.md file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            create_readme(project_path, "test-project")
            assert (project_path / "README.md").exists()

    def test_file_contains_project_name(self) -> None:
        """Test that README contains project name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            create_readme(project_path, "awesome-project")
            content = (project_path / "README.md").read_text()
            assert "# awesome-project" in content


class TestCreatePyTyped:
    """Tests for py.typed marker creation."""

    def test_creates_py_typed(self) -> None:
        """Test that py.typed marker is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            # Create necessary parent directories
            (project_path / "src" / "yolo_developer").mkdir(parents=True)
            create_py_typed(project_path)
            assert (project_path / "src" / "yolo_developer" / "py.typed").exists()


class TestReadmeTemplate:
    """Tests for README template."""

    def test_has_project_name_placeholder(self) -> None:
        """Test that README template has project name placeholder."""
        assert "{name}" in README_TEMPLATE

    def test_has_installation_section(self) -> None:
        """Test that README has installation section."""
        assert "## Installation" in README_TEMPLATE

    def test_has_usage_section(self) -> None:
        """Test that README has usage section."""
        assert "## Usage" in README_TEMPLATE

    def test_has_uv_sync_command(self) -> None:
        """Test that README mentions uv sync."""
        assert "uv sync" in README_TEMPLATE


class TestRunUvSync:
    """Tests for run_uv_sync function."""

    def test_run_uv_sync_success(self) -> None:
        """Test that run_uv_sync returns True on success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            # Create a minimal pyproject.toml
            (project_path / "pyproject.toml").write_text(
                '[project]\nname = "test"\nversion = "0.1.0"\n'
            )
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = None
                result = run_uv_sync(project_path)
                assert result is True
                mock_run.assert_called_once()

    def test_run_uv_sync_called_process_error(self) -> None:
        """Test that run_uv_sync returns False on CalledProcessError."""
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.CalledProcessError(1, "uv", stderr="error")
                result = run_uv_sync(project_path)
                assert result is False

    def test_run_uv_sync_file_not_found(self) -> None:
        """Test that run_uv_sync returns True when uv not found (graceful fallback)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError("uv not found")
                result = run_uv_sync(project_path)
                # Returns True because it's a graceful fallback
                assert result is True


class TestInitCommand:
    """Integration tests for init_command function."""

    def test_init_command_creates_project(self) -> None:
        """Test that init_command creates a complete project structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "test-project"
            with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                init_command(
                    path=str(project_path),
                    name="test-project",
                    author="Test Author",
                    email="test@example.com",
                )

            # Verify project structure
            assert project_path.exists()
            assert (project_path / "pyproject.toml").exists()
            assert (project_path / "README.md").exists()
            assert (project_path / "src" / "yolo_developer").exists()
            assert (project_path / "tests" / "unit").exists()

    def test_init_command_creates_pyproject_with_name(self) -> None:
        """Test that init_command creates pyproject.toml with correct name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "my-project"
            with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                init_command(
                    path=str(project_path),
                    name="my-cool-project",
                    author="Test Author",
                    email="test@example.com",
                )

            content = (project_path / "pyproject.toml").read_text()
            assert 'name = "my-cool-project"' in content

    def test_init_command_uses_defaults(self) -> None:
        """Test that init_command uses default author info when not provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "default-project"
            # Mock git config to return empty (no git config set)
            with (
                patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True),
                patch("yolo_developer.cli.commands.init.get_git_config", return_value=""),
            ):
                init_command(path=str(project_path))

            content = (project_path / "pyproject.toml").read_text()
            assert "YOLO Developer" in content
            assert "dev@example.com" in content
