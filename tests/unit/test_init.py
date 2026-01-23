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
    add_git_remote,
    check_gh_authenticated,
    check_gh_cli_available,
    check_git_initialized,
    create_directory_structure,
    create_github_repo,
    create_initial_commit,
    create_py_typed,
    create_pyproject_toml,
    create_readme,
    display_git_status,
    get_git_remotes,
    init_command,
    init_git_repository,
    push_to_remote,
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
                    skip_git=True,  # Skip git prompts in tests
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
                    skip_git=True,  # Skip git prompts in tests
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
                init_command(path=str(project_path), skip_git=True)  # Skip git prompts

            content = (project_path / "pyproject.toml").read_text()
            assert "YOLO Developer" in content
            assert "dev@example.com" in content


class TestCheckGitInitialized:
    """Tests for check_git_initialized function."""

    def test_returns_true_in_git_repo(self) -> None:
        """Test that check_git_initialized returns True in a git repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            # Initialize a git repository
            import subprocess

            subprocess.run(
                ["git", "init"],
                cwd=project_path,
                capture_output=True,
                check=True,
            )
            assert check_git_initialized(project_path) is True

    def test_returns_false_in_non_git_dir(self) -> None:
        """Test that check_git_initialized returns False in a non-git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            assert check_git_initialized(project_path) is False

    def test_returns_false_when_git_not_installed(self) -> None:
        """Test graceful handling when git is not installed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError("git not found")
                assert check_git_initialized(project_path) is False

    def test_uses_current_directory_when_path_is_none(self) -> None:
        """Test that check_git_initialized uses cwd when path is None."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = ".git"
            mock_run.return_value.returncode = 0
            check_git_initialized(None)
            # Verify cwd was None (uses current directory)
            mock_run.assert_called_once()
            assert mock_run.call_args.kwargs["cwd"] is None


class TestGetGitRemotes:
    """Tests for get_git_remotes function."""

    def test_returns_empty_dict_in_non_git_dir(self) -> None:
        """Test that get_git_remotes returns empty dict in non-git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            assert get_git_remotes(project_path) == {}

    def test_returns_empty_dict_when_no_remotes(self) -> None:
        """Test that get_git_remotes returns empty dict when no remotes configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            import subprocess

            subprocess.run(
                ["git", "init"],
                cwd=project_path,
                capture_output=True,
                check=True,
            )
            assert get_git_remotes(project_path) == {}

    def test_returns_remotes_when_configured(self) -> None:
        """Test that get_git_remotes returns configured remotes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            import subprocess

            subprocess.run(
                ["git", "init"],
                cwd=project_path,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "remote", "add", "origin", "https://github.com/user/repo.git"],
                cwd=project_path,
                capture_output=True,
                check=True,
            )
            remotes = get_git_remotes(project_path)
            assert "origin" in remotes
            assert remotes["origin"] == "https://github.com/user/repo.git"

    def test_returns_multiple_remotes(self) -> None:
        """Test that get_git_remotes returns all configured remotes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            import subprocess

            subprocess.run(
                ["git", "init"],
                cwd=project_path,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "remote", "add", "origin", "https://github.com/user/repo.git"],
                cwd=project_path,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "remote", "add", "upstream", "https://github.com/org/repo.git"],
                cwd=project_path,
                capture_output=True,
                check=True,
            )
            remotes = get_git_remotes(project_path)
            assert len(remotes) == 2
            assert remotes["origin"] == "https://github.com/user/repo.git"
            assert remotes["upstream"] == "https://github.com/org/repo.git"

    def test_returns_empty_dict_when_git_not_installed(self) -> None:
        """Test graceful handling when git is not installed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError("git not found")
                assert get_git_remotes(project_path) == {}

    def test_uses_current_directory_when_path_is_none(self) -> None:
        """Test that get_git_remotes uses cwd when path is None."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = ""
            mock_run.return_value.returncode = 0
            get_git_remotes(None)
            # Verify cwd was None (uses current directory)
            mock_run.assert_called_once()
            assert mock_run.call_args.kwargs["cwd"] is None


class TestInitGitRepository:
    """Tests for init_git_repository function."""

    def test_initializes_git_repo(self) -> None:
        """Test that init_git_repository creates a git repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            result = init_git_repository(project_path)

            assert result is True
            assert (project_path / ".git").exists()

    def test_creates_gitignore(self) -> None:
        """Test that init_git_repository creates a .gitignore file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            init_git_repository(project_path)

            gitignore_path = project_path / ".gitignore"
            assert gitignore_path.exists()
            content = gitignore_path.read_text()
            assert "__pycache__/" in content
            assert ".yolo/" in content
            assert ".venv" in content

    def test_does_not_overwrite_existing_gitignore(self) -> None:
        """Test that init_git_repository preserves existing .gitignore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            gitignore_path = project_path / ".gitignore"
            existing_content = "# Custom gitignore\nnode_modules/\n"
            gitignore_path.write_text(existing_content)

            init_git_repository(project_path)

            assert gitignore_path.read_text() == existing_content

    def test_returns_false_when_git_not_installed(self) -> None:
        """Test graceful handling when git is not installed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError("git not found")
                result = init_git_repository(project_path)
                assert result is False

    def test_returns_false_on_git_error(self) -> None:
        """Test handling of git init failure."""
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.CalledProcessError(
                    1, "git", stderr="error"
                )
                result = init_git_repository(project_path)
                assert result is False


class TestInitCommandWithGit:
    """Tests for init_command git integration."""

    def test_skip_git_option_skips_prompts(self) -> None:
        """Test that --skip-git prevents git initialization prompts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "test-project"
            with (
                patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True),
                patch("yolo_developer.cli.commands.init.init_git_repository") as mock_init_git,
                patch("typer.confirm") as mock_confirm,
            ):
                init_command(path=str(project_path), skip_git=True)

                # Should not call init_git_repository or prompt
                mock_init_git.assert_not_called()
                mock_confirm.assert_not_called()

    def test_no_input_skips_git_prompts(self) -> None:
        """Test that --no-input prevents git initialization prompts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "test-project"
            with (
                patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True),
                patch("yolo_developer.cli.commands.init.init_git_repository") as mock_init_git,
                patch("typer.confirm") as mock_confirm,
            ):
                init_command(path=str(project_path), no_input=True)

                # Should not prompt or initialize git
                mock_confirm.assert_not_called()
                mock_init_git.assert_not_called()


class TestAddGitRemote:
    """Tests for add_git_remote function."""

    def test_adds_remote_successfully(self) -> None:
        """Test that add_git_remote adds a remote to a git repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            import subprocess

            subprocess.run(
                ["git", "init"],
                cwd=project_path,
                capture_output=True,
                check=True,
            )

            result = add_git_remote(
                project_path, "origin", "https://github.com/user/repo.git"
            )

            assert result is True
            remotes = get_git_remotes(project_path)
            assert "origin" in remotes
            assert remotes["origin"] == "https://github.com/user/repo.git"

    def test_returns_false_when_git_not_installed(self) -> None:
        """Test graceful handling when git is not installed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError("git not found")
                result = add_git_remote(
                    project_path, "origin", "https://github.com/user/repo.git"
                )
                assert result is False

    def test_returns_false_on_git_error(self) -> None:
        """Test handling of git remote add failure."""
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.CalledProcessError(
                    1, "git", stderr="error"
                )
                result = add_git_remote(
                    project_path, "origin", "https://github.com/user/repo.git"
                )
                assert result is False


class TestDisplayGitStatus:
    """Tests for display_git_status function."""

    def test_displays_not_initialized(self, capsys) -> None:
        """Test display when git is not initialized."""
        display_git_status(False, {})
        captured = capsys.readouterr()
        assert "Not initialized" in captured.out or "not initialized" in captured.out.lower()

    def test_displays_initialized_no_remotes(self, capsys) -> None:
        """Test display when git is initialized but no remotes."""
        display_git_status(True, {})
        captured = capsys.readouterr()
        assert "None configured" in captured.out or "none" in captured.out.lower()

    def test_displays_initialized_with_remotes(self, capsys) -> None:
        """Test display when git is initialized with remotes."""
        display_git_status(True, {"origin": "https://github.com/user/repo.git"})
        captured = capsys.readouterr()
        assert "origin" in captured.out


class TestCheckGhCliAvailable:
    """Tests for check_gh_cli_available function."""

    def test_returns_true_when_gh_available(self) -> None:
        """Test returns True when gh CLI is available."""
        with patch("shutil.which", return_value="/usr/bin/gh"):
            assert check_gh_cli_available() is True

    def test_returns_false_when_gh_not_available(self) -> None:
        """Test returns False when gh CLI is not available."""
        with patch("shutil.which", return_value=None):
            assert check_gh_cli_available() is False


class TestCheckGhAuthenticated:
    """Tests for check_gh_authenticated function."""

    def test_returns_true_when_authenticated(self) -> None:
        """Test returns True when gh is authenticated."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            assert check_gh_authenticated() is True

    def test_returns_false_when_not_authenticated(self) -> None:
        """Test returns False when gh is not authenticated."""
        import subprocess

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "gh")
            assert check_gh_authenticated() is False

    def test_returns_false_when_gh_not_installed(self) -> None:
        """Test returns False when gh is not installed."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("gh not found")
            assert check_gh_authenticated() is False


class TestCreateGithubRepo:
    """Tests for create_github_repo function."""

    def test_returns_false_when_gh_not_available(self) -> None:
        """Test returns False when gh CLI is not available."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch(
                "yolo_developer.cli.commands.init.check_gh_cli_available",
                return_value=False,
            ),
        ):
            project_path = Path(tmpdir)
            success, url = create_github_repo(project_path, "test-repo")
            assert success is False
            assert url is None

    def test_returns_false_when_not_authenticated(self) -> None:
        """Test returns False when gh is not authenticated."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch(
                "yolo_developer.cli.commands.init.check_gh_cli_available",
                return_value=True,
            ),
            patch(
                "yolo_developer.cli.commands.init.check_gh_authenticated",
                return_value=False,
            ),
        ):
            project_path = Path(tmpdir)
            success, url = create_github_repo(project_path, "test-repo")
            assert success is False
            assert url is None

    def test_creates_repo_successfully(self) -> None:
        """Test successful repository creation."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch(
                "yolo_developer.cli.commands.init.check_gh_cli_available",
                return_value=True,
            ),
            patch(
                "yolo_developer.cli.commands.init.check_gh_authenticated",
                return_value=True,
            ),
            patch("subprocess.run") as mock_run,
            patch(
                "yolo_developer.cli.commands.init.get_git_remotes",
                return_value={"origin": "https://github.com/user/test-repo.git"},
            ),
        ):
            mock_run.return_value.stdout = "https://github.com/user/test-repo.git"
            mock_run.return_value.returncode = 0

            project_path = Path(tmpdir)
            success, url = create_github_repo(project_path, "test-repo")

            assert success is True
            assert url == "https://github.com/user/test-repo.git"

    def test_returns_false_on_error(self) -> None:
        """Test returns False on subprocess error."""
        import subprocess

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch(
                "yolo_developer.cli.commands.init.check_gh_cli_available",
                return_value=True,
            ),
            patch(
                "yolo_developer.cli.commands.init.check_gh_authenticated",
                return_value=True,
            ),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.side_effect = subprocess.CalledProcessError(1, "gh", stderr="error")

            project_path = Path(tmpdir)
            success, url = create_github_repo(project_path, "test-repo")

            assert success is False
            assert url is None


class TestCreateInitialCommit:
    """Tests for create_initial_commit function."""

    def test_creates_commit_successfully(self) -> None:
        """Test successful initial commit creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            import subprocess

            # Initialize git repo
            subprocess.run(
                ["git", "init"],
                cwd=project_path,
                capture_output=True,
                check=True,
            )

            # Configure git user for commit
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=project_path,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                cwd=project_path,
                capture_output=True,
                check=True,
            )

            # Create a file to commit
            (project_path / "README.md").write_text("# Test Project")

            result = create_initial_commit(project_path, "Initial commit")

            assert result is True

            # Verify commit exists
            log_result = subprocess.run(
                ["git", "log", "--oneline", "-1"],
                cwd=project_path,
                capture_output=True,
                text=True,
            )
            assert "Initial commit" in log_result.stdout

    def test_returns_true_when_nothing_to_commit(self) -> None:
        """Test returns True when there's nothing to commit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            import subprocess

            # Initialize git repo
            subprocess.run(
                ["git", "init"],
                cwd=project_path,
                capture_output=True,
                check=True,
            )

            # Configure git user
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=project_path,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                cwd=project_path,
                capture_output=True,
                check=True,
            )

            # Don't create any files - nothing to commit
            result = create_initial_commit(project_path)

            # Should return True (graceful handling)
            assert result is True

    def test_returns_false_when_git_not_installed(self) -> None:
        """Test returns False when git is not installed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError("git not found")
                result = create_initial_commit(project_path)
                assert result is False


class TestPushToRemote:
    """Tests for push_to_remote function."""

    def test_push_successfully(self) -> None:
        """Test successful push to remote."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("subprocess.run") as mock_run,
        ):
            # Mock git commands
            mock_run.return_value.stdout = "main"
            mock_run.return_value.returncode = 0

            project_path = Path(tmpdir)
            result = push_to_remote(project_path)

            assert result is True
            # Verify git push was called
            push_call = [call for call in mock_run.call_args_list if "push" in str(call)]
            assert len(push_call) > 0

    def test_returns_false_on_error(self) -> None:
        """Test returns False on push error."""
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            with patch("subprocess.run") as mock_run:
                # First call for branch detection succeeds
                mock_run.return_value.stdout = "main"
                mock_run.return_value.returncode = 0
                # Then fail on push
                mock_run.side_effect = [
                    type("Result", (), {"stdout": "main", "returncode": 0})(),
                    subprocess.CalledProcessError(1, "git", stderr="error"),
                ]
                result = push_to_remote(project_path)
                assert result is False

    def test_returns_false_when_git_not_installed(self) -> None:
        """Test returns False when git is not installed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError("git not found")
                result = push_to_remote(project_path)
                assert result is False
