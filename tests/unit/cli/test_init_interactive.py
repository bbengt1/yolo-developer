"""Tests for init command interactive mode and new flags (Story 12.2).

Tests cover:
- Interactive mode with Typer prompts (AC2)
- --no-input flag for non-interactive mode (AC3)
- Git config default detection (AC2, AC3)
- --existing flag for brownfield mode (AC4)
- yolo.yaml generation (AC5)
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from yolo_developer.cli.main import app

runner = CliRunner()


class TestInteractiveMode:
    """Tests for interactive mode prompts (AC2)."""

    def test_interactive_flag_prompts_for_project_name(self) -> None:
        """Test --interactive prompts for project name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate user input: project name, author, email
            result = runner.invoke(
                app,
                ["init", tmpdir, "--interactive"],
                input="my-project\nJohn Doe\njohn@example.com\n",
            )

            assert result.exit_code == 0
            # Check pyproject.toml has the entered name
            content = (Path(tmpdir) / "pyproject.toml").read_text()
            assert 'name = "my-project"' in content

    def test_interactive_flag_prompts_for_author(self) -> None:
        """Test --interactive prompts for author name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                ["init", tmpdir, "--interactive"],
                input="test-proj\nJane Smith\njane@test.com\n",
            )

            assert result.exit_code == 0
            content = (Path(tmpdir) / "pyproject.toml").read_text()
            assert "Jane Smith" in content

    def test_interactive_flag_prompts_for_email(self) -> None:
        """Test --interactive prompts for author email."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                ["init", tmpdir, "--interactive"],
                input="proj\nAuthor\nauthor@domain.com\n",
            )

            assert result.exit_code == 0
            content = (Path(tmpdir) / "pyproject.toml").read_text()
            assert "author@domain.com" in content

    def test_interactive_uses_git_config_as_defaults(self) -> None:
        """Test that interactive mode uses git config values as defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("yolo_developer.cli.commands.init.get_git_config") as mock_git:
                mock_git.side_effect = lambda key: {
                    "user.name": "Git User",
                    "user.email": "git@user.com",
                }.get(key, "")

                # Press enter to accept defaults
                result = runner.invoke(
                    app,
                    ["init", tmpdir, "--interactive"],
                    input="\n\n\n",  # Accept all defaults
                )

                assert result.exit_code == 0
                content = (Path(tmpdir) / "pyproject.toml").read_text()
                assert "Git User" in content
                assert "git@user.com" in content

    def test_interactive_short_flag(self) -> None:
        """Test -i short flag works for interactive mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                ["init", tmpdir, "-i"],
                input="proj\nAuthor\nemail@test.com\n",
            )

            assert result.exit_code == 0


class TestNoInputMode:
    """Tests for --no-input flag (AC3)."""

    def test_no_input_uses_directory_name(self) -> None:
        """Test --no-input uses directory name as project name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "my-awesome-project"
            project_dir.mkdir()

            with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                result = runner.invoke(app, ["init", str(project_dir), "--no-input"])

            assert result.exit_code == 0
            content = (project_dir / "pyproject.toml").read_text()
            assert 'name = "my-awesome-project"' in content

    def test_no_input_uses_git_config_defaults(self) -> None:
        """Test --no-input uses git config for author defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("yolo_developer.cli.commands.init.get_git_config") as mock_git:
                mock_git.side_effect = lambda key: {
                    "user.name": "Configured User",
                    "user.email": "configured@email.com",
                }.get(key, "")

                with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                    result = runner.invoke(app, ["init", tmpdir, "--no-input"])

                assert result.exit_code == 0
                content = (Path(tmpdir) / "pyproject.toml").read_text()
                assert "Configured User" in content
                assert "configured@email.com" in content

    def test_no_input_uses_fallback_when_no_git_config(self) -> None:
        """Test --no-input uses fallback values when git config not available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "yolo_developer.cli.commands.init.get_git_config",
                return_value="",
            ):
                with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                    result = runner.invoke(app, ["init", tmpdir, "--no-input"])

                assert result.exit_code == 0
                content = (Path(tmpdir) / "pyproject.toml").read_text()
                # Should use fallback values
                assert "Developer" in content or "YOLO Developer" in content

    def test_no_input_displays_no_prompts(self) -> None:
        """Test --no-input doesn't display any prompts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                result = runner.invoke(app, ["init", tmpdir, "--no-input"])

            assert result.exit_code == 0
            # Should not contain prompt text
            assert "Project name" not in result.output or ":" not in result.output

    def test_no_input_project_is_functional(self) -> None:
        """Test --no-input creates a fully functional project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                result = runner.invoke(app, ["init", tmpdir, "--no-input"])

            assert result.exit_code == 0
            # Verify essential files exist
            assert (Path(tmpdir) / "pyproject.toml").exists()
            assert (Path(tmpdir) / "README.md").exists()
            assert (Path(tmpdir) / "src" / "yolo_developer").exists()


class TestGitConfigDetection:
    """Tests for git config default detection."""

    def test_get_git_config_returns_user_name(self) -> None:
        """Test get_git_config retrieves user.name."""
        from yolo_developer.cli.commands.init import get_git_config

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="Test User\n",
                returncode=0,
            )

            result = get_git_config("user.name")

            assert result == "Test User"
            mock_run.assert_called_once()

    def test_get_git_config_returns_user_email(self) -> None:
        """Test get_git_config retrieves user.email."""
        from yolo_developer.cli.commands.init import get_git_config

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="test@email.com\n",
                returncode=0,
            )

            result = get_git_config("user.email")

            assert result == "test@email.com"

    def test_get_git_config_returns_empty_on_error(self) -> None:
        """Test get_git_config returns empty string on error."""
        from yolo_developer.cli.commands.init import get_git_config

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git")

            result = get_git_config("user.name")

            assert result == ""

    def test_get_git_config_returns_empty_when_not_set(self) -> None:
        """Test get_git_config returns empty string when config not set."""
        from yolo_developer.cli.commands.init import get_git_config

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", returncode=0)

            result = get_git_config("user.name")

            assert result == ""


class TestYoloYamlGeneration:
    """Tests for yolo.yaml generation (AC5)."""

    def test_init_creates_yolo_yaml(self) -> None:
        """Test that init creates yolo.yaml configuration file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                result = runner.invoke(app, ["init", tmpdir, "--no-input"])

            assert result.exit_code == 0
            assert (Path(tmpdir) / "yolo.yaml").exists()

    def test_yolo_yaml_has_project_name(self) -> None:
        """Test yolo.yaml contains project_name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "my-test-project"
            project_dir.mkdir()

            with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                runner.invoke(app, ["init", str(project_dir), "--no-input"])

            content = (project_dir / "yolo.yaml").read_text()
            assert "project_name:" in content
            assert "my-test-project" in content

    def test_yolo_yaml_has_llm_section(self) -> None:
        """Test yolo.yaml contains llm configuration section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                runner.invoke(app, ["init", tmpdir, "--no-input"])

            content = (Path(tmpdir) / "yolo.yaml").read_text()
            assert "llm:" in content
            assert "cheap_model:" in content

    def test_yolo_yaml_has_quality_section(self) -> None:
        """Test yolo.yaml contains quality configuration section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                runner.invoke(app, ["init", tmpdir, "--no-input"])

            content = (Path(tmpdir) / "yolo.yaml").read_text()
            assert "quality:" in content
            assert "test_coverage_threshold:" in content

    def test_yolo_yaml_has_memory_section(self) -> None:
        """Test yolo.yaml contains memory configuration section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                runner.invoke(app, ["init", tmpdir, "--no-input"])

            content = (Path(tmpdir) / "yolo.yaml").read_text()
            assert "memory:" in content
            assert "persist_path:" in content

    def test_yolo_yaml_has_comments(self) -> None:
        """Test yolo.yaml contains explanatory comments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                runner.invoke(app, ["init", tmpdir, "--no-input"])

            content = (Path(tmpdir) / "yolo.yaml").read_text()
            # Should have comment lines
            assert "#" in content

    def test_yolo_yaml_has_api_key_placeholders(self) -> None:
        """Test yolo.yaml indicates API keys should be set via env vars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                runner.invoke(app, ["init", tmpdir, "--no-input"])

            content = (Path(tmpdir) / "yolo.yaml").read_text()
            # Should mention env vars for API keys
            assert "YOLO_LLM__OPENAI_API_KEY" in content or "env" in content.lower()


class TestBrownfieldMode:
    """Tests for --existing brownfield mode (AC4)."""

    def test_existing_flag_exists(self) -> None:
        """Test --existing flag is recognized."""
        result = runner.invoke(app, ["init", "--help"])

        assert "--existing" in result.output

    def test_existing_does_not_overwrite_pyproject(self) -> None:
        """Test --existing doesn't overwrite existing pyproject.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing pyproject.toml
            existing_content = '[project]\nname = "existing-project"\nversion = "1.0.0"\n'
            (Path(tmpdir) / "pyproject.toml").write_text(existing_content)

            with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                result = runner.invoke(app, ["init", tmpdir, "--existing"])

            assert result.exit_code == 0
            content = (Path(tmpdir) / "pyproject.toml").read_text()
            # Original project name should be preserved
            assert 'name = "existing-project"' in content
            assert 'version = "1.0.0"' in content

    def test_existing_adds_yolo_dependencies(self) -> None:
        """Test --existing adds YOLO dependencies to existing pyproject.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal existing pyproject.toml
            existing_content = """[project]
name = "existing-project"
version = "1.0.0"
dependencies = ["requests"]
"""
            (Path(tmpdir) / "pyproject.toml").write_text(existing_content)

            with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                result = runner.invoke(app, ["init", tmpdir, "--existing"])

            assert result.exit_code == 0
            content = (Path(tmpdir) / "pyproject.toml").read_text()
            # Should still have original dependencies
            assert "requests" in content
            # And YOLO dependencies should be added
            assert "langgraph" in content or "yolo" in content.lower()

    def test_existing_creates_yolo_yaml(self) -> None:
        """Test --existing creates yolo.yaml even for brownfield projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing pyproject.toml
            (Path(tmpdir) / "pyproject.toml").write_text(
                '[project]\nname = "existing"\nversion = "1.0.0"\n'
            )

            with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                result = runner.invoke(app, ["init", tmpdir, "--existing"])

            assert result.exit_code == 0
            assert (Path(tmpdir) / "yolo.yaml").exists()

    def test_existing_skips_directory_creation_when_exists(self) -> None:
        """Test --existing doesn't recreate existing directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing structure
            (Path(tmpdir) / "src" / "myapp").mkdir(parents=True)
            (Path(tmpdir) / "src" / "myapp" / "__init__.py").write_text("# My app\n")
            (Path(tmpdir) / "pyproject.toml").write_text(
                '[project]\nname = "myapp"\nversion = "1.0.0"\n'
            )

            with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                result = runner.invoke(app, ["init", tmpdir, "--existing"])

            assert result.exit_code == 0
            # Original init file should be preserved
            content = (Path(tmpdir) / "src" / "myapp" / "__init__.py").read_text()
            assert "# My app" in content


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing behavior."""

    def test_init_without_flags_works(self) -> None:
        """Test init without new flags still works as before."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                result = runner.invoke(
                    app,
                    ["init", tmpdir, "--name", "test", "--author", "Author", "--email", "a@b.com"],
                )

            assert result.exit_code == 0
            assert (Path(tmpdir) / "pyproject.toml").exists()

    def test_existing_cli_options_still_work(self) -> None:
        """Test --name, --author, --email options still work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                result = runner.invoke(
                    app,
                    [
                        "init",
                        tmpdir,
                        "--name",
                        "my-project",
                        "-a",
                        "John",
                        "-e",
                        "john@test.com",
                    ],
                )

            assert result.exit_code == 0
            content = (Path(tmpdir) / "pyproject.toml").read_text()
            assert 'name = "my-project"' in content
            assert "John" in content
            assert "john@test.com" in content


class TestExtractPackageName:
    """Tests for _extract_package_name helper function."""

    def test_extract_simple_package_name(self) -> None:
        """Test extraction of simple package name."""
        from yolo_developer.cli.commands.init import _extract_package_name

        assert _extract_package_name("requests") == "requests"

    def test_extract_with_greater_equals(self) -> None:
        """Test extraction with >= version specifier."""
        from yolo_developer.cli.commands.init import _extract_package_name

        assert _extract_package_name("pydantic>=2.0.0") == "pydantic"

    def test_extract_with_double_equals(self) -> None:
        """Test extraction with == version specifier."""
        from yolo_developer.cli.commands.init import _extract_package_name

        assert _extract_package_name("django==4.2.0") == "django"

    def test_extract_with_tilde_equals(self) -> None:
        """Test extraction with ~= compatible release specifier."""
        from yolo_developer.cli.commands.init import _extract_package_name

        assert _extract_package_name("flask~=2.0") == "flask"

    def test_extract_with_less_than(self) -> None:
        """Test extraction with < version specifier."""
        from yolo_developer.cli.commands.init import _extract_package_name

        assert _extract_package_name("numpy<2.0") == "numpy"

    def test_extract_with_not_equals(self) -> None:
        """Test extraction with != version specifier."""
        from yolo_developer.cli.commands.init import _extract_package_name

        assert _extract_package_name("package!=1.0") == "package"

    def test_extract_with_extras(self) -> None:
        """Test extraction with extras like [security]."""
        from yolo_developer.cli.commands.init import _extract_package_name

        assert _extract_package_name("requests[security]>=2.0") == "requests"

    def test_extract_with_hyphen_in_name(self) -> None:
        """Test extraction with hyphenated package name."""
        from yolo_developer.cli.commands.init import _extract_package_name

        assert _extract_package_name("langchain-core>=0.1") == "langchain-core"

    def test_extract_with_underscore_in_name(self) -> None:
        """Test extraction with underscore package name."""
        from yolo_developer.cli.commands.init import _extract_package_name

        assert _extract_package_name("pydantic_settings") == "pydantic_settings"


class TestBrownfieldEdgeCases:
    """Tests for brownfield mode edge cases."""

    def test_existing_skips_existing_yolo_yaml(self) -> None:
        """Test --existing skips creation when yolo.yaml already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing pyproject.toml and yolo.yaml
            (Path(tmpdir) / "pyproject.toml").write_text(
                '[project]\nname = "existing"\nversion = "1.0.0"\n'
            )
            original_content = "# My existing yolo config\nproject_name: my-project\n"
            (Path(tmpdir) / "yolo.yaml").write_text(original_content)

            with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                result = runner.invoke(app, ["init", tmpdir, "--existing"])

            assert result.exit_code == 0
            # Original yolo.yaml should be preserved
            content = (Path(tmpdir) / "yolo.yaml").read_text()
            assert "# My existing yolo config" in content
            assert "already exists" in result.output

    def test_existing_handles_various_version_specifiers(self) -> None:
        """Test --existing handles existing deps with various version specifiers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create pyproject.toml with various version specifier styles
            existing_content = """[project]
name = "existing"
version = "1.0.0"
dependencies = [
    "pydantic==2.5.0",
    "typer~=0.9",
    "rich<14.0",
    "requests[security]>=2.0",
]
"""
            (Path(tmpdir) / "pyproject.toml").write_text(existing_content)

            with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                result = runner.invoke(app, ["init", tmpdir, "--existing"])

            assert result.exit_code == 0
            content = (Path(tmpdir) / "pyproject.toml").read_text()
            # Should not duplicate pydantic (pydantic-settings is a different package)
            # Count the specific dep string, not just the substring
            assert content.count('"pydantic==2.5.0"') == 1  # Original preserved
            assert "pydantic>=2.0.0" not in content  # YOLO's pydantic not added
            assert content.count('"typer') == 1  # Should not duplicate typer
            assert content.count('"rich') == 1  # Should not duplicate rich
            # Should add new deps like langgraph
            assert "langgraph" in content
            # pydantic-settings is a different package and should be added
            assert "pydantic-settings" in content

    def test_existing_handles_no_dependencies_section(self) -> None:
        """Test --existing adds dependencies section when none exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal pyproject.toml without dependencies
            existing_content = """[project]
name = "minimal"
version = "1.0.0"

[build-system]
requires = ["setuptools"]
"""
            (Path(tmpdir) / "pyproject.toml").write_text(existing_content)

            with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                result = runner.invoke(app, ["init", tmpdir, "--existing"])

            assert result.exit_code == 0
            content = (Path(tmpdir) / "pyproject.toml").read_text()
            # Should have added dependencies
            assert "dependencies = [" in content
            assert "langgraph" in content

    def test_existing_handles_compact_formatting(self) -> None:
        """Test --existing handles compact pyproject.toml formatting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create pyproject.toml with compact formatting (no spaces)
            existing_content = """[project]
name="compact"
version="1.0.0"
dependencies=["requests"]
"""
            (Path(tmpdir) / "pyproject.toml").write_text(existing_content)

            with patch("yolo_developer.cli.commands.init.run_uv_sync", return_value=True):
                result = runner.invoke(app, ["init", tmpdir, "--existing"])

            assert result.exit_code == 0
            content = (Path(tmpdir) / "pyproject.toml").read_text()
            # Should have added YOLO deps
            assert "langgraph" in content
