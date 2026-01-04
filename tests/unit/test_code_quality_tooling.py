"""Unit tests for code quality tooling configuration (Story 1.3)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import ClassVar


class TestRuffConfiguration:
    """Tests for ruff configuration in pyproject.toml."""

    EXPECTED_LINT_RULES: ClassVar[list[str]] = [
        "E",  # pycodestyle errors
        "F",  # pyflakes
        "I",  # isort
        "N",  # pep8-naming
        "W",  # pycodestyle warnings
        "UP",  # pyupgrade
        "B",  # flake8-bugbear
        "C4",  # flake8-comprehensions
        "DTZ",  # flake8-datetimez
        "RUF",  # ruff-specific rules
    ]

    def test_ruff_has_all_expected_lint_rules(self) -> None:
        """Verify all expected lint rules are configured."""
        pyproject = Path("pyproject.toml")
        content = pyproject.read_text()
        for rule in self.EXPECTED_LINT_RULES:
            assert f'"{rule}"' in content, f"Missing lint rule: {rule}"

    def test_ruff_section_exists(self) -> None:
        """Verify [tool.ruff] section exists in pyproject.toml."""
        pyproject = Path("pyproject.toml")
        content = pyproject.read_text()
        assert "[tool.ruff]" in content

    def test_ruff_lint_section_exists(self) -> None:
        """Verify [tool.ruff.lint] section exists in pyproject.toml."""
        pyproject = Path("pyproject.toml")
        content = pyproject.read_text()
        assert "[tool.ruff.lint]" in content

    def test_ruff_format_section_exists(self) -> None:
        """Verify [tool.ruff.format] section exists in pyproject.toml."""
        pyproject = Path("pyproject.toml")
        content = pyproject.read_text()
        assert "[tool.ruff.format]" in content

    def test_ruff_line_length_configured(self) -> None:
        """Verify ruff line-length is set to 100."""
        pyproject = Path("pyproject.toml")
        content = pyproject.read_text()
        assert "line-length = 100" in content

    def test_ruff_target_version_configured(self) -> None:
        """Verify ruff target-version is set to py310."""
        pyproject = Path("pyproject.toml")
        content = pyproject.read_text()
        assert 'target-version = "py310"' in content

    def test_ruff_isort_section_exists(self) -> None:
        """Verify [tool.ruff.lint.isort] section exists."""
        pyproject = Path("pyproject.toml")
        content = pyproject.read_text()
        assert "[tool.ruff.lint.isort]" in content

    def test_ruff_check_passes(self) -> None:
        """Verify ruff check passes on the codebase."""
        result = subprocess.run(
            ["uv", "run", "ruff", "check", "src/", "tests/"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Ruff check failed: {result.stdout}\n{result.stderr}"

    def test_ruff_format_no_changes_needed(self) -> None:
        """Verify ruff format reports no changes needed."""
        result = subprocess.run(
            ["uv", "run", "ruff", "format", "src/", "tests/", "--check"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Ruff format needed: {result.stdout}\n{result.stderr}"


class TestMypyConfiguration:
    """Tests for mypy configuration in pyproject.toml."""

    def test_mypy_section_exists(self) -> None:
        """Verify [tool.mypy] section exists in pyproject.toml."""
        pyproject = Path("pyproject.toml")
        content = pyproject.read_text()
        assert "[tool.mypy]" in content

    def test_mypy_strict_mode_enabled(self) -> None:
        """Verify mypy strict mode is enabled."""
        pyproject = Path("pyproject.toml")
        content = pyproject.read_text()
        assert "strict = true" in content

    def test_mypy_python_version_configured(self) -> None:
        """Verify mypy python_version is set to 3.10."""
        pyproject = Path("pyproject.toml")
        content = pyproject.read_text()
        assert 'python_version = "3.10"' in content

    def test_mypy_ignore_missing_imports(self) -> None:
        """Verify mypy ignore_missing_imports is enabled."""
        pyproject = Path("pyproject.toml")
        content = pyproject.read_text()
        assert "ignore_missing_imports = true" in content

    def test_mypy_show_error_codes(self) -> None:
        """Verify mypy show_error_codes is enabled."""
        pyproject = Path("pyproject.toml")
        content = pyproject.read_text()
        assert "show_error_codes = true" in content

    def test_mypy_test_overrides_exist(self) -> None:
        """Verify mypy has overrides for tests."""
        pyproject = Path("pyproject.toml")
        content = pyproject.read_text()
        assert "[[tool.mypy.overrides]]" in content
        assert 'module = "tests.*"' in content

    def test_mypy_passes(self) -> None:
        """Verify mypy passes on the codebase."""
        result = subprocess.run(
            ["uv", "run", "mypy", "src/"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Mypy failed: {result.stdout}\n{result.stderr}"


class TestPrecommitConfiguration:
    """Tests for pre-commit configuration."""

    EXPECTED_HOOKS: ClassVar[list[str]] = [
        "trailing-whitespace",
        "end-of-file-fixer",
        "check-yaml",
        "ruff",
        "ruff-format",
        "mypy",
    ]

    def test_precommit_config_exists(self) -> None:
        """Verify .pre-commit-config.yaml exists."""
        precommit = Path(".pre-commit-config.yaml")
        assert precommit.is_file(), ".pre-commit-config.yaml not found"

    def test_precommit_config_is_valid_yaml(self) -> None:
        """Verify .pre-commit-config.yaml is valid YAML."""
        import yaml

        precommit = Path(".pre-commit-config.yaml")
        content = precommit.read_text()
        parsed = yaml.safe_load(content)
        assert parsed is not None
        assert "repos" in parsed

    def test_precommit_has_required_hooks(self) -> None:
        """Verify pre-commit config has all required hooks."""
        import yaml

        precommit = Path(".pre-commit-config.yaml")
        content = precommit.read_text()
        parsed = yaml.safe_load(content)

        # Collect all hook IDs
        hook_ids: list[str] = []
        for repo in parsed.get("repos", []):
            for hook in repo.get("hooks", []):
                hook_ids.append(hook.get("id", ""))

        for expected_hook in self.EXPECTED_HOOKS:
            assert expected_hook in hook_ids, f"Missing hook: {expected_hook}"

    def test_precommit_has_ruff_repo(self) -> None:
        """Verify pre-commit config has ruff-pre-commit repo."""
        precommit = Path(".pre-commit-config.yaml")
        content = precommit.read_text()
        assert "astral-sh/ruff-pre-commit" in content

    def test_precommit_has_mypy_repo(self) -> None:
        """Verify pre-commit config has mypy repo."""
        precommit = Path(".pre-commit-config.yaml")
        content = precommit.read_text()
        assert "mirrors-mypy" in content

    def test_precommit_validate_config(self) -> None:
        """Verify pre-commit configuration is valid."""
        result = subprocess.run(
            ["uv", "run", "pre-commit", "validate-config"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Pre-commit config invalid: {result.stderr}"


class TestCodeQualityDevDependencies:
    """Tests for dev dependencies in pyproject.toml."""

    REQUIRED_DEV_DEPS: ClassVar[list[str]] = [
        "ruff",
        "mypy",
        "pre-commit",
        "pytest",
    ]

    def test_dev_dependencies_section_exists(self) -> None:
        """Verify [project.optional-dependencies] dev section exists."""
        pyproject = Path("pyproject.toml")
        content = pyproject.read_text()
        assert "[project.optional-dependencies]" in content
        assert "dev = [" in content

    def test_required_dev_dependencies_present(self) -> None:
        """Verify all required dev dependencies are present."""
        pyproject = Path("pyproject.toml")
        content = pyproject.read_text()
        for dep in self.REQUIRED_DEV_DEPS:
            assert f'"{dep}"' in content or f"'{dep}'" in content, f"Missing dev dep: {dep}"
