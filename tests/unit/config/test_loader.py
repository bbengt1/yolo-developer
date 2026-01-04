"""Unit tests for configuration loading (Story 1.5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_developer.config import ConfigurationError, load_config


class TestLoadConfigWithValidYAML:
    """Tests for load_config with valid YAML files (AC1, AC2, AC3)."""

    def test_load_valid_yaml_with_project_name(self, tmp_path: Path) -> None:
        """Load configuration from valid YAML file with project_name."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            """
project_name: test-project
"""
        )
        config = load_config(yaml_file)
        assert config.project_name == "test-project"

    def test_load_valid_yaml_with_nested_llm_config(self, tmp_path: Path) -> None:
        """Load configuration with nested LLM configuration (AC2)."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            """
project_name: test-project
llm:
  cheap_model: custom-model
  premium_model: custom-premium
  best_model: custom-best
"""
        )
        config = load_config(yaml_file)
        assert config.project_name == "test-project"
        assert config.llm.cheap_model == "custom-model"
        assert config.llm.premium_model == "custom-premium"
        assert config.llm.best_model == "custom-best"

    def test_load_valid_yaml_with_nested_quality_config(self, tmp_path: Path) -> None:
        """Load configuration with nested quality configuration (AC2)."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            """
project_name: test-project
quality:
  test_coverage_threshold: 0.85
  confidence_threshold: 0.92
"""
        )
        config = load_config(yaml_file)
        assert config.quality.test_coverage_threshold == 0.85
        assert config.quality.confidence_threshold == 0.92

    def test_load_valid_yaml_with_nested_memory_config(self, tmp_path: Path) -> None:
        """Load configuration with nested memory configuration (AC2)."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            """
project_name: test-project
memory:
  persist_path: /custom/path
  vector_store_type: chromadb
  graph_store_type: neo4j
"""
        )
        config = load_config(yaml_file)
        assert config.memory.persist_path == "/custom/path"
        assert config.memory.vector_store_type == "chromadb"
        assert config.memory.graph_store_type == "neo4j"

    def test_load_partial_yaml_uses_defaults(self, tmp_path: Path) -> None:
        """Partial YAML file uses defaults for missing values (AC3)."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            """
project_name: partial-project
llm:
  cheap_model: custom-cheap
"""
        )
        config = load_config(yaml_file)
        assert config.project_name == "partial-project"
        assert config.llm.cheap_model == "custom-cheap"
        # Defaults for non-specified values
        assert config.llm.premium_model == "claude-sonnet-4-20250514"
        assert config.llm.best_model == "claude-opus-4-5-20251101"
        assert config.quality.test_coverage_threshold == 0.80
        assert config.quality.confidence_threshold == 0.90
        assert config.memory.persist_path == ".yolo/memory"

    def test_yaml_overrides_defaults(self, tmp_path: Path) -> None:
        """YAML values override default values (AC1)."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            """
project_name: yaml-project
quality:
  test_coverage_threshold: 0.95
"""
        )
        config = load_config(yaml_file)
        assert config.quality.test_coverage_threshold == 0.95
        # Default for confidence is still used
        assert config.quality.confidence_threshold == 0.90


class TestLoadConfigMissingFile:
    """Tests for load_config with missing YAML file (AC5)."""

    def test_empty_yaml_file_uses_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty YAML file uses defaults with environment variable."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text("")  # Empty file
        monkeypatch.setenv("YOLO_PROJECT_NAME", "env-project")
        config = load_config(yaml_file)
        assert config.project_name == "env-project"
        # All other defaults apply
        assert config.llm.cheap_model == "gpt-4o-mini"
        assert config.quality.test_coverage_threshold == 0.80

    def test_yaml_file_with_only_comments_uses_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """YAML file with only comments uses defaults."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text("# This is a comment\n# Another comment\n")
        monkeypatch.setenv("YOLO_PROJECT_NAME", "env-project")
        config = load_config(yaml_file)
        assert config.project_name == "env-project"
        assert config.llm.cheap_model == "gpt-4o-mini"

    def test_missing_file_uses_defaults_with_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing YAML file uses defaults with environment variable (AC5)."""
        monkeypatch.setenv("YOLO_PROJECT_NAME", "env-project")
        config = load_config(tmp_path / "nonexistent.yaml")
        assert config.project_name == "env-project"
        # All other defaults apply
        assert config.llm.cheap_model == "gpt-4o-mini"
        assert config.quality.test_coverage_threshold == 0.80

    def test_missing_file_requires_project_name(self, tmp_path: Path) -> None:
        """Missing file without project_name env var raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            load_config(tmp_path / "nonexistent.yaml")
        assert "project_name" in str(exc_info.value)

    def test_default_path_is_yolo_yaml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default path is yolo.yaml in current directory."""
        monkeypatch.setenv("YOLO_PROJECT_NAME", "default-project")
        # This will look for yolo.yaml in cwd (likely doesn't exist)
        config = load_config()
        assert config.project_name == "default-project"


class TestLoadConfigSyntaxErrors:
    """Tests for load_config with YAML syntax errors (AC4)."""

    def test_yaml_syntax_error_includes_line_number(self, tmp_path: Path) -> None:
        """YAML syntax errors include line numbers (AC4)."""
        yaml_file = tmp_path / "yolo.yaml"
        # Invalid YAML - tab indentation mixed with spaces
        yaml_file.write_text(
            """project_name: test
llm:
\tcheap_model: bad-indent
"""
        )
        with pytest.raises(ConfigurationError) as exc_info:
            load_config(yaml_file)
        error_message = str(exc_info.value).lower()
        assert "line" in error_message

    def test_yaml_syntax_error_includes_column(self, tmp_path: Path) -> None:
        """YAML syntax errors include column information (AC4)."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            """project_name: test
llm: {invalid
"""
        )
        with pytest.raises(ConfigurationError) as exc_info:
            load_config(yaml_file)
        error_message = str(exc_info.value).lower()
        assert "column" in error_message

    def test_yaml_syntax_error_includes_hint(self, tmp_path: Path) -> None:
        """YAML syntax errors include helpful hints (AC4)."""
        yaml_file = tmp_path / "yolo.yaml"
        # Invalid YAML with unclosed bracket
        yaml_file.write_text(
            """project_name: test
llm: [invalid
  unclosed bracket
"""
        )
        with pytest.raises(ConfigurationError) as exc_info:
            load_config(yaml_file)
        error_message = str(exc_info.value)
        # Should have line number
        assert "line" in error_message.lower()
        # Should have helpful hint text (AC4 requirement)
        assert "hint" in error_message.lower() or "check" in error_message.lower()

    def test_yaml_syntax_error_includes_file_path(self, tmp_path: Path) -> None:
        """YAML syntax errors include the file path."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            """project_name: [invalid
"""
        )
        with pytest.raises(ConfigurationError) as exc_info:
            load_config(yaml_file)
        assert str(yaml_file) in str(exc_info.value)


class TestLoadConfigValidationErrors:
    """Tests for load_config with invalid configuration values."""

    def test_invalid_threshold_above_one(self, tmp_path: Path) -> None:
        """Threshold values above 1.0 raise ConfigurationError."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            """
project_name: test-project
quality:
  test_coverage_threshold: 1.5
"""
        )
        with pytest.raises(ConfigurationError) as exc_info:
            load_config(yaml_file)
        assert "test_coverage_threshold" in str(exc_info.value)

    def test_invalid_threshold_negative(self, tmp_path: Path) -> None:
        """Negative threshold values raise ConfigurationError."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            """
project_name: test-project
quality:
  confidence_threshold: -0.5
"""
        )
        with pytest.raises(ConfigurationError) as exc_info:
            load_config(yaml_file)
        assert "confidence_threshold" in str(exc_info.value)

    def test_invalid_value_error_is_descriptive(self, tmp_path: Path) -> None:
        """Validation error messages are descriptive."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            """
project_name: test-project
quality:
  test_coverage_threshold: 2.0
"""
        )
        with pytest.raises(ConfigurationError) as exc_info:
            load_config(yaml_file)
        error_str = str(exc_info.value)
        # Should mention the field
        assert "test_coverage_threshold" in error_str
        # Should be an invalid configuration error
        assert "Invalid" in error_str or "invalid" in error_str

    def test_missing_project_name_error(self, tmp_path: Path) -> None:
        """Missing project_name raises ConfigurationError."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            """
llm:
  cheap_model: custom-model
"""
        )
        with pytest.raises(ConfigurationError) as exc_info:
            load_config(yaml_file)
        assert "project_name" in str(exc_info.value)


class TestLoadConfigEnvironmentOverrides:
    """Tests for environment variable overrides (AC1)."""

    def test_env_overrides_yaml_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Environment variables override YAML values (AC1)."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            """
project_name: yaml-project
llm:
  cheap_model: yaml-model
"""
        )
        monkeypatch.setenv("YOLO_LLM__CHEAP_MODEL", "env-model")
        config = load_config(yaml_file)
        assert config.project_name == "yaml-project"  # From YAML
        assert config.llm.cheap_model == "env-model"  # Env overrides YAML

    def test_env_overrides_yaml_project_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Environment variables override YAML project_name."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            """
project_name: yaml-project
"""
        )
        monkeypatch.setenv("YOLO_PROJECT_NAME", "env-project")
        config = load_config(yaml_file)
        assert config.project_name == "env-project"

    def test_env_overrides_yaml_nested_quality(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Environment variables override YAML nested quality config."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            """
project_name: test-project
quality:
  test_coverage_threshold: 0.80
"""
        )
        monkeypatch.setenv("YOLO_QUALITY__TEST_COVERAGE_THRESHOLD", "0.95")
        config = load_config(yaml_file)
        assert config.quality.test_coverage_threshold == 0.95

    def test_priority_order_defaults_yaml_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Configuration priority: defaults → YAML → env (AC1)."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            """
project_name: yaml-project
llm:
  cheap_model: yaml-cheap
quality:
  test_coverage_threshold: 0.85
"""
        )
        # Override one YAML value with env
        monkeypatch.setenv("YOLO_QUALITY__TEST_COVERAGE_THRESHOLD", "0.99")

        config = load_config(yaml_file)

        # From YAML (overrides default)
        assert config.project_name == "yaml-project"
        assert config.llm.cheap_model == "yaml-cheap"
        assert config.quality.test_coverage_threshold == 0.99  # Env wins

        # Default values (not in YAML or env)
        assert config.llm.premium_model == "claude-sonnet-4-20250514"
        assert config.quality.confidence_threshold == 0.90


class TestLoadConfigModuleExports:
    """Tests for module exports."""

    def test_load_config_importable_from_config_module(self) -> None:
        """load_config can be imported from yolo_developer.config."""
        from yolo_developer.config import load_config as lc

        assert lc is not None
        assert callable(lc)

    def test_configuration_error_importable_from_config_module(self) -> None:
        """ConfigurationError can be imported from yolo_developer.config."""
        from yolo_developer.config import ConfigurationError as ConfError

        assert ConfError is not None
        assert issubclass(ConfError, Exception)

    def test_configuration_error_is_exception(self) -> None:
        """ConfigurationError is a subclass of Exception."""
        assert issubclass(ConfigurationError, Exception)


class TestLoadConfigFullExample:
    """Tests for complete YAML configuration example."""

    def test_full_yaml_example(self, tmp_path: Path) -> None:
        """Load complete YAML configuration matching Dev Notes example."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            """# yolo.yaml - YOLO Developer Configuration
project_name: my-awesome-project

llm:
  cheap_model: gpt-4o-mini
  premium_model: claude-sonnet-4-20250514
  best_model: claude-opus-4-5-20251101

quality:
  test_coverage_threshold: 0.85
  confidence_threshold: 0.92

memory:
  persist_path: .yolo/memory
  vector_store_type: chromadb
  graph_store_type: json
"""
        )
        config = load_config(yaml_file)

        assert config.project_name == "my-awesome-project"
        assert config.llm.cheap_model == "gpt-4o-mini"
        assert config.llm.premium_model == "claude-sonnet-4-20250514"
        assert config.llm.best_model == "claude-opus-4-5-20251101"
        assert config.quality.test_coverage_threshold == 0.85
        assert config.quality.confidence_threshold == 0.92
        assert config.memory.persist_path == ".yolo/memory"
        assert config.memory.vector_store_type == "chromadb"
        assert config.memory.graph_store_type == "json"


class TestLoaderCodeQuality:
    """Tests for code quality requirements."""

    @pytest.mark.slow
    def test_loader_passes_mypy(self) -> None:
        """Verify loader.py passes mypy type checking.

        This test is marked slow as it shells out to run mypy.
        Skip with: pytest -m "not slow"
        """
        import subprocess

        result = subprocess.run(
            ["uv", "run", "mypy", "src/yolo_developer/config/loader.py"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Mypy failed: {result.stdout}\n{result.stderr}"

    def test_loader_has_future_annotations(self) -> None:
        """Verify loader.py uses from __future__ import annotations."""
        import inspect

        from yolo_developer.config import loader

        # Get the actual file path from the module
        loader_file = Path(inspect.getfile(loader))
        content = loader_file.read_text()
        assert "from __future__ import annotations" in content
