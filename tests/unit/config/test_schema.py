"""Unit tests for configuration schema (Story 1.4)."""

from __future__ import annotations

from typing import get_type_hints

import pytest
from pydantic import ValidationError


class TestLLMConfig:
    """Tests for LLMConfig nested model (Task 1)."""

    def test_llm_config_exists(self) -> None:
        """Verify LLMConfig class exists and can be imported."""
        from yolo_developer.config.schema import LLMConfig

        assert LLMConfig is not None

    def test_llm_config_has_cheap_model_field(self) -> None:
        """Verify LLMConfig has cheap_model field with correct default."""
        from yolo_developer.config.schema import LLMConfig

        config = LLMConfig()
        assert hasattr(config, "cheap_model")
        assert config.cheap_model == "gpt-4o-mini"

    def test_llm_config_has_premium_model_field(self) -> None:
        """Verify LLMConfig has premium_model field with correct default."""
        from yolo_developer.config.schema import LLMConfig

        config = LLMConfig()
        assert hasattr(config, "premium_model")
        assert config.premium_model == "claude-sonnet-4-20250514"

    def test_llm_config_has_best_model_field(self) -> None:
        """Verify LLMConfig has best_model field with correct default."""
        from yolo_developer.config.schema import LLMConfig

        config = LLMConfig()
        assert hasattr(config, "best_model")
        assert config.best_model == "claude-opus-4-5-20251101"

    def test_llm_config_fields_have_type_hints(self) -> None:
        """Verify all LLMConfig fields have type hints."""
        from yolo_developer.config.schema import LLMConfig

        hints = get_type_hints(LLMConfig)
        assert "cheap_model" in hints
        assert "premium_model" in hints
        assert "best_model" in hints
        assert hints["cheap_model"] is str
        assert hints["premium_model"] is str
        assert hints["best_model"] is str

    def test_llm_config_fields_have_descriptions(self) -> None:
        """Verify LLMConfig fields have Field descriptions."""
        from yolo_developer.config.schema import LLMConfig

        # Check model_fields for descriptions
        assert LLMConfig.model_fields["cheap_model"].description is not None
        assert LLMConfig.model_fields["premium_model"].description is not None
        assert LLMConfig.model_fields["best_model"].description is not None


class TestQualityConfig:
    """Tests for QualityConfig nested model (Task 2)."""

    def test_quality_config_exists(self) -> None:
        """Verify QualityConfig class exists."""
        from yolo_developer.config.schema import QualityConfig

        assert QualityConfig is not None

    def test_quality_config_has_test_coverage_threshold(self) -> None:
        """Verify QualityConfig has test_coverage_threshold with 0.80 default."""
        from yolo_developer.config.schema import QualityConfig

        config = QualityConfig()
        assert hasattr(config, "test_coverage_threshold")
        assert config.test_coverage_threshold == 0.80

    def test_quality_config_has_confidence_threshold(self) -> None:
        """Verify QualityConfig has confidence_threshold with 0.90 default."""
        from yolo_developer.config.schema import QualityConfig

        config = QualityConfig()
        assert hasattr(config, "confidence_threshold")
        assert config.confidence_threshold == 0.90

    def test_quality_config_rejects_threshold_above_one(self) -> None:
        """Verify threshold values above 1.0 are rejected."""
        from yolo_developer.config.schema import QualityConfig

        with pytest.raises(ValidationError) as exc_info:
            QualityConfig(test_coverage_threshold=1.5)
        assert "test_coverage_threshold" in str(exc_info.value)

    def test_quality_config_rejects_negative_threshold(self) -> None:
        """Verify negative threshold values are rejected."""
        from yolo_developer.config.schema import QualityConfig

        with pytest.raises(ValidationError) as exc_info:
            QualityConfig(confidence_threshold=-0.1)
        assert "confidence_threshold" in str(exc_info.value)

    def test_quality_config_accepts_valid_thresholds(self) -> None:
        """Verify valid threshold values are accepted."""
        from yolo_developer.config.schema import QualityConfig

        config = QualityConfig(test_coverage_threshold=0.5, confidence_threshold=0.95)
        assert config.test_coverage_threshold == 0.5
        assert config.confidence_threshold == 0.95

    def test_quality_config_fields_have_type_hints(self) -> None:
        """Verify all QualityConfig fields have type hints."""
        from yolo_developer.config.schema import QualityConfig

        hints = get_type_hints(QualityConfig)
        assert "test_coverage_threshold" in hints
        assert "confidence_threshold" in hints
        assert hints["test_coverage_threshold"] is float
        assert hints["confidence_threshold"] is float


class TestMemoryConfig:
    """Tests for MemoryConfig nested model (Task 3)."""

    def test_memory_config_exists(self) -> None:
        """Verify MemoryConfig class exists."""
        from yolo_developer.config.schema import MemoryConfig

        assert MemoryConfig is not None

    def test_memory_config_has_persist_path(self) -> None:
        """Verify MemoryConfig has persist_path with correct default."""
        from yolo_developer.config.schema import MemoryConfig

        config = MemoryConfig()
        assert hasattr(config, "persist_path")
        assert config.persist_path == ".yolo/memory"

    def test_memory_config_has_vector_store_type(self) -> None:
        """Verify MemoryConfig has vector_store_type."""
        from yolo_developer.config.schema import MemoryConfig

        config = MemoryConfig()
        assert hasattr(config, "vector_store_type")
        assert config.vector_store_type == "chromadb"

    def test_memory_config_has_graph_store_type(self) -> None:
        """Verify MemoryConfig has graph_store_type with json default."""
        from yolo_developer.config.schema import MemoryConfig

        config = MemoryConfig()
        assert hasattr(config, "graph_store_type")
        assert config.graph_store_type == "json"

    def test_memory_config_accepts_neo4j_graph_store(self) -> None:
        """Verify MemoryConfig accepts neo4j as graph_store_type."""
        from yolo_developer.config.schema import MemoryConfig

        config = MemoryConfig(graph_store_type="neo4j")
        assert config.graph_store_type == "neo4j"

    def test_memory_config_fields_have_type_hints(self) -> None:
        """Verify all MemoryConfig fields have type hints."""
        from yolo_developer.config.schema import MemoryConfig

        hints = get_type_hints(MemoryConfig)
        assert "persist_path" in hints
        assert "vector_store_type" in hints
        assert "graph_store_type" in hints
        assert hints["persist_path"] is str


class TestYoloConfig:
    """Tests for main YoloConfig class (Task 4)."""

    def test_yolo_config_exists(self) -> None:
        """Verify YoloConfig class exists."""
        from yolo_developer.config.schema import YoloConfig

        assert YoloConfig is not None

    def test_yolo_config_inherits_from_base_settings(self) -> None:
        """Verify YoloConfig inherits from BaseSettings."""
        from pydantic_settings import BaseSettings

        from yolo_developer.config.schema import YoloConfig

        assert issubclass(YoloConfig, BaseSettings)

    def test_yolo_config_has_env_prefix(self) -> None:
        """Verify YoloConfig uses YOLO_ env prefix."""
        from yolo_developer.config.schema import YoloConfig

        assert YoloConfig.model_config.get("env_prefix") == "YOLO_"

    def test_yolo_config_has_env_nested_delimiter(self) -> None:
        """Verify YoloConfig uses __ as nested delimiter."""
        from yolo_developer.config.schema import YoloConfig

        assert YoloConfig.model_config.get("env_nested_delimiter") == "__"

    def test_yolo_config_instantiation_with_project_name(self) -> None:
        """Verify YoloConfig can be instantiated with project_name."""
        from yolo_developer.config.schema import YoloConfig

        config = YoloConfig(project_name="test-project")
        assert config.project_name == "test-project"

    def test_yolo_config_has_nested_llm_config(self) -> None:
        """Verify YoloConfig has nested llm configuration."""
        from yolo_developer.config.schema import YoloConfig

        config = YoloConfig(project_name="test")
        assert hasattr(config, "llm")
        assert config.llm.cheap_model == "gpt-4o-mini"

    def test_yolo_config_has_nested_quality_config(self) -> None:
        """Verify YoloConfig has nested quality configuration."""
        from yolo_developer.config.schema import YoloConfig

        config = YoloConfig(project_name="test")
        assert hasattr(config, "quality")
        assert config.quality.test_coverage_threshold == 0.80

    def test_yolo_config_has_nested_memory_config(self) -> None:
        """Verify YoloConfig has nested memory configuration."""
        from yolo_developer.config.schema import YoloConfig

        config = YoloConfig(project_name="test")
        assert hasattr(config, "memory")
        assert config.memory.persist_path == ".yolo/memory"

    def test_yolo_config_requires_project_name(self) -> None:
        """Verify YoloConfig requires project_name (no default)."""
        from yolo_developer.config.schema import YoloConfig

        with pytest.raises(ValidationError) as exc_info:
            YoloConfig()  # type: ignore[call-arg]
        assert "project_name" in str(exc_info.value)


class TestYoloConfigEnvironmentVariables:
    """Tests for environment variable overrides (Task 4)."""

    def test_env_override_project_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify YOLO_PROJECT_NAME env var overrides project_name."""
        from yolo_developer.config.schema import YoloConfig

        monkeypatch.setenv("YOLO_PROJECT_NAME", "env-project")
        config = YoloConfig()
        assert config.project_name == "env-project"

    def test_env_override_nested_llm_cheap_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify YOLO_LLM__CHEAP_MODEL env var overrides llm.cheap_model."""
        from yolo_developer.config.schema import YoloConfig

        monkeypatch.setenv("YOLO_PROJECT_NAME", "test")
        monkeypatch.setenv("YOLO_LLM__CHEAP_MODEL", "custom-model")
        config = YoloConfig()
        assert config.llm.cheap_model == "custom-model"

    def test_env_override_nested_quality_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify YOLO_QUALITY__TEST_COVERAGE_THRESHOLD env var works."""
        from yolo_developer.config.schema import YoloConfig

        monkeypatch.setenv("YOLO_PROJECT_NAME", "test")
        monkeypatch.setenv("YOLO_QUALITY__TEST_COVERAGE_THRESHOLD", "0.95")
        config = YoloConfig()
        assert config.quality.test_coverage_threshold == 0.95

    def test_env_override_nested_llm_premium_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify YOLO_LLM__PREMIUM_MODEL env var overrides llm.premium_model."""
        from yolo_developer.config.schema import YoloConfig

        monkeypatch.setenv("YOLO_PROJECT_NAME", "test")
        monkeypatch.setenv("YOLO_LLM__PREMIUM_MODEL", "custom-premium")
        config = YoloConfig()
        assert config.llm.premium_model == "custom-premium"

    def test_env_override_nested_llm_best_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify YOLO_LLM__BEST_MODEL env var overrides llm.best_model."""
        from yolo_developer.config.schema import YoloConfig

        monkeypatch.setenv("YOLO_PROJECT_NAME", "test")
        monkeypatch.setenv("YOLO_LLM__BEST_MODEL", "custom-best")
        config = YoloConfig()
        assert config.llm.best_model == "custom-best"

    def test_env_override_nested_memory_persist_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify YOLO_MEMORY__PERSIST_PATH env var overrides memory.persist_path."""
        from yolo_developer.config.schema import YoloConfig

        monkeypatch.setenv("YOLO_PROJECT_NAME", "test")
        monkeypatch.setenv("YOLO_MEMORY__PERSIST_PATH", "/custom/path")
        config = YoloConfig()
        assert config.memory.persist_path == "/custom/path"

    def test_env_override_nested_quality_confidence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify YOLO_QUALITY__CONFIDENCE_THRESHOLD env var works."""
        from yolo_developer.config.schema import YoloConfig

        monkeypatch.setenv("YOLO_PROJECT_NAME", "test")
        monkeypatch.setenv("YOLO_QUALITY__CONFIDENCE_THRESHOLD", "0.85")
        config = YoloConfig()
        assert config.quality.confidence_threshold == 0.85


class TestValidationErrorMessages:
    """Tests for clear validation error messages (Task 5)."""

    def test_error_includes_field_path(self) -> None:
        """Verify ValidationError includes the field path."""
        from yolo_developer.config.schema import QualityConfig

        with pytest.raises(ValidationError) as exc_info:
            QualityConfig(test_coverage_threshold=1.5)
        errors = exc_info.value.errors()
        assert len(errors) >= 1
        # Check that location is present
        assert "test_coverage_threshold" in str(errors[0].get("loc", ()))

    def test_error_message_is_descriptive(self) -> None:
        """Verify error message explains the constraint."""
        from yolo_developer.config.schema import QualityConfig

        with pytest.raises(ValidationError) as exc_info:
            QualityConfig(test_coverage_threshold=2.0)
        error_str = str(exc_info.value)
        # Should mention the constraint or value issue
        assert "test_coverage_threshold" in error_str


class TestConfigModuleExports:
    """Tests for config module public API exports."""

    def test_yoloconfig_importable_from_config_module(self) -> None:
        """Verify YoloConfig can be imported from yolo_developer.config."""
        from yolo_developer.config import YoloConfig

        assert YoloConfig is not None

    def test_llmconfig_importable_from_config_module(self) -> None:
        """Verify LLMConfig can be imported from yolo_developer.config."""
        from yolo_developer.config import LLMConfig

        assert LLMConfig is not None

    def test_qualityconfig_importable_from_config_module(self) -> None:
        """Verify QualityConfig can be imported from yolo_developer.config."""
        from yolo_developer.config import QualityConfig

        assert QualityConfig is not None

    def test_memoryconfig_importable_from_config_module(self) -> None:
        """Verify MemoryConfig can be imported from yolo_developer.config."""
        from yolo_developer.config import MemoryConfig

        assert MemoryConfig is not None


class TestLLMConfigAPIKeys:
    """Tests for LLMConfig API key fields (Story 1.6 - Task 1)."""

    def test_llm_config_has_openai_api_key_field(self) -> None:
        """Verify LLMConfig has openai_api_key field."""
        from yolo_developer.config.schema import LLMConfig

        config = LLMConfig()
        assert hasattr(config, "openai_api_key")
        assert config.openai_api_key is None  # Default is None

    def test_llm_config_has_anthropic_api_key_field(self) -> None:
        """Verify LLMConfig has anthropic_api_key field."""
        from yolo_developer.config.schema import LLMConfig

        config = LLMConfig()
        assert hasattr(config, "anthropic_api_key")
        assert config.anthropic_api_key is None  # Default is None

    def test_openai_api_key_is_secret_str_type(self) -> None:
        """Verify openai_api_key uses SecretStr for masking."""
        from pydantic import SecretStr

        from yolo_developer.config.schema import LLMConfig

        config = LLMConfig(openai_api_key="sk-test-key-12345")
        assert config.openai_api_key is not None
        assert isinstance(config.openai_api_key, SecretStr)

    def test_anthropic_api_key_is_secret_str_type(self) -> None:
        """Verify anthropic_api_key uses SecretStr for masking."""
        from pydantic import SecretStr

        from yolo_developer.config.schema import LLMConfig

        config = LLMConfig(anthropic_api_key="sk-ant-test-key")
        assert config.anthropic_api_key is not None
        assert isinstance(config.anthropic_api_key, SecretStr)

    def test_openai_api_key_get_secret_value(self) -> None:
        """Verify openai_api_key value is accessible via get_secret_value()."""
        from yolo_developer.config.schema import LLMConfig

        config = LLMConfig(openai_api_key="sk-test-key-12345")
        assert config.openai_api_key is not None
        assert config.openai_api_key.get_secret_value() == "sk-test-key-12345"

    def test_anthropic_api_key_get_secret_value(self) -> None:
        """Verify anthropic_api_key value is accessible via get_secret_value()."""
        from yolo_developer.config.schema import LLMConfig

        config = LLMConfig(anthropic_api_key="sk-ant-test-key")
        assert config.anthropic_api_key is not None
        assert config.anthropic_api_key.get_secret_value() == "sk-ant-test-key"

    def test_openai_api_key_masked_in_repr(self) -> None:
        """Verify openai_api_key is masked in repr output (AC4)."""
        from yolo_developer.config.schema import LLMConfig

        config = LLMConfig(openai_api_key="sk-secret-key")
        repr_output = repr(config)
        assert "sk-secret-key" not in repr_output
        assert "**********" in repr_output or "SecretStr" in repr_output

    def test_anthropic_api_key_masked_in_repr(self) -> None:
        """Verify anthropic_api_key is masked in repr output (AC4)."""
        from yolo_developer.config.schema import LLMConfig

        config = LLMConfig(anthropic_api_key="sk-ant-secret")
        repr_output = repr(config)
        assert "sk-ant-secret" not in repr_output
        assert "**********" in repr_output or "SecretStr" in repr_output

    def test_api_key_fields_have_descriptions(self) -> None:
        """Verify API key fields have Field descriptions."""
        from yolo_developer.config.schema import LLMConfig

        assert LLMConfig.model_fields["openai_api_key"].description is not None
        assert LLMConfig.model_fields["anthropic_api_key"].description is not None
        assert "env" in LLMConfig.model_fields["openai_api_key"].description.lower()
        assert "env" in LLMConfig.model_fields["anthropic_api_key"].description.lower()


class TestYoloConfigAPIKeyValidation:
    """Tests for API key validation warnings (Story 1.6 - AC5)."""

    def test_validate_api_keys_method_exists(self) -> None:
        """Verify YoloConfig has validate_api_keys method."""
        from yolo_developer.config.schema import YoloConfig

        config = YoloConfig(project_name="test")
        assert hasattr(config, "validate_api_keys")
        assert callable(config.validate_api_keys)

    def test_validate_api_keys_returns_warnings_when_no_keys(self) -> None:
        """Verify validate_api_keys returns warnings when no API keys are set (AC5)."""
        from yolo_developer.config.schema import YoloConfig

        config = YoloConfig(project_name="test")
        warnings = config.validate_api_keys()

        assert isinstance(warnings, list)
        assert len(warnings) > 0
        assert any("api key" in w.lower() for w in warnings)

    def test_validate_api_keys_returns_empty_when_openai_set(self) -> None:
        """Verify no warnings when OpenAI API key is set."""
        from yolo_developer.config.schema import LLMConfig, YoloConfig

        config = YoloConfig(
            project_name="test",
            llm=LLMConfig(openai_api_key="sk-test-key"),
        )
        warnings = config.validate_api_keys()

        assert len(warnings) == 0

    def test_validate_api_keys_returns_empty_when_anthropic_set(self) -> None:
        """Verify no warnings when Anthropic API key is set."""
        from yolo_developer.config.schema import LLMConfig, YoloConfig

        config = YoloConfig(
            project_name="test",
            llm=LLMConfig(anthropic_api_key="sk-ant-test"),
        )
        warnings = config.validate_api_keys()

        assert len(warnings) == 0

    def test_validate_api_keys_no_error_on_missing_keys(self) -> None:
        """Verify missing API keys do NOT raise an error (AC5)."""
        from yolo_developer.config.schema import YoloConfig

        # Should not raise - just returns warnings
        config = YoloConfig(project_name="test")
        warnings = config.validate_api_keys()

        # It's a warning, not an error
        assert isinstance(warnings, list)


class TestSchemaCodeQuality:
    """Tests for code quality requirements."""

    def test_schema_passes_mypy(self) -> None:
        """Verify schema.py passes mypy type checking."""
        import subprocess

        result = subprocess.run(
            ["uv", "run", "mypy", "src/yolo_developer/config/schema.py"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Mypy failed: {result.stdout}\n{result.stderr}"

    def test_schema_has_future_annotations(self) -> None:
        """Verify schema.py uses from __future__ import annotations."""
        from pathlib import Path

        schema_file = Path("src/yolo_developer/config/schema.py")
        content = schema_file.read_text()
        assert "from __future__ import annotations" in content
