"""Unit tests for configuration validators (Story 1.7).

Tests comprehensive validation of configuration beyond Pydantic field constraints.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from yolo_developer.config import (
    LLM_CHEAP_MODEL_DEFAULT,
    OPENAI_CHEAP_MODEL_DEFAULT,
    OPENAI_CODE_MODEL_DEFAULT,
    OPENAI_PREMIUM_MODEL_DEFAULT,
    load_config,
)
from yolo_developer.config.loader import ConfigurationError
from yolo_developer.config.schema import LLMConfig, YoloConfig
from yolo_developer.config.validators import (
    ValidationIssue,
    ValidationResult,
    validate_config,
)


class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_validation_issue_has_required_fields(self) -> None:
        """ValidationIssue has field and message attributes."""
        issue = ValidationIssue(field="test.field", message="Test message")
        assert issue.field == "test.field"
        assert issue.message == "Test message"

    def test_validation_issue_has_optional_value(self) -> None:
        """ValidationIssue has optional value attribute."""
        issue = ValidationIssue(
            field="test.field",
            message="Test message",
            value="actual_value",
        )
        assert issue.value == "actual_value"

    def test_validation_issue_has_optional_constraint(self) -> None:
        """ValidationIssue has optional constraint attribute."""
        issue = ValidationIssue(
            field="test.field",
            message="Test message",
            constraint="must be >= 0",
        )
        assert issue.constraint == "must be >= 0"

    def test_validation_issue_value_defaults_to_none(self) -> None:
        """ValidationIssue.value defaults to None."""
        issue = ValidationIssue(field="test", message="msg")
        assert issue.value is None

    def test_validation_issue_constraint_defaults_to_none(self) -> None:
        """ValidationIssue.constraint defaults to None."""
        issue = ValidationIssue(field="test", message="msg")
        assert issue.constraint is None


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_has_empty_errors_list_by_default(self) -> None:
        """ValidationResult has empty errors list by default."""
        result = ValidationResult()
        assert isinstance(result.errors, list)
        assert len(result.errors) == 0

    def test_validation_result_has_empty_warnings_list_by_default(self) -> None:
        """ValidationResult has empty warnings list by default."""
        result = ValidationResult()
        assert isinstance(result.warnings, list)
        assert len(result.warnings) == 0

    def test_validation_result_is_valid_when_no_errors(self) -> None:
        """ValidationResult.is_valid returns True when no errors."""
        result = ValidationResult()
        assert result.is_valid is True

    def test_validation_result_is_valid_with_warnings_only(self) -> None:
        """ValidationResult.is_valid returns True with warnings but no errors."""
        result = ValidationResult()
        result.warnings.append(ValidationIssue(field="test", message="warning"))
        assert result.is_valid is True

    def test_validation_result_is_not_valid_with_errors(self) -> None:
        """ValidationResult.is_valid returns False when errors exist."""
        result = ValidationResult()
        result.errors.append(ValidationIssue(field="test", message="error"))
        assert result.is_valid is False


class TestRequiredFieldValidation:
    """Tests for required field validation (AC1)."""

    def test_missing_project_name_produces_error(self, tmp_path: Path) -> None:
        """Missing project_name raises ConfigurationError."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(f"llm:\n  cheap_model: {LLM_CHEAP_MODEL_DEFAULT}\n")

        with pytest.raises(ConfigurationError) as exc_info:
            load_config(yaml_file)

        assert "project_name" in str(exc_info.value)

    def test_empty_project_name_is_accepted(self, tmp_path: Path) -> None:
        """Empty string project_name is accepted (no min_length constraint).

        Note: AC1 requires project_name "must be present" - empty string satisfies
        this requirement. Add min_length=1 to schema if empty should be rejected.
        """
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text('project_name: ""\n')

        config = load_config(yaml_file)
        assert config.project_name == ""


class TestValueRangeValidation:
    """Tests for value range validation (AC2)."""

    def test_coverage_threshold_above_one_produces_error(self, tmp_path: Path) -> None:
        """Coverage threshold > 1.0 raises ConfigurationError."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text("project_name: test\nquality:\n  test_coverage_threshold: 1.5\n")

        with pytest.raises(ConfigurationError) as exc_info:
            load_config(yaml_file)

        assert "test_coverage_threshold" in str(exc_info.value)
        assert "<= 1.0" in str(exc_info.value)

    def test_confidence_threshold_below_zero_produces_error(self, tmp_path: Path) -> None:
        """Confidence threshold < 0.0 raises ConfigurationError."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text("project_name: test\nquality:\n  confidence_threshold: -0.5\n")

        with pytest.raises(ConfigurationError) as exc_info:
            load_config(yaml_file)

        assert "confidence_threshold" in str(exc_info.value)
        assert ">= 0.0" in str(exc_info.value)

    def test_valid_threshold_values_pass(self, tmp_path: Path) -> None:
        """Valid threshold values pass validation."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            "project_name: test\n"
            "quality:\n"
            "  test_coverage_threshold: 0.80\n"
            "  confidence_threshold: 0.95\n"
        )

        config = load_config(yaml_file)
        assert config.quality.test_coverage_threshold == 0.80
        assert config.quality.confidence_threshold == 0.95

    def test_boundary_values_pass(self, tmp_path: Path) -> None:
        """Boundary values (0.0 and 1.0) pass validation."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            "project_name: test\n"
            "quality:\n"
            "  test_coverage_threshold: 1.0\n"
            "  confidence_threshold: 0.0\n"
        )

        config = load_config(yaml_file)
        assert config.quality.test_coverage_threshold == 1.0
        assert config.quality.confidence_threshold == 0.0


class TestPathValidation:
    """Tests for path validation (AC3)."""

    def test_relative_path_passes_validation(self, tmp_path: Path) -> None:
        """Relative path in writable directory passes validation."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text("project_name: test\nmemory:\n  persist_path: .yolo/memory\n")

        config = load_config(yaml_file)
        assert config.memory.persist_path == ".yolo/memory"

    def test_absolute_path_in_writable_dir_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Absolute path in writable directory passes validation."""
        yaml_file = tmp_path / "yolo.yaml"
        persist_path = tmp_path / "memory"
        yaml_file.write_text(f"project_name: test\nmemory:\n  persist_path: {persist_path}\n")

        # Set API keys to avoid unrelated warnings
        monkeypatch.setenv("YOLO_LLM__OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("YOLO_LLM__ANTHROPIC_API_KEY", "sk-ant-test")

        config = load_config(yaml_file)
        assert config.memory.persist_path == str(persist_path)

    def test_non_writable_path_produces_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Non-writable path produces warning (not error)."""
        yaml_file = tmp_path / "yolo.yaml"
        # Create a non-writable directory for deterministic test
        non_writable_dir = tmp_path / "readonly"
        non_writable_dir.mkdir()
        non_writable_dir.chmod(0o444)  # Read-only

        try:
            persist_path = non_writable_dir / "memory"
            yaml_file.write_text(f"project_name: test\nmemory:\n  persist_path: {persist_path}\n")

            with caplog.at_level(logging.WARNING):
                config = load_config(yaml_file)

            # Should succeed (not raise error) but log warning about path access
            assert config.memory.persist_path == str(persist_path)
            warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
            # Check for either "writable" or "accessible" warning
            path_warnings = [
                msg
                for msg in warning_messages
                if "writable" in msg.lower() or "accessible" in msg.lower()
            ]
            assert len(path_warnings) == 1, f"Expected 1 path warning, got: {warning_messages}"
        finally:
            # Restore permissions for cleanup
            non_writable_dir.chmod(0o755)


class TestAPIKeyValidation:
    """Tests for API key validation (AC4)."""

    def test_openai_model_without_key_produces_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """OpenAI model without API key logs warning."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            "project_name: test\n"
            "llm:\n"
            f"  cheap_model: {OPENAI_CHEAP_MODEL_DEFAULT}\n"
            f"  premium_model: {OPENAI_PREMIUM_MODEL_DEFAULT}\n"
            f"  best_model: {OPENAI_CODE_MODEL_DEFAULT}\n"
        )

        with caplog.at_level(logging.WARNING):
            load_config(yaml_file)

        # Should have logged warning about missing OpenAI key
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("openai" in msg.lower() for msg in warning_messages)
        assert any(
            OPENAI_CHEAP_MODEL_DEFAULT in msg
            or OPENAI_PREMIUM_MODEL_DEFAULT in msg
            or OPENAI_CODE_MODEL_DEFAULT in msg
            for msg in warning_messages
        )

    def test_anthropic_model_without_key_produces_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Anthropic model without API key logs warning."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            "project_name: test\n"
            "llm:\n"
            "  cheap_model: claude-3-haiku-20240307\n"
            "  premium_model: claude-sonnet-4-20250514\n"
            "  best_model: claude-opus-4-5-20251101\n"
        )

        # Set OpenAI key to isolate Anthropic warning
        monkeypatch.setenv("YOLO_LLM__OPENAI_API_KEY", "sk-test")

        with caplog.at_level(logging.WARNING):
            load_config(yaml_file)

        # Should have logged warning about missing Anthropic key
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("anthropic" in msg.lower() for msg in warning_messages)

    def test_no_warning_when_openai_key_set(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No warning when OpenAI key is set for OpenAI models."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            "project_name: test\n"
            "llm:\n"
            f"  cheap_model: {OPENAI_CHEAP_MODEL_DEFAULT}\n"
            f"  premium_model: {OPENAI_PREMIUM_MODEL_DEFAULT}\n"
            f"  best_model: {OPENAI_CODE_MODEL_DEFAULT}\n"
        )
        monkeypatch.setenv("YOLO_LLM__OPENAI_API_KEY", "sk-test-key")

        with caplog.at_level(logging.WARNING):
            load_config(yaml_file)

        # Should NOT have logged OpenAI API key warning
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        openai_warnings = [msg for msg in warning_messages if "openai" in msg.lower()]
        assert len(openai_warnings) == 0

    def test_no_warning_when_anthropic_key_set(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No warning when Anthropic key is set for Anthropic models."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            "project_name: test\n"
            "llm:\n"
            "  cheap_model: claude-3-haiku-20240307\n"
            "  premium_model: claude-sonnet-4-20250514\n"
            "  best_model: claude-opus-4-5-20251101\n"
        )
        monkeypatch.setenv("YOLO_LLM__ANTHROPIC_API_KEY", "sk-ant-test-key")

        with caplog.at_level(logging.WARNING):
            load_config(yaml_file)

        # Should NOT have logged Anthropic API key warning
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        anthropic_warnings = [msg for msg in warning_messages if "anthropic" in msg.lower()]
        assert len(anthropic_warnings) == 0


class TestComprehensiveErrorCollection:
    """Tests for collecting multiple validation errors (AC5)."""

    def test_multiple_errors_collected_together(self, tmp_path: Path) -> None:
        """Multiple Pydantic validation errors are reported together."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            "project_name: test\n"
            "quality:\n"
            "  test_coverage_threshold: 1.5\n"
            "  confidence_threshold: -0.5\n"
        )

        with pytest.raises(ConfigurationError) as exc_info:
            load_config(yaml_file)

        error_msg = str(exc_info.value)
        # Both errors should be in the message
        assert "test_coverage_threshold" in error_msg
        assert "confidence_threshold" in error_msg

    def test_valid_config_passes_all_validation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Valid configuration passes all validation."""
        yaml_file = tmp_path / "yolo.yaml"
        yaml_file.write_text(
            "project_name: test-project\n"
            "llm:\n"
            f"  cheap_model: {LLM_CHEAP_MODEL_DEFAULT}\n"
            "  premium_model: claude-sonnet-4-20250514\n"
            "  best_model: claude-opus-4-5-20251101\n"
            "quality:\n"
            "  test_coverage_threshold: 0.80\n"
            "  confidence_threshold: 0.90\n"
            "memory:\n"
            "  persist_path: .yolo/memory\n"
            "  vector_store_type: chromadb\n"
            "  graph_store_type: json\n"
        )
        monkeypatch.setenv("YOLO_LLM__OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("YOLO_LLM__ANTHROPIC_API_KEY", "sk-ant-test-key")

        config = load_config(yaml_file)

        assert config.project_name == "test-project"
        assert config.llm.cheap_model == LLM_CHEAP_MODEL_DEFAULT
        assert config.quality.test_coverage_threshold == 0.80


class TestValidateConfigFunction:
    """Tests for the validate_config function directly."""

    def test_validate_config_returns_validation_result(self) -> None:
        """validate_config returns ValidationResult."""
        config = YoloConfig(project_name="test")
        result = validate_config(config)
        assert isinstance(result, ValidationResult)

    def test_validate_config_collects_api_key_warnings(self) -> None:
        """validate_config collects API key warnings."""
        config = YoloConfig(
            project_name="test",
            llm=LLMConfig(
                cheap_model=LLM_CHEAP_MODEL_DEFAULT,
                premium_model="claude-sonnet-4-20250514",
                best_model="claude-opus-4-5-20251101",
            ),
        )
        result = validate_config(config)

        # Should have warnings for both missing keys
        assert len(result.warnings) >= 2
        openai_warning = any("openai" in w.field.lower() for w in result.warnings)
        anthropic_warning = any("anthropic" in w.field.lower() for w in result.warnings)
        assert openai_warning
        assert anthropic_warning

    def test_validate_config_is_valid_with_warnings(self) -> None:
        """validate_config returns is_valid=True even with warnings."""
        config = YoloConfig(
            project_name="test",
            llm=LLMConfig(cheap_model=LLM_CHEAP_MODEL_DEFAULT),
        )
        result = validate_config(config)

        # Warnings exist but result is still valid
        assert len(result.warnings) > 0
        assert result.is_valid is True


class TestValidatorsModuleExports:
    """Tests for validators module exports."""

    def test_validation_issue_importable_from_config_module(self) -> None:
        """ValidationIssue is importable from yolo_developer.config."""
        from yolo_developer.config import ValidationIssue

        issue = ValidationIssue(field="test", message="test")
        assert issue.field == "test"

    def test_validation_result_importable_from_config_module(self) -> None:
        """ValidationResult is importable from yolo_developer.config."""
        from yolo_developer.config import ValidationResult

        result = ValidationResult()
        assert result.is_valid is True

    def test_validate_config_importable_from_config_module(self) -> None:
        """validate_config is importable from yolo_developer.config."""
        from yolo_developer.config import validate_config

        assert callable(validate_config)


class TestValidatorsCodeQuality:
    """Tests for validators code quality requirements."""

    def test_validators_passes_mypy(self) -> None:
        """validators.py passes mypy type checking."""
        import subprocess

        from tests.conftest import UV_AVAILABLE, UV_CMD

        if not UV_AVAILABLE:
            pytest.skip("uv not available")

        result = subprocess.run(
            [*UV_CMD, "run", "mypy", "src/yolo_developer/config/validators.py"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"mypy failed:\n{result.stdout}\n{result.stderr}"

    def test_validators_has_future_annotations(self) -> None:
        """validators.py uses __future__ annotations."""
        validators_path = Path("src/yolo_developer/config/validators.py")
        content = validators_path.read_text()
        assert "from __future__ import annotations" in content
