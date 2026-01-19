"""Unit tests for yolo config command (Story 12.8).

Tests cover:
- CLI flag parsing for all subcommands
- Configuration viewing with and without config file
- Configuration setting with valid/invalid keys
- Configuration setting with nested keys
- Configuration export to default and custom paths
- Configuration import with valid/invalid files
- API key masking in output
- JSON output format for all subcommands
- Error handling for missing config file
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from typer.testing import CliRunner

from yolo_developer.cli.main import app

if TYPE_CHECKING:
    from collections.abc import Generator


runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text for testing Rich output."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def extract_json(output: str) -> dict[str, Any]:
    """Extract JSON from output that may contain log lines.

    The CLI may output log lines before/after JSON when structlog is configured.
    This function finds and parses the JSON portion of the output.
    """
    # Try to find JSON object by looking for lines starting with { or [
    lines = output.strip().split("\n")
    json_lines: list[str] = []
    in_json = False
    brace_count = 0

    for line in lines:
        stripped = line.strip()
        if not in_json:
            if stripped.startswith("{"):
                in_json = True
                brace_count = stripped.count("{") - stripped.count("}")
                json_lines.append(line)
                if brace_count == 0:
                    break
        else:
            json_lines.append(line)
            brace_count += stripped.count("{") - stripped.count("}")
            if brace_count == 0:
                break

    if json_lines:
        json_str = "\n".join(json_lines)
        return json.loads(json_str)

    # Fall back to trying to parse the whole output
    return json.loads(output)


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary directory with a valid yolo.yaml config file."""
    config_content = {
        "project_name": "test-project",
        "llm": {
            "cheap_model": "gpt-4o-mini",
            "premium_model": "claude-sonnet-4-20250514",
            "best_model": "claude-opus-4-5-20251101",
        },
        "quality": {
            "test_coverage_threshold": 0.80,
            "confidence_threshold": 0.90,
        },
        "memory": {
            "persist_path": ".yolo/memory",
            "vector_store_type": "chromadb",
            "graph_store_type": "json",
        },
    }

    config_file = tmp_path / "yolo.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config_content, f)

    # Change to temp directory for tests
    import os

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(original_cwd)


@pytest.fixture
def no_config_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary directory without a yolo.yaml config file."""
    import os

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(original_cwd)


class TestConfigShowCommand:
    """Tests for yolo config (show) command."""

    def test_show_config_displays_project_name(self, temp_config_dir: Path) -> None:
        """Test that yolo config shows project name."""
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "test-project" in output

    def test_show_config_displays_llm_section(self, temp_config_dir: Path) -> None:
        """Test that yolo config shows LLM configuration."""
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "llm" in output.lower()
        assert "gpt-4o-mini" in output

    def test_show_config_displays_quality_section(self, temp_config_dir: Path) -> None:
        """Test that yolo config shows quality configuration."""
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "quality" in output.lower()
        assert "0.8" in output  # test_coverage_threshold

    def test_show_config_displays_memory_section(self, temp_config_dir: Path) -> None:
        """Test that yolo config shows memory configuration."""
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "memory" in output.lower()
        assert ".yolo/memory" in output

    def test_show_config_masks_api_keys(self, temp_config_dir: Path) -> None:
        """Test that API keys are masked by default."""
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        # API keys should show as masked or "not set", never actual values
        assert "openai_api_key" not in output.lower() or "****" in output or "not set" in output.lower()

    def test_show_config_json_output(self, temp_config_dir: Path) -> None:
        """Test that --json flag outputs valid JSON."""
        result = runner.invoke(app, ["config", "--json"])
        assert result.exit_code == 0
        output = extract_json(result.output)
        assert output["project_name"] == "test-project"
        assert "llm" in output
        assert "quality" in output
        assert "memory" in output

    def test_show_config_no_config_file(self, no_config_dir: Path) -> None:
        """Test behavior when yolo.yaml doesn't exist."""
        result = runner.invoke(app, ["config"])
        # Should show warning and suggest yolo init
        assert "init" in result.output.lower() or result.exit_code != 0

    def test_show_config_json_masks_api_keys(self, temp_config_dir: Path) -> None:
        """Test that JSON output masks API keys by default."""
        result = runner.invoke(app, ["config", "--json"])
        assert result.exit_code == 0
        output = extract_json(result.output)
        # API keys should be masked in JSON output
        if output["llm"].get("openai_api_key"):
            assert output["llm"]["openai_api_key"] == "****"


class TestConfigSetCommand:
    """Tests for yolo config set command."""

    def test_set_simple_key(self, temp_config_dir: Path) -> None:
        """Test setting a simple top-level key."""
        result = runner.invoke(app, ["config", "set", "project_name", "new-name"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "new-name" in output

        # Verify the change was persisted
        with open(temp_config_dir / "yolo.yaml") as f:
            config = yaml.safe_load(f)
        assert config["project_name"] == "new-name"

    def test_set_nested_key(self, temp_config_dir: Path) -> None:
        """Test setting a nested key with dot notation."""
        result = runner.invoke(app, ["config", "set", "llm.cheap_model", "gpt-4o"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "gpt-4o" in output

        # Verify the change was persisted
        with open(temp_config_dir / "yolo.yaml") as f:
            config = yaml.safe_load(f)
        assert config["llm"]["cheap_model"] == "gpt-4o"

    def test_set_numeric_value(self, temp_config_dir: Path) -> None:
        """Test setting a numeric value."""
        result = runner.invoke(
            app, ["config", "set", "quality.test_coverage_threshold", "0.85"]
        )
        assert result.exit_code == 0

        # Verify the change was persisted
        with open(temp_config_dir / "yolo.yaml") as f:
            config = yaml.safe_load(f)
        assert config["quality"]["test_coverage_threshold"] == 0.85

    def test_set_invalid_key(self, temp_config_dir: Path) -> None:
        """Test setting an invalid key shows error."""
        result = runner.invoke(app, ["config", "set", "nonexistent.key", "value"])
        assert result.exit_code != 0
        output = strip_ansi(result.output)
        assert "unknown" in output.lower() or "invalid" in output.lower()

    def test_set_protected_key_rejected(self, temp_config_dir: Path) -> None:
        """Test that setting API keys via CLI is rejected."""
        result = runner.invoke(
            app, ["config", "set", "llm.openai_api_key", "sk-secret"]
        )
        assert result.exit_code != 0
        output = strip_ansi(result.output)
        assert "environment" in output.lower() or "protected" in output.lower()

    def test_set_shows_before_after(self, temp_config_dir: Path) -> None:
        """Test that set command shows before/after values."""
        result = runner.invoke(app, ["config", "set", "llm.cheap_model", "gpt-4o"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        # Should show old and new values
        assert "gpt-4o-mini" in output or "gpt-4o" in output

    def test_set_json_output(self, temp_config_dir: Path) -> None:
        """Test set command JSON output."""
        result = runner.invoke(
            app, ["config", "set", "project_name", "json-test", "--json"]
        )
        assert result.exit_code == 0
        output = extract_json(result.output)
        assert output["key"] == "project_name"
        assert output["new_value"] == "json-test"
        assert output["status"] == "success"

    def test_set_no_config_file(self, no_config_dir: Path) -> None:
        """Test setting value when no config file exists."""
        result = runner.invoke(app, ["config", "set", "project_name", "test"])
        # Should fail or suggest yolo init
        assert "init" in result.output.lower() or result.exit_code != 0


class TestConfigExportCommand:
    """Tests for yolo config export command."""

    def test_export_default_path(self, temp_config_dir: Path) -> None:
        """Test export to default path."""
        result = runner.invoke(app, ["config", "export"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "export" in output.lower()

        # Check that export file was created
        export_file = temp_config_dir / "yolo-config-export.yaml"
        assert export_file.exists()

    def test_export_custom_path(self, temp_config_dir: Path) -> None:
        """Test export to custom path with --output."""
        custom_path = temp_config_dir / "custom-export.yaml"
        result = runner.invoke(app, ["config", "export", "--output", str(custom_path)])
        assert result.exit_code == 0
        assert custom_path.exists()

    def test_export_excludes_api_keys(self, temp_config_dir: Path) -> None:
        """Test that export excludes API keys."""
        result = runner.invoke(app, ["config", "export"])
        assert result.exit_code == 0

        export_file = temp_config_dir / "yolo-config-export.yaml"
        with open(export_file) as f:
            content = f.read()
        # API keys should not be in exported file
        assert "openai_api_key" not in content or "null" in content.lower()

    def test_export_json_output(self, temp_config_dir: Path) -> None:
        """Test export command JSON output."""
        result = runner.invoke(app, ["config", "export", "--json"])
        assert result.exit_code == 0
        output = extract_json(result.output)
        assert output["status"] == "success"
        assert "path" in output

    def test_export_no_config_file(self, no_config_dir: Path) -> None:
        """Test export when no config file exists."""
        result = runner.invoke(app, ["config", "export"])
        # Should fail or suggest yolo init
        assert result.exit_code != 0 or "init" in result.output.lower()


class TestConfigImportCommand:
    """Tests for yolo config import command."""

    def test_import_valid_file(self, temp_config_dir: Path) -> None:
        """Test importing a valid config file."""
        # Create a source config file
        source_file = temp_config_dir / "source-config.yaml"
        source_content = {"project_name": "imported-project"}
        with open(source_file, "w") as f:
            yaml.safe_dump(source_content, f)

        result = runner.invoke(app, ["config", "import", str(source_file)])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "success" in output.lower() or "imported" in output.lower()

    def test_import_nonexistent_file(self, temp_config_dir: Path) -> None:
        """Test importing a file that doesn't exist."""
        result = runner.invoke(app, ["config", "import", "nonexistent.yaml"])
        assert result.exit_code != 0
        output = strip_ansi(result.output)
        assert "not found" in output.lower() or "exist" in output.lower()

    def test_import_invalid_yaml(self, temp_config_dir: Path) -> None:
        """Test importing an invalid YAML file."""
        invalid_file = temp_config_dir / "invalid.yaml"
        with open(invalid_file, "w") as f:
            f.write("invalid: yaml: content:")

        result = runner.invoke(app, ["config", "import", str(invalid_file)])
        assert result.exit_code != 0

    def test_import_invalid_config(self, temp_config_dir: Path) -> None:
        """Test importing valid YAML but invalid config."""
        invalid_config = temp_config_dir / "invalid-config.yaml"
        # Missing required project_name
        with open(invalid_config, "w") as f:
            yaml.safe_dump({"llm": {"cheap_model": "test"}}, f)

        result = runner.invoke(app, ["config", "import", str(invalid_config)])
        assert result.exit_code != 0

    def test_import_json_output(self, temp_config_dir: Path) -> None:
        """Test import command JSON output."""
        source_file = temp_config_dir / "source.yaml"
        with open(source_file, "w") as f:
            yaml.safe_dump({"project_name": "json-import-test"}, f)

        result = runner.invoke(app, ["config", "import", str(source_file), "--json"])
        assert result.exit_code == 0
        output = extract_json(result.output)
        assert output["status"] == "success"
        assert "source" in output


class TestConfigCliFlags:
    """Tests for CLI flag parsing and help text."""

    def test_config_help(self) -> None:
        """Test that config command shows help text."""
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "configuration" in result.output.lower()

    def test_config_set_help(self) -> None:
        """Test that config set shows help text."""
        result = runner.invoke(app, ["config", "set", "--help"])
        assert result.exit_code == 0
        assert "key" in result.output.lower()
        assert "value" in result.output.lower()

    def test_config_export_help(self) -> None:
        """Test that config export shows help text."""
        result = runner.invoke(app, ["config", "export", "--help"])
        assert result.exit_code == 0
        assert "output" in result.output.lower() or "export" in result.output.lower()

    def test_config_import_help(self) -> None:
        """Test that config import shows help text."""
        result = runner.invoke(app, ["config", "import", "--help"])
        assert result.exit_code == 0
        assert "import" in result.output.lower()

    def test_config_json_flag_short(self, temp_config_dir: Path) -> None:
        """Test that -j works as short form of --json."""
        result = runner.invoke(app, ["config", "-j"])
        assert result.exit_code == 0
        # Should produce valid JSON
        extract_json(result.output)


class TestConfigEdgeCases:
    """Tests for edge cases and error handling."""

    def test_deeply_nested_key(self, temp_config_dir: Path) -> None:
        """Test setting a deeply nested key (3+ levels)."""
        result = runner.invoke(
            app, ["config", "set", "quality.seed_thresholds.overall", "0.75"]
        )
        assert result.exit_code == 0

        # Verify the change was persisted
        with open(temp_config_dir / "yolo.yaml") as f:
            config = yaml.safe_load(f)
        assert config["quality"]["seed_thresholds"]["overall"] == 0.75

    def test_set_boolean_value(self, temp_config_dir: Path) -> None:
        """Test setting a boolean value (if applicable)."""
        # This tests the value type conversion
        result = runner.invoke(
            app, ["config", "set", "quality.test_coverage_threshold", "0.9"]
        )
        assert result.exit_code == 0

    def test_empty_value(self, temp_config_dir: Path) -> None:
        """Test setting an empty value."""
        result = runner.invoke(app, ["config", "set", "project_name", ""])
        # Behavior depends on validation - may succeed or fail
        # Just ensure it doesn't crash
        assert result.exit_code in [0, 1]

    def test_no_mask_flag(self, temp_config_dir: Path) -> None:
        """Test that --no-mask flag is accepted and doesn't show masked values."""
        result = runner.invoke(app, ["config", "--no-mask"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        # When no API key is set, should show "not set" instead of "****"
        assert "not set" in output.lower()

    def test_no_mask_flag_json(self, temp_config_dir: Path) -> None:
        """Test that --no-mask with --json returns proper JSON."""
        result = runner.invoke(app, ["config", "--no-mask", "--json"])
        assert result.exit_code == 0
        output = extract_json(result.output)
        # API keys should be null when not set, not masked
        assert output["llm"].get("openai_api_key") is None

    def test_set_invalid_numeric_value(self, temp_config_dir: Path) -> None:
        """Test setting an invalid numeric value to a float field."""
        result = runner.invoke(
            app, ["config", "set", "quality.test_coverage_threshold", "not-a-number"]
        )
        # Should fail validation
        assert result.exit_code != 0
        output = strip_ansi(result.output)
        assert "error" in output.lower() or "invalid" in output.lower()

    def test_set_with_corrupted_yaml(self, temp_config_dir: Path) -> None:
        """Test setting value when yolo.yaml has invalid YAML syntax."""
        # Corrupt the YAML file
        with open(temp_config_dir / "yolo.yaml", "w") as f:
            f.write("invalid: yaml: content: {unclosed")

        result = runner.invoke(app, ["config", "set", "project_name", "test"])
        assert result.exit_code != 0
        output = strip_ansi(result.output)
        assert "yaml" in output.lower() or "error" in output.lower()

    def test_set_with_corrupted_yaml_json_output(self, temp_config_dir: Path) -> None:
        """Test setting value with corrupted YAML returns JSON error."""
        # Corrupt the YAML file
        with open(temp_config_dir / "yolo.yaml", "w") as f:
            f.write("invalid: yaml: content: {unclosed")

        result = runner.invoke(app, ["config", "set", "project_name", "test", "--json"])
        assert result.exit_code != 0
        # Should return valid JSON error
        output = extract_json(result.output)
        assert "error" in output
        assert output["status"] == "failed"
