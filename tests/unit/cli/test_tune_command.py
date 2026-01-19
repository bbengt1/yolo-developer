"""Unit tests for yolo tune command (Story 12.7).

Tests cover:
- CLI flag parsing for all options
- Agent template registry completeness
- Template viewing output
- Custom template save/load
- Template reset functionality
- Export/import round-trip
- Validation of invalid templates
- --list output structure
- JSON output format

References:
    - Story 12.7: yolo tune command
    - FR91: Users can customize agent templates and rules
    - FR103: Users can modify agent templates via yolo tune command
"""

from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from rich.console import Console


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_pattern = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_pattern.sub("", text)

from yolo_developer.cli.commands.tune import (
    AGENT_COLORS,
    CONFIGURABLE_AGENTS,
    TEMPLATES_DIR,
    VALID_AGENTS,
    _display_agent_template,
    _display_agents_list,
    _edit_template,
    _export_template,
    _get_effective_template,
    _get_template_path,
    _get_templates_dir,
    _import_template,
    _load_default_template,
    has_custom_template,
    load_custom_template,
    reset_template,
    save_custom_template,
    tune_command,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create a temp directory and change to it."""
    return tmp_path


@pytest.fixture
def templates_dir(temp_dir: Path) -> Path:
    """Create a templates directory in temp."""
    tpl_dir = temp_dir / ".yolo" / "templates"
    tpl_dir.mkdir(parents=True)
    return tpl_dir


@pytest.fixture
def mock_cwd(temp_dir: Path) -> Any:
    """Mock Path.cwd() to return temp_dir."""
    with patch.object(Path, "cwd", return_value=temp_dir):
        yield temp_dir


@pytest.fixture
def sample_template() -> dict[str, Any]:
    """Sample template data for testing."""
    return {
        "agent": "analyst",
        "version": "1.0",
        "customized_at": "2026-01-19T10:00:00+00:00",
        "system_prompt": "Custom analyst system prompt",
        "user_prompt_template": "Custom analyst user prompt: {seed_content}",
        "guidelines": {"custom_rule": "Custom rule content"},
    }


# =============================================================================
# Registry Tests (Task 2)
# =============================================================================


class TestAgentRegistry:
    """Tests for agent template registry."""

    def test_all_agents_defined(self) -> None:
        """All 6 agents should be in CONFIGURABLE_AGENTS."""
        expected = {"analyst", "pm", "architect", "dev", "sm", "tea"}
        assert set(CONFIGURABLE_AGENTS.keys()) == expected

    def test_all_agents_have_description(self) -> None:
        """Every agent must have a non-empty description."""
        for agent_name, config in CONFIGURABLE_AGENTS.items():
            assert "description" in config, f"{agent_name} missing description"
            assert config["description"], f"{agent_name} has empty description"

    def test_valid_agents_matches_configurable(self) -> None:
        """VALID_AGENTS should match CONFIGURABLE_AGENTS keys."""
        assert VALID_AGENTS == frozenset(CONFIGURABLE_AGENTS.keys())

    def test_all_agents_have_colors(self) -> None:
        """Every agent should have a color defined."""
        for agent_name in CONFIGURABLE_AGENTS:
            assert agent_name in AGENT_COLORS, f"{agent_name} missing color"


# =============================================================================
# Template Storage Tests (Task 4)
# =============================================================================


class TestTemplateStorage:
    """Tests for template storage functions."""

    def test_get_templates_dir(self, mock_cwd: Path) -> None:
        """_get_templates_dir returns correct path."""
        expected = mock_cwd / TEMPLATES_DIR
        assert _get_templates_dir() == expected

    def test_get_template_path(self, mock_cwd: Path) -> None:
        """_get_template_path returns correct file path."""
        expected = mock_cwd / TEMPLATES_DIR / "analyst.yaml"
        assert _get_template_path("analyst") == expected

    def test_has_custom_template_false(self, mock_cwd: Path) -> None:
        """has_custom_template returns False when no template exists."""
        assert has_custom_template("analyst") is False

    def test_has_custom_template_true(
        self, mock_cwd: Path, templates_dir: Path
    ) -> None:
        """has_custom_template returns True when template exists."""
        (templates_dir / "analyst.yaml").write_text("agent: analyst")
        assert has_custom_template("analyst") is True

    def test_load_custom_template_none(self, mock_cwd: Path) -> None:
        """load_custom_template returns None when no template exists."""
        assert load_custom_template("analyst") is None

    def test_load_custom_template_success(
        self, mock_cwd: Path, templates_dir: Path, sample_template: dict[str, Any]
    ) -> None:
        """load_custom_template loads YAML correctly."""
        (templates_dir / "analyst.yaml").write_text(yaml.dump(sample_template))
        loaded = load_custom_template("analyst")
        assert loaded is not None
        assert loaded["agent"] == "analyst"
        assert loaded["system_prompt"] == sample_template["system_prompt"]

    def test_load_custom_template_invalid_yaml(
        self, mock_cwd: Path, templates_dir: Path
    ) -> None:
        """load_custom_template returns None for invalid YAML."""
        (templates_dir / "analyst.yaml").write_text("{ invalid: yaml: :")
        assert load_custom_template("analyst") is None

    def test_save_custom_template(self, mock_cwd: Path) -> None:
        """save_custom_template creates template file."""
        path = save_custom_template(
            agent_name="analyst",
            system_prompt="Test system prompt",
            user_prompt_template="Test user prompt: {content}",
            guidelines={"rule1": "Rule content"},
        )
        assert path.exists()
        loaded = yaml.safe_load(path.read_text())
        assert loaded["agent"] == "analyst"
        assert loaded["system_prompt"] == "Test system prompt"
        assert loaded["user_prompt_template"] == "Test user prompt: {content}"
        assert loaded["guidelines"]["rule1"] == "Rule content"
        assert "customized_at" in loaded

    def test_save_custom_template_creates_directory(self, mock_cwd: Path) -> None:
        """save_custom_template creates .yolo/templates if missing."""
        tpl_dir = mock_cwd / ".yolo" / "templates"
        assert not tpl_dir.exists()
        save_custom_template("analyst", "prompt", None)
        assert tpl_dir.exists()

    def test_reset_template_exists(
        self, mock_cwd: Path, templates_dir: Path
    ) -> None:
        """reset_template removes existing custom template."""
        (templates_dir / "analyst.yaml").write_text("agent: analyst")
        assert has_custom_template("analyst")
        result = reset_template("analyst")
        assert result is True
        assert not has_custom_template("analyst")

    def test_reset_template_not_exists(self, mock_cwd: Path) -> None:
        """reset_template returns False when no template exists."""
        result = reset_template("analyst")
        assert result is False


# =============================================================================
# Template Loading Tests (Task 3)
# =============================================================================


class TestTemplateLoading:
    """Tests for template loading functions."""

    def test_load_default_template_analyst(self) -> None:
        """_load_default_template loads analyst prompts."""
        template = _load_default_template("analyst")
        assert template["system_prompt"] is not None
        assert "Requirements Analyst" in template["system_prompt"]
        assert template["user_prompt_template"] is not None
        assert "{seed_content}" in template["user_prompt_template"]

    def test_load_default_template_pm(self) -> None:
        """_load_default_template loads PM prompts."""
        template = _load_default_template("pm")
        assert template["system_prompt"] is not None
        assert "Product Manager" in template["system_prompt"]

    def test_load_default_template_dev(self) -> None:
        """_load_default_template loads dev guidelines."""
        template = _load_default_template("dev")
        # Dev uses guidelines instead of system_prompt
        assert template["guidelines"]
        assert "MAINTAINABILITY_GUIDELINES" in template["guidelines"]
        assert "PROJECT_CONVENTIONS" in template["guidelines"]

    def test_load_default_template_no_prompts(self) -> None:
        """_load_default_template handles agents without prompts."""
        # SM and TEA don't have centralized prompts
        for agent in ["sm", "tea"]:
            template = _load_default_template(agent)
            assert template["system_prompt"] is None
            assert template["user_prompt_template"] is None

    def test_load_default_template_unknown_agent(self) -> None:
        """_load_default_template returns empty dict for unknown agent."""
        template = _load_default_template("unknown")
        assert template == {"system_prompt": None, "user_prompt_template": None, "guidelines": {}}

    def test_get_effective_template_default(self, mock_cwd: Path) -> None:
        """_get_effective_template returns default when no custom exists."""
        template, is_customized = _get_effective_template("analyst")
        assert is_customized is False
        assert template["system_prompt"] is not None

    def test_get_effective_template_custom(
        self, mock_cwd: Path, templates_dir: Path, sample_template: dict[str, Any]
    ) -> None:
        """_get_effective_template returns custom when exists."""
        (templates_dir / "analyst.yaml").write_text(yaml.dump(sample_template))
        template, is_customized = _get_effective_template("analyst")
        assert is_customized is True
        assert template["system_prompt"] == sample_template["system_prompt"]


# =============================================================================
# Display Tests (Task 3, Task 7)
# =============================================================================


class TestDisplay:
    """Tests for display functions."""

    def test_display_agent_template_json(self, mock_cwd: Path) -> None:
        """_display_agent_template outputs valid JSON."""
        output = StringIO()
        test_console = Console(file=output, force_terminal=True)
        with patch("yolo_developer.cli.commands.tune.console", test_console):
            _display_agent_template("analyst", json_output=True)

        result = strip_ansi(output.getvalue())
        # Extract JSON from output
        assert '"agent": "analyst"' in result
        assert '"status": "default"' in result

    def test_display_agent_template_rich(self, mock_cwd: Path) -> None:
        """_display_agent_template outputs Rich formatted content."""
        output = StringIO()
        test_console = Console(file=output, force_terminal=True)
        with patch("yolo_developer.cli.commands.tune.console", test_console):
            _display_agent_template("analyst", json_output=False)

        result = strip_ansi(output.getvalue())
        assert "ANALYST Agent Template" in result
        assert "(default)" in result

    def test_display_agents_list_json(self, mock_cwd: Path) -> None:
        """_display_agents_list outputs valid JSON."""
        output = StringIO()
        test_console = Console(file=output, force_terminal=True)
        with patch("yolo_developer.cli.commands.tune.console", test_console):
            _display_agents_list(json_output=True)

        result = strip_ansi(output.getvalue())
        assert '"agents":' in result
        assert '"analyst"' in result
        assert '"pm"' in result

    def test_display_agents_list_rich(self, mock_cwd: Path) -> None:
        """_display_agents_list outputs Rich table."""
        output = StringIO()
        test_console = Console(file=output, force_terminal=True)
        with patch("yolo_developer.cli.commands.tune.console", test_console):
            _display_agents_list(json_output=False)

        result = strip_ansi(output.getvalue())
        assert "Configurable Agents" in result
        assert "analyst" in result
        assert "default" in result


# =============================================================================
# Export/Import Tests (Task 6)
# =============================================================================


class TestExportImport:
    """Tests for export/import functions."""

    def test_export_template(self, mock_cwd: Path) -> None:
        """_export_template creates valid YAML file."""
        export_path = mock_cwd / "exported.yaml"
        result = _export_template("analyst", export_path)
        assert result is True
        assert export_path.exists()

        loaded = yaml.safe_load(export_path.read_text())
        assert loaded["agent"] == "analyst"
        assert loaded["status"] == "default"
        assert "exported_at" in loaded

    def test_export_template_custom(
        self, mock_cwd: Path, templates_dir: Path, sample_template: dict[str, Any]
    ) -> None:
        """_export_template exports customized template with correct status."""
        (templates_dir / "analyst.yaml").write_text(yaml.dump(sample_template))

        export_path = mock_cwd / "exported.yaml"
        _export_template("analyst", export_path)

        loaded = yaml.safe_load(export_path.read_text())
        assert loaded["status"] == "customized"
        assert loaded["system_prompt"] == sample_template["system_prompt"]

    def test_import_template(self, mock_cwd: Path) -> None:
        """_import_template loads template from file."""
        import_path = mock_cwd / "import.yaml"
        import_data = {
            "agent": "analyst",
            "version": "1.0",
            "system_prompt": "Imported prompt",
        }
        import_path.write_text(yaml.dump(import_data))

        result = _import_template("analyst", import_path)
        assert result is True
        assert has_custom_template("analyst")

        loaded = load_custom_template("analyst")
        assert loaded is not None
        assert loaded["system_prompt"] == "Imported prompt"

    def test_import_template_not_found(self, mock_cwd: Path) -> None:
        """_import_template handles missing file."""
        result = _import_template("analyst", mock_cwd / "nonexistent.yaml")
        assert result is False

    def test_import_template_agent_mismatch(self, mock_cwd: Path) -> None:
        """_import_template rejects mismatched agent."""
        import_path = mock_cwd / "import.yaml"
        import_data = {"agent": "pm", "system_prompt": "PM prompt"}
        import_path.write_text(yaml.dump(import_data))

        result = _import_template("analyst", import_path)
        assert result is False

    def test_import_template_empty_file(self, mock_cwd: Path) -> None:
        """_import_template handles empty file."""
        import_path = mock_cwd / "empty.yaml"
        import_path.write_text("")

        result = _import_template("analyst", import_path)
        assert result is False

    def test_import_template_invalid_yaml(self, mock_cwd: Path) -> None:
        """_import_template handles invalid YAML."""
        import_path = mock_cwd / "invalid.yaml"
        import_path.write_text("{ invalid: yaml: :")

        result = _import_template("analyst", import_path)
        assert result is False

    def test_export_import_roundtrip(self, mock_cwd: Path) -> None:
        """Export then import preserves template content."""
        # Save a custom template
        save_custom_template(
            "analyst",
            system_prompt="Roundtrip test prompt",
            user_prompt_template="Roundtrip user: {content}",
        )

        # Export it
        export_path = mock_cwd / "roundtrip.yaml"
        _export_template("analyst", export_path)

        # Reset and verify gone
        reset_template("analyst")
        assert not has_custom_template("analyst")

        # Import
        _import_template("analyst", export_path)
        assert has_custom_template("analyst")

        loaded = load_custom_template("analyst")
        assert loaded is not None
        assert loaded["system_prompt"] == "Roundtrip test prompt"


# =============================================================================
# Edit Template Tests (Task 5)
# =============================================================================


class TestEditTemplate:
    """Tests for _edit_template function."""

    def test_edit_template_success(self, mock_cwd: Path) -> None:
        """_edit_template saves template when editor exits successfully."""
        output = StringIO()
        test_console = Console(file=output, force_terminal=True)

        def mock_editor(temp_path: str) -> None:
            """Simulate editing the temp file."""
            with open(temp_path, "w") as f:
                yaml.dump({
                    "agent": "analyst",
                    "version": "1.0",
                    "system_prompt": "Edited system prompt",
                }, f)

        with (
            patch("yolo_developer.cli.commands.tune.console", test_console),
            patch("yolo_developer.cli.display.console", test_console),
            patch("subprocess.run") as mock_run,
            patch.dict("os.environ", {"EDITOR": "mock_editor"}),
        ):
            # Simulate editor modifying the file
            def run_side_effect(args: list[str], check: bool = False) -> Any:
                mock_editor(args[1])  # args[1] is the temp file path
                mock_result = type("MockResult", (), {"returncode": 0})()
                return mock_result

            mock_run.side_effect = run_side_effect
            result = _edit_template("analyst")

        assert result is True
        assert has_custom_template("analyst")
        loaded = load_custom_template("analyst")
        assert loaded is not None
        assert loaded["system_prompt"] == "Edited system prompt"

    def test_edit_template_editor_nonzero_exit(self, mock_cwd: Path) -> None:
        """_edit_template returns False when editor exits with non-zero code."""
        output = StringIO()
        test_console = Console(file=output, force_terminal=True)

        with (
            patch("yolo_developer.cli.commands.tune.console", test_console),
            patch("yolo_developer.cli.display.console", test_console),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = type("MockResult", (), {"returncode": 1})()
            result = _edit_template("analyst")

        assert result is False
        assert not has_custom_template("analyst")
        assert "Edit Cancelled" in strip_ansi(output.getvalue())

    def test_edit_template_empty_result(self, mock_cwd: Path) -> None:
        """_edit_template returns False when edited file is empty."""
        output = StringIO()
        test_console = Console(file=output, force_terminal=True)

        def mock_editor_empty(temp_path: str) -> None:
            """Simulate emptying the temp file."""
            with open(temp_path, "w") as f:
                f.write("")

        with (
            patch("yolo_developer.cli.commands.tune.console", test_console),
            patch("yolo_developer.cli.display.console", test_console),
            patch("subprocess.run") as mock_run,
        ):
            def run_side_effect(args: list[str], check: bool = False) -> Any:
                mock_editor_empty(args[1])
                return type("MockResult", (), {"returncode": 0})()

            mock_run.side_effect = run_side_effect
            result = _edit_template("analyst")

        assert result is False
        assert not has_custom_template("analyst")
        assert "Empty template" in strip_ansi(output.getvalue())

    def test_edit_template_agent_mismatch(self, mock_cwd: Path) -> None:
        """_edit_template returns False when agent name is changed in edited file."""
        output = StringIO()
        test_console = Console(file=output, force_terminal=True)

        def mock_editor_wrong_agent(temp_path: str) -> None:
            """Simulate changing agent name in temp file."""
            with open(temp_path, "w") as f:
                yaml.dump({
                    "agent": "pm",  # Wrong agent!
                    "system_prompt": "Edited prompt",
                }, f)

        with (
            patch("yolo_developer.cli.commands.tune.console", test_console),
            patch("yolo_developer.cli.display.console", test_console),
            patch("subprocess.run") as mock_run,
        ):
            def run_side_effect(args: list[str], check: bool = False) -> Any:
                mock_editor_wrong_agent(args[1])
                return type("MockResult", (), {"returncode": 0})()

            mock_run.side_effect = run_side_effect
            result = _edit_template("analyst")

        assert result is False
        assert not has_custom_template("analyst")
        assert "Invalid Template" in strip_ansi(output.getvalue())

    def test_edit_template_invalid_yaml(self, mock_cwd: Path) -> None:
        """_edit_template returns False when edited file has invalid YAML."""
        output = StringIO()
        test_console = Console(file=output, force_terminal=True)

        def mock_editor_invalid_yaml(temp_path: str) -> None:
            """Simulate writing invalid YAML to temp file."""
            with open(temp_path, "w") as f:
                f.write("{ invalid: yaml: syntax: :")

        with (
            patch("yolo_developer.cli.commands.tune.console", test_console),
            patch("yolo_developer.cli.display.console", test_console),
            patch("subprocess.run") as mock_run,
        ):
            def run_side_effect(args: list[str], check: bool = False) -> Any:
                mock_editor_invalid_yaml(args[1])
                return type("MockResult", (), {"returncode": 0})()

            mock_run.side_effect = run_side_effect
            result = _edit_template("analyst")

        assert result is False
        assert not has_custom_template("analyst")
        assert "Invalid YAML" in strip_ansi(output.getvalue())

    def test_edit_template_missing_agent_key(self, mock_cwd: Path) -> None:
        """_edit_template returns False when agent key is missing from edited file."""
        output = StringIO()
        test_console = Console(file=output, force_terminal=True)

        def mock_editor_no_agent(temp_path: str) -> None:
            """Simulate removing agent key from temp file."""
            with open(temp_path, "w") as f:
                yaml.dump({
                    "system_prompt": "Edited prompt",
                    # Missing "agent" key!
                }, f)

        with (
            patch("yolo_developer.cli.commands.tune.console", test_console),
            patch("yolo_developer.cli.display.console", test_console),
            patch("subprocess.run") as mock_run,
        ):
            def run_side_effect(args: list[str], check: bool = False) -> Any:
                mock_editor_no_agent(args[1])
                return type("MockResult", (), {"returncode": 0})()

            mock_run.side_effect = run_side_effect
            result = _edit_template("analyst")

        assert result is False
        assert not has_custom_template("analyst")
        assert "Invalid Template" in strip_ansi(output.getvalue())


# =============================================================================
# Command Tests
# =============================================================================


class TestTuneCommand:
    """Tests for tune_command function."""

    def test_tune_command_list(self, mock_cwd: Path) -> None:
        """tune_command with --list shows all agents."""
        output = StringIO()
        test_console = Console(file=output, force_terminal=True)
        with patch("yolo_developer.cli.commands.tune.console", test_console):
            tune_command(list_agents=True)

        result = strip_ansi(output.getvalue())
        assert "Configurable Agents" in result

    def test_tune_command_list_json(self, mock_cwd: Path) -> None:
        """tune_command with --list --json outputs JSON."""
        output = StringIO()
        test_console = Console(file=output, force_terminal=True)
        with patch("yolo_developer.cli.commands.tune.console", test_console):
            tune_command(list_agents=True, json_output=True)

        result = strip_ansi(output.getvalue())
        assert '"agents":' in result

    def test_tune_command_no_args(self, mock_cwd: Path) -> None:
        """tune_command with no args shows help."""
        output = StringIO()
        test_console = Console(file=output, force_terminal=True)
        with (
            patch("yolo_developer.cli.commands.tune.console", test_console),
            patch("yolo_developer.cli.display.console", test_console),
        ):
            tune_command()

        result = strip_ansi(output.getvalue())
        assert "yolo tune --list" in result

    def test_tune_command_no_args_json(self, mock_cwd: Path) -> None:
        """tune_command with no args and --json outputs error JSON."""
        output = StringIO()
        test_console = Console(file=output, force_terminal=True)
        with (
            patch("yolo_developer.cli.commands.tune.console", test_console),
            patch("yolo_developer.cli.display.console", test_console),
        ):
            tune_command(json_output=True)

        result = strip_ansi(output.getvalue())
        assert '"error":' in result

    def test_tune_command_view_agent(self, mock_cwd: Path) -> None:
        """tune_command with agent name shows template."""
        output = StringIO()
        test_console = Console(file=output, force_terminal=True)
        with patch("yolo_developer.cli.commands.tune.console", test_console):
            tune_command(agent_name="analyst")

        result = strip_ansi(output.getvalue())
        assert "ANALYST Agent Template" in result

    def test_tune_command_view_agent_json(self, mock_cwd: Path) -> None:
        """tune_command with agent and --json outputs JSON."""
        output = StringIO()
        test_console = Console(file=output, force_terminal=True)
        with patch("yolo_developer.cli.commands.tune.console", test_console):
            tune_command(agent_name="analyst", json_output=True)

        result = strip_ansi(output.getvalue())
        assert '"agent": "analyst"' in result

    def test_tune_command_invalid_agent(self, mock_cwd: Path) -> None:
        """tune_command with invalid agent shows error and exits with code 1."""
        import typer

        output = StringIO()
        test_console = Console(file=output, force_terminal=True)
        with (
            patch("yolo_developer.cli.commands.tune.console", test_console),
            patch("yolo_developer.cli.display.console", test_console),
            pytest.raises(typer.Exit) as exc_info,
        ):
            tune_command(agent_name="unknown")

        assert exc_info.value.exit_code == 1
        result = strip_ansi(output.getvalue())
        assert "Unknown agent" in result or "Invalid Agent" in result

    def test_tune_command_case_insensitive(self, mock_cwd: Path) -> None:
        """tune_command handles case-insensitive agent names."""
        output = StringIO()
        test_console = Console(file=output, force_terminal=True)
        with patch("yolo_developer.cli.commands.tune.console", test_console):
            tune_command(agent_name="ANALYST")

        result = strip_ansi(output.getvalue())
        assert "ANALYST Agent Template" in result

    def test_tune_command_reset(
        self, mock_cwd: Path, templates_dir: Path
    ) -> None:
        """tune_command with --reset removes custom template."""
        (templates_dir / "analyst.yaml").write_text("agent: analyst")
        assert has_custom_template("analyst")

        output = StringIO()
        test_console = Console(file=output, force_terminal=True)
        with (
            patch("yolo_developer.cli.commands.tune.console", test_console),
            patch("yolo_developer.cli.display.console", test_console),
        ):
            tune_command(agent_name="analyst", reset=True)

        assert not has_custom_template("analyst")
        result = strip_ansi(output.getvalue())
        assert "reset" in result.lower()

    def test_tune_command_reset_no_custom(self, mock_cwd: Path) -> None:
        """tune_command with --reset and no custom shows info."""
        output = StringIO()
        test_console = Console(file=output, force_terminal=True)
        with (
            patch("yolo_developer.cli.commands.tune.console", test_console),
            patch("yolo_developer.cli.display.console", test_console),
        ):
            tune_command(agent_name="analyst", reset=True)

        result = strip_ansi(output.getvalue())
        assert "default" in result.lower()

    def test_tune_command_export(self, mock_cwd: Path) -> None:
        """tune_command with --export creates file."""
        export_path = mock_cwd / "test_export.yaml"

        output = StringIO()
        test_console = Console(file=output, force_terminal=True)
        with (
            patch("yolo_developer.cli.commands.tune.console", test_console),
            patch("yolo_developer.cli.display.console", test_console),
        ):
            tune_command(agent_name="analyst", export_path=export_path)

        assert export_path.exists()
        result = strip_ansi(output.getvalue())
        assert "Export" in result

    def test_tune_command_import(self, mock_cwd: Path) -> None:
        """tune_command with --import loads template."""
        import_path = mock_cwd / "test_import.yaml"
        import_path.write_text(yaml.dump({
            "agent": "analyst",
            "system_prompt": "Imported via command",
        }))

        output = StringIO()
        test_console = Console(file=output, force_terminal=True)
        with (
            patch("yolo_developer.cli.commands.tune.console", test_console),
            patch("yolo_developer.cli.display.console", test_console),
        ):
            tune_command(agent_name="analyst", import_path=import_path)

        assert has_custom_template("analyst")
        result = strip_ansi(output.getvalue())
        assert "Import" in result


# =============================================================================
# JSON Output Structure Tests
# =============================================================================


class TestJsonOutputStructure:
    """Tests verifying JSON output matches documented schema."""

    def test_json_template_structure(self, mock_cwd: Path) -> None:
        """JSON output for single agent matches schema."""
        output = StringIO()
        test_console = Console(file=output, force_terminal=True)
        with patch("yolo_developer.cli.commands.tune.console", test_console):
            _display_agent_template("analyst", json_output=True)

        # Parse JSON from output
        result = strip_ansi(output.getvalue().strip())
        data = json.loads(result)

        assert "agent" in data
        assert "status" in data
        assert "customized_at" in data
        assert "template" in data
        assert "system_prompt" in data["template"]
        assert "user_prompt_template" in data["template"]
        assert "guidelines" in data["template"]

    def test_json_list_structure(self, mock_cwd: Path) -> None:
        """JSON output for --list matches schema."""
        output = StringIO()
        test_console = Console(file=output, force_terminal=True)
        with patch("yolo_developer.cli.commands.tune.console", test_console):
            _display_agents_list(json_output=True)

        result = strip_ansi(output.getvalue().strip())
        data = json.loads(result)

        assert "agents" in data
        assert isinstance(data["agents"], list)
        assert len(data["agents"]) == 6

        for agent in data["agents"]:
            assert "name" in agent
            assert "description" in agent
            assert "status" in agent
            assert agent["status"] in ("default", "customized")


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_all_agent_templates_loadable(self) -> None:
        """All agents with prompts can load their default templates."""
        for agent_name in VALID_AGENTS:
            template = _load_default_template(agent_name)
            # Should not raise
            assert isinstance(template, dict)

    def test_custom_template_yaml_error_handling(
        self, mock_cwd: Path, templates_dir: Path
    ) -> None:
        """load_custom_template handles malformed YAML gracefully."""
        (templates_dir / "analyst.yaml").write_text("invalid: yaml: content: :")
        result = load_custom_template("analyst")
        assert result is None

    def test_save_template_with_none_values(self, mock_cwd: Path) -> None:
        """save_custom_template handles None values correctly."""
        path = save_custom_template("analyst", None, None, None)
        loaded = yaml.safe_load(path.read_text())
        assert "system_prompt" not in loaded
        assert "user_prompt_template" not in loaded
        assert "guidelines" not in loaded
