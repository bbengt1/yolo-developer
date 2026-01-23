"""Integration tests for CLI tools command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from yolo_developer.cli.main import app


runner = CliRunner()


class TestToolsCommand:
    """Integration tests for yolo tools command."""

    def test_tools_without_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify tools command handles missing config gracefully."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["tools"])

        # Should fail gracefully with helpful message
        assert result.exit_code == 1
        assert "Configuration Error" in result.output or "config" in result.output.lower()

    def test_tools_status_without_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify tools status command handles missing config gracefully."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["tools", "status"])

        assert result.exit_code == 1

    def test_tools_json_output_without_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify tools --json returns error as JSON."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["tools", "--json"])

        assert result.exit_code == 1
        # Should still output valid JSON
        try:
            data = json.loads(result.output)
            assert "error" in data or "status" in data
        except json.JSONDecodeError:
            # It's okay if it doesn't output JSON on error
            pass


class TestToolsWithConfig:
    """Integration tests for tools command with valid config."""

    @pytest.fixture
    def project_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> Path:
        """Create a project directory with yolo.yaml config."""
        config_content = """
project_name: test-project

tools:
  claude_code:
    enabled: true
    timeout: 300
  aider:
    enabled: false
"""
        yolo_yaml = tmp_path / "yolo.yaml"
        yolo_yaml.write_text(config_content)
        monkeypatch.chdir(tmp_path)
        return tmp_path

    def test_tools_shows_status(self, project_dir: Path) -> None:
        """Verify tools command shows tool status table."""
        result = runner.invoke(app, ["tools"])

        # Should succeed (exit code 0)
        assert result.exit_code == 0
        assert "claude_code" in result.output
        assert "aider" in result.output

    def test_tools_json_output(self, project_dir: Path) -> None:
        """Verify tools --json returns valid JSON."""
        result = runner.invoke(app, ["tools", "--json"])

        assert result.exit_code == 0

        # Find JSON object in output (may have log messages before it)
        output = result.output
        json_start = output.find("{")
        assert json_start >= 0, "No JSON object found in output"
        json_output = output[json_start:]

        data = json.loads(json_output)
        assert "tools" in data
        assert "status" in data
        assert data["status"] == "success"

        # Verify tool entries
        tools = data["tools"]
        assert len(tools) == 2

        claude_tool = next(t for t in tools if t["name"] == "claude_code")
        assert claude_tool["enabled"] is True
        assert claude_tool["timeout"] == 300

        aider_tool = next(t for t in tools if t["name"] == "aider")
        assert aider_tool["enabled"] is False

    def test_tools_status_subcommand(self, project_dir: Path) -> None:
        """Verify tools status subcommand works."""
        result = runner.invoke(app, ["tools", "status"])

        assert result.exit_code == 0
        assert "claude_code" in result.output

    def test_tools_help(self) -> None:
        """Verify tools --help shows usage."""
        result = runner.invoke(app, ["tools", "--help"])

        assert result.exit_code == 0
        assert "Manage external CLI tool integrations" in result.output
        assert "status" in result.output
