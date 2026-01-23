"""Tests for main CLI commands (Story 12.1).

Tests cover:
- run command calls run_command function
- status command calls status_command function
- logs command calls logs_command function
- tune command calls tune_command function
- config command calls config_command function
- All commands are registered with Typer app
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from yolo_developer.cli.main import app

runner = CliRunner()


class TestRunCommand:
    """Tests for run command."""

    @patch("yolo_developer.cli.commands.run.run_command")
    def test_run_command_invokes_function(self, mock_run_command: MagicMock) -> None:
        """Test yolo run invokes run_command function."""
        result = runner.invoke(app, ["run"])

        mock_run_command.assert_called_once()
        assert result.exit_code == 0

    def test_run_command_has_help(self) -> None:
        """Test yolo run --help shows help."""
        result = runner.invoke(app, ["run", "--help"])

        assert result.exit_code == 0
        assert "Execute an autonomous sprint" in result.output


class TestStatusCommand:
    """Tests for status command."""

    @patch("yolo_developer.cli.commands.status.status_command")
    def test_status_command_invokes_function(self, mock_status_command: MagicMock) -> None:
        """Test yolo status invokes status_command function."""
        result = runner.invoke(app, ["status"])

        mock_status_command.assert_called_once()
        assert result.exit_code == 0

    def test_status_command_has_help(self) -> None:
        """Test yolo status --help shows help."""
        result = runner.invoke(app, ["status", "--help"])

        assert result.exit_code == 0
        assert "sprint progress" in result.output.lower()


class TestLogsCommand:
    """Tests for logs command."""

    @patch("yolo_developer.cli.commands.logs.logs_command")
    def test_logs_command_invokes_function(self, mock_logs_command: MagicMock) -> None:
        """Test yolo logs invokes logs_command function."""
        result = runner.invoke(app, ["logs"])

        mock_logs_command.assert_called_once()
        assert result.exit_code == 0

    def test_logs_command_has_help(self) -> None:
        """Test yolo logs --help shows help."""
        result = runner.invoke(app, ["logs", "--help"])

        assert result.exit_code == 0
        assert "audit trail" in result.output.lower()


class TestTuneCommand:
    """Tests for tune command."""

    @patch("yolo_developer.cli.commands.tune.tune_command")
    def test_tune_command_invokes_function(self, mock_tune_command: MagicMock) -> None:
        """Test yolo tune invokes tune_command function."""
        result = runner.invoke(app, ["tune"])

        mock_tune_command.assert_called_once()
        assert result.exit_code == 0

    def test_tune_command_has_help(self) -> None:
        """Test yolo tune --help shows help."""
        result = runner.invoke(app, ["tune", "--help"])

        assert result.exit_code == 0
        assert "agent" in result.output.lower()


class TestConfigCommand:
    """Tests for config command."""

    @patch("yolo_developer.cli.commands.config.show_config")
    def test_config_command_invokes_function(self, mock_show_config: MagicMock) -> None:
        """Test yolo config invokes show_config function."""
        result = runner.invoke(app, ["config"])

        mock_show_config.assert_called_once()
        assert result.exit_code == 0

    def test_config_command_has_help(self) -> None:
        """Test yolo config --help shows help."""
        result = runner.invoke(app, ["config", "--help"])

        assert result.exit_code == 0
        assert "configuration" in result.output.lower()


class TestExistingCommandsRegression:
    """Regression tests to ensure existing commands still work (AC4)."""

    def test_version_command_works(self) -> None:
        """Test yolo version still works after changes."""
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "YOLO Developer" in result.output

    def test_init_command_has_help(self) -> None:
        """Test yolo init --help still works after changes."""
        result = runner.invoke(app, ["init", "--help"])

        assert result.exit_code == 0
        assert "Initialize a new YOLO Developer project" in result.output

    def test_seed_command_has_help(self) -> None:
        """Test yolo seed --help still works after changes."""
        result = runner.invoke(app, ["seed", "--help"])

        assert result.exit_code == 0
        assert "Parse a seed document" in result.output


class TestAppCommands:
    """Tests for CLI app command registration."""

    def test_app_has_all_commands(self) -> None:
        """Test all expected commands are registered."""
        # Get registered command names (direct commands)
        command_names = [cmd.name for cmd in app.registered_commands]
        # Get registered group names (sub-apps like config)
        group_names = [group.name for group in app.registered_groups]

        # Commands registered directly
        expected_commands = [
            "chat",
            "init",
            "version",
            "seed",
            "run",
            "status",
            "logs",
            "tune",
        ]
        for cmd in expected_commands:
            assert cmd in command_names, f"Command '{cmd}' not registered"

        # Commands registered as groups (sub-apps)
        expected_groups = ["config"]
        expected_groups.append("integrate")
        for grp in expected_groups:
            assert grp in group_names, f"Command group '{grp}' not registered"

    def test_app_help_lists_all_commands(self) -> None:
        """Test yolo --help lists all commands."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        # Check all commands appear in help
        for cmd in [
            "chat",
            "init",
            "version",
            "seed",
            "run",
            "status",
            "logs",
            "tune",
            "config",
            "integrate",
        ]:
            assert cmd in result.output


class TestChatCommand:
    """Tests for chat command routing."""

    @patch("yolo_developer.cli.commands.chat.read_piped_input", return_value=None)
    @patch("yolo_developer.cli.commands.chat.chat_command")
    def test_chat_command_invokes_function(
        self,
        mock_chat_command: MagicMock,
        _mock_read: MagicMock,
    ) -> None:
        """Test yolo chat invokes chat_command function."""
        result = runner.invoke(app, ["chat"])

        mock_chat_command.assert_called_once_with(prompt=None, interactive=True)
        assert result.exit_code == 0

    @patch("yolo_developer.cli.commands.chat.read_piped_input", return_value=None)
    @patch("yolo_developer.cli.commands.chat.chat_command")
    def test_root_no_args_starts_chat(
        self,
        mock_chat_command: MagicMock,
        _mock_read: MagicMock,
    ) -> None:
        """Test yolo (no args) starts interactive chat."""
        result = runner.invoke(app, [])

        mock_chat_command.assert_called_once_with(prompt=None, interactive=True)
        assert result.exit_code == 0

    @patch("yolo_developer.cli.commands.chat.read_piped_input", return_value=None)
    @patch("yolo_developer.cli.commands.chat.chat_command")
    def test_root_args_run_one_shot(
        self,
        mock_chat_command: MagicMock,
        _mock_read: MagicMock,
    ) -> None:
        """Test yolo <prompt> runs one-shot chat."""
        result = runner.invoke(app, ["summarize", "this"])

        mock_chat_command.assert_called_once_with(prompt="summarize this", interactive=False)
        assert result.exit_code == 0

    @patch("yolo_developer.cli.commands.chat.read_piped_input", return_value="from-pipe")
    @patch("yolo_developer.cli.commands.chat.chat_command")
    def test_root_piped_input_runs_one_shot(
        self,
        mock_chat_command: MagicMock,
        _mock_read: MagicMock,
    ) -> None:
        """Test piped stdin runs one-shot chat."""
        result = runner.invoke(app, [])

        mock_chat_command.assert_called_once_with(prompt="from-pipe", interactive=False)
        assert result.exit_code == 0


class TestCLIModuleExports:
    """Tests for CLI module exports."""

    def test_cli_exports_app(self) -> None:
        """Test app is exported from CLI module."""
        from yolo_developer.cli import app as cli_app

        assert cli_app is not None

    def test_cli_exports_display_functions(self) -> None:
        """Test display functions are exported from CLI module."""
        from yolo_developer.cli import (
            coming_soon,
            console,
            create_table,
            error_panel,
            info_panel,
            success_panel,
            warning_panel,
        )

        assert callable(success_panel)
        assert callable(error_panel)
        assert callable(info_panel)
        assert callable(warning_panel)
        assert callable(coming_soon)
        assert callable(create_table)
        assert console is not None
