"""Tests for placeholder command modules (Story 12.1).

Tests cover:
- Each placeholder command module calls coming_soon
- Each placeholder command module logs debug message

Note: run_command tests moved to test_run_command.py after Story 12.4 implementation.
Note: status_command tests moved to test_status_command.py after Story 12.5 implementation.
Note: logs_command tests moved to test_logs_command.py after Story 12.6 implementation.
Note: tune_command tests moved to test_tune_command.py after Story 12.7 implementation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestConfigCommand:
    """Tests for config command module."""

    @patch("yolo_developer.cli.commands.config.coming_soon")
    @patch("yolo_developer.cli.commands.config.logger")
    def test_config_command_calls_coming_soon(
        self, mock_logger: MagicMock, mock_coming_soon: MagicMock
    ) -> None:
        """Test config_command calls coming_soon with 'config'."""
        from yolo_developer.cli.commands.config import config_command

        config_command()

        mock_coming_soon.assert_called_once_with("config")

    @patch("yolo_developer.cli.commands.config.coming_soon")
    @patch("yolo_developer.cli.commands.config.logger")
    def test_config_command_logs_invocation(
        self, mock_logger: MagicMock, mock_coming_soon: MagicMock
    ) -> None:
        """Test config_command logs debug message."""
        from yolo_developer.cli.commands.config import config_command

        config_command()

        mock_logger.debug.assert_called_once_with("config_command_invoked")


class TestCommandModuleExports:
    """Tests for commands module exports."""

    def test_all_commands_exported(self) -> None:
        """Test all command functions are exported from commands module."""
        from yolo_developer.cli.commands import (
            config_command,
            init_command,
            logs_command,
            run_command,
            seed_command,
            status_command,
            tune_command,
        )

        assert callable(config_command)
        assert callable(init_command)
        assert callable(logs_command)
        assert callable(run_command)
        assert callable(seed_command)
        assert callable(status_command)
        assert callable(tune_command)
