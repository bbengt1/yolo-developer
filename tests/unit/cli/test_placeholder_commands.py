"""Tests for placeholder command modules (Story 12.1).

Tests cover:
- Command module exports are available

Note: run_command tests moved to test_run_command.py after Story 12.4 implementation.
Note: status_command tests moved to test_status_command.py after Story 12.5 implementation.
Note: logs_command tests moved to test_logs_command.py after Story 12.6 implementation.
Note: tune_command tests moved to test_tune_command.py after Story 12.7 implementation.
Note: config_command tests moved to test_config_command.py after Story 12.8 implementation.
"""

from __future__ import annotations


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
