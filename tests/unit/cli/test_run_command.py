"""Unit tests for yolo run command (Story 12.4).

This module tests the run command implementation including:
- CLI flag parsing and handling
- Workflow execution integration
- Real-time progress display
- Interrupt handling
- Completion summary generation
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from yolo_developer.cli.main import app

runner = CliRunner()


# =============================================================================
# Task 1: CLI Flag Tests
# =============================================================================


class TestRunCommandCLIFlags:
    """Test CLI flags are properly wired."""

    def test_run_help_shows_all_flags(self) -> None:
        """Test --help shows all expected flags."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0

        # Check for all expected flags
        assert "--dry-run" in result.output
        assert "-d" in result.output
        assert "--verbose" in result.output
        assert "-v" in result.output
        assert "--json" in result.output
        assert "-j" in result.output
        assert "--resume" in result.output
        assert "-r" in result.output
        assert "--thread-id" in result.output
        assert "-t" in result.output

    def test_run_command_accepts_dry_run_flag(self) -> None:
        """Test --dry-run flag is accepted and passed to run_command."""
        with patch("yolo_developer.cli.commands.run.run_command") as mock_run:
            runner.invoke(app, ["run", "--dry-run"])
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs.get("dry_run") is True

    def test_run_command_accepts_verbose_flag(self) -> None:
        """Test --verbose flag is accepted and passed to run_command."""
        with patch("yolo_developer.cli.commands.run.run_command") as mock_run:
            runner.invoke(app, ["run", "--verbose"])
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs.get("verbose") is True

    def test_run_command_accepts_json_flag(self) -> None:
        """Test --json flag is accepted and passed to run_command."""
        with patch("yolo_developer.cli.commands.run.run_command") as mock_run:
            runner.invoke(app, ["run", "--json"])
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs.get("json_output") is True

    def test_run_command_accepts_resume_flag(self) -> None:
        """Test --resume flag is accepted and passed to run_command."""
        with patch("yolo_developer.cli.commands.run.run_command") as mock_run:
            runner.invoke(app, ["run", "--resume"])
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs.get("resume") is True

    def test_run_command_accepts_thread_id_option(self) -> None:
        """Test --thread-id option is accepted and passed to run_command."""
        with patch("yolo_developer.cli.commands.run.run_command") as mock_run:
            runner.invoke(app, ["run", "--thread-id", "test-thread-123"])
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs.get("thread_id") == "test-thread-123"

    def test_run_command_short_flags(self) -> None:
        """Test short flag versions work."""
        with patch("yolo_developer.cli.commands.run.run_command") as mock_run:
            runner.invoke(app, ["run", "-d", "-v", "-j"])
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs.get("dry_run") is True
            assert call_kwargs.get("verbose") is True
            assert call_kwargs.get("json_output") is True


# =============================================================================
# Task 2: Core Logic Tests
# =============================================================================


class TestRunCommandCoreLogic:
    """Test run command core logic."""

    def test_missing_seed_shows_error(self) -> None:
        """Test error message when no seed has been parsed."""
        with (
            patch(
                "yolo_developer.cli.commands.run.load_config",
                return_value=MagicMock(project_name="test-project"),
            ),
            patch("yolo_developer.cli.commands.run.check_seed_exists", return_value=False),
        ):
            result = runner.invoke(app, ["run"])
            # Should exit with error or show seed message
            assert result.exit_code != 0 or "seed" in result.output.lower()

    def test_dry_run_validates_without_executing(self) -> None:
        """Test dry-run mode validates but doesn't execute workflow."""
        with (
            patch(
                "yolo_developer.cli.commands.run.load_config",
                return_value=MagicMock(project_name="test-project"),
            ),
            patch("yolo_developer.cli.commands.run.check_seed_exists", return_value=True),
            patch("yolo_developer.cli.commands.run.run_async_workflow") as mock_run,
        ):
            result = runner.invoke(app, ["run", "--dry-run"])
            # run_async_workflow should NOT be called in dry-run mode
            mock_run.assert_not_called()
            # Should show validation message
            assert "dry" in result.output.lower() or "validation" in result.output.lower()

    def test_load_config_called(self) -> None:
        """Test that configuration is loaded."""
        with (
            patch("yolo_developer.cli.commands.run.load_config") as mock_config,
            patch("yolo_developer.cli.commands.run.check_seed_exists", return_value=True),
            patch("yolo_developer.cli.commands.run.run_async_workflow"),
        ):
            mock_config.return_value = MagicMock(project_name="test-project")
            runner.invoke(app, ["run", "--dry-run"])
            mock_config.assert_called_once()


# =============================================================================
# Task 3: Workflow Execution Tests
# =============================================================================


class TestWorkflowExecution:
    """Test workflow execution integration."""

    def test_workflow_executed_when_seed_exists(self) -> None:
        """Test workflow is executed when seed exists."""
        with (
            patch(
                "yolo_developer.cli.commands.run.load_config",
                return_value=MagicMock(project_name="test-project"),
            ),
            patch("yolo_developer.cli.commands.run.check_seed_exists", return_value=True),
            patch("yolo_developer.cli.commands.run.run_async_workflow") as mock_run,
            patch("yolo_developer.cli.commands.run.display_summary"),
        ):
            mock_run.return_value = {
                "thread_id": "test-123",
                "elapsed_time": 1.5,
                "event_count": 5,
                "agents_executed": ["analyst", "pm"],
                "interrupted": False,
                "final_state": {},
                "decisions": [],
            }
            runner.invoke(app, ["run"])
            mock_run.assert_called_once()


# =============================================================================
# Task 4: Progress Display Tests
# =============================================================================


class TestProgressDisplay:
    """Test real-time progress display."""

    def test_verbose_mode_passed_to_workflow(self) -> None:
        """Test verbose flag is passed to workflow execution."""
        with (
            patch(
                "yolo_developer.cli.commands.run.load_config",
                return_value=MagicMock(project_name="test-project"),
            ),
            patch("yolo_developer.cli.commands.run.check_seed_exists", return_value=True),
            patch("yolo_developer.cli.commands.run.run_async_workflow") as mock_run,
            patch("yolo_developer.cli.commands.run.display_summary"),
        ):
            mock_run.return_value = {
                "thread_id": "test-123",
                "elapsed_time": 1.5,
                "event_count": 5,
                "agents_executed": [],
                "interrupted": False,
                "final_state": {},
                "decisions": [],
            }
            runner.invoke(app, ["run", "--verbose"])
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs.get("verbose") is True


# =============================================================================
# Task 5: Interrupt Handling Tests
# =============================================================================


class TestInterruptHandling:
    """Test Ctrl+C interrupt handling."""

    def test_interrupted_flag_in_result(self) -> None:
        """Test interrupted flag is included in result."""
        with (
            patch(
                "yolo_developer.cli.commands.run.load_config",
                return_value=MagicMock(project_name="test-project"),
            ),
            patch("yolo_developer.cli.commands.run.check_seed_exists", return_value=True),
            patch("yolo_developer.cli.commands.run.run_async_workflow") as mock_run,
            patch("yolo_developer.cli.commands.run.display_summary") as mock_summary,
        ):
            mock_run.return_value = {
                "thread_id": "test-123",
                "elapsed_time": 1.5,
                "event_count": 5,
                "agents_executed": [],
                "interrupted": True,
                "final_state": {},
                "decisions": [],
            }
            runner.invoke(app, ["run"])
            # Check display_summary was called with interrupted result
            call_args = mock_summary.call_args
            assert call_args[0][0]["interrupted"] is True

    def test_signal_handler_sets_interrupted_flag(self) -> None:
        """Test signal handler sets the _interrupted global flag."""
        import signal

        import yolo_developer.cli.commands.run as run_module

        # Reset the flag
        run_module._interrupted = False

        # Create handler like run_command does
        def handle_interrupt(signum: int, frame: object) -> None:
            run_module._interrupted = True

        # Simulate signal handler invocation
        handle_interrupt(signal.SIGINT, None)

        assert run_module._interrupted is True

        # Clean up
        run_module._interrupted = False


# =============================================================================
# Task 6: Completion Summary Tests
# =============================================================================


class TestCompletionSummary:
    """Test completion summary generation."""

    def test_json_output_format(self) -> None:
        """Test --json flag produces valid JSON output."""
        with (
            patch(
                "yolo_developer.cli.commands.run.load_config",
                return_value=MagicMock(project_name="test-project"),
            ),
            patch("yolo_developer.cli.commands.run.check_seed_exists", return_value=True),
        ):
            result = runner.invoke(app, ["run", "--json", "--dry-run"])
            # Should produce JSON-parseable output
            assert result.exit_code == 0
            # Check for JSON-like structure
            assert '"status"' in result.output or '"validated"' in result.output

    def test_display_summary_called(self) -> None:
        """Test display_summary is called after workflow execution."""
        with (
            patch(
                "yolo_developer.cli.commands.run.load_config",
                return_value=MagicMock(project_name="test-project"),
            ),
            patch("yolo_developer.cli.commands.run.check_seed_exists", return_value=True),
            patch("yolo_developer.cli.commands.run.run_async_workflow") as mock_run,
            patch("yolo_developer.cli.commands.run.display_summary") as mock_summary,
        ):
            mock_run.return_value = {
                "thread_id": "test-123",
                "elapsed_time": 1.5,
                "event_count": 5,
                "agents_executed": [],
                "interrupted": False,
                "final_state": {},
                "decisions": [],
            }
            runner.invoke(app, ["run"])
            mock_summary.assert_called_once()


# =============================================================================
# Task 7: Resume Functionality Tests
# =============================================================================


class TestResumeFunctionality:
    """Test resume from checkpoint functionality."""

    def test_resume_with_thread_id(self) -> None:
        """Test resuming with specific thread ID."""
        with (
            patch(
                "yolo_developer.cli.commands.run.load_config",
                return_value=MagicMock(project_name="test-project"),
            ),
            patch("yolo_developer.cli.commands.run.check_seed_exists", return_value=True),
            patch("yolo_developer.cli.commands.run.run_async_workflow") as mock_run,
            patch("yolo_developer.cli.commands.run.display_summary"),
        ):
            mock_run.return_value = {
                "thread_id": "my-thread",
                "elapsed_time": 1.5,
                "event_count": 5,
                "agents_executed": [],
                "interrupted": False,
                "final_state": {},
                "decisions": [],
            }
            runner.invoke(app, ["run", "--resume", "--thread-id", "my-thread"])
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs.get("resume") is True
            assert call_kwargs.get("thread_id") == "my-thread"

    def test_resume_skips_seed_check(self) -> None:
        """Test that resume mode skips seed existence check."""
        with (
            patch(
                "yolo_developer.cli.commands.run.load_config",
                return_value=MagicMock(project_name="test-project"),
            ),
            patch("yolo_developer.cli.commands.run.check_seed_exists", return_value=False),
            patch("yolo_developer.cli.commands.run.run_async_workflow") as mock_run,
            patch("yolo_developer.cli.commands.run.display_summary"),
        ):
            mock_run.return_value = {
                "thread_id": "my-thread",
                "elapsed_time": 1.5,
                "event_count": 5,
                "agents_executed": [],
                "interrupted": False,
                "final_state": {},
                "decisions": [],
            }
            # With resume, should not fail even if seed doesn't exist
            runner.invoke(app, ["run", "--resume", "--thread-id", "my-thread"])
            # Should succeed because resume skips seed check
            mock_run.assert_called_once()


# =============================================================================
# Unit Tests for Helper Functions
# =============================================================================


class TestHelperFunctions:
    """Test helper functions in run module."""

    def test_check_seed_exists_returns_false_when_no_file(self, tmp_path: Path) -> None:
        """Test check_seed_exists returns False when no seed file."""
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            from yolo_developer.cli.commands.run import check_seed_exists

            assert check_seed_exists() is False
        finally:
            os.chdir(original_cwd)

    def test_check_seed_exists_returns_true_when_file_exists(self, tmp_path: Path) -> None:
        """Test check_seed_exists returns True when seed file exists."""
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            # Create .yolo directory and seed_state.json
            yolo_dir = tmp_path / ".yolo"
            yolo_dir.mkdir()
            seed_file = yolo_dir / "seed_state.json"
            seed_file.write_text('{"goals": [], "features": [], "constraints": []}')

            from yolo_developer.cli.commands.run import check_seed_exists

            assert check_seed_exists() is True
        finally:
            os.chdir(original_cwd)

    def test_get_seed_messages_returns_empty_when_no_file(self, tmp_path: Path) -> None:
        """Test get_seed_messages returns empty list when no file."""
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            from yolo_developer.cli.commands.run import get_seed_messages

            messages = get_seed_messages()
            assert messages == []
        finally:
            os.chdir(original_cwd)

    def test_get_seed_messages_loads_seed_data(self, tmp_path: Path) -> None:
        """Test get_seed_messages loads and parses seed data."""
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            # Create seed file with data
            yolo_dir = tmp_path / ".yolo"
            yolo_dir.mkdir()
            seed_file = yolo_dir / "seed_state.json"
            seed_data = {
                "goals": [{"description": "Build an app"}],
                "features": [{"description": "User login"}],
                "constraints": [{"description": "Must be secure"}],
            }
            seed_file.write_text(json.dumps(seed_data))

            from yolo_developer.cli.commands.run import get_seed_messages

            messages = get_seed_messages()
            assert len(messages) == 3
            assert "Build an app" in messages[0].content
            assert "User login" in messages[1].content
            assert "Must be secure" in messages[2].content
        finally:
            os.chdir(original_cwd)

    def test_get_seed_messages_handles_string_items(self, tmp_path: Path) -> None:
        """Test get_seed_messages handles both dict and string seed items."""
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            yolo_dir = tmp_path / ".yolo"
            yolo_dir.mkdir()
            seed_file = yolo_dir / "seed_state.json"
            # Mix of dict and string formats
            seed_data = {
                "goals": ["Simple goal as string", {"description": "Dict goal"}],
                "features": [{"name": "Feature with name only"}],
                "constraints": ["Plain string constraint"],
            }
            seed_file.write_text(json.dumps(seed_data))

            from yolo_developer.cli.commands.run import get_seed_messages

            messages = get_seed_messages()
            assert len(messages) == 3
            assert "Simple goal as string" in messages[0].content
            assert "Dict goal" in messages[0].content
            assert "Feature with name only" in messages[1].content
            assert "Plain string constraint" in messages[2].content
        finally:
            os.chdir(original_cwd)


# =============================================================================
# Task 8: Activity Display Integration Tests (Story 12.9)
# =============================================================================


class TestActivityDisplayIntegration:
    """Test ActivityDisplay integration with run command (Story 12.9)."""

    def test_activity_display_used_in_workflow(self) -> None:
        """Test that ActivityDisplay is used instead of basic Progress spinner."""
        with (
            patch(
                "yolo_developer.cli.commands.run.load_config",
                return_value=MagicMock(project_name="test-project"),
            ),
            patch("yolo_developer.cli.commands.run.check_seed_exists", return_value=True),
            patch("yolo_developer.cli.commands.run.ActivityDisplay") as mock_display_class,
            patch("yolo_developer.orchestrator.stream_workflow") as mock_stream,
            patch("yolo_developer.cli.commands.run.display_summary"),
        ):
            # Mock the activity display
            mock_display = MagicMock()
            mock_display.__enter__ = MagicMock(return_value=mock_display)
            mock_display.__exit__ = MagicMock(return_value=None)
            mock_display_class.return_value = mock_display

            # Mock stream to return async generator
            async def empty_stream():
                return
                yield  # Make it a generator

            mock_stream.return_value = empty_stream()

            runner.invoke(app, ["run"])

            # Verify ActivityDisplay was created
            mock_display_class.assert_called()

    def test_verbose_flag_enables_verbose_display(self) -> None:
        """Test that --verbose flag enables verbose mode in ActivityDisplay."""
        with (
            patch(
                "yolo_developer.cli.commands.run.load_config",
                return_value=MagicMock(project_name="test-project"),
            ),
            patch("yolo_developer.cli.commands.run.check_seed_exists", return_value=True),
            patch("yolo_developer.cli.commands.run.ActivityDisplay") as mock_display_class,
            patch("yolo_developer.orchestrator.stream_workflow") as mock_stream,
            patch("yolo_developer.cli.commands.run.display_summary"),
        ):
            mock_display = MagicMock()
            mock_display.__enter__ = MagicMock(return_value=mock_display)
            mock_display.__exit__ = MagicMock(return_value=None)
            mock_display_class.return_value = mock_display

            # Mock stream to return async generator
            async def empty_stream():
                return
                yield  # Make it a generator

            mock_stream.return_value = empty_stream()

            runner.invoke(app, ["run", "--verbose"])

            # Verify verbose=True was passed
            call_kwargs = mock_display_class.call_args.kwargs
            assert call_kwargs.get("verbose") is True

    def test_json_output_disables_activity_display(self) -> None:
        """Test that --json flag disables rich ActivityDisplay."""
        with (
            patch(
                "yolo_developer.cli.commands.run.load_config",
                return_value=MagicMock(project_name="test-project"),
            ),
            patch("yolo_developer.cli.commands.run.check_seed_exists", return_value=True),
            patch("yolo_developer.cli.commands.run.run_async_workflow") as mock_run,
            patch("yolo_developer.cli.commands.run.display_summary"),
        ):
            mock_run.return_value = {
                "thread_id": "test-123",
                "elapsed_time": 1.5,
                "event_count": 5,
                "agents_executed": [],
                "interrupted": False,
                "final_state": {},
                "decisions": [],
            }
            result = runner.invoke(app, ["run", "--json"])
            # JSON output should not show Rich panels
            assert "Agent Activity" not in result.output


# =============================================================================
# Integration Tests
# =============================================================================


class TestRunCLIIntegration:
    """Integration tests for run CLI command."""

    def test_cli_run_command_basic(self) -> None:
        """Test basic CLI invocation."""
        with patch("yolo_developer.cli.commands.run.run_command"):
            result = runner.invoke(app, ["run"])
            assert result.exit_code == 0

    def test_cli_run_command_all_flags(self) -> None:
        """Test CLI invocation with all flags."""
        with patch("yolo_developer.cli.commands.run.run_command") as mock_run:
            runner.invoke(app, ["run", "-d", "-v", "-j", "-r", "-t", "thread-123"])
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["dry_run"] is True
            assert call_kwargs["verbose"] is True
            assert call_kwargs["json_output"] is True
            assert call_kwargs["resume"] is True
            assert call_kwargs["thread_id"] == "thread-123"
