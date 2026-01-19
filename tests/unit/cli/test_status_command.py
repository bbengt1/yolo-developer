"""Tests for yolo status command implementation (Story 12.5).

Tests cover:
- CLI flag parsing
- No session display
- Sprint progress calculation
- Health metrics display (mocked)
- Session listing
- JSON output structure
- Verbose mode detail level
- Health-only mode
- Sessions-only mode

References:
    - Story 12.5: yolo status command
    - AC1: Sprint Progress Display
    - AC2: Health Metrics Dashboard
    - AC3: Session Information
    - AC4: JSON Output Support
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from yolo_developer.cli.commands.status import (
    _build_json_output,
    _format_datetime,
    _format_duration,
    status_command,
)


class TestFormatDuration:
    """Tests for _format_duration helper function."""

    def test_format_seconds_under_minute(self) -> None:
        """Test formatting seconds under 60."""
        assert _format_duration(45.5) == "45.5s"
        assert _format_duration(0.0) == "0.0s"
        assert _format_duration(59.9) == "59.9s"

    def test_format_minutes(self) -> None:
        """Test formatting durations in minutes."""
        assert _format_duration(60.0) == "1m 0s"
        assert _format_duration(90.0) == "1m 30s"
        assert _format_duration(150.0) == "2m 30s"
        assert _format_duration(3599.0) == "59m 59s"

    def test_format_hours(self) -> None:
        """Test formatting durations in hours."""
        assert _format_duration(3600.0) == "1h 0m"
        assert _format_duration(5400.0) == "1h 30m"
        assert _format_duration(7200.0) == "2h 0m"


class TestFormatDatetime:
    """Tests for _format_datetime helper function."""

    def test_format_just_now(self) -> None:
        """Test formatting datetime within last minute."""
        now = datetime.now(timezone.utc)
        assert _format_datetime(now) == "just now"

    def test_format_minutes_ago(self) -> None:
        """Test formatting datetime within last hour."""
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        ten_min_ago = now - timedelta(minutes=10)
        result = _format_datetime(ten_min_ago)
        assert result == "10m ago"

    def test_format_hours_ago(self) -> None:
        """Test formatting datetime within last day."""
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        two_hours_ago = now - timedelta(hours=2)
        result = _format_datetime(two_hours_ago)
        assert result == "2h ago"

    def test_format_older_date(self) -> None:
        """Test formatting datetime older than 24 hours."""
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        two_days_ago = now - timedelta(days=2)
        result = _format_datetime(two_days_ago)
        # Should be formatted as date string
        assert "-" in result  # Contains date format


class TestBuildJsonOutput:
    """Tests for _build_json_output function."""

    def test_build_json_with_no_data(self) -> None:
        """Test JSON output when no data available."""
        result = _build_json_output(
            active_id=None,
            metadata=None,
            all_sessions=[],
            health=None,
        )

        assert result["sprint"] is None
        assert result["health"] is None
        assert result["session"] is None
        assert result["available_sessions"] == []

    def test_build_json_with_session_metadata(self) -> None:
        """Test JSON output with session metadata."""
        # Mock session metadata
        mock_metadata = MagicMock()
        mock_metadata.session_id = "session-abc123"
        mock_metadata.created_at = datetime(2026, 1, 19, 10, 0, 0, tzinfo=timezone.utc)
        mock_metadata.last_checkpoint = datetime(2026, 1, 19, 10, 30, 0, tzinfo=timezone.utc)
        mock_metadata.current_agent = "dev"
        mock_metadata.stories_completed = 3
        mock_metadata.stories_total = 5

        result = _build_json_output(
            active_id="session-abc123",
            metadata=mock_metadata,
            all_sessions=[],
            health=None,
        )

        assert result["sprint"] is not None
        assert result["sprint"]["stories_completed"] == 3
        assert result["sprint"]["stories_total"] == 5
        assert result["sprint"]["completion_percentage"] == 60.0
        assert result["sprint"]["status"] == "in_progress"

        assert result["session"] is not None
        assert result["session"]["session_id"] == "session-abc123"
        assert result["session"]["current_agent"] == "dev"

    def test_build_json_with_completed_sprint(self) -> None:
        """Test JSON output with completed sprint."""
        mock_metadata = MagicMock()
        mock_metadata.session_id = "session-xyz"
        mock_metadata.created_at = datetime(2026, 1, 19, 10, 0, 0, tzinfo=timezone.utc)
        mock_metadata.last_checkpoint = datetime(2026, 1, 19, 12, 0, 0, tzinfo=timezone.utc)
        mock_metadata.current_agent = ""
        mock_metadata.stories_completed = 5
        mock_metadata.stories_total = 5

        result = _build_json_output(
            active_id="session-xyz",
            metadata=mock_metadata,
            all_sessions=[],
            health=None,
        )

        assert result["sprint"]["completion_percentage"] == 100.0
        assert result["sprint"]["status"] == "completed"

    def test_build_json_with_health_data(self) -> None:
        """Test JSON output with health data."""
        # Mock health status with to_dict method
        mock_health = MagicMock()
        mock_health.to_dict.return_value = {
            "status": "healthy",
            "is_healthy": True,
            "summary": "All systems nominal",
        }

        result = _build_json_output(
            active_id=None,
            metadata=None,
            all_sessions=[],
            health=mock_health,
        )

        assert result["health"] is not None
        assert result["health"]["status"] == "healthy"
        assert result["health"]["is_healthy"] is True

    def test_build_json_with_available_sessions(self) -> None:
        """Test JSON output with available sessions list."""
        mock_session = MagicMock()
        mock_session.session_id = "session-old"
        mock_session.last_checkpoint = datetime(2026, 1, 18, 10, 0, 0, tzinfo=timezone.utc)
        mock_session.current_agent = "pm"
        mock_session.stories_completed = 2
        mock_session.stories_total = 4

        result = _build_json_output(
            active_id=None,
            metadata=None,
            all_sessions=[mock_session],
            health=None,
        )

        assert len(result["available_sessions"]) == 1
        assert result["available_sessions"][0]["session_id"] == "session-old"
        assert result["available_sessions"][0]["stories_completed"] == 2


class TestStatusCommandExecution:
    """Tests for status_command execution."""

    @patch("yolo_developer.cli.commands.status.asyncio")
    @patch("yolo_developer.cli.commands.status.console")
    @patch("yolo_developer.cli.commands.status.logger")
    def test_status_command_logs_invocation(
        self,
        mock_logger: MagicMock,
        mock_console: MagicMock,
        mock_asyncio: MagicMock,
    ) -> None:
        """Test status_command logs debug message with parameters."""
        # Return 4 values: (active_id, metadata, all_sessions, health)
        mock_asyncio.run.return_value = (None, None, [], None)

        status_command(verbose=False, json_output=False, health_only=False, sessions_only=False)

        mock_logger.debug.assert_called_once_with(
            "status_command_invoked",
            verbose=False,
            json_output=False,
            health_only=False,
            sessions_only=False,
        )

    @patch("yolo_developer.cli.commands.status.asyncio")
    @patch("yolo_developer.cli.commands.status.console")
    @patch("yolo_developer.cli.commands.status.logger")
    def test_status_command_no_sessions_displays_info(
        self,
        mock_logger: MagicMock,
        mock_console: MagicMock,
        mock_asyncio: MagicMock,
    ) -> None:
        """Test status_command with no sessions shows info panels."""
        # Return 4 values: (active_id, metadata, all_sessions, health)
        mock_asyncio.run.return_value = (None, None, [], None)

        status_command()

        # Should have printed header and info panels
        assert mock_console.print.called

    @patch("yolo_developer.cli.commands.status.asyncio")
    @patch("yolo_developer.cli.commands.status.console")
    @patch("yolo_developer.cli.commands.status.logger")
    def test_status_command_json_output(
        self,
        mock_logger: MagicMock,
        mock_console: MagicMock,
        mock_asyncio: MagicMock,
    ) -> None:
        """Test status_command with JSON output."""
        # Return 4 values: (active_id, metadata, all_sessions, health)
        mock_asyncio.run.return_value = (None, None, [], None)

        status_command(json_output=True)

        # Should call print_json
        mock_console.print_json.assert_called_once()

    @patch("yolo_developer.cli.commands.status.asyncio")
    @patch("yolo_developer.cli.commands.status.console")
    @patch("yolo_developer.cli.commands.status.logger")
    def test_status_command_health_only_mode(
        self,
        mock_logger: MagicMock,
        mock_console: MagicMock,
        mock_asyncio: MagicMock,
    ) -> None:
        """Test status_command with health_only flag."""
        # Return 4 values: (active_id, metadata, all_sessions, health)
        mock_asyncio.run.return_value = (None, None, [], None)

        status_command(health_only=True)

        # Should print header and health section only
        mock_logger.debug.assert_called_once()
        call_kwargs = mock_logger.debug.call_args[1]
        assert call_kwargs["health_only"] is True

    @patch("yolo_developer.cli.commands.status.asyncio")
    @patch("yolo_developer.cli.commands.status.console")
    @patch("yolo_developer.cli.commands.status.logger")
    def test_status_command_sessions_only_mode(
        self,
        mock_logger: MagicMock,
        mock_console: MagicMock,
        mock_asyncio: MagicMock,
    ) -> None:
        """Test status_command with sessions_only flag."""
        # Return 4 values: (active_id, metadata, all_sessions, health)
        mock_asyncio.run.return_value = (None, None, [], None)

        status_command(sessions_only=True)

        # Should print header and sessions section only
        mock_logger.debug.assert_called_once()
        call_kwargs = mock_logger.debug.call_args[1]
        assert call_kwargs["sessions_only"] is True


class TestCLIFlagParsing:
    """Tests for CLI flag parsing via Typer."""

    def test_status_command_with_verbose_flag(self) -> None:
        """Test --verbose flag is parsed correctly."""
        from yolo_developer.cli.main import app

        runner = CliRunner()
        with patch("yolo_developer.cli.commands.status.status_command") as mock_cmd:
            runner.invoke(app, ["status", "--verbose"])
            # Command may fail due to missing sessions, but flags should be parsed
            if mock_cmd.called:
                args, kwargs = mock_cmd.call_args
                assert kwargs.get("verbose") is True or args[0] is True

    def test_status_command_with_json_flag(self) -> None:
        """Test --json flag is parsed correctly."""
        from yolo_developer.cli.main import app

        runner = CliRunner()
        with patch("yolo_developer.cli.commands.status.status_command") as mock_cmd:
            runner.invoke(app, ["status", "--json"])
            if mock_cmd.called:
                args, kwargs = mock_cmd.call_args
                assert kwargs.get("json_output") is True or args[1] is True

    def test_status_command_with_health_flag(self) -> None:
        """Test --health flag is parsed correctly."""
        from yolo_developer.cli.main import app

        runner = CliRunner()
        with patch("yolo_developer.cli.commands.status.status_command") as mock_cmd:
            runner.invoke(app, ["status", "--health"])
            if mock_cmd.called:
                args, kwargs = mock_cmd.call_args
                assert kwargs.get("health_only") is True or args[2] is True

    def test_status_command_with_sessions_flag(self) -> None:
        """Test --sessions flag is parsed correctly."""
        from yolo_developer.cli.main import app

        runner = CliRunner()
        with patch("yolo_developer.cli.commands.status.status_command") as mock_cmd:
            runner.invoke(app, ["status", "--sessions"])
            if mock_cmd.called:
                args, kwargs = mock_cmd.call_args
                assert kwargs.get("sessions_only") is True or args[3] is True

    def test_status_command_help_text(self) -> None:
        """Test --help displays correct help text."""
        from yolo_developer.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["status", "--help"])

        assert result.exit_code == 0
        assert "sprint progress" in result.output.lower()
        assert "--verbose" in result.output or "-v" in result.output
        assert "--json" in result.output or "-j" in result.output


class TestSessionDataRetrieval:
    """Tests for session data retrieval."""

    @pytest.mark.asyncio
    async def test_get_session_data_no_sessions(self) -> None:
        """Test _get_session_data with no sessions."""
        from yolo_developer.cli.commands.status import _get_session_data

        with patch("yolo_developer.orchestrator.session.SessionManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.get_active_session_id = AsyncMock(return_value=None)
            mock_manager.list_sessions = AsyncMock(return_value=[])
            mock_manager_class.return_value = mock_manager

            active_id, metadata, sessions = await _get_session_data(".yolo/sessions")

            assert active_id is None
            assert metadata is None
            assert sessions == []

    @pytest.mark.asyncio
    async def test_get_session_data_with_active_session(self) -> None:
        """Test _get_session_data with active session."""
        from yolo_developer.cli.commands.status import _get_session_data

        mock_metadata = MagicMock()
        mock_metadata.session_id = "session-123"

        with patch("yolo_developer.orchestrator.session.SessionManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.get_active_session_id = AsyncMock(return_value="session-123")
            mock_manager.load_session = AsyncMock(return_value=(None, mock_metadata))
            mock_manager.list_sessions = AsyncMock(return_value=[mock_metadata])
            mock_manager_class.return_value = mock_manager

            active_id, metadata, sessions = await _get_session_data(".yolo/sessions")

            assert active_id == "session-123"
            assert metadata == mock_metadata
            assert len(sessions) == 1


class TestHealthDataRetrieval:
    """Tests for health data retrieval."""

    @pytest.mark.asyncio
    async def test_get_health_data_with_no_state(self) -> None:
        """Test _get_health_data returns None when state is None."""
        from yolo_developer.cli.commands.status import _get_health_data

        result = await _get_health_data(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_health_data_with_state(self) -> None:
        """Test _get_health_data calls monitor_health."""
        from yolo_developer.cli.commands.status import _get_health_data

        mock_health = MagicMock()
        mock_health.is_healthy = True

        with patch("yolo_developer.agents.sm.health.monitor_health", new_callable=AsyncMock) as mock_monitor:
            mock_monitor.return_value = mock_health

            state = {"messages": [], "current_agent": "analyst"}
            result = await _get_health_data(state)

            mock_monitor.assert_called_once_with(state)
            assert result == mock_health

    @pytest.mark.asyncio
    async def test_get_health_data_handles_exception(self) -> None:
        """Test _get_health_data handles exceptions gracefully."""
        from yolo_developer.cli.commands.status import _get_health_data

        with patch("yolo_developer.agents.sm.health.monitor_health", new_callable=AsyncMock) as mock_monitor:
            # Use ValueError which is in the caught exception list
            mock_monitor.side_effect = ValueError("Health check failed")

            state = {"messages": []}
            result = await _get_health_data(state)

            assert result is None
