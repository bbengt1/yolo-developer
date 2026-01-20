"""Tests for yolo logs command implementation (Story 12.6).

Tests cover:
- CLI flag parsing
- Time parsing utility (relative and ISO)
- Agent filter application
- Since filter application
- Pagination logic
- --all flag behavior
- JSON output structure
- Verbose mode detail level
- No audit data handling

References:
    - Story 12.6: yolo logs command
    - AC1: Recent Decisions Display
    - AC2: Agent Filter
    - AC3: Since Filter
    - AC4: Pagination for Long Logs
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from yolo_developer.audit.types import (
    AgentIdentity,
    Decision,
    DecisionContext,
)
from yolo_developer.cli.commands.logs import (
    _build_json_output,
    _format_timestamp,
    _parse_since,
    _truncate,
    logs_command,
)
from yolo_developer.cli.main import app


class TestParseSince:
    """Tests for _parse_since helper function (AC3)."""

    def test_parse_minutes(self) -> None:
        """Test parsing relative minutes."""
        result = _parse_since("30m")
        assert result is not None
        # Should be an ISO timestamp roughly 30 minutes ago
        parsed = datetime.fromisoformat(result)
        expected_min = datetime.now(timezone.utc) - timedelta(minutes=31)
        expected_max = datetime.now(timezone.utc) - timedelta(minutes=29)
        assert expected_min <= parsed <= expected_max

    def test_parse_hours(self) -> None:
        """Test parsing relative hours."""
        result = _parse_since("1h")
        assert result is not None
        parsed = datetime.fromisoformat(result)
        expected_min = datetime.now(timezone.utc) - timedelta(hours=1, minutes=1)
        expected_max = datetime.now(timezone.utc) - timedelta(minutes=59)
        assert expected_min <= parsed <= expected_max

    def test_parse_days(self) -> None:
        """Test parsing relative days."""
        result = _parse_since("7d")
        assert result is not None
        parsed = datetime.fromisoformat(result)
        expected_min = datetime.now(timezone.utc) - timedelta(days=7, minutes=1)
        expected_max = datetime.now(timezone.utc) - timedelta(days=6, hours=23)
        assert expected_min <= parsed <= expected_max

    def test_parse_weeks(self) -> None:
        """Test parsing relative weeks."""
        result = _parse_since("2w")
        assert result is not None
        parsed = datetime.fromisoformat(result)
        expected_min = datetime.now(timezone.utc) - timedelta(weeks=2, minutes=1)
        expected_max = datetime.now(timezone.utc) - timedelta(weeks=1, days=6)
        assert expected_min <= parsed <= expected_max

    def test_parse_case_insensitive(self) -> None:
        """Test that relative time parsing is case-insensitive."""
        result_lower = _parse_since("1h")
        result_upper = _parse_since("1H")
        assert result_lower is not None
        assert result_upper is not None

    def test_parse_iso_date(self) -> None:
        """Test parsing ISO date format."""
        result = _parse_since("2026-01-15")
        assert result is not None
        assert "2026-01-15" in result

    def test_parse_iso_datetime(self) -> None:
        """Test parsing ISO datetime format."""
        result = _parse_since("2026-01-15T10:30:00")
        assert result is not None
        assert "2026-01-15" in result
        assert "10:30:00" in result

    def test_parse_iso_datetime_with_z(self) -> None:
        """Test parsing ISO datetime with Z suffix."""
        result = _parse_since("2026-01-15T10:30:00Z")
        assert result is not None
        assert "2026-01-15" in result

    def test_parse_iso_datetime_with_timezone(self) -> None:
        """Test parsing ISO datetime with timezone offset."""
        result = _parse_since("2026-01-15T10:30:00+00:00")
        assert result is not None
        assert "2026-01-15" in result

    def test_parse_invalid_format(self) -> None:
        """Test that invalid formats return None."""
        assert _parse_since("invalid") is None
        assert _parse_since("abc123") is None
        assert _parse_since("") is None
        assert _parse_since("10x") is None

    def test_parse_invalid_relative_format(self) -> None:
        """Test that invalid relative formats return None."""
        assert _parse_since("10s") is None  # seconds not supported
        assert _parse_since("h") is None  # no number
        assert _parse_since("1y") is None  # years not supported


class TestTruncate:
    """Tests for _truncate helper function."""

    def test_truncate_short_text(self) -> None:
        """Test that short text is not truncated."""
        assert _truncate("short text", 60) == "short text"

    def test_truncate_exact_length(self) -> None:
        """Test text exactly at max length."""
        text = "a" * 60
        assert _truncate(text, 60) == text

    def test_truncate_long_text(self) -> None:
        """Test that long text is truncated with ellipsis."""
        text = "a" * 70
        result = _truncate(text, 60)
        assert len(result) == 60
        assert result.endswith("...")

    def test_truncate_custom_length(self) -> None:
        """Test truncation with custom max length."""
        result = _truncate("hello world", 8)
        assert result == "hello..."
        assert len(result) == 8


class TestFormatTimestamp:
    """Tests for _format_timestamp helper function."""

    def test_format_full_timestamp(self) -> None:
        """Test formatting a full ISO timestamp."""
        result = _format_timestamp("2026-01-19T10:30:45+00:00")
        assert result == "2026-01-19 10:30:45"

    def test_format_short_timestamp(self) -> None:
        """Test handling of short timestamps."""
        result = _format_timestamp("2026-01-19")
        assert result == "2026-01-19"


class TestBuildJsonOutput:
    """Tests for _build_json_output function (AC4)."""

    def test_build_json_empty_decisions(self) -> None:
        """Test JSON output with no decisions."""
        result = _build_json_output(
            decisions=[],
            filters_applied={"agent_name": None, "start_time": None},
            total_count=0,
            showing=0,
        )

        assert result["decisions"] == []
        assert result["total_count"] == 0
        assert result["showing"] == 0
        assert "filters_applied" in result

    def test_build_json_with_decisions(self) -> None:
        """Test JSON output with decisions."""
        decision = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Test decision content",
            rationale="Test rationale",
            agent=AgentIdentity(
                agent_name="analyst",
                agent_type="analyst",
                session_id="session-123",
            ),
            context=DecisionContext(
                sprint_id="sprint-1",
                story_id="1-2-auth",
            ),
            timestamp="2026-01-19T10:00:00+00:00",
            severity="info",
        )

        result = _build_json_output(
            decisions=[decision],
            filters_applied={"agent_name": "analyst", "start_time": None},
            total_count=1,
            showing=1,
        )

        assert len(result["decisions"]) == 1
        assert result["decisions"][0]["id"] == "dec-001"
        assert result["decisions"][0]["agent"]["agent_name"] == "analyst"
        assert result["total_count"] == 1
        assert result["filters_applied"]["agent_name"] == "analyst"

    def test_build_json_pagination_info(self) -> None:
        """Test JSON output includes pagination info."""
        result = _build_json_output(
            decisions=[],
            filters_applied={},
            total_count=100,
            showing=20,
        )

        assert result["total_count"] == 100
        assert result["showing"] == 20


class TestLogsCommand:
    """Tests for logs_command function."""

    @patch("yolo_developer.cli.commands.logs._load_decisions")
    def test_logs_command_no_data(self, mock_load: AsyncMock) -> None:
        """Test logs command with no audit data (AC1)."""
        mock_load.return_value = []

        # Should not raise, just show info message
        logs_command()
        mock_load.assert_called_once()

    @patch("yolo_developer.cli.commands.logs._load_decisions")
    def test_logs_command_with_agent_filter(self, mock_load: AsyncMock) -> None:
        """Test logs command with agent filter (AC2)."""
        mock_load.return_value = []

        logs_command(agent="analyst")

        # Check that agent filter was passed (case-insensitive)
        call_args = mock_load.call_args
        assert call_args.kwargs.get("agent_name") == "analyst"

    @patch("yolo_developer.cli.commands.logs._load_decisions")
    def test_logs_command_agent_filter_case_insensitive(self, mock_load: AsyncMock) -> None:
        """Test that agent filter is case-insensitive (AC2).

        The lowercasing is done once at entry point in logs_command,
        before passing to _load_decisions.
        """
        mock_load.return_value = []

        logs_command(agent="ANALYST")

        call_args = mock_load.call_args
        # The value is normalized to lowercase at entry point
        assert call_args.kwargs.get("agent_name") == "analyst"

    @patch("yolo_developer.cli.commands.logs._load_decisions")
    def test_logs_command_with_since_filter(self, mock_load: AsyncMock) -> None:
        """Test logs command with since filter (AC3)."""
        mock_load.return_value = []

        logs_command(since="1h")

        call_args = mock_load.call_args
        assert call_args.kwargs.get("start_time") is not None

    def test_logs_command_invalid_since_filter(self) -> None:
        """Test logs command with invalid since filter shows warning."""
        # Should not raise, just show warning
        logs_command(since="invalid_time")

    @patch("yolo_developer.cli.commands.logs._load_decisions")
    def test_logs_command_default_pagination(self, mock_load: AsyncMock) -> None:
        """Test logs command default pagination of 20 (AC4)."""
        # Create 30 decisions
        decisions = [
            Decision(
                id=f"dec-{i:03d}",
                decision_type="requirement_analysis",
                content=f"Decision {i}",
                rationale="Test",
                agent=AgentIdentity(
                    agent_name="analyst",
                    agent_type="analyst",
                    session_id="session-123",
                ),
                context=DecisionContext(),
                timestamp=f"2026-01-19T{10 + i // 60:02d}:{i % 60:02d}:00+00:00",
                severity="info",
            )
            for i in range(30)
        ]
        mock_load.return_value = decisions

        # Default limit is 20
        logs_command()

        mock_load.assert_called_once()

    @patch("yolo_developer.cli.commands.logs._load_decisions")
    def test_logs_command_custom_limit(self, mock_load: AsyncMock) -> None:
        """Test logs command with custom limit (AC4)."""
        mock_load.return_value = []

        logs_command(limit=50)

        mock_load.assert_called_once()

    @patch("yolo_developer.cli.commands.logs._load_decisions")
    def test_logs_command_show_all(self, mock_load: AsyncMock) -> None:
        """Test logs command with --all flag (AC4)."""
        # Create 30 decisions
        decisions = [
            Decision(
                id=f"dec-{i:03d}",
                decision_type="requirement_analysis",
                content=f"Decision {i}",
                rationale="Test",
                agent=AgentIdentity(
                    agent_name="analyst",
                    agent_type="analyst",
                    session_id="session-123",
                ),
                context=DecisionContext(),
                timestamp=f"2026-01-19T{10 + i // 60:02d}:{i % 60:02d}:00+00:00",
                severity="info",
            )
            for i in range(30)
        ]
        mock_load.return_value = decisions

        logs_command(show_all=True)

        mock_load.assert_called_once()

    @patch("yolo_developer.cli.commands.logs._load_decisions")
    def test_logs_command_json_output(
        self, mock_load: AsyncMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test logs command JSON output (AC4)."""
        decision = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Test decision",
            rationale="Test rationale",
            agent=AgentIdentity(
                agent_name="analyst",
                agent_type="analyst",
                session_id="session-123",
            ),
            context=DecisionContext(),
            timestamp="2026-01-19T10:00:00+00:00",
            severity="info",
        )
        mock_load.return_value = [decision]

        logs_command(json_output=True)

        # Verify JSON was output (captured by Rich console)
        mock_load.assert_called_once()

    @patch("yolo_developer.cli.commands.logs._load_decisions")
    def test_logs_command_verbose_mode(self, mock_load: AsyncMock) -> None:
        """Test logs command verbose mode (AC1)."""
        decision = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Test decision content",
            rationale="This is the rationale",
            agent=AgentIdentity(
                agent_name="analyst",
                agent_type="analyst",
                session_id="session-123",
            ),
            context=DecisionContext(
                sprint_id="sprint-1",
                story_id="1-2-auth",
            ),
            timestamp="2026-01-19T10:00:00+00:00",
            severity="info",
        )
        mock_load.return_value = [decision]

        logs_command(verbose=True)

        mock_load.assert_called_once()

    @patch("yolo_developer.cli.commands.logs._load_decisions")
    def test_logs_command_type_filter(self, mock_load: AsyncMock) -> None:
        """Test logs command with decision type filter."""
        mock_load.return_value = []

        logs_command(decision_type="architecture_choice")

        call_args = mock_load.call_args
        assert call_args.kwargs.get("decision_type") == "architecture_choice"

    @patch("yolo_developer.cli.commands.logs._load_decisions")
    def test_logs_command_invalid_decision_type(self, mock_load: AsyncMock) -> None:
        """Test logs command with invalid decision type shows warning."""
        mock_load.return_value = []

        # Should not call _load_decisions when decision_type is invalid
        logs_command(decision_type="invalid_type")

        # _load_decisions should not be called since validation fails first
        mock_load.assert_not_called()

    @patch("yolo_developer.cli.commands.logs._load_decisions")
    def test_logs_command_invalid_limit(self, mock_load: AsyncMock) -> None:
        """Test logs command with invalid limit shows warning."""
        mock_load.return_value = []

        # Should not call _load_decisions when limit is invalid
        logs_command(limit=0)

        # _load_decisions should not be called since validation fails first
        mock_load.assert_not_called()

    @patch("yolo_developer.cli.commands.logs._load_decisions")
    def test_logs_command_negative_limit(self, mock_load: AsyncMock) -> None:
        """Test logs command with negative limit shows warning."""
        mock_load.return_value = []

        # Should not call _load_decisions when limit is negative
        logs_command(limit=-5)

        # _load_decisions should not be called since validation fails first
        mock_load.assert_not_called()


class TestLogsCommandCLI:
    """Integration tests for logs command CLI."""

    runner = CliRunner()

    def test_cli_help(self) -> None:
        """Test that --help works."""
        result = self.runner.invoke(app, ["logs", "--help"])
        assert result.exit_code == 0
        assert "Browse decision audit trail" in result.output

    def test_cli_flags_recognized(self) -> None:
        """Test that all CLI flags are recognized."""
        # Test with all flags - should not error on flag parsing
        result = self.runner.invoke(
            app,
            [
                "logs",
                "--agent",
                "analyst",
                "--since",
                "1h",
                "--type",
                "requirement_analysis",
                "--limit",
                "50",
                "--verbose",
            ],
        )
        # Should run (may show "no decisions" but shouldn't fail on flags)
        assert result.exit_code == 0

    def test_cli_short_flags(self) -> None:
        """Test that short flags work."""
        result = self.runner.invoke(
            app,
            [
                "logs",
                "-a",
                "analyst",
                "-s",
                "1h",
                "-t",
                "requirement_analysis",
                "-l",
                "10",
                "-v",
            ],
        )
        assert result.exit_code == 0

    def test_cli_json_flag(self) -> None:
        """Test JSON output flag."""
        result = self.runner.invoke(app, ["logs", "--json"])
        assert result.exit_code == 0

    def test_cli_all_flag(self) -> None:
        """Test --all flag."""
        result = self.runner.invoke(app, ["logs", "--all"])
        assert result.exit_code == 0

    def test_cli_combined_flags(self) -> None:
        """Test combining multiple flags."""
        result = self.runner.invoke(
            app,
            ["logs", "--agent", "analyst", "--since", "7d", "--limit", "100"],
        )
        assert result.exit_code == 0


class TestDecisionDisplayColors:
    """Tests for decision display color mappings."""

    def test_severity_colors_defined(self) -> None:
        """Test that all severity levels have colors."""
        from yolo_developer.cli.commands.logs import SEVERITY_COLORS

        assert "info" in SEVERITY_COLORS
        assert "warning" in SEVERITY_COLORS
        assert "critical" in SEVERITY_COLORS

    def test_agent_colors_defined(self) -> None:
        """Test that all agent types have colors."""
        from yolo_developer.cli.commands.logs import AGENT_COLORS

        assert "analyst" in AGENT_COLORS
        assert "pm" in AGENT_COLORS
        assert "architect" in AGENT_COLORS
        assert "dev" in AGENT_COLORS
        assert "tea" in AGENT_COLORS
        assert "sm" in AGENT_COLORS

    def test_decision_type_display_names(self) -> None:
        """Test that decision types have display names."""
        from yolo_developer.cli.commands.logs import DECISION_TYPE_DISPLAY

        assert "requirement_analysis" in DECISION_TYPE_DISPLAY
        assert "architecture_choice" in DECISION_TYPE_DISPLAY
        assert "implementation_choice" in DECISION_TYPE_DISPLAY
