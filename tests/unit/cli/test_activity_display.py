"""Unit tests for the activity display module (Story 12.9).

Tests cover:
- ActivityDisplay class initialization and configuration
- Event handling and display updates
- Agent transition visualization
- Event throttling behavior
- Verbose vs normal mode output
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch


class TestActivityEvent:
    """Tests for ActivityEvent dataclass."""

    def test_activity_event_creation(self) -> None:
        """Test creating an ActivityEvent with all fields."""
        from yolo_developer.cli.activity import ActivityEvent

        event = ActivityEvent(
            event_type="start",
            agent="analyst",
            description="Analyzing requirements",
            timestamp=1234567890.0,
        )

        assert event.event_type == "start"
        assert event.agent == "analyst"
        assert event.description == "Analyzing requirements"
        assert event.timestamp == 1234567890.0
        assert event.details is None
        assert event.previous_agent is None

    def test_activity_event_with_optional_fields(self) -> None:
        """Test ActivityEvent with optional fields populated."""
        from yolo_developer.cli.activity import ActivityEvent

        event = ActivityEvent(
            event_type="transition",
            agent="pm",
            description="Transitioning to PM",
            timestamp=1234567890.0,
            details={"reason": "requirements_ready"},
            previous_agent="analyst",
        )

        assert event.event_type == "transition"
        assert event.details == {"reason": "requirements_ready"}
        assert event.previous_agent == "analyst"

    def test_activity_event_types(self) -> None:
        """Test all valid event types can be created."""
        from yolo_developer.cli.activity import ActivityEvent

        for event_type in ["start", "progress", "complete", "transition"]:
            event = ActivityEvent(
                event_type=event_type,  # type: ignore[arg-type]
                agent="test",
                description="test",
                timestamp=0.0,
            )
            assert event.event_type == event_type


class TestActivityDisplay:
    """Tests for ActivityDisplay class."""

    def test_activity_display_initialization(self) -> None:
        """Test ActivityDisplay initializes with correct defaults."""
        from yolo_developer.cli.activity import ActivityDisplay

        display = ActivityDisplay()

        assert display.current_agent == ""
        assert display.current_description == ""
        assert display.elapsed_time == 0.0
        assert display.event_count == 0
        assert display.transitions == []
        assert display.verbose is False
        assert display.refresh_rate == 4

    def test_activity_display_with_verbose(self) -> None:
        """Test ActivityDisplay with verbose mode enabled."""
        from yolo_developer.cli.activity import ActivityDisplay

        display = ActivityDisplay(verbose=True)

        assert display.verbose is True

    def test_activity_display_with_custom_refresh_rate(self) -> None:
        """Test ActivityDisplay with custom refresh rate."""
        from yolo_developer.cli.activity import ActivityDisplay

        display = ActivityDisplay(refresh_rate=10)

        assert display.refresh_rate == 10

    def test_update_agent_status(self) -> None:
        """Test updating agent status."""
        from yolo_developer.cli.activity import ActivityDisplay

        display = ActivityDisplay()
        display.update("analyst", "Analyzing requirements", 5.0)

        assert display.current_agent == "analyst"
        assert display.current_description == "Analyzing requirements"
        assert display.elapsed_time == 5.0

    def test_update_increments_event_count(self) -> None:
        """Test that update increments event count."""
        from yolo_developer.cli.activity import ActivityDisplay

        display = ActivityDisplay()
        assert display.event_count == 0

        display.update("analyst", "test", 1.0)
        assert display.event_count == 1

        display.update("analyst", "test2", 2.0)
        assert display.event_count == 2

    def test_add_transition(self) -> None:
        """Test adding agent transitions."""
        from yolo_developer.cli.activity import ActivityDisplay

        display = ActivityDisplay()
        display.add_transition("analyst", "pm", "requirements_ready", 10.0)

        assert len(display.transitions) == 1
        transition = display.transitions[0]
        assert transition["from_agent"] == "analyst"
        assert transition["to_agent"] == "pm"
        assert transition["reason"] == "requirements_ready"
        assert transition["elapsed"] == 10.0

    def test_multiple_transitions(self) -> None:
        """Test tracking multiple transitions."""
        from yolo_developer.cli.activity import ActivityDisplay

        display = ActivityDisplay()
        display.add_transition("analyst", "pm", "ready", 10.0)
        display.add_transition("pm", "dev", "stories_created", 20.0)

        assert len(display.transitions) == 2
        assert display.transitions[0]["from_agent"] == "analyst"
        assert display.transitions[1]["from_agent"] == "pm"


class TestActivityDisplayRendering:
    """Tests for ActivityDisplay rendering methods."""

    def test_render_returns_panel(self) -> None:
        """Test render returns a Rich Panel."""
        from rich.panel import Panel

        from yolo_developer.cli.activity import ActivityDisplay

        display = ActivityDisplay()
        display.update("analyst", "Testing", 5.0)

        result = display.render()

        assert isinstance(result, Panel)

    def test_render_includes_agent_name(self) -> None:
        """Test render output includes current agent name."""
        from io import StringIO

        from rich.console import Console

        from yolo_developer.cli.activity import ActivityDisplay

        display = ActivityDisplay()
        display.update("analyst", "Analyzing", 5.0)

        result = display.render()

        # Render to string to check content
        console = Console(file=StringIO(), force_terminal=True, width=80)
        console.print(result)
        output = console.file.getvalue()  # type: ignore[union-attr]

        assert "analyst" in output

    def test_render_includes_elapsed_time(self) -> None:
        """Test render output includes elapsed time."""
        from io import StringIO

        from rich.console import Console

        from yolo_developer.cli.activity import ActivityDisplay

        display = ActivityDisplay()
        display.update("analyst", "Testing", 65.5)  # 1:05.5

        result = display.render()

        # Render to string to check content
        console = Console(file=StringIO(), force_terminal=True, width=80)
        console.print(result)
        output = console.file.getvalue()  # type: ignore[union-attr]

        # Should show formatted time (01:05)
        assert "01:05" in output

    def test_render_verbose_shows_event_count(self) -> None:
        """Test verbose mode shows event count."""
        from io import StringIO

        from rich.console import Console

        from yolo_developer.cli.activity import ActivityDisplay

        display = ActivityDisplay(verbose=True)
        display.update("analyst", "Testing", 5.0)
        display.update("analyst", "Testing", 6.0)
        display.update("analyst", "Testing", 7.0)

        result = display.render()

        # Render to string to check content
        console = Console(file=StringIO(), force_terminal=True, width=80)
        console.print(result)
        output = console.file.getvalue()  # type: ignore[union-attr]

        # In verbose mode, should show events count
        assert "Events: 3" in output

    def test_render_shows_transitions(self) -> None:
        """Test render shows transitions when present."""
        from io import StringIO

        from rich.console import Console

        from yolo_developer.cli.activity import ActivityDisplay

        display = ActivityDisplay()
        display.add_transition("analyst", "pm", "ready", 10.0)
        display.update("pm", "Working", 15.0)

        result = display.render()

        # Render to string to check content
        console = Console(file=StringIO(), force_terminal=True, width=80)
        console.print(result)
        output = console.file.getvalue()  # type: ignore[union-attr]

        # Should show transition indicator
        assert "analyst" in output and "pm" in output


class TestEventThrottling:
    """Tests for event throttling behavior."""

    def test_should_update_always_true_for_first_event(self) -> None:
        """Test first event always passes throttle check."""
        from yolo_developer.cli.activity import ActivityDisplay

        display = ActivityDisplay(refresh_rate=4)

        # First event should always be allowed
        assert display.should_update() is True

    def test_should_update_respects_refresh_rate(self) -> None:
        """Test throttling respects refresh rate."""
        from yolo_developer.cli.activity import ActivityDisplay

        display = ActivityDisplay(refresh_rate=4)  # 4 updates/sec = 250ms interval

        # First update always allowed
        assert display.should_update() is True
        display._last_update = time.time()

        # Immediate second update should be throttled
        assert display.should_update() is False

    def test_should_update_allows_after_interval(self) -> None:
        """Test update allowed after throttle interval passes."""
        from yolo_developer.cli.activity import ActivityDisplay

        display = ActivityDisplay(refresh_rate=4)
        display._last_update = time.time() - 1.0  # 1 second ago

        # Should be allowed since more than 250ms has passed
        assert display.should_update() is True

    def test_transition_events_bypass_throttle(self) -> None:
        """Test transition events are never throttled."""
        from yolo_developer.cli.activity import ActivityDisplay

        display = ActivityDisplay(refresh_rate=4)
        display._last_update = time.time()

        # Regular update would be throttled
        assert display.should_update() is False

        # But transition events bypass throttle
        assert display.should_update(is_transition=True) is True


class TestActivityDisplayContextManager:
    """Tests for ActivityDisplay context manager functionality."""

    def test_context_manager_start_stop(self) -> None:
        """Test ActivityDisplay works as context manager."""
        from yolo_developer.cli.activity import ActivityDisplay

        display = ActivityDisplay()

        # Should support context manager protocol
        with patch.object(display, "start") as mock_start, patch.object(
            display, "stop"
        ) as mock_stop:
            with display:
                pass

            mock_start.assert_called_once()
            mock_stop.assert_called_once()

    def test_start_initializes_live_display(self) -> None:
        """Test start initializes the Rich Live display."""
        from yolo_developer.cli.activity import ActivityDisplay

        display = ActivityDisplay()

        with patch("yolo_developer.cli.activity.Live") as mock_live_class:
            mock_live_instance = MagicMock()
            mock_live_class.return_value = mock_live_instance

            display.start()

            mock_live_class.assert_called_once()
            mock_live_instance.start.assert_called_once()

    def test_stop_cleans_up_live_display(self) -> None:
        """Test stop properly cleans up the Live display."""
        from yolo_developer.cli.activity import ActivityDisplay

        display = ActivityDisplay()

        mock_live = MagicMock()
        display._live = mock_live

        display.stop()

        mock_live.stop.assert_called_once()


class TestFormatHelpers:
    """Tests for formatting helper functions."""

    def test_format_elapsed_time_seconds(self) -> None:
        """Test formatting elapsed time in seconds."""
        from yolo_developer.cli.activity import format_elapsed_time

        assert format_elapsed_time(5.0) == "00:05"
        assert format_elapsed_time(59.9) == "00:59"

    def test_format_elapsed_time_minutes(self) -> None:
        """Test formatting elapsed time with minutes."""
        from yolo_developer.cli.activity import format_elapsed_time

        assert format_elapsed_time(60.0) == "01:00"
        assert format_elapsed_time(125.5) == "02:05"

    def test_format_elapsed_time_hours(self) -> None:
        """Test formatting elapsed time with hours."""
        from yolo_developer.cli.activity import format_elapsed_time

        assert format_elapsed_time(3665.0) == "01:01:05"
