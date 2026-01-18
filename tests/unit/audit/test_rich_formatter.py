"""Tests for Rich-based audit formatter (Story 11.3).

Tests cover:
- RichAuditFormatter implements AuditFormatter protocol
- format_decision produces Rich-formatted output
- format_decisions orders chronologically
- format_trace_chain produces tree visualization
- format_coverage_report produces table output
- format_summary produces summary statistics
- Severity color coding
- Agent type color coding
- Different FormatterStyle options (minimal, standard, verbose)
"""

from __future__ import annotations

import pytest
from rich.console import Console

from yolo_developer.audit.types import (
    AgentIdentity,
    Decision,
    DecisionContext,
)
from yolo_developer.audit.traceability_types import TraceableArtifact, TraceLink


@pytest.fixture
def console() -> Console:
    """Create a Rich console with no color for testing."""
    return Console(force_terminal=True, no_color=True, width=120)


@pytest.fixture
def sample_decision() -> Decision:
    """Create a sample decision for testing."""
    return Decision(
        id="dec-001",
        decision_type="requirement_analysis",
        content="OAuth2 authentication required for API access",
        rationale="Industry standard security protocol with broad library support",
        agent=AgentIdentity(
            agent_name="analyst",
            agent_type="analyst",
            session_id="session-123",
        ),
        context=DecisionContext(
            sprint_id="sprint-1",
            story_id="1-2-user-auth",
        ),
        timestamp="2026-01-18T10:00:00Z",
        severity="critical",
    )


@pytest.fixture
def sample_decisions() -> list[Decision]:
    """Create multiple decisions for testing chronological ordering."""
    agent = AgentIdentity(
        agent_name="analyst",
        agent_type="analyst",
        session_id="session-123",
    )
    context = DecisionContext(sprint_id="sprint-1")

    return [
        Decision(
            id="dec-003",
            decision_type="requirement_analysis",
            content="Third decision",
            rationale="Third rationale",
            agent=agent,
            context=context,
            timestamp="2026-01-18T12:00:00Z",
            severity="info",
        ),
        Decision(
            id="dec-001",
            decision_type="story_creation",
            content="First decision",
            rationale="First rationale",
            agent=AgentIdentity(
                agent_name="pm",
                agent_type="pm",
                session_id="session-123",
            ),
            context=context,
            timestamp="2026-01-18T10:00:00Z",
            severity="warning",
        ),
        Decision(
            id="dec-002",
            decision_type="architecture_choice",
            content="Second decision",
            rationale="Second rationale",
            agent=AgentIdentity(
                agent_name="architect",
                agent_type="architect",
                session_id="session-123",
            ),
            context=context,
            timestamp="2026-01-18T11:00:00Z",
            severity="critical",
        ),
    ]


@pytest.fixture
def sample_artifacts() -> list[TraceableArtifact]:
    """Create sample artifacts for trace chain testing."""
    return [
        TraceableArtifact(
            id="req-001",
            artifact_type="requirement",
            name="User Authentication",
            description="Users must authenticate before accessing the system",
            created_at="2026-01-18T09:00:00Z",
        ),
        TraceableArtifact(
            id="story-001",
            artifact_type="story",
            name="Implement login flow",
            description="As a user, I want to log in",
            created_at="2026-01-18T10:00:00Z",
        ),
        TraceableArtifact(
            id="code-001",
            artifact_type="code",
            name="auth_handler.py",
            description="Authentication handler module",
            created_at="2026-01-18T11:00:00Z",
        ),
    ]


@pytest.fixture
def sample_links() -> list[TraceLink]:
    """Create sample links for trace chain testing."""
    return [
        TraceLink(
            id="link-001",
            source_id="story-001",
            source_type="story",
            target_id="req-001",
            target_type="requirement",
            link_type="derives_from",
            created_at="2026-01-18T10:00:00Z",
        ),
        TraceLink(
            id="link-002",
            source_id="code-001",
            source_type="code",
            target_id="story-001",
            target_type="story",
            link_type="implements",
            created_at="2026-01-18T11:00:00Z",
        ),
    ]


class TestRichAuditFormatter:
    """Tests for RichAuditFormatter class."""

    def test_rich_formatter_exists(self) -> None:
        """Test that RichAuditFormatter class exists."""
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        assert RichAuditFormatter is not None

    def test_rich_formatter_implements_protocol(self, console: Console) -> None:
        """Test that RichAuditFormatter implements AuditFormatter protocol."""
        from yolo_developer.audit.formatter_protocol import AuditFormatter
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        formatter = RichAuditFormatter(console)
        assert isinstance(formatter, AuditFormatter)

    def test_rich_formatter_requires_console(self) -> None:
        """Test that RichAuditFormatter requires a Console instance."""
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        console = Console()
        formatter = RichAuditFormatter(console)
        assert formatter._console is console


class TestFormatDecision:
    """Tests for format_decision method."""

    def test_format_decision_returns_string(
        self, console: Console, sample_decision: Decision
    ) -> None:
        """Test that format_decision returns a string."""
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        formatter = RichAuditFormatter(console)
        result = formatter.format_decision(sample_decision)
        assert isinstance(result, str)

    def test_format_decision_contains_id(self, console: Console, sample_decision: Decision) -> None:
        """Test that formatted output contains decision ID."""
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        formatter = RichAuditFormatter(console)
        result = formatter.format_decision(sample_decision)
        assert "dec-001" in result

    def test_format_decision_contains_content(
        self, console: Console, sample_decision: Decision
    ) -> None:
        """Test that formatted output contains decision content."""
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        formatter = RichAuditFormatter(console)
        result = formatter.format_decision(sample_decision)
        assert "OAuth2" in result

    def test_format_decision_contains_agent(
        self, console: Console, sample_decision: Decision
    ) -> None:
        """Test that formatted output contains agent information."""
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        formatter = RichAuditFormatter(console)
        result = formatter.format_decision(sample_decision)
        assert "analyst" in result

    def test_format_decision_contains_severity(
        self, console: Console, sample_decision: Decision
    ) -> None:
        """Test that formatted output contains severity."""
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        formatter = RichAuditFormatter(console)
        result = formatter.format_decision(sample_decision)
        assert "critical" in result.lower()


class TestFormatDecisions:
    """Tests for format_decisions method."""

    def test_format_decisions_returns_string(
        self, console: Console, sample_decisions: list[Decision]
    ) -> None:
        """Test that format_decisions returns a string."""
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        formatter = RichAuditFormatter(console)
        result = formatter.format_decisions(sample_decisions)
        assert isinstance(result, str)

    def test_format_decisions_orders_chronologically(
        self, console: Console, sample_decisions: list[Decision]
    ) -> None:
        """Test that decisions are ordered by timestamp (oldest first)."""
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        formatter = RichAuditFormatter(console)
        result = formatter.format_decisions(sample_decisions)

        # First decision should appear before second, second before third
        first_pos = result.find("dec-001")
        second_pos = result.find("dec-002")
        third_pos = result.find("dec-003")

        assert first_pos < second_pos < third_pos

    def test_format_decisions_empty_list(self, console: Console) -> None:
        """Test that format_decisions handles empty list."""
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        formatter = RichAuditFormatter(console)
        result = formatter.format_decisions([])
        assert isinstance(result, str)
        assert "no decisions" in result.lower() or result == ""


class TestFormatTraceChain:
    """Tests for format_trace_chain method."""

    def test_format_trace_chain_returns_string(
        self,
        console: Console,
        sample_artifacts: list[TraceableArtifact],
        sample_links: list[TraceLink],
    ) -> None:
        """Test that format_trace_chain returns a string."""
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        formatter = RichAuditFormatter(console)
        result = formatter.format_trace_chain(sample_artifacts, sample_links)
        assert isinstance(result, str)

    def test_format_trace_chain_contains_artifact_names(
        self,
        console: Console,
        sample_artifacts: list[TraceableArtifact],
        sample_links: list[TraceLink],
    ) -> None:
        """Test that trace chain contains artifact names."""
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        formatter = RichAuditFormatter(console)
        result = formatter.format_trace_chain(sample_artifacts, sample_links)
        assert "User Authentication" in result
        assert "Implement login flow" in result

    def test_format_trace_chain_empty(self, console: Console) -> None:
        """Test that format_trace_chain handles empty input."""
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        formatter = RichAuditFormatter(console)
        result = formatter.format_trace_chain([], [])
        assert isinstance(result, str)

    def test_format_trace_chain_accepts_options(
        self,
        console: Console,
        sample_artifacts: list[TraceableArtifact],
        sample_links: list[TraceLink],
    ) -> None:
        """Test that format_trace_chain accepts FormatOptions parameter."""
        from yolo_developer.audit.formatter_types import FormatOptions
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        formatter = RichAuditFormatter(console)
        options = FormatOptions(style="verbose")
        result = formatter.format_trace_chain(sample_artifacts, sample_links, options)
        assert isinstance(result, str)
        assert "User Authentication" in result


class TestFormatCoverageReport:
    """Tests for format_coverage_report method."""

    def test_format_coverage_report_returns_string(self, console: Console) -> None:
        """Test that format_coverage_report returns a string."""
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        formatter = RichAuditFormatter(console)
        report = {
            "total_requirements": 10,
            "covered_requirements": 8,
            "coverage_percentage": 80.0,
            "unlinked_requirements": ["req-009", "req-010"],
        }
        result = formatter.format_coverage_report(report)
        assert isinstance(result, str)

    def test_format_coverage_report_contains_stats(self, console: Console) -> None:
        """Test that coverage report contains statistics."""
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        formatter = RichAuditFormatter(console)
        report = {
            "total_requirements": 10,
            "covered_requirements": 8,
            "coverage_percentage": 80.0,
        }
        result = formatter.format_coverage_report(report)
        assert "10" in result
        assert "8" in result
        assert "80" in result

    def test_format_coverage_report_accepts_options(self, console: Console) -> None:
        """Test that format_coverage_report accepts FormatOptions parameter."""
        from yolo_developer.audit.formatter_types import FormatOptions
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        formatter = RichAuditFormatter(console)
        report = {
            "total_requirements": 10,
            "covered_requirements": 8,
            "coverage_percentage": 80.0,
        }
        options = FormatOptions(style="verbose")
        result = formatter.format_coverage_report(report, options)
        assert isinstance(result, str)
        assert "10" in result


class TestFormatSummary:
    """Tests for format_summary method."""

    def test_format_summary_returns_string(
        self, console: Console, sample_decisions: list[Decision]
    ) -> None:
        """Test that format_summary returns a string."""
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        formatter = RichAuditFormatter(console)
        result = formatter.format_summary(sample_decisions)
        assert isinstance(result, str)

    def test_format_summary_contains_counts(
        self, console: Console, sample_decisions: list[Decision]
    ) -> None:
        """Test that summary contains decision counts."""
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        formatter = RichAuditFormatter(console)
        result = formatter.format_summary(sample_decisions)
        # Should contain total count
        assert "3" in result


class TestFormatterStyles:
    """Tests for different FormatterStyle options."""

    def test_minimal_style_shorter_output(
        self, console: Console, sample_decision: Decision
    ) -> None:
        """Test that minimal style produces shorter output."""
        from yolo_developer.audit.formatter_types import FormatOptions
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        formatter = RichAuditFormatter(console)

        minimal_options = FormatOptions(style="minimal")
        verbose_options = FormatOptions(style="verbose")

        minimal_result = formatter.format_decision(sample_decision, minimal_options)
        verbose_result = formatter.format_decision(sample_decision, verbose_options)

        # Minimal should be shorter than verbose
        assert len(minimal_result) < len(verbose_result)

    def test_verbose_style_includes_metadata(
        self, console: Console, sample_decision: Decision
    ) -> None:
        """Test that verbose style includes metadata when available."""
        from yolo_developer.audit.formatter_types import FormatOptions
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        # Create decision with metadata
        decision = Decision(
            id="dec-meta",
            decision_type="requirement_analysis",
            content="Test content",
            rationale="Test rationale",
            agent=sample_decision.agent,
            context=sample_decision.context,
            timestamp="2026-01-18T10:00:00Z",
            metadata={"key": "value"},
        )

        formatter = RichAuditFormatter(console)
        verbose_options = FormatOptions(style="verbose", show_metadata=True)

        result = formatter.format_decision(decision, verbose_options)
        # Metadata should be visible in verbose mode
        assert "key" in result or "value" in result


class TestColorCoding:
    """Tests for severity and agent color coding."""

    def test_formatter_has_color_scheme(self, console: Console) -> None:
        """Test that formatter has a color scheme."""
        from yolo_developer.audit.formatter_types import ColorScheme
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        formatter = RichAuditFormatter(console)
        assert hasattr(formatter, "_color_scheme")
        assert isinstance(formatter._color_scheme, ColorScheme)

    def test_custom_color_scheme(self, console: Console) -> None:
        """Test that custom color scheme can be provided."""
        from yolo_developer.audit.formatter_types import ColorScheme
        from yolo_developer.audit.rich_formatter import RichAuditFormatter

        custom_scheme = ColorScheme(severity_critical="bright_red")
        formatter = RichAuditFormatter(console, color_scheme=custom_scheme)
        assert formatter._color_scheme.severity_critical == "bright_red"
