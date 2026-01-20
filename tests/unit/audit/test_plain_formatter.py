"""Tests for plain text audit formatter (Story 11.3).

Tests cover:
- PlainAuditFormatter implements AuditFormatter protocol
- format_decision produces plain text output
- format_decisions orders chronologically
- format_trace_chain produces ASCII tree
- All format methods produce valid plain text
"""

from __future__ import annotations

import pytest

from yolo_developer.audit.traceability_types import TraceableArtifact, TraceLink
from yolo_developer.audit.types import (
    AgentIdentity,
    Decision,
    DecisionContext,
)


@pytest.fixture
def sample_decision() -> Decision:
    """Create a sample decision for testing."""
    return Decision(
        id="dec-001",
        decision_type="requirement_analysis",
        content="OAuth2 authentication required for API access",
        rationale="Industry standard security protocol",
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
            agent=agent,
            context=context,
            timestamp="2026-01-18T10:00:00Z",
            severity="warning",
        ),
        Decision(
            id="dec-002",
            decision_type="architecture_choice",
            content="Second decision",
            rationale="Second rationale",
            agent=agent,
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
            description="Users must authenticate",
            created_at="2026-01-18T09:00:00Z",
        ),
        TraceableArtifact(
            id="story-001",
            artifact_type="story",
            name="Implement login flow",
            description="As a user, I want to log in",
            created_at="2026-01-18T10:00:00Z",
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
    ]


class TestPlainAuditFormatter:
    """Tests for PlainAuditFormatter class."""

    def test_plain_formatter_exists(self) -> None:
        """Test that PlainAuditFormatter class exists."""
        from yolo_developer.audit.plain_formatter import PlainAuditFormatter

        assert PlainAuditFormatter is not None

    def test_plain_formatter_implements_protocol(self) -> None:
        """Test that PlainAuditFormatter implements AuditFormatter protocol."""
        from yolo_developer.audit.formatter_protocol import AuditFormatter
        from yolo_developer.audit.plain_formatter import PlainAuditFormatter

        formatter = PlainAuditFormatter()
        assert isinstance(formatter, AuditFormatter)


class TestFormatDecision:
    """Tests for format_decision method."""

    def test_format_decision_returns_string(self, sample_decision: Decision) -> None:
        """Test that format_decision returns a string."""
        from yolo_developer.audit.plain_formatter import PlainAuditFormatter

        formatter = PlainAuditFormatter()
        result = formatter.format_decision(sample_decision)
        assert isinstance(result, str)

    def test_format_decision_contains_id(self, sample_decision: Decision) -> None:
        """Test that formatted output contains decision ID."""
        from yolo_developer.audit.plain_formatter import PlainAuditFormatter

        formatter = PlainAuditFormatter()
        result = formatter.format_decision(sample_decision)
        assert "dec-001" in result

    def test_format_decision_contains_content(self, sample_decision: Decision) -> None:
        """Test that formatted output contains decision content."""
        from yolo_developer.audit.plain_formatter import PlainAuditFormatter

        formatter = PlainAuditFormatter()
        result = formatter.format_decision(sample_decision)
        assert "OAuth2" in result

    def test_format_decision_contains_severity_marker(self, sample_decision: Decision) -> None:
        """Test that formatted output contains severity marker."""
        from yolo_developer.audit.plain_formatter import PlainAuditFormatter

        formatter = PlainAuditFormatter()
        result = formatter.format_decision(sample_decision)
        # Should have [CRITICAL] or similar marker
        assert "CRITICAL" in result.upper()

    def test_format_decision_is_plain_text(self, sample_decision: Decision) -> None:
        """Test that output is plain ASCII text (no Rich markup)."""
        from yolo_developer.audit.plain_formatter import PlainAuditFormatter

        formatter = PlainAuditFormatter()
        result = formatter.format_decision(sample_decision)
        # Should not contain Rich markup
        assert "[/" not in result
        assert "[bold]" not in result


class TestFormatDecisions:
    """Tests for format_decisions method."""

    def test_format_decisions_returns_string(self, sample_decisions: list[Decision]) -> None:
        """Test that format_decisions returns a string."""
        from yolo_developer.audit.plain_formatter import PlainAuditFormatter

        formatter = PlainAuditFormatter()
        result = formatter.format_decisions(sample_decisions)
        assert isinstance(result, str)

    def test_format_decisions_orders_chronologically(
        self, sample_decisions: list[Decision]
    ) -> None:
        """Test that decisions are ordered by timestamp (oldest first)."""
        from yolo_developer.audit.plain_formatter import PlainAuditFormatter

        formatter = PlainAuditFormatter()
        result = formatter.format_decisions(sample_decisions)

        # First decision should appear before second, second before third
        first_pos = result.find("dec-001")
        second_pos = result.find("dec-002")
        third_pos = result.find("dec-003")

        assert first_pos < second_pos < third_pos

    def test_format_decisions_empty_list(self) -> None:
        """Test that format_decisions handles empty list."""
        from yolo_developer.audit.plain_formatter import PlainAuditFormatter

        formatter = PlainAuditFormatter()
        result = formatter.format_decisions([])
        assert isinstance(result, str)


class TestFormatTraceChain:
    """Tests for format_trace_chain method."""

    def test_format_trace_chain_returns_string(
        self,
        sample_artifacts: list[TraceableArtifact],
        sample_links: list[TraceLink],
    ) -> None:
        """Test that format_trace_chain returns a string."""
        from yolo_developer.audit.plain_formatter import PlainAuditFormatter

        formatter = PlainAuditFormatter()
        result = formatter.format_trace_chain(sample_artifacts, sample_links)
        assert isinstance(result, str)

    def test_format_trace_chain_contains_artifact_names(
        self,
        sample_artifacts: list[TraceableArtifact],
        sample_links: list[TraceLink],
    ) -> None:
        """Test that trace chain contains artifact names."""
        from yolo_developer.audit.plain_formatter import PlainAuditFormatter

        formatter = PlainAuditFormatter()
        result = formatter.format_trace_chain(sample_artifacts, sample_links)
        assert "User Authentication" in result
        assert "Implement login flow" in result

    def test_format_trace_chain_uses_ascii(
        self,
        sample_artifacts: list[TraceableArtifact],
        sample_links: list[TraceLink],
    ) -> None:
        """Test that trace chain uses ASCII characters."""
        from yolo_developer.audit.plain_formatter import PlainAuditFormatter

        formatter = PlainAuditFormatter()
        result = formatter.format_trace_chain(sample_artifacts, sample_links)
        # Should use ASCII tree characters like +, |, -, or indentation
        assert result.isascii()

    def test_format_trace_chain_empty(self) -> None:
        """Test that format_trace_chain handles empty input."""
        from yolo_developer.audit.plain_formatter import PlainAuditFormatter

        formatter = PlainAuditFormatter()
        result = formatter.format_trace_chain([], [])
        assert isinstance(result, str)

    def test_format_trace_chain_accepts_options(
        self,
        sample_artifacts: list[TraceableArtifact],
        sample_links: list[TraceLink],
    ) -> None:
        """Test that format_trace_chain accepts FormatOptions parameter."""
        from yolo_developer.audit.formatter_types import FormatOptions
        from yolo_developer.audit.plain_formatter import PlainAuditFormatter

        formatter = PlainAuditFormatter()
        options = FormatOptions(style="verbose")
        result = formatter.format_trace_chain(sample_artifacts, sample_links, options)
        assert isinstance(result, str)
        assert "User Authentication" in result


class TestFormatCoverageReport:
    """Tests for format_coverage_report method."""

    def test_format_coverage_report_returns_string(self) -> None:
        """Test that format_coverage_report returns a string."""
        from yolo_developer.audit.plain_formatter import PlainAuditFormatter

        formatter = PlainAuditFormatter()
        report = {
            "total_requirements": 10,
            "covered_requirements": 8,
            "coverage_percentage": 80.0,
        }
        result = formatter.format_coverage_report(report)
        assert isinstance(result, str)

    def test_format_coverage_report_contains_stats(self) -> None:
        """Test that coverage report contains statistics."""
        from yolo_developer.audit.plain_formatter import PlainAuditFormatter

        formatter = PlainAuditFormatter()
        report = {
            "total_requirements": 10,
            "covered_requirements": 8,
            "coverage_percentage": 80.0,
        }
        result = formatter.format_coverage_report(report)
        assert "10" in result
        assert "8" in result
        assert "80" in result

    def test_format_coverage_report_accepts_options(self) -> None:
        """Test that format_coverage_report accepts FormatOptions parameter."""
        from yolo_developer.audit.formatter_types import FormatOptions
        from yolo_developer.audit.plain_formatter import PlainAuditFormatter

        formatter = PlainAuditFormatter()
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

    def test_format_summary_returns_string(self, sample_decisions: list[Decision]) -> None:
        """Test that format_summary returns a string."""
        from yolo_developer.audit.plain_formatter import PlainAuditFormatter

        formatter = PlainAuditFormatter()
        result = formatter.format_summary(sample_decisions)
        assert isinstance(result, str)

    def test_format_summary_contains_counts(self, sample_decisions: list[Decision]) -> None:
        """Test that summary contains decision counts."""
        from yolo_developer.audit.plain_formatter import PlainAuditFormatter

        formatter = PlainAuditFormatter()
        result = formatter.format_summary(sample_decisions)
        # Should contain total count
        assert "3" in result

    def test_format_summary_is_plain_text(self, sample_decisions: list[Decision]) -> None:
        """Test that summary is plain ASCII text."""
        from yolo_developer.audit.plain_formatter import PlainAuditFormatter

        formatter = PlainAuditFormatter()
        result = formatter.format_summary(sample_decisions)
        assert result.isascii()
