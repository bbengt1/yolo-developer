"""Tests for audit module exports (Story 11.1, 11.2, 11.3).

Tests that the audit module exports all public types, protocols, and functions
for decision logging (FR81), requirement traceability (FR82), and
human-readable view (FR83).
"""

from __future__ import annotations


class TestAuditModuleExports:
    """Tests for audit module exports."""

    def test_exports_decision_type(self) -> None:
        """Should export DecisionType."""
        from yolo_developer.audit import DecisionType

        assert DecisionType is not None

    def test_exports_decision_severity(self) -> None:
        """Should export DecisionSeverity."""
        from yolo_developer.audit import DecisionSeverity

        assert DecisionSeverity is not None

    def test_exports_valid_decision_types(self) -> None:
        """Should export VALID_DECISION_TYPES constant."""
        from yolo_developer.audit import VALID_DECISION_TYPES

        assert isinstance(VALID_DECISION_TYPES, frozenset)
        assert "requirement_analysis" in VALID_DECISION_TYPES

    def test_exports_valid_decision_severities(self) -> None:
        """Should export VALID_DECISION_SEVERITIES constant."""
        from yolo_developer.audit import VALID_DECISION_SEVERITIES

        assert isinstance(VALID_DECISION_SEVERITIES, frozenset)
        assert "info" in VALID_DECISION_SEVERITIES

    def test_exports_agent_identity(self) -> None:
        """Should export AgentIdentity."""
        from yolo_developer.audit import AgentIdentity

        agent = AgentIdentity(
            agent_name="analyst",
            agent_type="analyst",
            session_id="session-1",
        )
        assert agent.agent_name == "analyst"

    def test_exports_decision_context(self) -> None:
        """Should export DecisionContext."""
        from yolo_developer.audit import DecisionContext

        context = DecisionContext(sprint_id="sprint-1")
        assert context.sprint_id == "sprint-1"

    def test_exports_decision(self) -> None:
        """Should export Decision."""
        from yolo_developer.audit import AgentIdentity, Decision, DecisionContext

        decision = Decision(
            id="dec-001",
            decision_type="requirement_analysis",
            content="Content",
            rationale="Rationale",
            agent=AgentIdentity("analyst", "analyst", "session-1"),
            context=DecisionContext(),
            timestamp="2026-01-17T12:00:00Z",
        )
        assert decision.id == "dec-001"

    def test_exports_decision_filters(self) -> None:
        """Should export DecisionFilters."""
        from yolo_developer.audit import DecisionFilters

        filters = DecisionFilters(agent_name="analyst")
        assert filters.agent_name == "analyst"

    def test_exports_decision_store_protocol(self) -> None:
        """Should export DecisionStore protocol."""
        from yolo_developer.audit import DecisionStore

        assert DecisionStore is not None

    def test_exports_in_memory_decision_store(self) -> None:
        """Should export InMemoryDecisionStore."""
        from yolo_developer.audit import InMemoryDecisionStore

        store = InMemoryDecisionStore()
        assert store is not None

    def test_exports_decision_logger(self) -> None:
        """Should export DecisionLogger."""
        from yolo_developer.audit import DecisionLogger, InMemoryDecisionStore

        store = InMemoryDecisionStore()
        logger = DecisionLogger(store)
        assert logger is not None

    def test_exports_get_logger(self) -> None:
        """Should export get_logger factory function."""
        from yolo_developer.audit import get_logger

        logger = get_logger()
        assert logger is not None

    def test_module_docstring_mentions_fr81(self) -> None:
        """Module docstring should reference FR81."""
        import yolo_developer.audit as audit_module

        assert audit_module.__doc__ is not None
        assert "FR81" in audit_module.__doc__

    def test_module_docstring_mentions_fr82(self) -> None:
        """Module docstring should reference FR82 (Story 11.2)."""
        import yolo_developer.audit as audit_module

        assert audit_module.__doc__ is not None
        assert "FR82" in audit_module.__doc__

    def test_module_docstring_mentions_fr83(self) -> None:
        """Module docstring should reference FR83 (Story 11.3)."""
        import yolo_developer.audit as audit_module

        assert audit_module.__doc__ is not None
        assert "FR83" in audit_module.__doc__


class TestAuditModuleAll:
    """Tests for __all__ exports."""

    def test_all_exports_are_defined(self) -> None:
        """__all__ should be defined and contain all public exports."""
        from yolo_developer import audit

        assert hasattr(audit, "__all__")
        assert isinstance(audit.__all__, list)

    def test_all_contains_expected_exports(self) -> None:
        """__all__ should contain all expected public exports."""
        from yolo_developer import audit

        expected = [
            # Types
            "DecisionType",
            "DecisionSeverity",
            "VALID_DECISION_TYPES",
            "VALID_DECISION_SEVERITIES",
            "AgentIdentity",
            "DecisionContext",
            "Decision",
            # Store
            "DecisionFilters",
            "DecisionStore",
            "InMemoryDecisionStore",
            # Logger
            "DecisionLogger",
            "get_logger",
            # Traceability types (Story 11.2)
            "ArtifactType",
            "LinkType",
            "VALID_ARTIFACT_TYPES",
            "VALID_LINK_TYPES",
            "TraceableArtifact",
            "TraceLink",
            # Traceability store (Story 11.2)
            "TraceabilityStore",
            "InMemoryTraceabilityStore",
            # Traceability service (Story 11.2)
            "TraceabilityService",
            "get_traceability_service",
            # Formatter types (Story 11.3)
            "FormatterStyle",
            "ColorScheme",
            "FormatOptions",
            "VALID_FORMATTER_STYLES",
            "DEFAULT_COLOR_SCHEME",
            "DEFAULT_FORMAT_OPTIONS",
            # Formatter protocol (Story 11.3)
            "AuditFormatter",
            # Formatters (Story 11.3)
            "RichAuditFormatter",
            "PlainAuditFormatter",
            # View service (Story 11.3)
            "AuditViewService",
            "get_audit_view_service",
        ]

        for name in expected:
            assert name in audit.__all__, f"{name} not in __all__"


class TestTraceabilityExports:
    """Tests for traceability exports (Story 11.2 - Task 6)."""

    def test_exports_artifact_type(self) -> None:
        """Should export ArtifactType."""
        from yolo_developer.audit import ArtifactType

        assert ArtifactType is not None

    def test_exports_link_type(self) -> None:
        """Should export LinkType."""
        from yolo_developer.audit import LinkType

        assert LinkType is not None

    def test_exports_valid_artifact_types(self) -> None:
        """Should export VALID_ARTIFACT_TYPES constant."""
        from yolo_developer.audit import VALID_ARTIFACT_TYPES

        assert isinstance(VALID_ARTIFACT_TYPES, frozenset)
        assert "requirement" in VALID_ARTIFACT_TYPES
        assert "code" in VALID_ARTIFACT_TYPES

    def test_exports_valid_link_types(self) -> None:
        """Should export VALID_LINK_TYPES constant."""
        from yolo_developer.audit import VALID_LINK_TYPES

        assert isinstance(VALID_LINK_TYPES, frozenset)
        assert "derives_from" in VALID_LINK_TYPES
        assert "implements" in VALID_LINK_TYPES

    def test_exports_traceable_artifact(self) -> None:
        """Should export TraceableArtifact."""
        from yolo_developer.audit import TraceableArtifact

        artifact = TraceableArtifact(
            id="req-001",
            artifact_type="requirement",
            name="Test Requirement",
            description="A test requirement",
            created_at="2026-01-17T12:00:00Z",
        )
        assert artifact.id == "req-001"

    def test_exports_trace_link(self) -> None:
        """Should export TraceLink."""
        from yolo_developer.audit import TraceLink

        link = TraceLink(
            id="link-001",
            source_id="story-001",
            source_type="story",
            target_id="req-001",
            target_type="requirement",
            link_type="derives_from",
            created_at="2026-01-17T12:00:00Z",
        )
        assert link.id == "link-001"

    def test_exports_traceability_store_protocol(self) -> None:
        """Should export TraceabilityStore protocol."""
        from yolo_developer.audit import TraceabilityStore

        assert TraceabilityStore is not None

    def test_exports_in_memory_traceability_store(self) -> None:
        """Should export InMemoryTraceabilityStore."""
        from yolo_developer.audit import InMemoryTraceabilityStore

        store = InMemoryTraceabilityStore()
        assert store is not None

    def test_exports_traceability_service(self) -> None:
        """Should export TraceabilityService."""
        from yolo_developer.audit import InMemoryTraceabilityStore, TraceabilityService

        store = InMemoryTraceabilityStore()
        service = TraceabilityService(store)
        assert service is not None

    def test_exports_get_traceability_service(self) -> None:
        """Should export get_traceability_service factory function."""
        from yolo_developer.audit import get_traceability_service

        service = get_traceability_service()
        assert service is not None


class TestHumanReadableViewExports:
    """Tests for human-readable view exports (Story 11.3 - Task 6)."""

    def test_exports_formatter_style(self) -> None:
        """Should export FormatterStyle."""
        from yolo_developer.audit import FormatterStyle

        assert FormatterStyle is not None

    def test_exports_color_scheme(self) -> None:
        """Should export ColorScheme."""
        from yolo_developer.audit import ColorScheme

        scheme = ColorScheme()
        assert scheme is not None

    def test_exports_format_options(self) -> None:
        """Should export FormatOptions."""
        from yolo_developer.audit import FormatOptions

        options = FormatOptions()
        assert options is not None

    def test_exports_valid_formatter_styles(self) -> None:
        """Should export VALID_FORMATTER_STYLES constant."""
        from yolo_developer.audit import VALID_FORMATTER_STYLES

        assert isinstance(VALID_FORMATTER_STYLES, frozenset)
        assert "minimal" in VALID_FORMATTER_STYLES
        assert "standard" in VALID_FORMATTER_STYLES
        assert "verbose" in VALID_FORMATTER_STYLES

    def test_exports_default_color_scheme(self) -> None:
        """Should export DEFAULT_COLOR_SCHEME constant."""
        from yolo_developer.audit import DEFAULT_COLOR_SCHEME, ColorScheme

        assert isinstance(DEFAULT_COLOR_SCHEME, ColorScheme)

    def test_exports_default_format_options(self) -> None:
        """Should export DEFAULT_FORMAT_OPTIONS constant."""
        from yolo_developer.audit import DEFAULT_FORMAT_OPTIONS, FormatOptions

        assert isinstance(DEFAULT_FORMAT_OPTIONS, FormatOptions)

    def test_exports_audit_formatter_protocol(self) -> None:
        """Should export AuditFormatter protocol."""
        from yolo_developer.audit import AuditFormatter

        assert AuditFormatter is not None

    def test_exports_rich_audit_formatter(self) -> None:
        """Should export RichAuditFormatter."""
        from rich.console import Console

        from yolo_developer.audit import RichAuditFormatter

        formatter = RichAuditFormatter(Console())
        assert formatter is not None

    def test_exports_plain_audit_formatter(self) -> None:
        """Should export PlainAuditFormatter."""
        from yolo_developer.audit import PlainAuditFormatter

        formatter = PlainAuditFormatter()
        assert formatter is not None

    def test_exports_audit_view_service(self) -> None:
        """Should export AuditViewService."""
        from yolo_developer.audit import (
            AuditViewService,
            InMemoryDecisionStore,
            InMemoryTraceabilityStore,
        )

        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()
        service = AuditViewService(
            decision_store=decision_store,
            traceability_store=traceability_store,
        )
        assert service is not None

    def test_exports_get_audit_view_service(self) -> None:
        """Should export get_audit_view_service factory function."""
        from yolo_developer.audit import (
            InMemoryDecisionStore,
            InMemoryTraceabilityStore,
            get_audit_view_service,
        )

        decision_store = InMemoryDecisionStore()
        traceability_store = InMemoryTraceabilityStore()
        service = get_audit_view_service(
            decision_store=decision_store,
            traceability_store=traceability_store,
        )
        assert service is not None
