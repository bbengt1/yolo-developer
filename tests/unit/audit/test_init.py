"""Tests for audit module exports (Story 11.1 - Task 5).

Tests that the audit module exports all public types, protocols, and functions.
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
        ]

        for name in expected:
            assert name in audit.__all__, f"{name} not in __all__"
