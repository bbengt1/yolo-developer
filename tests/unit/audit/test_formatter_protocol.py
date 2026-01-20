"""Tests for audit formatter protocol (Story 11.3).

Tests cover:
- AuditFormatter Protocol definition
- Protocol method signatures
- Runtime checkability
"""

from __future__ import annotations

from typing import Any


class TestAuditFormatterProtocol:
    """Tests for AuditFormatter Protocol."""

    def test_audit_formatter_protocol_exists(self) -> None:
        """Test that AuditFormatter protocol exists."""
        from yolo_developer.audit.formatter_protocol import AuditFormatter

        assert AuditFormatter is not None

    def test_audit_formatter_is_protocol(self) -> None:
        """Test that AuditFormatter is a Protocol."""
        from typing import Protocol

        from yolo_developer.audit.formatter_protocol import AuditFormatter

        assert issubclass(AuditFormatter, Protocol)

    def test_audit_formatter_is_runtime_checkable(self) -> None:
        """Test that AuditFormatter is runtime_checkable."""
        from yolo_developer.audit.formatter_protocol import AuditFormatter
        from yolo_developer.audit.traceability_types import TraceableArtifact, TraceLink
        from yolo_developer.audit.types import (
            Decision,
        )

        # Create a mock formatter that implements the protocol
        class MockFormatter:
            def format_decision(self, decision: Decision) -> str:
                return "decision"

            def format_decisions(self, decisions: list[Decision]) -> str:
                return "decisions"

            def format_trace_chain(
                self, artifacts: list[TraceableArtifact], links: list[TraceLink]
            ) -> str:
                return "trace_chain"

            def format_coverage_report(self, report: dict[str, Any]) -> str:
                return "coverage"

            def format_summary(self, decisions: list[Decision]) -> str:
                return "summary"

        formatter = MockFormatter()
        assert isinstance(formatter, AuditFormatter)

    def test_audit_formatter_has_format_decision_method(self) -> None:
        """Test that AuditFormatter has format_decision method signature."""
        from yolo_developer.audit.formatter_protocol import AuditFormatter

        # Check method exists in protocol
        assert hasattr(AuditFormatter, "format_decision")

    def test_audit_formatter_has_format_decisions_method(self) -> None:
        """Test that AuditFormatter has format_decisions method signature."""
        from yolo_developer.audit.formatter_protocol import AuditFormatter

        assert hasattr(AuditFormatter, "format_decisions")

    def test_audit_formatter_has_format_trace_chain_method(self) -> None:
        """Test that AuditFormatter has format_trace_chain method signature."""
        from yolo_developer.audit.formatter_protocol import AuditFormatter

        assert hasattr(AuditFormatter, "format_trace_chain")

    def test_audit_formatter_has_format_coverage_report_method(self) -> None:
        """Test that AuditFormatter has format_coverage_report method signature."""
        from yolo_developer.audit.formatter_protocol import AuditFormatter

        assert hasattr(AuditFormatter, "format_coverage_report")

    def test_audit_formatter_has_format_summary_method(self) -> None:
        """Test that AuditFormatter has format_summary method signature."""
        from yolo_developer.audit.formatter_protocol import AuditFormatter

        assert hasattr(AuditFormatter, "format_summary")

    def test_non_conforming_class_not_instance(self) -> None:
        """Test that a class not implementing protocol methods is not an instance."""
        from yolo_developer.audit.formatter_protocol import AuditFormatter

        class NotAFormatter:
            pass

        obj = NotAFormatter()
        assert not isinstance(obj, AuditFormatter)

    def test_partial_implementation_not_instance(self) -> None:
        """Test that partial implementation is not an instance."""
        from yolo_developer.audit.formatter_protocol import AuditFormatter
        from yolo_developer.audit.types import Decision

        class PartialFormatter:
            def format_decision(self, decision: Decision) -> str:
                return "decision"

            # Missing other methods

        obj = PartialFormatter()
        assert not isinstance(obj, AuditFormatter)
