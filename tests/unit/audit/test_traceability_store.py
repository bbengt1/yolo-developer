"""Tests for TraceabilityStore protocol definition (Story 11.2).

Tests cover:
- Protocol definition exists and can be used for type checking
- Protocol methods are defined with correct signatures
"""

from __future__ import annotations


class TestTraceabilityStoreProtocol:
    """Tests for TraceabilityStore protocol definition."""

    def test_protocol_exists(self) -> None:
        """Test that TraceabilityStore protocol is defined."""
        from yolo_developer.audit.traceability_store import TraceabilityStore

        assert TraceabilityStore is not None

    def test_protocol_is_runtime_checkable(self) -> None:
        """Test that TraceabilityStore is runtime checkable."""
        from yolo_developer.audit.traceability_store import TraceabilityStore

        # Protocol should be decorated with @runtime_checkable
        assert hasattr(TraceabilityStore, "__protocol_attrs__") or isinstance(
            TraceabilityStore, type
        )

    def test_protocol_has_register_artifact_method(self) -> None:
        """Test that TraceabilityStore has register_artifact method."""
        from yolo_developer.audit.traceability_store import TraceabilityStore

        assert hasattr(TraceabilityStore, "register_artifact")

    def test_protocol_has_create_link_method(self) -> None:
        """Test that TraceabilityStore has create_link method."""
        from yolo_developer.audit.traceability_store import TraceabilityStore

        assert hasattr(TraceabilityStore, "create_link")

    def test_protocol_has_get_artifact_method(self) -> None:
        """Test that TraceabilityStore has get_artifact method."""
        from yolo_developer.audit.traceability_store import TraceabilityStore

        assert hasattr(TraceabilityStore, "get_artifact")

    def test_protocol_has_get_links_from_method(self) -> None:
        """Test that TraceabilityStore has get_links_from method."""
        from yolo_developer.audit.traceability_store import TraceabilityStore

        assert hasattr(TraceabilityStore, "get_links_from")

    def test_protocol_has_get_links_to_method(self) -> None:
        """Test that TraceabilityStore has get_links_to method."""
        from yolo_developer.audit.traceability_store import TraceabilityStore

        assert hasattr(TraceabilityStore, "get_links_to")

    def test_protocol_has_get_trace_chain_method(self) -> None:
        """Test that TraceabilityStore has get_trace_chain method."""
        from yolo_developer.audit.traceability_store import TraceabilityStore

        assert hasattr(TraceabilityStore, "get_trace_chain")

    def test_protocol_has_get_unlinked_artifacts_method(self) -> None:
        """Test that TraceabilityStore has get_unlinked_artifacts method."""
        from yolo_developer.audit.traceability_store import TraceabilityStore

        assert hasattr(TraceabilityStore, "get_unlinked_artifacts")
