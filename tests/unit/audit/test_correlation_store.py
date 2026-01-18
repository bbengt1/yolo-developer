"""Tests for correlation store protocol (Story 11.5 - Task 2).

Tests the CorrelationStore protocol and CorrelationFilters dataclass.

References:
    - FR85: System can correlate decisions across agent boundaries
    - AC #1: Decision chains are identified
    - AC #4: Correlations are searchable
"""

from __future__ import annotations

import json

import pytest


class TestCorrelationFilters:
    """Tests for CorrelationFilters dataclass."""

    def test_correlation_filters_exists(self) -> None:
        """Test CorrelationFilters class exists."""
        from yolo_developer.audit.correlation_store import CorrelationFilters

        assert CorrelationFilters is not None

    def test_correlation_filters_creation(self) -> None:
        """Test CorrelationFilters can be created with all fields."""
        from yolo_developer.audit.correlation_store import CorrelationFilters

        filters = CorrelationFilters(
            agent_name="analyst",
            session_id="session-123",
            start_time="2026-01-18T00:00:00Z",
            end_time="2026-01-18T23:59:59Z",
            chain_type="session",
        )

        assert filters.agent_name == "analyst"
        assert filters.session_id == "session-123"
        assert filters.start_time == "2026-01-18T00:00:00Z"
        assert filters.end_time == "2026-01-18T23:59:59Z"
        assert filters.chain_type == "session"

    def test_correlation_filters_default_values(self) -> None:
        """Test CorrelationFilters has None defaults for all fields."""
        from yolo_developer.audit.correlation_store import CorrelationFilters

        filters = CorrelationFilters()

        assert filters.agent_name is None
        assert filters.session_id is None
        assert filters.start_time is None
        assert filters.end_time is None
        assert filters.chain_type is None

    def test_correlation_filters_is_frozen(self) -> None:
        """Test CorrelationFilters is immutable (frozen dataclass)."""
        from yolo_developer.audit.correlation_store import CorrelationFilters

        filters = CorrelationFilters(agent_name="analyst")

        with pytest.raises(AttributeError):
            filters.agent_name = "pm"  # type: ignore[misc]

    def test_correlation_filters_to_dict(self) -> None:
        """Test CorrelationFilters to_dict produces JSON-serializable output."""
        from yolo_developer.audit.correlation_store import CorrelationFilters

        filters = CorrelationFilters(
            agent_name="analyst",
            session_id="session-123",
            chain_type="causal",
        )

        result = filters.to_dict()

        assert result["agent_name"] == "analyst"
        assert result["session_id"] == "session-123"
        assert result["chain_type"] == "causal"
        assert result["start_time"] is None
        assert result["end_time"] is None

        # Verify JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None


class TestCorrelationStoreProtocol:
    """Tests for CorrelationStore protocol."""

    def test_correlation_store_protocol_exists(self) -> None:
        """Test CorrelationStore protocol exists."""
        from yolo_developer.audit.correlation_store import CorrelationStore

        assert CorrelationStore is not None

    def test_correlation_store_is_runtime_checkable(self) -> None:
        """Test CorrelationStore is runtime checkable."""
        from yolo_developer.audit.correlation_store import CorrelationStore

        # Protocol should be runtime_checkable
        assert hasattr(CorrelationStore, "__protocol_attrs__") or hasattr(
            CorrelationStore, "_is_protocol"
        )

    def test_correlation_store_has_store_chain_method(self) -> None:
        """Test CorrelationStore defines store_chain method."""
        from yolo_developer.audit.correlation_store import CorrelationStore

        # Check method is defined in protocol
        assert hasattr(CorrelationStore, "store_chain")

    def test_correlation_store_has_store_causal_relation_method(self) -> None:
        """Test CorrelationStore defines store_causal_relation method."""
        from yolo_developer.audit.correlation_store import CorrelationStore

        assert hasattr(CorrelationStore, "store_causal_relation")

    def test_correlation_store_has_store_transition_method(self) -> None:
        """Test CorrelationStore defines store_transition method."""
        from yolo_developer.audit.correlation_store import CorrelationStore

        assert hasattr(CorrelationStore, "store_transition")

    def test_correlation_store_has_get_chain_method(self) -> None:
        """Test CorrelationStore defines get_chain method."""
        from yolo_developer.audit.correlation_store import CorrelationStore

        assert hasattr(CorrelationStore, "get_chain")

    def test_correlation_store_has_get_chains_for_decision_method(self) -> None:
        """Test CorrelationStore defines get_chains_for_decision method."""
        from yolo_developer.audit.correlation_store import CorrelationStore

        assert hasattr(CorrelationStore, "get_chains_for_decision")

    def test_correlation_store_has_get_causal_relations_method(self) -> None:
        """Test CorrelationStore defines get_causal_relations method."""
        from yolo_developer.audit.correlation_store import CorrelationStore

        assert hasattr(CorrelationStore, "get_causal_relations")

    def test_correlation_store_has_get_transitions_by_session_method(self) -> None:
        """Test CorrelationStore defines get_transitions_by_session method."""
        from yolo_developer.audit.correlation_store import CorrelationStore

        assert hasattr(CorrelationStore, "get_transitions_by_session")

    def test_correlation_store_has_search_correlations_method(self) -> None:
        """Test CorrelationStore defines search_correlations method."""
        from yolo_developer.audit.correlation_store import CorrelationStore

        assert hasattr(CorrelationStore, "search_correlations")
