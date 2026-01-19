"""Tests for JSON file-based decision store (Story 13.3)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from yolo_developer.audit import JsonDecisionStore
from yolo_developer.audit.store import DecisionFilters
from yolo_developer.audit.types import (
    AgentIdentity,
    Decision,
    DecisionContext,
)


class TestJsonDecisionStore:
    """Tests for JsonDecisionStore."""

    def _create_decision(
        self,
        decision_id: str = "dec-001",
        agent_name: str = "analyst",
        decision_type: str = "requirement_analysis",
        timestamp: str | None = None,
    ) -> Decision:
        """Create a test decision."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        return Decision(
            id=decision_id,
            decision_type=decision_type,  # type: ignore[arg-type]
            content="Test decision content",
            rationale="Test rationale",
            agent=AgentIdentity(
                agent_name=agent_name,
                agent_type=agent_name,
                session_id="session-123",
            ),
            context=DecisionContext(
                sprint_id="sprint-1",
                story_id="story-1",
            ),
            timestamp=timestamp,
            metadata={"key": "value"},
            severity="info",
        )

    @pytest.mark.asyncio
    async def test_log_decision_creates_file(self, tmp_path: Path) -> None:
        """Test that log_decision creates the JSON file."""
        decisions_file = tmp_path / "audit" / "decisions.json"
        store = JsonDecisionStore(decisions_file)

        decision = self._create_decision()
        result = await store.log_decision(decision)

        assert result == "dec-001"
        assert decisions_file.exists()

    @pytest.mark.asyncio
    async def test_get_decision_returns_stored_decision(self, tmp_path: Path) -> None:
        """Test that get_decision returns a previously stored decision."""
        decisions_file = tmp_path / "audit" / "decisions.json"
        store = JsonDecisionStore(decisions_file)

        decision = self._create_decision()
        await store.log_decision(decision)

        retrieved = await store.get_decision("dec-001")

        assert retrieved is not None
        assert retrieved.id == "dec-001"
        assert retrieved.content == "Test decision content"
        assert retrieved.agent.agent_name == "analyst"

    @pytest.mark.asyncio
    async def test_get_decision_returns_none_for_missing(
        self, tmp_path: Path
    ) -> None:
        """Test that get_decision returns None for non-existent decisions."""
        decisions_file = tmp_path / "audit" / "decisions.json"
        store = JsonDecisionStore(decisions_file)

        retrieved = await store.get_decision("non-existent")

        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_decisions_returns_all_decisions(
        self, tmp_path: Path
    ) -> None:
        """Test that get_decisions returns all stored decisions."""
        decisions_file = tmp_path / "audit" / "decisions.json"
        store = JsonDecisionStore(decisions_file)

        await store.log_decision(self._create_decision("dec-001"))
        await store.log_decision(self._create_decision("dec-002"))
        await store.log_decision(self._create_decision("dec-003"))

        decisions = await store.get_decisions()

        assert len(decisions) == 3

    @pytest.mark.asyncio
    async def test_get_decisions_filters_by_agent_name(
        self, tmp_path: Path
    ) -> None:
        """Test filtering decisions by agent name."""
        decisions_file = tmp_path / "audit" / "decisions.json"
        store = JsonDecisionStore(decisions_file)

        await store.log_decision(self._create_decision("dec-001", agent_name="analyst"))
        await store.log_decision(self._create_decision("dec-002", agent_name="pm"))
        await store.log_decision(self._create_decision("dec-003", agent_name="analyst"))

        filters = DecisionFilters(agent_name="analyst")
        decisions = await store.get_decisions(filters)

        assert len(decisions) == 2
        assert all(d.agent.agent_name == "analyst" for d in decisions)

    @pytest.mark.asyncio
    async def test_get_decisions_filters_by_decision_type(
        self, tmp_path: Path
    ) -> None:
        """Test filtering decisions by decision type."""
        decisions_file = tmp_path / "audit" / "decisions.json"
        store = JsonDecisionStore(decisions_file)

        await store.log_decision(
            self._create_decision("dec-001", decision_type="requirement_analysis")
        )
        await store.log_decision(
            self._create_decision("dec-002", decision_type="story_creation")
        )

        filters = DecisionFilters(decision_type="requirement_analysis")
        decisions = await store.get_decisions(filters)

        assert len(decisions) == 1
        assert decisions[0].decision_type == "requirement_analysis"

    @pytest.mark.asyncio
    async def test_get_decision_count(self, tmp_path: Path) -> None:
        """Test that get_decision_count returns correct count."""
        decisions_file = tmp_path / "audit" / "decisions.json"
        store = JsonDecisionStore(decisions_file)

        assert await store.get_decision_count() == 0

        await store.log_decision(self._create_decision("dec-001"))
        assert await store.get_decision_count() == 1

        await store.log_decision(self._create_decision("dec-002"))
        assert await store.get_decision_count() == 2

    @pytest.mark.asyncio
    async def test_persistence_across_store_instances(
        self, tmp_path: Path
    ) -> None:
        """Test that decisions persist across store instances (AC5)."""
        decisions_file = tmp_path / "audit" / "decisions.json"

        # First store instance - log decision
        store1 = JsonDecisionStore(decisions_file)
        await store1.log_decision(self._create_decision("dec-001"))

        # Second store instance - should see the decision
        store2 = JsonDecisionStore(decisions_file)
        decisions = await store2.get_decisions()

        assert len(decisions) == 1
        assert decisions[0].id == "dec-001"

    @pytest.mark.asyncio
    async def test_decisions_sorted_by_timestamp(self, tmp_path: Path) -> None:
        """Test that get_decisions returns decisions in chronological order."""
        decisions_file = tmp_path / "audit" / "decisions.json"
        store = JsonDecisionStore(decisions_file)

        # Add decisions in reverse chronological order
        await store.log_decision(
            self._create_decision("dec-003", timestamp="2026-01-03T00:00:00Z")
        )
        await store.log_decision(
            self._create_decision("dec-001", timestamp="2026-01-01T00:00:00Z")
        )
        await store.log_decision(
            self._create_decision("dec-002", timestamp="2026-01-02T00:00:00Z")
        )

        decisions = await store.get_decisions()

        assert decisions[0].id == "dec-001"
        assert decisions[1].id == "dec-002"
        assert decisions[2].id == "dec-003"

    @pytest.mark.asyncio
    async def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Test that store creates parent directories if they don't exist."""
        decisions_file = tmp_path / "deep" / "nested" / "path" / "decisions.json"
        store = JsonDecisionStore(decisions_file)

        await store.log_decision(self._create_decision())

        assert decisions_file.exists()
        assert decisions_file.parent.exists()

    @pytest.mark.asyncio
    async def test_handles_empty_file(self, tmp_path: Path) -> None:
        """Test that store handles empty JSON file gracefully."""
        decisions_file = tmp_path / "audit" / "decisions.json"
        decisions_file.parent.mkdir(parents=True, exist_ok=True)
        decisions_file.write_text("")

        store = JsonDecisionStore(decisions_file)
        decisions = await store.get_decisions()

        assert decisions == []

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self, tmp_path: Path) -> None:
        """Test that store handles invalid JSON file gracefully."""
        decisions_file = tmp_path / "audit" / "decisions.json"
        decisions_file.parent.mkdir(parents=True, exist_ok=True)
        decisions_file.write_text("not valid json")

        store = JsonDecisionStore(decisions_file)
        decisions = await store.get_decisions()

        assert decisions == []

    @pytest.mark.asyncio
    async def test_get_decisions_filters_by_time_range(
        self, tmp_path: Path
    ) -> None:
        """Test filtering decisions by time range (AC1)."""
        decisions_file = tmp_path / "audit" / "decisions.json"
        store = JsonDecisionStore(decisions_file)

        # Create decisions at different timestamps
        await store.log_decision(
            self._create_decision("dec-001", timestamp="2026-01-01T00:00:00Z")
        )
        await store.log_decision(
            self._create_decision("dec-002", timestamp="2026-01-02T00:00:00Z")
        )
        await store.log_decision(
            self._create_decision("dec-003", timestamp="2026-01-03T00:00:00Z")
        )

        # Filter by start_time only
        filters = DecisionFilters(start_time="2026-01-02T00:00:00Z")
        decisions = await store.get_decisions(filters)

        assert len(decisions) == 2
        assert all(d.timestamp >= "2026-01-02T00:00:00Z" for d in decisions)

        # Filter by end_time only
        filters = DecisionFilters(end_time="2026-01-02T00:00:00Z")
        decisions = await store.get_decisions(filters)

        assert len(decisions) == 2
        assert all(d.timestamp <= "2026-01-02T00:00:00Z" for d in decisions)

        # Filter by both start_time and end_time
        filters = DecisionFilters(
            start_time="2026-01-01T12:00:00Z",
            end_time="2026-01-02T12:00:00Z",
        )
        decisions = await store.get_decisions(filters)

        assert len(decisions) == 1
        assert decisions[0].id == "dec-002"
