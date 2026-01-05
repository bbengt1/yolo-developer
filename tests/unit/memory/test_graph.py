"""Unit tests for JSONGraphStore.

Tests cover:
- store_relationship creates edges
- store_relationship handles duplicates
- get_relationships filters by source, target, relation
- get_related transitive queries with cycle handling
- persistence saves to and loads from file
- concurrent access doesn't corrupt data
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from yolo_developer.memory.graph import (
    JSONGraphError,
    JSONGraphStore,
    Relationship,
    RelationshipResult,
)


class TestRelationship:
    """Tests for the Relationship dataclass."""

    def test_relationship_creation(self):
        """Relationship should store source, target, and relation."""
        edge = Relationship(source="story-001", target="req-001", relation="implements")
        assert edge.source == "story-001"
        assert edge.target == "req-001"
        assert edge.relation == "implements"

    def test_relationship_is_frozen(self):
        """Relationship should be immutable."""
        edge = Relationship(source="a", target="b", relation="c")
        with pytest.raises(AttributeError):
            edge.source = "x"  # type: ignore[misc]

    def test_relationship_hashable(self):
        """Relationship should be hashable for use in sets."""
        edge1 = Relationship(source="a", target="b", relation="c")
        edge2 = Relationship(source="a", target="b", relation="c")
        edge_set = {edge1, edge2}
        assert len(edge_set) == 1

    def test_relationship_equality(self):
        """Relationships with same values should be equal."""
        edge1 = Relationship(source="a", target="b", relation="c")
        edge2 = Relationship(source="a", target="b", relation="c")
        assert edge1 == edge2

    def test_relationship_inequality(self):
        """Relationships with different values should not be equal."""
        edge1 = Relationship(source="a", target="b", relation="c")
        edge2 = Relationship(source="a", target="b", relation="d")
        assert edge1 != edge2


class TestRelationshipResult:
    """Tests for the RelationshipResult dataclass."""

    def test_relationship_result_creation(self):
        """RelationshipResult should store all fields."""
        result = RelationshipResult(
            source="story-001",
            target="req-001",
            relation="implements",
            path=["story-001", "req-001"],
        )
        assert result.source == "story-001"
        assert result.target == "req-001"
        assert result.relation == "implements"
        assert result.path == ["story-001", "req-001"]

    def test_relationship_result_path_optional(self):
        """RelationshipResult path should be optional."""
        result = RelationshipResult(
            source="story-001",
            target="req-001",
            relation="implements",
        )
        assert result.path is None


class TestJSONGraphStore:
    """Tests for the JSONGraphStore class."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> JSONGraphStore:
        """Create a JSONGraphStore with a temporary persist path."""
        return JSONGraphStore(persist_path=str(tmp_path / "graph.json"))

    @pytest.fixture
    def graph_file(self, tmp_path: Path) -> Path:
        """Return path to the graph JSON file."""
        return tmp_path / "graph.json"

    # --- store_relationship tests ---

    @pytest.mark.asyncio
    async def test_store_relationship_creates_edge(self, store: JSONGraphStore):
        """store_relationship should create a new edge."""
        await store.store_relationship("story-001", "req-001", "implements")
        results = await store.get_relationships(source="story-001")
        assert len(results) == 1
        assert results[0].source == "story-001"
        assert results[0].target == "req-001"
        assert results[0].relation == "implements"

    @pytest.mark.asyncio
    async def test_store_relationship_prevents_duplicates(self, store: JSONGraphStore):
        """store_relationship should ignore duplicate edges."""
        await store.store_relationship("story-001", "req-001", "implements")
        await store.store_relationship("story-001", "req-001", "implements")
        results = await store.get_relationships()
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_store_relationship_allows_different_edges(self, store: JSONGraphStore):
        """store_relationship should allow edges with different values."""
        await store.store_relationship("story-001", "req-001", "implements")
        await store.store_relationship("story-001", "req-002", "implements")
        await store.store_relationship("story-001", "req-001", "depends_on")
        results = await store.get_relationships()
        assert len(results) == 3

    # --- get_relationships filter tests ---

    @pytest.mark.asyncio
    async def test_get_relationships_filters_by_source(self, store: JSONGraphStore):
        """get_relationships should filter by source."""
        await store.store_relationship("story-001", "req-001", "implements")
        await store.store_relationship("story-002", "req-002", "implements")

        results = await store.get_relationships(source="story-001")
        assert len(results) == 1
        assert results[0].source == "story-001"

    @pytest.mark.asyncio
    async def test_get_relationships_filters_by_target(self, store: JSONGraphStore):
        """get_relationships should filter by target."""
        await store.store_relationship("story-001", "req-001", "implements")
        await store.store_relationship("story-002", "req-001", "implements")
        await store.store_relationship("story-003", "req-002", "implements")

        results = await store.get_relationships(target="req-001")
        assert len(results) == 2
        assert all(r.target == "req-001" for r in results)

    @pytest.mark.asyncio
    async def test_get_relationships_filters_by_relation(self, store: JSONGraphStore):
        """get_relationships should filter by relation type."""
        await store.store_relationship("story-001", "req-001", "implements")
        await store.store_relationship("story-002", "req-002", "depends_on")
        await store.store_relationship("story-003", "req-003", "implements")

        results = await store.get_relationships(relation="implements")
        assert len(results) == 2
        assert all(r.relation == "implements" for r in results)

    @pytest.mark.asyncio
    async def test_get_relationships_multiple_filters(self, store: JSONGraphStore):
        """get_relationships should combine multiple filters with AND logic."""
        await store.store_relationship("story-001", "req-001", "implements")
        await store.store_relationship("story-001", "req-002", "implements")
        await store.store_relationship("story-001", "req-001", "depends_on")

        results = await store.get_relationships(source="story-001", target="req-001")
        assert len(results) == 2

        results = await store.get_relationships(
            source="story-001", target="req-001", relation="implements"
        )
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_relationships_returns_all_when_no_filters(self, store: JSONGraphStore):
        """get_relationships should return all relationships when no filters specified."""
        await store.store_relationship("a", "b", "r1")
        await store.store_relationship("c", "d", "r2")
        await store.store_relationship("e", "f", "r3")

        results = await store.get_relationships()
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_get_relationships_returns_empty_list_no_matches(self, store: JSONGraphStore):
        """get_relationships should return empty list when no matches."""
        await store.store_relationship("story-001", "req-001", "implements")
        results = await store.get_relationships(source="nonexistent")
        assert results == []

    # --- get_related transitive query tests ---

    @pytest.mark.asyncio
    async def test_get_related_depth_1(self, store: JSONGraphStore):
        """get_related with depth=1 should return direct neighbors only."""
        await store.store_relationship("A", "B", "links")
        await store.store_relationship("B", "C", "links")

        related = await store.get_related("A", depth=1)
        assert set(related) == {"B"}

    @pytest.mark.asyncio
    async def test_get_related_depth_2(self, store: JSONGraphStore):
        """get_related with depth=2 should return neighbors and their neighbors."""
        await store.store_relationship("A", "B", "links")
        await store.store_relationship("B", "C", "links")
        await store.store_relationship("C", "D", "links")

        related = await store.get_related("A", depth=2)
        assert set(related) == {"B", "C"}

    @pytest.mark.asyncio
    async def test_get_related_handles_cycles(self, store: JSONGraphStore):
        """get_related should handle cycles without infinite loops."""
        await store.store_relationship("A", "B", "links")
        await store.store_relationship("B", "C", "links")
        await store.store_relationship("C", "A", "links")  # Cycle back to A

        related = await store.get_related("A", depth=10)
        assert set(related) == {"B", "C"}

    @pytest.mark.asyncio
    async def test_get_related_branching(self, store: JSONGraphStore):
        """get_related should find all branches."""
        await store.store_relationship("A", "B", "links")
        await store.store_relationship("A", "C", "links")
        await store.store_relationship("B", "D", "links")

        related = await store.get_related("A", depth=2)
        assert set(related) == {"B", "C", "D"}

    @pytest.mark.asyncio
    async def test_get_related_excludes_start_node(self, store: JSONGraphStore):
        """get_related should not include the starting node."""
        await store.store_relationship("A", "B", "links")

        related = await store.get_related("A", depth=1)
        assert "A" not in related

    @pytest.mark.asyncio
    async def test_get_related_no_outgoing_edges(self, store: JSONGraphStore):
        """get_related should return empty for node with no outgoing edges."""
        await store.store_relationship("A", "B", "links")

        related = await store.get_related("B", depth=1)
        assert related == []

    @pytest.mark.asyncio
    async def test_get_related_nonexistent_node(self, store: JSONGraphStore):
        """get_related should return empty for nonexistent node."""
        await store.store_relationship("A", "B", "links")

        related = await store.get_related("X", depth=1)
        assert related == []

    # --- Persistence tests ---

    @pytest.mark.asyncio
    async def test_persistence_saves_to_file(self, store: JSONGraphStore, graph_file: Path):
        """store_relationship should persist to JSON file."""
        await store.store_relationship("story-001", "req-001", "implements")

        assert graph_file.exists()
        data = json.loads(graph_file.read_text())
        assert "edges" in data
        assert len(data["edges"]) == 1
        assert data["edges"][0]["source"] == "story-001"

    @pytest.mark.asyncio
    async def test_persistence_new_instance_loads_data(self, tmp_path: Path):
        """New JSONGraphStore instance should load previously stored data."""
        persist_path = str(tmp_path / "graph.json")

        # Store data with first instance
        store1 = JSONGraphStore(persist_path=persist_path)
        await store1.store_relationship("story-001", "req-001", "implements")
        await store1.store_relationship("story-002", "req-002", "depends_on")

        # Create new instance and verify data is loaded
        store2 = JSONGraphStore(persist_path=persist_path)
        results = await store2.get_relationships()
        assert len(results) == 2

    def test_load_handles_missing_file(self, tmp_path: Path):
        """JSONGraphStore should handle missing file gracefully."""
        store = JSONGraphStore(persist_path=str(tmp_path / "nonexistent.json"))
        # Should not raise, should have empty edges
        assert len(store._edges) == 0

    def test_load_handles_invalid_json(self, tmp_path: Path):
        """JSONGraphStore should handle invalid JSON gracefully."""
        graph_file = tmp_path / "graph.json"
        graph_file.write_text("not valid json")

        store = JSONGraphStore(persist_path=str(graph_file))
        # Should not raise, should have empty edges
        assert len(store._edges) == 0

    def test_load_handles_invalid_structure(self, tmp_path: Path):
        """JSONGraphStore should handle invalid data structure gracefully."""
        graph_file = tmp_path / "graph.json"
        # Missing required keys in edge
        graph_file.write_text('{"edges": [{"source": "a"}]}')

        store = JSONGraphStore(persist_path=str(graph_file))
        # Should not raise, should have empty edges
        assert len(store._edges) == 0

    # --- Concurrent access tests ---

    @pytest.mark.asyncio
    async def test_concurrent_store_no_corruption(self, store: JSONGraphStore):
        """Concurrent store_relationship calls should not corrupt data."""
        # Store many edges concurrently
        tasks = [
            store.store_relationship(f"story-{i}", f"req-{i}", "implements") for i in range(50)
        ]
        await asyncio.gather(*tasks)

        results = await store.get_relationships()
        assert len(results) == 50

    @pytest.mark.asyncio
    async def test_concurrent_store_preserves_all_edges(self, tmp_path: Path):
        """Concurrent stores should preserve all unique edges."""
        persist_path = str(tmp_path / "graph.json")
        store = JSONGraphStore(persist_path=persist_path)

        # Store edges concurrently with some duplicates
        tasks = []
        for i in range(10):
            for j in range(5):
                tasks.append(store.store_relationship(f"source-{i}", f"target-{j}", "link"))
        await asyncio.gather(*tasks)

        results = await store.get_relationships()
        assert len(results) == 50  # 10 * 5 unique combinations


class TestJSONGraphError:
    """Tests for the JSONGraphError exception class."""

    def test_json_graph_error_creation(self):
        """JSONGraphError should store message, operation, and original error."""
        original = OSError("disk full")
        error = JSONGraphError(
            message="Failed to save graph",
            operation="save",
            original_error=original,
        )

        assert str(error) == "Failed to save graph"
        assert error.operation == "save"
        assert error.original_error is original

    def test_json_graph_error_is_exception(self):
        """JSONGraphError should be an Exception."""
        error = JSONGraphError("test", "op", ValueError("val"))
        assert isinstance(error, Exception)
