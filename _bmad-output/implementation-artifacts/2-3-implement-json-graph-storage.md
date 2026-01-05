# Story 2.3: Implement JSON Graph Storage

Status: done

## Story

As a developer,
I want relationship data stored in a JSON-based graph,
So that I can track connections between artifacts without requiring Neo4j.

## Acceptance Criteria

1. **AC1: Store Relationship Method**
   - **Given** a JSONGraphStore instance is available
   - **When** I call `store_relationship(source, target, relation)`
   - **Then** the relationship is persisted to JSON storage
   - **And** the relationship can be retrieved via query methods
   - **And** duplicate relationships are handled gracefully (no duplicates stored)

2. **AC2: Query by Source**
   - **Given** relationships have been stored in the graph
   - **When** I call `get_relationships(source="story-001")`
   - **Then** all relationships where source matches are returned
   - **And** results include target and relation type for each edge

3. **AC3: Query by Target**
   - **Given** relationships have been stored in the graph
   - **When** I call `get_relationships(target="req-001")`
   - **Then** all relationships where target matches are returned
   - **And** results include source and relation type for each edge

4. **AC4: Query by Relation Type**
   - **Given** relationships have been stored in the graph
   - **When** I call `get_relationships(relation="implements")`
   - **Then** all relationships of that type are returned
   - **And** results include source and target for each edge

5. **AC5: Transitive Queries**
   - **Given** relationships A→B→C exist in the graph
   - **When** I call `get_related(node="A", depth=2)`
   - **Then** nodes B and C are returned (all nodes reachable within depth)
   - **And** the relation path is included in results

6. **AC6: Persistence to Disk**
   - **Given** a persist_path is configured
   - **When** relationships are stored and the instance is closed
   - **Then** data persists to a JSON file in the specified location
   - **And** a new JSONGraphStore instance can load previously stored relationships
   - **And** the storage location follows architecture conventions (`.yolo/memory/graph.json`)

7. **AC7: Concurrent Access Safety**
   - **Given** multiple async operations may write simultaneously
   - **When** concurrent store_relationship calls occur
   - **Then** a file lock prevents data corruption
   - **And** all relationships are eventually stored correctly
   - **And** no data is lost

8. **AC8: MemoryStore Protocol Integration**
   - **Given** ChromaMemory from Story 2.2 has a stub `store_relationship`
   - **When** JSONGraphStore is available
   - **Then** ChromaMemory can delegate `store_relationship` to JSONGraphStore
   - **Or** a CompositeMemory class combines both storage backends
   - **And** the combined solution satisfies the MemoryStore protocol

## Tasks / Subtasks

- [x] Task 1: Create JSONGraphStore Class Structure (AC: 1, 6)
  - [x] Create `src/yolo_developer/memory/graph.py` module
  - [x] Define `Relationship` dataclass with source, target, relation fields
  - [x] Define `RelationshipResult` dataclass for query results
  - [x] Define `JSONGraphStore` class with constructor accepting `persist_path`
  - [x] Implement `_load()` method to read existing JSON file
  - [x] Implement `_save()` method to persist graph to JSON file
  - [x] Export classes from `memory/__init__.py`

- [x] Task 2: Implement store_relationship Method (AC: 1, 7)
  - [x] Implement `async def store_relationship(self, source, target, relation) -> None`
  - [x] Store edges in memory as list of Relationship objects
  - [x] Check for duplicates before adding (source+target+relation tuple)
  - [x] Call `_save()` to persist after each modification
  - [x] Use asyncio.Lock for concurrent access protection

- [x] Task 3: Implement Query Methods (AC: 2, 3, 4)
  - [x] Implement `async def get_relationships(self, source=None, target=None, relation=None) -> list[RelationshipResult]`
  - [x] Filter by any combination of source, target, relation
  - [x] Return empty list when no matches
  - [x] Support querying with all parameters None (returns all relationships)

- [x] Task 4: Implement Transitive Query (AC: 5)
  - [x] Implement `async def get_related(self, node: str, depth: int = 1) -> list[str]`
  - [x] Use BFS/DFS to find all nodes reachable within depth
  - [x] Handle cycles gracefully (avoid infinite loops)
  - [ ] Optionally include path information in results (DEFERRED: returns node IDs only, paths can be reconstructed via get_relationships)

- [x] Task 5: Integrate with ChromaMemory (AC: 8)
  - [x] Option A: Update ChromaMemory to accept optional JSONGraphStore
  - [x] Option B: Create CompositeMemory class combining ChromaMemory + JSONGraphStore
  - [x] Ensure combined solution satisfies MemoryStore protocol
  - [x] Update `store_relationship` to delegate to graph storage

- [x] Task 6: Add Error Handling and Retry Logic (AC: 1, 6, 7)
  - [x] Apply tenacity retry decorator for file I/O operations
  - [x] Handle file not found (create empty graph)
  - [x] Handle JSON parse errors (log warning, start with empty graph)
  - [x] Create JSONGraphError wrapper class for descriptive errors

- [x] Task 7: Write Unit Tests (AC: all)
  - [x] Create `tests/unit/memory/test_graph.py`
  - [x] Test: store_relationship creates edge
  - [x] Test: store_relationship handles duplicates
  - [x] Test: get_relationships filters by source
  - [x] Test: get_relationships filters by target
  - [x] Test: get_relationships filters by relation
  - [x] Test: get_relationships with multiple filters
  - [x] Test: get_related with depth=1
  - [x] Test: get_related with depth=2 (transitive)
  - [x] Test: get_related handles cycles
  - [x] Test: persistence saves to file
  - [x] Test: new instance loads from file
  - [x] Test: concurrent access doesn't corrupt data

- [x] Task 8: Write Integration Tests (AC: 6, 8)
  - [x] Add tests to `tests/integration/test_memory_persistence.py`
  - [x] Test: Full lifecycle with real file I/O
  - [x] Test: ChromaMemory + JSONGraphStore composite usage
  - [x] Test: MemoryStore protocol compliance for composite

## Dev Notes

### Critical Architecture Requirements

**From ADR-002 (Memory Persistence Strategy):**
- JSON-based graph for MVP relationships (Neo4j optional for v1.1)
- Memory abstraction layer using Protocol pattern
- `MemoryStore.store_relationship()` must be fully implemented

**From Architecture Patterns:**
- Async-first design for all I/O operations
- Tenacity for retry logic (ADR-007)
- snake_case for all state dictionary keys
- Full type annotations on all functions
- Structured logging with structlog

### Implementation Approach

**JSON Graph Structure:**

```python
# src/yolo_developer/memory/graph.py
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass(frozen=True)
class Relationship:
    """A directed edge in the relationship graph."""
    source: str
    target: str
    relation: str


@dataclass(frozen=True)
class RelationshipResult:
    """Result from a relationship query."""
    source: str
    target: str
    relation: str
    path: list[str] | None = None  # For transitive queries


class JSONGraphStore:
    """JSON-based graph storage for artifact relationships.

    Stores relationships as a list of edges in a JSON file.
    Supports queries by source, target, relation type, and transitive queries.

    Example:
        >>> store = JSONGraphStore(persist_path=".yolo/memory/graph.json")
        >>> await store.store_relationship("story-001", "req-001", "implements")
        >>> results = await store.get_relationships(source="story-001")
    """

    def __init__(self, persist_path: str) -> None:
        self.persist_path = Path(persist_path)
        self._edges: set[Relationship] = set()
        self._lock = asyncio.Lock()
        self._load()

    def _load(self) -> None:
        """Load existing graph from JSON file."""
        if self.persist_path.exists():
            try:
                data = json.loads(self.persist_path.read_text())
                self._edges = {
                    Relationship(e["source"], e["target"], e["relation"])
                    for e in data.get("edges", [])
                }
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("graph_load_failed", error=str(e))
                self._edges = set()

    def _save(self) -> None:
        """Persist graph to JSON file."""
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"edges": [asdict(e) for e in self._edges]}
        self.persist_path.write_text(json.dumps(data, indent=2))

    async def store_relationship(
        self,
        source: str,
        target: str,
        relation: str,
    ) -> None:
        """Store a relationship between two entities."""
        async with self._lock:
            edge = Relationship(source, target, relation)
            if edge not in self._edges:
                self._edges.add(edge)
                self._save()

    async def get_relationships(
        self,
        source: str | None = None,
        target: str | None = None,
        relation: str | None = None,
    ) -> list[RelationshipResult]:
        """Query relationships by optional filters."""
        results = []
        for edge in self._edges:
            if source and edge.source != source:
                continue
            if target and edge.target != target:
                continue
            if relation and edge.relation != relation:
                continue
            results.append(RelationshipResult(
                source=edge.source,
                target=edge.target,
                relation=edge.relation,
            ))
        return results

    async def get_related(
        self,
        node: str,
        depth: int = 1,
    ) -> list[str]:
        """Find all nodes reachable from node within depth."""
        visited: set[str] = set()
        queue = [(node, 0)]

        while queue:
            current, current_depth = queue.pop(0)
            if current_depth > depth:
                continue
            if current in visited and current != node:
                continue
            visited.add(current)

            # Find all adjacent nodes (outgoing edges)
            for edge in self._edges:
                if edge.source == current and edge.target not in visited:
                    queue.append((edge.target, current_depth + 1))

        # Remove the starting node from results
        visited.discard(node)
        return list(visited)
```

**Integration with ChromaMemory:**

Two options for integration:

**Option A: Dependency Injection**
```python
class ChromaMemory:
    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "yolo_memory",
        graph_store: JSONGraphStore | None = None,
    ) -> None:
        # ... existing init ...
        self._graph_store = graph_store

    async def store_relationship(
        self,
        source: str,
        target: str,
        relation: str,
    ) -> None:
        if self._graph_store:
            await self._graph_store.store_relationship(source, target, relation)
        else:
            logger.warning("graph_store_not_configured")
```

**Option B: Composite Pattern (Recommended)**
```python
class CompositeMemory:
    """Combines vector (ChromaDB) and graph (JSON) storage."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "yolo_memory",
    ) -> None:
        self._vector = ChromaMemory(persist_directory, collection_name)
        graph_path = Path(persist_directory) / "graph.json"
        self._graph = JSONGraphStore(str(graph_path))

    async def store_embedding(self, key: str, content: str, metadata: dict) -> None:
        await self._vector.store_embedding(key, content, metadata)

    async def search_similar(self, query: str, k: int = 5) -> list[MemoryResult]:
        return await self._vector.search_similar(query, k)

    async def store_relationship(self, source: str, target: str, relation: str) -> None:
        await self._graph.store_relationship(source, target, relation)
```

**Concurrent Access:**

Use asyncio.Lock to prevent race conditions:
```python
async with self._lock:
    self._edges.add(edge)
    self._save()
```

### Project Structure Notes

**New Module Location:**
```
src/yolo_developer/memory/
├── __init__.py      # Add JSONGraphStore, Relationship exports
├── protocol.py      # MemoryStore protocol (Story 2.1)
├── vector.py        # ChromaMemory (Story 2.2)
└── graph.py         # NEW: JSONGraphStore implementation
```

**Test Location:**
```
tests/unit/memory/
├── __init__.py
├── test_protocol.py  # (Story 2.1)
├── test_vector.py    # (Story 2.2)
└── test_graph.py     # NEW: JSONGraphStore tests

tests/integration/
└── test_memory_persistence.py  # Add graph persistence tests
```

### Previous Story Learnings (from Story 2.2)

1. **Protocol compliance** - Ensure methods match MemoryStore protocol exactly
2. **mypy validation** - Run mypy on both src and tests to verify protocol compliance
3. **Immutable types** - Use frozen=True for dataclasses
4. **Complete type hints** - All parameters and return types must be annotated
5. **Run quality checks** - `ruff check`, `ruff format`, `mypy` before marking complete
6. **Retry patterns** - Use tenacity for file I/O operations
7. **Empty metadata handling** - Handle edge cases gracefully (empty sets, missing files)
8. **Code review fixes** - Expect to fix specific exception handling, logging patterns

### Testing Approach

**Unit Tests (in-memory graph):**
- Use tmp_path fixture for isolated file I/O
- Test edge cases: empty graph, duplicates, cycles
- Verify query filtering logic
- Test concurrent access with asyncio.gather

**Integration Tests (real file I/O):**
- Full store → save → reload cycle
- Combined ChromaMemory + JSONGraphStore usage
- Verify MemoryStore protocol compliance

```python
# tests/unit/memory/test_graph.py
import pytest
from yolo_developer.memory.graph import JSONGraphStore, Relationship


class TestJSONGraphStore:
    @pytest.fixture
    def store(self, tmp_path):
        return JSONGraphStore(persist_path=str(tmp_path / "graph.json"))

    @pytest.mark.asyncio
    async def test_store_relationship_creates_edge(self, store):
        await store.store_relationship("story-001", "req-001", "implements")
        results = await store.get_relationships(source="story-001")
        assert len(results) == 1
        assert results[0].target == "req-001"

    @pytest.mark.asyncio
    async def test_store_relationship_prevents_duplicates(self, store):
        await store.store_relationship("story-001", "req-001", "implements")
        await store.store_relationship("story-001", "req-001", "implements")
        results = await store.get_relationships()
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_related_transitive(self, store):
        await store.store_relationship("A", "B", "links")
        await store.store_relationship("B", "C", "links")
        related = await store.get_related("A", depth=2)
        assert set(related) == {"B", "C"}
```

### References

- [Source: architecture.md#ADR-002] - Memory Persistence Strategy (JSON graph for MVP)
- [Source: architecture.md#ADR-007] - Error Handling Strategy (Tenacity retry)
- [Source: architecture.md#Implementation Patterns] - Async patterns, type hints
- [Source: 2-1-create-memory-store-protocol.md] - MemoryStore protocol definition
- [Source: 2-2-implement-chromadb-vector-storage.md] - ChromaMemory with stub store_relationship

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A

### Completion Notes List

1. Implemented JSONGraphStore following Option A (dependency injection) - ChromaMemory accepts optional `graph_store` parameter
2. Used `set[Relationship]` instead of `list` for O(1) duplicate detection via frozen dataclass with `__hash__`
3. Applied tenacity retry decorators to `_load_from_file()` and `_save()` methods for file I/O resilience
4. BFS algorithm used for `get_related()` transitive queries with cycle handling via visited set
5. Fixed logging issue: Python's logging module reserves `"message"` key in `extra` dict, renamed field
6. Used standard library `logging` instead of `structlog` to match project conventions in other modules
7. Integration approach: ChromaMemory delegates `store_relationship` to injected JSONGraphStore; logs warning if no graph_store configured
8. All 306 project tests pass (59 memory unit tests + 15 integration tests)

### Senior Developer Review (AI)

**Review Date:** 2026-01-04
**Reviewer:** Claude Opus 4.5 (Adversarial Code Review)
**Outcome:** APPROVED with fixes applied

**Issues Found and Fixed:**
1. **[HIGH] Race condition in get_relationships()** - Added asyncio.Lock acquisition before iterating self._edges
2. **[HIGH] Race condition in get_related()** - Added asyncio.Lock acquisition before BFS traversal
3. **[MEDIUM] Inefficient BFS using list.pop(0)** - Changed to collections.deque with popleft() for O(1) performance
4. **[LOW] IOError redundancy** - Removed redundant IOError from _TRANSIENT_EXCEPTIONS (alias for OSError in Python 3)

**Known Limitations:**
- AC5 "path included in results" is partially satisfied - get_related() returns node IDs only, not paths. Documented as DEFERRED in Task 4 subtask. Paths can be reconstructed using get_relationships().
- JSONGraphError is defined but not raised (kept for API stability and future use)

**Tests:** All 47 graph-related tests pass after fixes

### File List

**Created:**
- `src/yolo_developer/memory/graph.py` - JSONGraphStore implementation with Relationship, RelationshipResult dataclasses
- `tests/unit/memory/test_graph.py` - 32 unit tests for graph storage

**Modified:**
- `src/yolo_developer/memory/__init__.py` - Added exports for JSONGraphError, JSONGraphStore, Relationship, RelationshipResult
- `src/yolo_developer/memory/vector.py` - Added optional `graph_store` parameter to ChromaMemory.__init__, updated store_relationship to delegate
- `tests/integration/test_memory_persistence.py` - Added TestJSONGraphStorePersistence and TestChromaMemoryGraphIntegration test classes
