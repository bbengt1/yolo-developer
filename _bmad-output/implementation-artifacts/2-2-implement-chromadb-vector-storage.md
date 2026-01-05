# Story 2.2: Implement ChromaDB Vector Storage

Status: done

## Story

As a developer,
I want vector embeddings stored in ChromaDB,
So that the system can perform semantic similarity searches efficiently.

## Acceptance Criteria

1. **AC1: ChromaDB Collection Management**
   - **Given** ChromaDB is configured with a persist directory
   - **When** a ChromaMemory instance is created
   - **Then** it creates or connects to a ChromaDB PersistentClient
   - **And** it uses a named collection for storing embeddings
   - **And** the collection is created if it doesn't exist

2. **AC2: Store Embeddings**
   - **Given** a ChromaMemory instance is available
   - **When** I call `store_embedding(key, content, metadata)`
   - **Then** the content is added to the ChromaDB collection with the key as document ID
   - **And** metadata is stored alongside the embedding
   - **And** ChromaDB's default embedding function is used (or configurable)
   - **And** duplicate keys update the existing document (upsert behavior)

3. **AC3: Semantic Similarity Search**
   - **Given** embeddings have been stored in the collection
   - **When** I call `search_similar(query, k=5)`
   - **Then** ChromaDB performs a vector similarity search
   - **And** results are returned as `list[MemoryResult]` ordered by similarity (highest first)
   - **And** the score field reflects ChromaDB's distance metric (converted to similarity)
   - **And** results include stored metadata

4. **AC4: Persistence with Local Directory**
   - **Given** a persist_directory is configured
   - **When** embeddings are stored and the client is closed
   - **Then** data persists to disk in the specified directory
   - **And** a new ChromaMemory instance can access previously stored data
   - **And** the storage location follows architecture conventions (`.yolo/memory/`)

5. **AC5: Error Handling and Retries**
   - **Given** ChromaDB operations may fail (e.g., connection issues, disk errors)
   - **When** a transient error occurs
   - **Then** operations are retried up to 3 times with exponential backoff
   - **And** permanent failures raise descriptive exceptions
   - **And** error states don't corrupt existing data

6. **AC6: Protocol Compliance**
   - **Given** the MemoryStore Protocol defined in Story 2.1
   - **When** ChromaMemory is implemented
   - **Then** it satisfies the MemoryStore protocol (mypy validates)
   - **And** it can be used wherever MemoryStore is expected
   - **And** `store_relationship` method exists (delegates to graph store or no-ops for MVP)

## Tasks / Subtasks

- [x] Task 1: Create ChromaMemory Class Structure (AC: 1, 6)
  - [x] Create `src/yolo_developer/memory/vector.py` module
  - [x] Define `ChromaMemory` class implementing MemoryStore protocol
  - [x] Add constructor accepting `persist_directory` and optional `collection_name`
  - [x] Initialize ChromaDB PersistentClient in constructor
  - [x] Create or get collection with configured name
  - [x] Export `ChromaMemory` from `memory/__init__.py`

- [x] Task 2: Implement store_embedding Method (AC: 2)
  - [x] Implement `async def store_embedding(self, key, content, metadata) -> None`
  - [x] Use ChromaDB's `collection.upsert()` for idempotent storage
  - [x] Map key to ChromaDB's `ids` parameter
  - [x] Map content to `documents` parameter
  - [x] Map metadata to `metadatas` parameter
  - [x] Handle empty metadata dict gracefully

- [x] Task 3: Implement search_similar Method (AC: 3)
  - [x] Implement `async def search_similar(self, query, k=5) -> list[MemoryResult]`
  - [x] Use ChromaDB's `collection.query()` with `query_texts`
  - [x] Convert ChromaDB distances to similarity scores (1 - distance for L2/cosine)
  - [x] Create `MemoryResult` objects from query results
  - [x] Handle empty results gracefully (return empty list)
  - [x] Ensure results are ordered by similarity (highest first)

- [x] Task 4: Implement store_relationship Stub (AC: 6)
  - [x] Implement `async def store_relationship(self, source, target, relation) -> None`
  - [x] For MVP, implement as no-op or log warning (graph storage is Story 2.3)
  - [x] Add docstring noting this will be implemented in Story 2.3

- [x] Task 5: Add Error Handling and Retries (AC: 5)
  - [x] Import tenacity retry decorator directly from tenacity package
  - [x] Apply retry logic to ChromaDB operations (3 attempts, exponential backoff)
  - [x] Handle transient exceptions (OSError, RuntimeError, ConnectionError, TimeoutError)
  - [x] Add retry logging for observability via standard logging module
  - [x] Create ChromaDBError wrapper class for descriptive error context
  - [x] Ensure failed operations don't leave partial state

- [x] Task 6: Write Unit Tests (AC: all)
  - [x] Create `tests/unit/memory/test_vector.py`
  - [x] Test: ChromaMemory satisfies MemoryStore protocol (mypy validation)
  - [x] Test: store_embedding stores content and metadata correctly
  - [x] Test: store_embedding updates existing key (upsert behavior)
  - [x] Test: search_similar returns results ordered by similarity
  - [x] Test: search_similar returns empty list when no matches
  - [x] Test: search_similar respects k parameter
  - [x] Test: Persistence works (store, close, reopen, search finds data)
  - [x] Test: MemoryResult fields are populated correctly from ChromaDB
  - [x] Test: store_relationship exists and doesn't raise

- [x] Task 7: Write Integration Test (AC: 4, 5)
  - [x] Create `tests/integration/test_memory_persistence.py` (if doesn't exist)
  - [x] Test: Full lifecycle with real ChromaDB persistence
  - [x] Test: Error recovery with retry behavior

## Dev Notes

### Critical Architecture Requirements

**From ADR-002 (Memory Persistence Strategy):**
- ChromaDB 1.2.x for vector storage (embedded mode)
- Memory abstraction layer using Protocol pattern (Story 2.1 completed)
- `MemoryStore.store_embedding()`, `search_similar()`, `store_relationship()`

**From Architecture Patterns:**
- Async-first design for all I/O operations
- Tenacity for retry logic (ADR-007)
- snake_case for all state dictionary keys
- Full type annotations on all functions
- Structured logging with structlog

### Implementation Approach

**ChromaDB Integration:**

```python
# src/yolo_developer/memory/vector.py
from __future__ import annotations

import chromadb
from chromadb.config import Settings

from yolo_developer.memory.protocol import MemoryResult, MemoryStore


class ChromaMemory:
    """ChromaDB implementation of MemoryStore for vector embeddings.

    Uses ChromaDB's PersistentClient for local storage with automatic
    embedding generation via the default embedding function.
    """

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "yolo_memory",
    ) -> None:
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

    async def store_embedding(
        self,
        key: str,
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        """Store content with vector embedding."""
        self.collection.upsert(
            ids=[key],
            documents=[content],
            metadatas=[metadata] if metadata else None,
        )

    async def search_similar(
        self,
        query: str,
        k: int = 5,
    ) -> list[MemoryResult]:
        """Search for semantically similar content."""
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to MemoryResult objects
        memory_results = []
        if results["ids"] and results["ids"][0]:
            for i, id_ in enumerate(results["ids"][0]):
                # Convert distance to similarity (1 - distance for cosine)
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1.0 - distance

                memory_results.append(MemoryResult(
                    key=id_,
                    content=results["documents"][0][i] if results["documents"] else "",
                    score=score,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                ))

        return memory_results

    async def store_relationship(
        self,
        source: str,
        target: str,
        relation: str,
    ) -> None:
        """Store relationship (delegates to graph store in Story 2.3)."""
        # No-op for MVP - graph storage implemented in Story 2.3
        pass
```

**Async Considerations:**

ChromaDB operations are synchronous, but our protocol requires async methods. Options:
1. Use `asyncio.to_thread()` to run sync operations in thread pool
2. Keep methods async but call sync ChromaDB directly (acceptable for embedded mode)

For MVP, option 2 is simpler since ChromaDB embedded mode is fast and doesn't block event loop significantly.

**Distance to Similarity Conversion:**

ChromaDB returns distances (lower = more similar). We need to convert:
- For cosine distance: `similarity = 1 - distance`
- Distances are in range [0, 2] for cosine, so similarity is in [-1, 1]
- Clamp to [0, 1] for consistency with MemoryResult.score documentation

### Project Structure Notes

**New Module Location:**
```
src/yolo_developer/memory/
├── __init__.py      # Add ChromaMemory export
├── protocol.py      # Already exists (Story 2.1)
└── vector.py        # NEW: ChromaDB implementation
```

**Test Location:**
```
tests/unit/memory/
├── __init__.py
├── test_protocol.py  # Already exists (Story 2.1)
└── test_vector.py    # NEW: ChromaMemory tests

tests/integration/
└── test_memory_persistence.py  # NEW: Persistence tests
```

### Previous Story Learnings (from Story 2.1)

1. **Protocol compliance** - Ensure all 3 methods match MemoryStore protocol exactly
2. **mypy validation** - Run mypy on both src and tests to verify protocol compliance
3. **Immutable types** - MemoryResult is frozen, don't try to modify instances
4. **Complete type hints** - All parameters and return types must be annotated
5. **Run quality checks** - `ruff check`, `ruff format`, `mypy` before marking complete

### Web Research Findings (2025-2026)

From ChromaDB research:
- ChromaDB uses a three-tier storage architecture (brute force buffer → HNSW cache → Arrow disk)
- 2025 Rust-core rewrite provides 4x performance boost
- `PersistentClient` uses SQLite for ACID transactions
- `AsyncHttpClient` available for server mode (not needed for embedded)
- Default embedding function works well for MVP (can configure later)

### Testing Approach

**Unit Tests (mocked ChromaDB):**
- Use pytest fixtures for ChromaDB client
- Use temp directories for persistence tests
- Verify MemoryResult construction
- Verify protocol compliance

**Integration Tests (real ChromaDB):**
- Full store → search → persist cycle
- Verify data survives client restart

```python
# tests/unit/memory/test_vector.py
import pytest
import tempfile
from yolo_developer.memory import ChromaMemory, MemoryResult, MemoryStore


class TestChromaMemory:
    @pytest.fixture
    def memory(self, tmp_path):
        return ChromaMemory(persist_directory=str(tmp_path))

    def test_satisfies_protocol(self, memory: ChromaMemory) -> None:
        """ChromaMemory should satisfy MemoryStore protocol."""
        store: MemoryStore = memory  # Should pass mypy
        assert store is not None

    @pytest.mark.asyncio
    async def test_store_and_search(self, memory: ChromaMemory) -> None:
        """Should store and retrieve similar content."""
        await memory.store_embedding(
            key="test-1",
            content="Python is a programming language",
            metadata={"type": "fact"}
        )

        results = await memory.search_similar("programming languages", k=1)

        assert len(results) == 1
        assert results[0].key == "test-1"
        assert results[0].content == "Python is a programming language"
        assert results[0].metadata == {"type": "fact"}
        assert 0 <= results[0].score <= 1
```

### References

- [Source: architecture.md#ADR-002] - Memory Persistence Strategy (ChromaDB 1.2.x)
- [Source: architecture.md#ADR-007] - Error Handling Strategy (Tenacity retry)
- [Source: architecture.md#Implementation Patterns] - Async patterns, type hints
- [Source: 2-1-create-memory-store-protocol.md] - MemoryStore protocol definition
- [ChromaDB PyPI](https://pypi.org/project/chromadb/) - Latest package info
- [ChromaDB GitHub](https://github.com/chroma-core/chroma) - Open-source search and retrieval database
- [Real Python ChromaDB Tutorial](https://realpython.com/chromadb-vector-database/) - Embeddings and vector databases

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A

### Completion Notes List

- Implemented ChromaMemory class with full MemoryStore protocol compliance
- Added tenacity retry logic (3 attempts, exponential backoff) for transient error handling
- Cosine distance converted to similarity score using formula: 1.0 - (distance / 2.0)
- Empty metadata handled by passing None to ChromaDB (avoids validation errors)
- All 18 unit tests and 8 integration tests passing
- mypy, ruff check, and ruff format all pass

**Code Review Fixes Applied:**
- Added specific exception handling for transient errors only (OSError, RuntimeError, ConnectionError, TimeoutError)
- Created ChromaDBError wrapper class for descriptive error context
- Added retry attempt logging via standard logging module
- Refactored integration tests to not test private methods
- Updated Task 5 documentation to accurately reflect implementation

### File List

- `src/yolo_developer/memory/vector.py` - ChromaMemory implementation with retry logic
- `src/yolo_developer/memory/__init__.py` - Updated exports (includes ChromaDBError)
- `tests/unit/memory/test_vector.py` - 18 unit tests
- `tests/integration/test_memory_persistence.py` - 8 integration tests
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Updated story status
