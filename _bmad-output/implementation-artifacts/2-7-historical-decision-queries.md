# Story 2.7: Historical Decision Queries

Status: done

## Story

As a developer,
I want to query past decisions,
So that agents can learn from previous similar situations.

## Acceptance Criteria

1. **AC1: Decision Storage Infrastructure**
   - **Given** decisions have been made during agent execution
   - **When** a decision is logged
   - **Then** the decision is stored with its rationale
   - **And** metadata includes agent type, timestamp, and artifact references
   - **And** decisions are embedded for semantic search

2. **AC2: Semantic Similarity Search**
   - **Given** decisions have been stored from previous sprints
   - **When** an agent faces a similar situation
   - **Then** it can query for semantically related historical decisions
   - **And** similarity matching uses vector embeddings
   - **And** results are ranked by relevance score

3. **AC3: Decision Rationale Inclusion**
   - **Given** a historical decision is retrieved
   - **When** the result is returned
   - **Then** the full decision rationale is included
   - **And** the original context is preserved
   - **And** the outcome/result is recorded if available

4. **AC4: Filtering Capabilities**
   - **Given** a query for historical decisions
   - **When** filters are applied
   - **Then** results can be filtered by agent type (Analyst, PM, Architect, Dev, SM, TEA)
   - **And** results can be filtered by time range
   - **And** results can be filtered by artifact type (requirement, story, design, code)

5. **AC5: Integration with Memory Layer**
   - **Given** the existing MemoryStore protocol
   - **When** decision queries are implemented
   - **Then** they use the existing ChromaDB infrastructure
   - **And** decisions are stored in a separate collection for isolation
   - **And** retrieval performance is under 500ms

## Tasks / Subtasks

- [x] Task 1: Define Decision Data Structures (AC: 1, 3)
  - [x] Create `src/yolo_developer/memory/decisions.py` module
  - [x] Define `Decision` dataclass with id, agent_type, timestamp, context, rationale, outcome
  - [x] Define `DecisionType` enum (REQUIREMENT_CLARIFICATION, STORY_PRIORITIZATION, ARCHITECTURE_CHOICE, IMPLEMENTATION_APPROACH, etc.)
  - [x] Define `DecisionResult` dataclass wrapping Decision with similarity score
  - [x] Add `to_embedding_text()` method for semantic search
  - [x] Export from `memory/__init__.py`

- [x] Task 2: Extend MemoryStore Protocol (AC: 5)
  - [x] Add `store_decision(decision: Decision) -> str` to MemoryStore protocol
  - [x] Add `search_decisions(query: str, filters: DecisionFilter, k: int) -> list[DecisionResult]`
  - [x] Add `get_decisions_by_agent(agent_type: str, limit: int) -> list[Decision]`
  - [x] Define `DecisionFilter` dataclass with agent_type, time_range, artifact_type fields

- [x] Task 3: Implement ChromaDB Decision Store (AC: 1, 2, 5)
  - [x] Create `src/yolo_developer/memory/decision_store.py` module
  - [x] Implement `ChromaDecisionStore` class
  - [x] Create separate ChromaDB collection for decisions (project-isolated)
  - [x] Implement `store_decision()` with embedding generation
  - [x] Implement metadata storage for filtering support
  - [x] Add tenacity retry logic for ChromaDB operations

- [x] Task 4: Implement Semantic Search with Filters (AC: 2, 4)
  - [x] Implement `search_decisions()` with ChromaDB where clause
  - [x] Support agent_type filter via metadata
  - [x] Support time_range filter via timestamp metadata
  - [x] Support artifact_type filter via metadata
  - [x] Implement combined semantic + metadata filtering
  - [x] Return results ranked by similarity score

- [x] Task 5: Implement Decision Retrieval Methods (AC: 3, 4)
  - [x] Implement `get_decisions_by_agent()` for agent-specific history
  - [x] Implement `get_decision_by_id()` for specific decision lookup
  - [x] Ensure rationale and context are fully preserved in results
  - [x] Add pagination support for large result sets

- [x] Task 6: Create Decision Query Interface (AC: 2, 3, 4)
  - [x] Create `src/yolo_developer/memory/decision_queries.py` module
  - [x] Implement `DecisionQueryEngine` class as high-level interface
  - [x] Add `find_similar_decisions(context: str, k: int = 5) -> list[DecisionResult]`
  - [x] Add `get_agent_decision_history(agent_type: str, limit: int = 10) -> list[Decision]`
  - [x] Add `search_with_filters(query: str, filters: DecisionFilter) -> list[DecisionResult]`

- [x] Task 7: Write Unit Tests (AC: all)
  - [x] Create `tests/unit/memory/test_decisions.py` for data structures
  - [x] Create `tests/unit/memory/test_decision_store.py` for storage
  - [x] Create `tests/unit/memory/test_decision_queries.py` for query engine
  - [x] Test edge cases: empty results, invalid filters, large result sets

- [x] Task 8: Write Integration Tests (AC: all)
  - [x] Create `tests/integration/test_decision_queries.py`
  - [x] Test full decision lifecycle: store -> query -> retrieve
  - [x] Test filtering by agent type, time range, artifact type
  - [x] Test semantic similarity matching
  - [x] Verify retrieval performance < 500ms (AC5)

## Dev Notes

### Architecture Compliance

- **ADR-002 (Memory Persistence):** Use ChromaDB embedded for decision storage, separate collection from artifacts
- **ADR-001 (State Management):** Use dataclass for Decision (frozen=True for immutability)
- **Pattern from Story 2.6:** Follow same ChromaDB collection isolation pattern used for patterns

### Technical Requirements

- **Embedding Model:** Use same sentence-transformer as artifact embeddings (consistency)
- **Collection Naming:** `decisions_{project_id}` for project isolation
- **Metadata Schema:** Store agent_type, timestamp (ISO 8601), artifact_type, decision_type as ChromaDB metadata
- **Performance Target:** Query latency < 500ms per AC5

### Library/Framework Requirements

- **ChromaDB 1.2.x:** Already installed, use `collection.query()` with `where` clause for filtering
- **Tenacity:** Apply retry decorator for ChromaDB operations (pattern from Story 2.2)
- **asyncio.to_thread:** Wrap blocking ChromaDB calls (pattern from Story 2.5)

### File Structure Requirements

```
src/yolo_developer/memory/
├── decisions.py          # NEW: Decision data structures
├── decision_store.py     # NEW: ChromaDB decision storage
├── decision_queries.py   # NEW: High-level query interface
├── __init__.py           # UPDATE: Export new classes
└── protocol.py           # UPDATE: Add decision methods
```

### Testing Standards

- Use pytest-asyncio for async tests
- Use tmp_path fixture for isolated ChromaDB instances
- Test performance with time.perf_counter()
- Mock ChromaDB for unit tests, use real ChromaDB for integration

### Previous Story Intelligence (from Story 2.6)

**Learnings to Apply:**
1. ChromaDB collection creation via `get_or_create_collection()` with project-specific names
2. Use `asyncio.to_thread()` for blocking ChromaDB operations
3. Implement caching for frequently-queried data (TTL-based)
4. Metadata filtering via ChromaDB `where` clause
5. Use `to_embedding_text()` pattern for semantic search content
6. Apply tenacity retry for all ChromaDB operations

**Files to Reference:**
- `src/yolo_developer/memory/pattern_store.py` - ChromaDB storage pattern
- `src/yolo_developer/memory/learning.py` - Caching implementation
- `src/yolo_developer/memory/patterns.py` - Data structure patterns

### Git Intelligence (Recent Commits)

Recent implementation patterns from Stories 2.2-2.6:
- Consistent use of `@dataclass(frozen=True)` for immutable data
- Protocol-based abstractions in `protocol.py`
- Separate store implementations (ChromaDB) from orchestrators
- Integration tests verify full lifecycle

### Project Structure Notes

- Alignment with `src/yolo_developer/memory/` module organization
- Decision storage extends existing memory layer without breaking changes
- Follows snake_case naming convention (from project patterns)

### References

- [Source: architecture.md#ADR-002] - ChromaDB for vector storage
- [Source: epics.md#Story-2.7] - Historical Decision Queries requirements
- [Source: prd.md#FR33] - System can query historical decisions for similar situations
- [ChromaDB Filtering](https://docs.trychroma.com/guides#filtering-by-metadata)
- [Story 2.6 Implementation] - Pattern for ChromaDB collection management

## Dev Agent Record

### Agent Model Used
Claude Opus 4.5

### Debug Log References
- 67 tests passing (19 unit test_decisions + 16 unit test_decision_store + 16 unit test_decision_queries + 16 integration)
- All code quality checks pass (ruff check, ruff format, mypy)
- Performance tests confirm < 500ms query latency

### Completion Notes List
- Implemented frozen dataclasses following Story 2.6 patterns
- Used tenacity retry decorator for ChromaDB resilience
- Implemented `to_chromadb_where()` method on DecisionFilter for clean filter conversion
- DecisionQueryEngine provides high-level interface with `record_decision()` convenience method
- Full project isolation via `decisions_{project_id}` collection naming
- All 5 acceptance criteria met

### File List
- `src/yolo_developer/memory/decisions.py` (NEW) - Decision, DecisionType, DecisionResult, DecisionFilter dataclasses
- `src/yolo_developer/memory/decision_store.py` (NEW) - ChromaDecisionStore implementation
- `src/yolo_developer/memory/decision_queries.py` (NEW) - DecisionQueryEngine high-level interface
- `src/yolo_developer/memory/protocol.py` (MODIFIED) - Added decision methods to MemoryStore protocol
- `src/yolo_developer/memory/__init__.py` (MODIFIED) - Exports new classes
- `src/yolo_developer/memory/learning.py` (MODIFIED) - Fixed mypy type error, added cast import
- `src/yolo_developer/memory/pattern_store.py` (MODIFIED) - Code formatting (ruff)
- `src/yolo_developer/memory/scanner.py` (MODIFIED) - Code formatting (ruff)
- `tests/unit/memory/test_decisions.py` (NEW) - 19 unit tests for data structures
- `tests/unit/memory/test_decision_store.py` (NEW) - 16 unit tests for storage
- `tests/unit/memory/test_decision_queries.py` (NEW) - 16 unit tests for query engine
- `tests/integration/test_decision_queries.py` (NEW) - 16 integration tests
- `tests/integration/test_pattern_learning.py` (MODIFIED) - Code formatting (ruff)
- `tests/unit/memory/test_analyzers.py` (MODIFIED) - Fixed unused variable, removed duplicate import
- `tests/unit/memory/test_learning.py` (MODIFIED) - Code formatting (ruff)
- `tests/unit/memory/test_pattern_store.py` (MODIFIED) - Code formatting (ruff)

## Senior Developer Review (AI)

### Review Date
2026-01-05

### Reviewer
Claude Opus 4.5 (Code Review Workflow)

### Review Outcome
**APPROVED** - All HIGH and MEDIUM issues fixed

### Issues Found and Resolved

| Severity | Issue | Resolution |
|----------|-------|------------|
| HIGH | H1/H2: Story File List incomplete - 8 files modified but not documented | Updated File List to include all modified files |
| HIGH | H3: No input validation on agent_type field | Added `VALID_AGENT_TYPES` constant, `validate_agent_type()` function, and `__post_init__` validation in Decision class |
| MEDIUM | M3: Missing error handling for invalid DecisionType | Added try/except with logging in `_metadata_to_decision()` |
| MEDIUM | M4: Unused mock fixture in test_decision_store.py | Removed unused `mock_chromadb_client` fixture |

### Tests After Review
- 71 tests passing (23 unit test_decisions + 16 unit test_decision_store + 16 unit test_decision_queries + 16 integration)
- All code quality checks pass (ruff check, ruff format, mypy)

### Files Modified During Review
- `src/yolo_developer/memory/decisions.py` - Added VALID_AGENT_TYPES, validate_agent_type(), Decision.__post_init__
- `src/yolo_developer/memory/decision_store.py` - Added DecisionType parsing error handling
- `src/yolo_developer/memory/__init__.py` - Export VALID_AGENT_TYPES and validate_agent_type
- `tests/unit/memory/test_decisions.py` - Added 4 tests for agent_type validation
- `tests/unit/memory/test_decision_store.py` - Removed unused mock fixture

