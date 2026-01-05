# Story 2.8: Project Isolation

Status: done

## Story

As a developer working on multiple projects,
I want memory completely isolated between projects,
So that one project's context never leaks into another.

## Acceptance Criteria

1. **AC1: Project-Scoped Memory Factory**
   - **Given** a project identifier
   - **When** memory stores are instantiated
   - **Then** a unified MemoryFactory creates all stores with consistent project_id
   - **And** all stores use the same project_id for collection/file naming
   - **And** factory validates project_id format (alphanumeric, hyphens, underscores)

2. **AC2: ChromaDB Collection Isolation**
   - **Given** two different project identifiers
   - **When** ChromaMemory instances are created for each
   - **Then** each project uses separate ChromaDB collections
   - **And** collection names follow pattern `{store_type}_{project_id}` (e.g., `yolo_memory_my-project`)
   - **And** queries on Project A never return results from Project B

3. **AC3: JSON Graph Storage Isolation**
   - **Given** two different project identifiers
   - **When** JSONGraphStore instances are created for each
   - **Then** each project uses separate JSON files
   - **And** file paths follow pattern `{base_dir}/{project_id}/graph.json`
   - **And** relationships from Project A are invisible to Project B

4. **AC4: Pattern and Decision Store Isolation**
   - **Given** two different project identifiers
   - **When** ChromaPatternStore and ChromaDecisionStore are created
   - **Then** patterns stored in Project A are isolated from Project B
   - **And** decisions stored in Project A are isolated from Project B
   - **And** each uses collection names with project_id prefix

5. **AC5: Context Switching**
   - **Given** a MemoryFactory configured for Project A
   - **When** the user switches to Project B
   - **Then** a new factory instance is created with Project B's identifier
   - **And** all subsequent operations use Project B's isolated stores
   - **And** Project A's stores are no longer accessible in the current context

6. **AC6: Project ID Validation**
   - **Given** an invalid project identifier (empty, special chars, too long)
   - **When** creating memory stores with that identifier
   - **Then** a clear error is raised explaining valid formats
   - **And** valid formats include: alphanumeric, hyphens, underscores, 1-64 chars

## Tasks / Subtasks

- [x] Task 1: Define Project Isolation Types (AC: 1, 6)
  - [x] Create `src/yolo_developer/memory/isolation.py` module
  - [x] Define `ProjectId` type alias with validation
  - [x] Create `validate_project_id(project_id: str) -> str` function
  - [x] Define `InvalidProjectIdError` exception class
  - [x] Add regex pattern for valid project IDs (alphanumeric, hyphens, underscores, 1-64 chars)

- [x] Task 2: Implement MemoryFactory (AC: 1, 2, 3, 4, 5)
  - [x] Create `src/yolo_developer/memory/factory.py` module
  - [x] Implement `MemoryFactory` class with project_id parameter
  - [x] Add `create_vector_store() -> ChromaMemory` method
  - [x] Add `create_graph_store() -> JSONGraphStore` method
  - [x] Add `create_pattern_store() -> ChromaPatternStore` method
  - [x] Add `create_decision_store() -> ChromaDecisionStore` method
  - [x] Add `get_all_stores()` convenience method returning all stores
  - [x] Validate project_id in `__init__`

- [x] Task 3: Update ChromaMemory for Project Isolation (AC: 2)
  - [x] Modify `ChromaMemory.__init__` to accept `project_id` parameter
  - [x] Update default collection name from `yolo_memory` to `yolo_memory_{project_id}`
  - [x] Ensure backward compatibility with existing code (optional project_id)
  - [x] Update docstrings with project isolation examples

- [x] Task 4: Update JSONGraphStore for Project Isolation (AC: 3)
  - [x] JSONGraphStore unchanged - factory constructs project-scoped paths
  - [x] Create project-scoped directory structure: `{base_dir}/{project_id}/graph.json`
  - [x] Backward compatibility maintained via direct `persist_path` usage
  - [x] Factory handles path construction with project_id

- [x] Task 5: Verify Pattern and Decision Store Isolation (AC: 4)
  - [x] Verify `ChromaPatternStore` already uses `patterns_{project_id}` naming (Story 2.6)
  - [x] Verify `ChromaDecisionStore` already uses `decisions_{project_id}` naming (Story 2.7)
  - [x] Add factory methods to ensure consistent project_id across all stores
  - [x] Add integration test confirming pattern/decision isolation

- [x] Task 6: Export from Memory Module (AC: 1)
  - [x] Update `src/yolo_developer/memory/__init__.py` with new exports
  - [x] Export `MemoryFactory`, `validate_project_id`, `InvalidProjectIdError`
  - [x] Update module docstring with factory usage example

- [x] Task 7: Write Unit Tests (AC: all)
  - [x] Create `tests/unit/memory/test_isolation.py` for validation functions
  - [x] Create `tests/unit/memory/test_factory.py` for MemoryFactory
  - [x] Test project_id validation: valid patterns, invalid patterns, edge cases
  - [x] Test factory creates stores with correct project_id

- [x] Task 8: Write Integration Tests (AC: 2, 3, 4, 5)
  - [x] Create `tests/integration/test_project_isolation.py`
  - [x] Test ChromaMemory isolation: store in A, query in B returns empty
  - [x] Test JSONGraphStore isolation: relationships in A invisible to B
  - [x] Test PatternStore isolation: patterns in A not found in B
  - [x] Test DecisionStore isolation: decisions in A not found in B
  - [x] Test context switching: switch projects, verify isolation

## Dev Notes

### Architecture Compliance

- **ADR-002 (Memory Persistence):** ChromaDB embedded for vector storage, JSON for graph
- **FR35 (PRD):** System can isolate memory between different projects
- **Security Requirement:** Project isolation - no cross-project data access

### Technical Requirements

- **Project ID Format:** Alphanumeric, hyphens, underscores, 1-64 characters
- **Collection Naming:** `{store_type}_{project_id}` pattern
- **Directory Structure:** `{base_dir}/{project_id}/` for file-based stores
- **Backward Compatibility:** Existing code without project_id should continue working with "default"

### Library/Framework Requirements

- **ChromaDB 1.2.x:** Collection names must be valid ChromaDB identifiers (3-63 chars, alphanumeric/underscore)
- **Tenacity:** Retry logic already in place (from Stories 2.2-2.7)
- **asyncio.to_thread:** Wrap blocking operations (pattern from Stories 2.5-2.7)

### File Structure Requirements

```
src/yolo_developer/memory/
├── isolation.py        # NEW: Project ID validation and types
├── factory.py          # NEW: MemoryFactory for unified store creation
├── vector.py           # MODIFY: Add project_id parameter
├── graph.py            # MODIFY: Add project_id-based paths
├── pattern_store.py    # VERIFY: Already uses project_id
├── decision_store.py   # VERIFY: Already uses project_id
├── __init__.py         # MODIFY: Export new classes
└── protocol.py         # NO CHANGE
```

### Testing Standards

- Use pytest-asyncio for async tests
- Use tmp_path fixture for isolated test directories
- Create separate project_ids for each test to ensure isolation
- Test both positive (isolation works) and negative (no cross-leak) cases

### Previous Story Intelligence (from Story 2.7)

**Learnings to Apply:**
1. ChromaDB collection naming: `{type}_{project_id}` pattern established
2. Use `get_or_create_collection()` for safe collection access
3. Tenacity retry for all ChromaDB operations
4. `asyncio.to_thread()` for blocking ChromaDB calls
5. Frozen dataclasses for immutable data structures
6. Export validation functions from `__init__.py`

**Files to Reference:**
- `src/yolo_developer/memory/decision_store.py` - Project isolation pattern (collection naming)
- `src/yolo_developer/memory/pattern_store.py` - Project isolation pattern
- `src/yolo_developer/memory/decisions.py` - Validation function pattern (validate_agent_type)
- `tests/integration/test_decision_queries.py` - Integration test patterns

### Git Intelligence (Recent Commits)

Recent implementation patterns from Stories 2.2-2.7:
- Collection naming: `{type}_{project_id}` (pattern_store, decision_store)
- Default project_id: `"default"` when not specified
- Validation functions with descriptive error messages
- Export from `__init__.py` for public API
- Integration tests verify full lifecycle

### Project Structure Notes

- Alignment with `src/yolo_developer/memory/` module organization
- Factory pattern provides centralized store creation
- Follows snake_case naming convention (from architecture patterns)
- No changes to MemoryStore protocol (protocol remains storage-agnostic)

### References

- [Source: architecture.md#ADR-002] - Memory Persistence Strategy
- [Source: prd.md#FR35] - System can isolate memory between different projects
- [Source: prd.md#Security] - Project isolation, no cross-project data access
- [Source: epics.md#Story-2.8] - Project Isolation requirements
- [Story 2.6 Implementation] - ChromaPatternStore project isolation pattern
- [Story 2.7 Implementation] - ChromaDecisionStore project isolation pattern
- [ChromaDB Collections](https://docs.trychroma.com/guides#creating-inspecting-and-deleting-collections)

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

None

### Completion Notes List

1. **Task 1 Complete**: Created `isolation.py` with project ID validation
   - `DEFAULT_PROJECT_ID = "default"` for backward compatibility
   - `validate_project_id()` function with regex validation
   - `InvalidProjectIdError` exception with descriptive messages
   - 24 unit tests all passing

2. **Task 2 Complete**: Created `factory.py` with MemoryFactory class
   - Factory creates all stores with consistent project_id
   - Methods: `create_vector_store()`, `create_graph_store()`, `create_pattern_store()`, `create_decision_store()`, `get_all_stores()`
   - 18 unit tests all passing

3. **Task 3 Complete**: Updated ChromaMemory for project isolation
   - Added `project_id` parameter to `__init__`
   - Collection names now include project_id: `{collection_name}_{project_id}`
   - Backward compatible with DEFAULT_PROJECT_ID
   - All 27 existing tests still pass

4. **Task 4 Complete**: JSONGraphStore isolation via factory
   - Factory constructs project-scoped paths: `{base_dir}/{project_id}/graph.json`
   - JSONGraphStore unchanged - backward compatible
   - Integration tests verify isolation

5. **Task 5 Complete**: Verified Pattern and Decision Store isolation
   - Both stores already use project_id in collection naming
   - Fixed ChromaDB client type inconsistency (changed from `Client()` to `PersistentClient()`)
   - This fixed test conflicts when creating multiple stores in same directory

6. **Task 6 Complete**: Exported new classes from memory module
   - Added `MemoryFactory`, `InvalidProjectIdError`, `validate_project_id` to `__all__`
   - Updated module docstring with factory usage example

7. **Task 7 Complete**: Unit tests written and passing
   - `test_isolation.py`: 24 tests for project ID validation
   - `test_factory.py`: 18 tests for MemoryFactory
   - All 267 memory unit tests pass

8. **Task 8 Complete**: Integration tests written and passing
   - `test_project_isolation.py`: 12 comprehensive integration tests
   - Tests verify ChromaMemory, JSONGraphStore, PatternStore, DecisionStore isolation
   - Tests verify context switching between projects
   - Tests verify backward compatibility with default project_id

**Total Tests**: 279 tests passing (267 unit + 12 integration)

### File List

**New Files:**
- `src/yolo_developer/memory/isolation.py` - Project ID validation and types
- `src/yolo_developer/memory/factory.py` - MemoryFactory for unified store creation
- `tests/unit/memory/test_isolation.py` - Unit tests for isolation module
- `tests/unit/memory/test_factory.py` - Unit tests for MemoryFactory
- `tests/integration/test_project_isolation.py` - Integration tests for project isolation

**Modified Files:**
- `src/yolo_developer/memory/vector.py` - Added project_id parameter to ChromaMemory
- `src/yolo_developer/memory/pattern_store.py` - Changed to PersistentClient
- `src/yolo_developer/memory/decision_store.py` - Changed to PersistentClient
- `src/yolo_developer/memory/__init__.py` - Added new exports
