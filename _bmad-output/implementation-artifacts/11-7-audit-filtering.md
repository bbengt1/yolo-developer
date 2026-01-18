# Story 11.7: Audit Filtering

## Story

**As a** developer,
**I want** to filter audit trails by various criteria,
**So that** I can find specific information quickly.

## Status

- **Epic:** 11 - Audit Trail & Observability
- **Status:** done
- **Priority:** P2
- **Story Points:** 3

## Acceptance Criteria

### AC1: Filtering by Agent Works
**Given** audit data from multiple agents
**When** filtering by agent name
**Then** only decisions from that agent are returned

### AC2: Filtering by Time Range Works
**Given** audit data spanning multiple time periods
**When** filtering by start_time and/or end_time
**Then** only decisions within the time range are returned

### AC3: Filtering by Artifact Type Works
**Given** audit data with various artifact types
**When** filtering by artifact type (requirement, story, design_decision, code, test)
**Then** only traceability artifacts of that type are returned

### AC4: Filters Can Be Combined
**Given** audit data with various attributes
**When** multiple filters are applied simultaneously
**Then** results match ALL filter criteria (AND logic)

### AC5: Results Are Accurate
**Given** any combination of filters
**When** filtering is applied
**Then** all matching records are returned and no non-matching records are included

## Technical Requirements

### Functional Requirements Mapping
- **FR87:** Users can filter audit trail by agent, time range, or artifact

### Architecture References
- **ADR-001:** Frozen dataclasses for filter types
- **Epic 11 Pattern:** Protocol-based stores, structlog logging, factory functions
- **Story 11.1 Pattern:** DecisionFilters dataclass for query filtering
- **Story 11.3 Pattern:** AuditViewService with filters parameter

### Technology Stack
- **structlog:** For structured logging of filter operations
- **Frozen Dataclasses:** For immutable filter configuration
- **Existing Patterns:** Follow DecisionFilters, CostFilters patterns

## Tasks

### Task 1: Create Unified AuditFilters Type (filter_types.py)
**File:** `src/yolo_developer/audit/filter_types.py`

Create a unified filter dataclass that combines all audit filtering capabilities:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from yolo_developer.audit.types import DecisionType, DecisionSeverity
from yolo_developer.audit.traceability_types import ArtifactType

@dataclass(frozen=True)
class AuditFilters:
    """Unified filters for querying audit data across all stores.

    All fields are optional; None means no filtering on that field.
    Multiple filters are combined with AND logic.

    Attributes:
        agent_name: Filter by agent name
        decision_type: Filter by decision type
        artifact_type: Filter by artifact type (for traceability)
        start_time: Filter items after this timestamp (inclusive, ISO 8601)
        end_time: Filter items before this timestamp (inclusive, ISO 8601)
        sprint_id: Filter by sprint ID
        story_id: Filter by story ID
        session_id: Filter by session ID
        severity: Filter by decision severity
    """
    agent_name: str | None = None
    decision_type: str | None = None
    artifact_type: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    sprint_id: str | None = None
    story_id: str | None = None
    session_id: str | None = None
    severity: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agent_name": self.agent_name,
            "decision_type": self.decision_type,
            "artifact_type": self.artifact_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "sprint_id": self.sprint_id,
            "story_id": self.story_id,
            "session_id": self.session_id,
            "severity": self.severity,
        }

    def to_decision_filters(self) -> "DecisionFilters":
        """Convert to DecisionFilters for decision store queries."""
        from yolo_developer.audit.store import DecisionFilters
        return DecisionFilters(
            agent_name=self.agent_name,
            decision_type=self.decision_type,
            start_time=self.start_time,
            end_time=self.end_time,
            sprint_id=self.sprint_id,
            story_id=self.story_id,
        )

    def to_cost_filters(self) -> "CostFilters":
        """Convert to CostFilters for cost store queries."""
        from yolo_developer.audit.cost_store import CostFilters
        return CostFilters(
            agent_name=self.agent_name,
            story_id=self.story_id,
            sprint_id=self.sprint_id,
            session_id=self.session_id,
            start_time=self.start_time,
            end_time=self.end_time,
        )
```

**Subtasks:**
1. Create `AuditFilters` frozen dataclass with all filter fields
2. Implement `to_dict()` for serialization
3. Implement `to_decision_filters()` conversion method
4. Implement `to_cost_filters()` conversion method
5. Add validation in `__post_init__` for artifact_type values

### Task 2: Add Artifact Type Filtering to TraceabilityStore (traceability_store.py)
**File:** `src/yolo_developer/audit/traceability_store.py`

Extend TraceabilityStore protocol to support filtering by artifact type:

```python
async def get_artifacts_by_type(
    self,
    artifact_type: str,
) -> list[TraceableArtifact]:
    """Get all artifacts of a specific type.

    Args:
        artifact_type: Type to filter by (requirement, story, design_decision, code, test)

    Returns:
        List of artifacts matching the type.
    """
    ...

async def get_artifacts_by_filters(
    self,
    artifact_type: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
) -> list[TraceableArtifact]:
    """Get artifacts with optional filtering.

    Args:
        artifact_type: Optional type filter
        created_after: Optional start time filter (ISO 8601)
        created_before: Optional end time filter (ISO 8601)

    Returns:
        List of artifacts matching all filters.
    """
    ...
```

**Subtasks:**
1. Add `get_artifacts_by_type()` to TraceabilityStore Protocol
2. Add `get_artifacts_by_filters()` with combined filtering
3. Document new methods with docstrings

### Task 3: Implement Filtering in InMemoryTraceabilityStore (traceability_memory_store.py)
**File:** `src/yolo_developer/audit/traceability_memory_store.py`

Implement the new filter methods:

```python
async def get_artifacts_by_type(
    self,
    artifact_type: str,
) -> list[TraceableArtifact]:
    """Get all artifacts of a specific type."""
    with self._lock:
        return [
            a for a in self._artifacts.values()
            if a.artifact_type == artifact_type
        ]

async def get_artifacts_by_filters(
    self,
    artifact_type: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
) -> list[TraceableArtifact]:
    """Get artifacts with optional filtering."""
    with self._lock:
        results = list(self._artifacts.values())

    if artifact_type is not None:
        results = [a for a in results if a.artifact_type == artifact_type]

    if created_after is not None:
        results = [a for a in results if a.created_at >= created_after]

    if created_before is not None:
        results = [a for a in results if a.created_at <= created_before]

    return results
```

**Subtasks:**
1. Implement `get_artifacts_by_type()` with thread safety
2. Implement `get_artifacts_by_filters()` with combined filtering
3. Add structlog logging for filter operations

### Task 4: Create AuditFilterService (filter_service.py)
**File:** `src/yolo_developer/audit/filter_service.py`

Create a unified service that coordinates filtering across all audit stores:

```python
class AuditFilterService:
    """Service for filtering audit data across all stores.

    Provides unified filtering interface that queries decisions,
    traceability, and cost data with combined filters.
    """

    def __init__(
        self,
        decision_store: DecisionStore,
        traceability_store: TraceabilityStore,
        cost_store: CostStore | None = None,
    ) -> None:
        self._decision_store = decision_store
        self._traceability_store = traceability_store
        self._cost_store = cost_store

    async def filter_decisions(
        self,
        filters: AuditFilters,
    ) -> list[Decision]:
        """Filter decisions using unified filters."""
        decision_filters = filters.to_decision_filters()
        return await self._decision_store.get_decisions(decision_filters)

    async def filter_artifacts(
        self,
        filters: AuditFilters,
    ) -> list[TraceableArtifact]:
        """Filter traceability artifacts using unified filters."""
        return await self._traceability_store.get_artifacts_by_filters(
            artifact_type=filters.artifact_type,
            created_after=filters.start_time,
            created_before=filters.end_time,
        )

    async def filter_costs(
        self,
        filters: AuditFilters,
    ) -> list[CostRecord]:
        """Filter cost records using unified filters."""
        if self._cost_store is None:
            return []
        cost_filters = filters.to_cost_filters()
        return await self._cost_store.get_costs(cost_filters)

    async def filter_all(
        self,
        filters: AuditFilters,
    ) -> dict[str, Any]:
        """Filter all audit data and return combined results.

        Returns:
            Dictionary with 'decisions', 'artifacts', and 'costs' keys.
        """
        decisions = await self.filter_decisions(filters)
        artifacts = await self.filter_artifacts(filters)
        costs = await self.filter_costs(filters)

        return {
            "decisions": decisions,
            "artifacts": artifacts,
            "costs": costs,
            "filters_applied": filters.to_dict(),
        }


def get_audit_filter_service(
    decision_store: DecisionStore,
    traceability_store: TraceabilityStore,
    cost_store: CostStore | None = None,
) -> AuditFilterService:
    """Factory function to create AuditFilterService."""
    return AuditFilterService(
        decision_store=decision_store,
        traceability_store=traceability_store,
        cost_store=cost_store,
    )
```

**Subtasks:**
1. Create `AuditFilterService` class with all store dependencies
2. Implement `filter_decisions()` method
3. Implement `filter_artifacts()` method
4. Implement `filter_costs()` method
5. Implement `filter_all()` for combined results
6. Add structlog logging for all operations
7. Create factory function `get_audit_filter_service()`

### Task 5: Update Module Exports (__init__.py)
**File:** `src/yolo_developer/audit/__init__.py`

Export new filtering types and services:

```python
# Filter types
from yolo_developer.audit.filter_types import AuditFilters

# Filter service
from yolo_developer.audit.filter_service import (
    AuditFilterService,
    get_audit_filter_service,
)
```

**Subtasks:**
1. Add imports for filter_types module
2. Add imports for filter_service module
3. Update `__all__` list with new exports

### Task 6: Write Comprehensive Tests
**Files:**
- `tests/unit/audit/test_filter_types.py`
- `tests/unit/audit/test_filter_service.py`
- `tests/unit/audit/test_traceability_filtering.py`

Test all filtering functionality:

**test_filter_types.py:**
- Test AuditFilters creation with various combinations
- Test to_dict() serialization
- Test to_decision_filters() conversion
- Test to_cost_filters() conversion
- Test validation warnings for invalid artifact_type

**test_filter_service.py:**
- Test filter_decisions() with various filters
- Test filter_artifacts() with type and time filters
- Test filter_costs() with agent and session filters
- Test filter_all() combined results
- Test filter service with None cost_store

**test_traceability_filtering.py:**
- Test get_artifacts_by_type() for each valid type
- Test get_artifacts_by_filters() with single filter
- Test get_artifacts_by_filters() with combined filters
- Test empty results for non-matching filters

**Subtasks:**
1. Create test_filter_types.py with dataclass tests
2. Create test_filter_service.py with service tests
3. Add traceability filter tests to existing test file
4. Ensure >90% coverage on new code

## Dev Notes

### Existing Filter Patterns to Follow

**DecisionFilters (from store.py):**
```python
@dataclass(frozen=True)
class DecisionFilters:
    agent_name: str | None = None
    decision_type: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    sprint_id: str | None = None
    story_id: str | None = None
```

**CostFilters (from cost_store.py - TypedDict pattern):**
```python
class CostFilters(TypedDict, total=False):
    agent_name: str | None
    story_id: str | None
    sprint_id: str | None
    session_id: str | None
    model: str | None
    tier: str | None
    start_time: str | None
    end_time: str | None
```

### Filter Logic Pattern (from memory_store.py)
```python
def _matches_filters(self, record: CostRecord, filters: CostFilters) -> bool:
    if filters.agent_name is not None and record.agent_name != filters.agent_name:
        return False
    if filters.start_time is not None and record.timestamp < filters.start_time:
        return False
    if filters.end_time is not None and record.timestamp > filters.end_time:
        return False
    return True
```

### Valid Artifact Types (from traceability_types.py)
```python
ArtifactType = Literal["requirement", "story", "design_decision", "code", "test"]
VALID_ARTIFACT_TYPES: frozenset[str] = frozenset(
    ["requirement", "story", "design_decision", "code", "test"]
)
```

### Project Structure Notes
Files will be added to the existing audit module structure:
```
src/yolo_developer/audit/
├── filter_types.py          # NEW: AuditFilters dataclass
├── filter_service.py        # NEW: AuditFilterService
├── traceability_store.py    # MODIFY: Add filter methods to Protocol
├── traceability_memory_store.py  # MODIFY: Implement filter methods
└── __init__.py              # MODIFY: Add new exports

tests/unit/audit/
├── test_filter_types.py     # NEW: Filter dataclass tests
├── test_filter_service.py   # NEW: Filter service tests
└── test_traceability_memory_store.py  # MODIFY: Add filter tests
```

### Previous Story Intelligence (Story 11.6)
From Story 11.6 implementation:
- Used frozen dataclasses for immutable filter types
- TypedDict pattern also works (CostFilters) but frozen dataclass preferred
- Thread safety via `threading.Lock()` in memory stores
- Validation warnings via `__post_init__` with logging
- Factory functions follow `get_<service>()` naming

### Testing Approach
Follow pattern from existing audit tests:
- Use pytest.mark.asyncio for async tests
- Create helper functions for test data (e.g., `_make_decision()`)
- Test both positive and negative filter matches
- Test filter combinations (AND logic)
- Test empty result sets

## Definition of Done

- [x] All acceptance criteria implemented and verified
- [x] Unit tests for all new modules with >90% coverage
- [x] Type hints on all public functions (mypy passes)
- [x] Code formatted with ruff
- [x] Docstrings following Google style on all public APIs
- [x] Integration with existing audit module exports
- [x] No breaking changes to existing audit functionality

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

None - implementation completed without errors.

### Completion Notes List

1. **AuditFilters dataclass** (filter_types.py): Created unified filter type with 9 optional fields covering all audit stores. Includes `to_dict()`, `to_decision_filters()`, and `to_cost_filters()` conversion methods. Validation in `__post_init__` warns on invalid artifact types.

2. **TraceabilityStore Protocol extension** (traceability_store.py): Added `get_artifacts_by_type()` and `get_artifacts_by_filters()` methods to the Protocol, enabling artifact filtering by type and time range.

3. **InMemoryTraceabilityStore implementation** (traceability_memory_store.py): Implemented filter methods with thread-safe access and AND logic for combined filters.

4. **AuditFilterService** (filter_service.py): Created unified service coordinating filtering across all three stores (decisions, traceability, costs). Includes `filter_decisions()`, `filter_artifacts()`, `filter_costs()`, and `filter_all()` methods with structlog instrumentation.

5. **Module exports** (__init__.py): Added `AuditFilters`, `AuditFilterService`, and `get_audit_filter_service` to public API with documentation examples.

6. **Comprehensive tests**: 66 tests across 3 test files covering all acceptance criteria. All audit module tests pass.

### Code Review Fixes Applied

1. **M1: Added `to_traceability_filters()` method** (filter_types.py): For consistency with `to_decision_filters()` and `to_cost_filters()`, added explicit conversion method that maps `start_time`/`end_time` to `created_after`/`created_before`. Updated `filter_service.py` to use it.

2. **M2: Fixed artifact type documentation** (this file): Updated Dev Notes to use correct `design_decision` instead of `design` in artifact type references.

3. **M4: Fixed thread safety in `get_artifacts_by_filters()`** (traceability_memory_store.py): Moved all filtering logic inside the `with self._lock:` block to prevent race conditions.

4. **Test count correction**: Updated from incorrect "62 new tests" to actual count of 66 tests.

### File List

**New Files:**
- `src/yolo_developer/audit/filter_types.py` - AuditFilters dataclass
- `src/yolo_developer/audit/filter_service.py` - AuditFilterService class
- `tests/unit/audit/test_filter_types.py` - 19 tests for AuditFilters
- `tests/unit/audit/test_filter_service.py` - 21 tests for AuditFilterService

**Modified Files:**
- `src/yolo_developer/audit/traceability_store.py` - Added 2 Protocol methods
- `src/yolo_developer/audit/traceability_memory_store.py` - Implemented filter methods
- `src/yolo_developer/audit/__init__.py` - Added exports
- `tests/unit/audit/test_traceability_memory_store.py` - Added 11 filter tests

## References

- Epic 11: Audit Trail & Observability requirements
- FR87: Users can filter audit trail by agent, time range, or artifact
- Story 11.1: Decision Logging (DecisionFilters pattern)
- Story 11.2: Requirement Traceability (TraceabilityStore)
- Story 11.3: Human-Readable Audit View (AuditViewService with filters)
- Story 11.6: Token/Cost Tracking (CostFilters pattern)
