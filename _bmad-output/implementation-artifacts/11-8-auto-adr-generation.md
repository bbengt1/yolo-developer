# Story 11.8: Auto ADR Generation

## Story

**As a** developer,
**I want** Architecture Decision Records generated automatically from audit trail decisions,
**So that** design documentation stays current without manual effort.

## Status

- **Epic:** 11 - Audit Trail & Observability
- **Status:** done
- **Priority:** P2
- **Story Points:** 3

## Acceptance Criteria

### AC1: ADRs Generated from Architectural Decisions
**Given** architectural decisions made during execution (logged via Story 11.1)
**When** ADR generation runs
**Then** ADRs are created in standard format (Title, Status, Context, Decision, Consequences)
**And** only `architecture_choice` decision types trigger ADR generation

### AC2: ADRs Capture Context, Decision, and Consequences
**Given** a Decision record from the audit trail
**When** an ADR is generated
**Then** the ADR context explains why the decision was needed
**And** the decision section states what was chosen
**And** consequences document positive/negative effects and trade-offs

### AC3: ADRs Linked to Relevant Stories
**Given** a Decision record with context.story_id populated
**When** an ADR is generated
**Then** the ADR is linked to all relevant story IDs from the decision context
**And** the links are navigable (story_id → ADR relationship)

### AC4: ADRs Stored in Project
**Given** generated ADRs
**When** storage completes
**Then** ADRs are persisted to the audit store
**And** ADRs can be queried by ID, story_id, or time range
**And** ADRs are included in audit trail exports (Story 11.4)

### AC5: ADR Generation Service Integration
**Given** the audit module ecosystem
**When** ADR generation is invoked
**Then** it integrates with DecisionStore from Story 11.1
**And** it follows the protocol pattern (ADRStore protocol + InMemoryADRStore)
**And** it uses structlog for logging like other Epic 11 components

## Technical Requirements

### Functional Requirements Mapping
- **FR88:** System can generate Architecture Decision Records automatically

### Architecture References
- **ADR-001:** Frozen dataclasses for ADR types
- **Story 7.3 Pattern:** Existing ADR generation in `agents/architect/adr_generator.py`
- **Story 11.1 Pattern:** DecisionStore protocol with InMemoryDecisionStore
- **Story 11.2 Pattern:** TraceabilityStore for linking artifacts
- **Epic 11 Pattern:** Protocol-based stores, structlog logging, factory functions

### Technology Stack
- **structlog:** For structured logging of ADR operations
- **Frozen Dataclasses:** For immutable ADR configuration
- **Existing Patterns:** Follow DecisionStore, CostStore patterns from Epic 11

## Tasks

### Task 1: Create ADR Types (adr_types.py)
**File:** `src/yolo_developer/audit/adr_types.py`

Create types for auto-generated ADRs:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

ADRStatus = Literal["proposed", "accepted", "deprecated", "superseded"]

@dataclass(frozen=True)
class AutoADR:
    """Architecture Decision Record auto-generated from audit decisions.

    Attributes:
        id: Unique ADR identifier (format: ADR-{number:03d})
        title: Descriptive ADR title
        status: ADR status (proposed by default for auto-generated)
        context: Why this decision was needed
        decision: What was decided
        consequences: Positive/negative effects and trade-offs
        source_decision_id: ID of the Decision that triggered this ADR
        story_ids: Stories this ADR relates to
        created_at: ISO 8601 timestamp when ADR was generated
    """
    id: str
    title: str
    status: ADRStatus
    context: str
    decision: str
    consequences: str
    source_decision_id: str
    story_ids: tuple[str, ...] = ()
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "context": self.context,
            "decision": self.decision,
            "consequences": self.consequences,
            "source_decision_id": self.source_decision_id,
            "story_ids": list(self.story_ids),
            "created_at": self.created_at,
        }
```

**Subtasks:**
1. Create `AutoADR` frozen dataclass with all fields
2. Add `to_dict()` for serialization
3. Add validation in `__post_init__` for required fields
4. Export from `audit/__init__.py`

### Task 2: Create ADR Store Protocol (adr_store.py)
**File:** `src/yolo_developer/audit/adr_store.py`

Create the ADRStore protocol following DecisionStore pattern:

```python
from __future__ import annotations

from typing import Protocol

from yolo_developer.audit.adr_types import AutoADR

class ADRStore(Protocol):
    """Protocol for ADR storage implementations.

    Defines the interface for storing and retrieving auto-generated ADRs.
    """

    async def store_adr(self, adr: AutoADR) -> str:
        """Store a new ADR.

        Args:
            adr: The ADR to store.

        Returns:
            The ADR ID.
        """
        ...

    async def get_adr(self, adr_id: str) -> AutoADR | None:
        """Retrieve an ADR by ID.

        Args:
            adr_id: The ID of the ADR to retrieve.

        Returns:
            The ADR if found, None otherwise.
        """
        ...

    async def get_adrs_by_story(self, story_id: str) -> list[AutoADR]:
        """Get all ADRs linked to a story.

        Args:
            story_id: The story ID to filter by.

        Returns:
            List of ADRs linked to the story.
        """
        ...

    async def get_all_adrs(self) -> list[AutoADR]:
        """Get all stored ADRs.

        Returns:
            List of all ADRs, ordered by created_at descending.
        """
        ...

    async def get_next_adr_number(self) -> int:
        """Get the next available ADR number for ID generation.

        Returns:
            The next sequential ADR number.
        """
        ...
```

**Subtasks:**
1. Create `ADRStore` Protocol with all methods
2. Document all methods with docstrings
3. Export from `audit/__init__.py`

### Task 3: Implement InMemoryADRStore (adr_memory_store.py)
**File:** `src/yolo_developer/audit/adr_memory_store.py`

Implement the in-memory ADR store following InMemoryDecisionStore pattern:

```python
from __future__ import annotations

import threading

from yolo_developer.audit.adr_types import AutoADR


class InMemoryADRStore:
    """In-memory implementation of ADRStore protocol.

    Stores ADRs in memory with thread-safe access.
    Maintains indices for fast lookup by story_id.
    """

    def __init__(self) -> None:
        """Initialize the in-memory ADR store."""
        self._adrs: dict[str, AutoADR] = {}
        self._story_index: dict[str, list[str]] = {}  # story_id -> [adr_id]
        self._adr_counter: int = 0
        self._lock = threading.Lock()

    async def store_adr(self, adr: AutoADR) -> str:
        """Store a new ADR."""
        with self._lock:
            self._adrs[adr.id] = adr
            # Update story index
            for story_id in adr.story_ids:
                if story_id not in self._story_index:
                    self._story_index[story_id] = []
                self._story_index[story_id].append(adr.id)
        return adr.id

    async def get_adr(self, adr_id: str) -> AutoADR | None:
        """Retrieve an ADR by ID."""
        with self._lock:
            return self._adrs.get(adr_id)

    async def get_adrs_by_story(self, story_id: str) -> list[AutoADR]:
        """Get all ADRs linked to a story."""
        with self._lock:
            adr_ids = self._story_index.get(story_id, [])
            return [self._adrs[adr_id] for adr_id in adr_ids if adr_id in self._adrs]

    async def get_all_adrs(self) -> list[AutoADR]:
        """Get all stored ADRs, ordered by created_at descending."""
        with self._lock:
            adrs = list(self._adrs.values())
            return sorted(adrs, key=lambda a: a.created_at, reverse=True)

    async def get_next_adr_number(self) -> int:
        """Get the next available ADR number."""
        with self._lock:
            self._adr_counter += 1
            return self._adr_counter
```

**Subtasks:**
1. Implement `InMemoryADRStore` with thread-safe access
2. Implement all protocol methods
3. Add story index for fast lookup
4. Add structlog logging
5. Export from `audit/__init__.py`

### Task 4: Create ADR Generation Service (adr_service.py)
**File:** `src/yolo_developer/audit/adr_service.py`

Create the service that generates ADRs from audit decisions:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from yolo_developer.audit.adr_types import AutoADR

if TYPE_CHECKING:
    from yolo_developer.audit.adr_store import ADRStore
    from yolo_developer.audit.store import DecisionStore
    from yolo_developer.audit.types import Decision

_logger = structlog.get_logger(__name__)


class ADRGenerationService:
    """Service for auto-generating ADRs from audit decisions.

    Monitors decision store for architecture_choice decisions
    and generates corresponding ADRs.
    """

    def __init__(
        self,
        decision_store: DecisionStore,
        adr_store: ADRStore,
    ) -> None:
        """Initialize the ADR generation service."""
        self._decision_store = decision_store
        self._adr_store = adr_store

    async def generate_adr_from_decision(self, decision: Decision) -> AutoADR | None:
        """Generate an ADR from a single decision.

        Only generates ADRs for architecture_choice decisions.

        Args:
            decision: The Decision to potentially convert to ADR.

        Returns:
            The generated AutoADR, or None if decision type doesn't warrant ADR.
        """
        if decision.decision_type != "architecture_choice":
            _logger.debug(
                "skipping_non_architectural_decision",
                decision_id=decision.id,
                decision_type=decision.decision_type,
            )
            return None

        adr_number = await self._adr_store.get_next_adr_number()

        # Build story_ids tuple from context
        story_ids: tuple[str, ...] = ()
        if decision.context.story_id:
            story_ids = (decision.context.story_id,)

        adr = AutoADR(
            id=f"ADR-{adr_number:03d}",
            title=_generate_title(decision),
            status="proposed",
            context=_generate_context(decision),
            decision=_generate_decision_text(decision),
            consequences=_generate_consequences(decision),
            source_decision_id=decision.id,
            story_ids=story_ids,
        )

        await self._adr_store.store_adr(adr)

        _logger.info(
            "adr_generated",
            adr_id=adr.id,
            source_decision_id=decision.id,
            story_ids=story_ids,
        )

        return adr

    async def generate_adrs_for_session(self, session_id: str) -> list[AutoADR]:
        """Generate ADRs for all architectural decisions in a session.

        Args:
            session_id: The session ID to process.

        Returns:
            List of generated ADRs.
        """
        from yolo_developer.audit.store import DecisionFilters

        filters = DecisionFilters(decision_type="architecture_choice")
        decisions = await self._decision_store.get_decisions(filters)

        # Filter by session (decisions don't have session in filters, check agent)
        session_decisions = [
            d for d in decisions
            if d.agent.session_id == session_id
        ]

        adrs: list[AutoADR] = []
        for decision in session_decisions:
            adr = await self.generate_adr_from_decision(decision)
            if adr is not None:
                adrs.append(adr)

        _logger.info(
            "session_adrs_generated",
            session_id=session_id,
            adr_count=len(adrs),
        )

        return adrs


def _generate_title(decision: Decision) -> str:
    """Generate ADR title from decision content."""
    # Extract first sentence or use truncated content
    content = decision.content
    if ". " in content:
        title = content.split(". ")[0]
    else:
        title = content[:50] + "..." if len(content) > 50 else content
    return title


def _generate_context(decision: Decision) -> str:
    """Generate ADR context section."""
    parts = []
    parts.append(f"An architectural decision was needed: {decision.content}")

    if decision.context.story_id:
        parts.append(f"This decision was made for story {decision.context.story_id}.")

    if decision.context.sprint_id:
        parts.append(f"Sprint: {decision.context.sprint_id}.")

    return " ".join(parts)


def _generate_decision_text(decision: Decision) -> str:
    """Generate ADR decision section."""
    parts = []
    parts.append(f"Decision: {decision.content}")
    parts.append(f"Rationale: {decision.rationale}")
    return " ".join(parts)


def _generate_consequences(decision: Decision) -> str:
    """Generate ADR consequences section."""
    # Default consequences based on decision metadata
    parts = []
    parts.append("Positive: Addresses architectural need with documented rationale.")

    if decision.severity == "critical":
        parts.append("Note: This is a critical decision that significantly impacts system behavior.")
    elif decision.severity == "warning":
        parts.append("Note: This decision may need attention and should be reviewed.")

    parts.append("Trade-offs: Requires careful implementation and monitoring.")
    return " ".join(parts)


def get_adr_generation_service(
    decision_store: DecisionStore,
    adr_store: ADRStore,
) -> ADRGenerationService:
    """Factory function to create ADRGenerationService."""
    return ADRGenerationService(
        decision_store=decision_store,
        adr_store=adr_store,
    )
```

**Subtasks:**
1. Create `ADRGenerationService` class
2. Implement `generate_adr_from_decision()` method
3. Implement `generate_adrs_for_session()` method
4. Add helper functions for content generation
5. Add structlog logging
6. Create factory function
7. Export from `audit/__init__.py`

### Task 5: Update Module Exports (__init__.py)
**File:** `src/yolo_developer/audit/__init__.py`

Export new ADR types and services:

```python
# ADR types
from yolo_developer.audit.adr_types import AutoADR, ADRStatus

# ADR store
from yolo_developer.audit.adr_store import ADRStore
from yolo_developer.audit.adr_memory_store import InMemoryADRStore

# ADR service
from yolo_developer.audit.adr_service import (
    ADRGenerationService,
    get_adr_generation_service,
)
```

**Subtasks:**
1. Add imports for adr_types module
2. Add imports for adr_store module
3. Add imports for adr_memory_store module
4. Add imports for adr_service module
5. Update `__all__` list with new exports

### Task 6: Write Comprehensive Tests
**Files:**
- `tests/unit/audit/test_adr_types.py`
- `tests/unit/audit/test_adr_memory_store.py`
- `tests/unit/audit/test_adr_service.py`

Test all ADR functionality:

**test_adr_types.py:**
- Test AutoADR creation with all fields
- Test to_dict() serialization
- Test frozen dataclass immutability
- Test default values (created_at, status)

**test_adr_memory_store.py:**
- Test store_adr() returns ADR ID
- Test get_adr() retrieves correct ADR
- Test get_adrs_by_story() filtering
- Test get_all_adrs() ordering
- Test get_next_adr_number() incrementing
- Test thread safety

**test_adr_service.py:**
- Test generate_adr_from_decision() for architecture_choice
- Test generate_adr_from_decision() skips non-architectural
- Test generate_adrs_for_session()
- Test ADR content generation helpers
- Test integration with decision store

**Subtasks:**
1. Create test_adr_types.py with type tests
2. Create test_adr_memory_store.py with store tests
3. Create test_adr_service.py with service tests
4. Ensure >90% coverage on new code

## Dev Notes

### Relationship to Story 7.3 ADR Generation

Story 7.3 implemented ADR generation in the Architect Agent (`agents/architect/adr_generator.py`) for design decisions made during architectural analysis. That implementation:
- Generates ADRs from `DesignDecision` objects during architect node execution
- Uses LLM for content generation with pattern-based fallback
- Creates ADRs with 12-Factor compliance analysis

**Story 11.8 (this story)** focuses on a different use case:
- Generates ADRs from audit trail `Decision` records with `decision_type="architecture_choice"`
- Part of the audit/observability system (Epic 11)
- Does NOT use LLM - uses pattern-based generation only (simpler scope)
- Integrates with DecisionStore and audit filtering

The two systems are complementary:
- 7.3: Proactive ADR generation during design phase
- 11.8: Reactive ADR generation from logged audit decisions

### Existing Epic 11 Patterns to Follow

**From Story 11.1 (Decision Logging):**
```python
@dataclass(frozen=True)
class Decision:
    id: str
    decision_type: DecisionType  # includes "architecture_choice"
    content: str
    rationale: str
    agent: AgentIdentity
    context: DecisionContext
    timestamp: str
    ...
```

**From Story 11.6 (Cost Tracking):**
```python
class CostStore(Protocol):
    async def store_cost(self, record: CostRecord) -> str: ...
    async def get_costs(self, filters: CostFilters) -> list[CostRecord]: ...

class InMemoryCostStore:
    def __init__(self) -> None:
        self._records: dict[str, CostRecord] = {}
        self._lock = threading.Lock()
```

### Valid Decision Types (from types.py)

```python
DecisionType = Literal[
    "requirement_analysis",
    "story_creation",
    "architecture_choice",  # ← This triggers ADR generation
    "implementation_choice",
    "test_strategy",
    "orchestration",
    "quality_gate",
    "escalation",
]
```

### Project Structure Notes

Files will be added to the existing audit module structure:
```
src/yolo_developer/audit/
├── adr_types.py          # NEW: AutoADR dataclass
├── adr_store.py          # NEW: ADRStore protocol
├── adr_memory_store.py   # NEW: InMemoryADRStore
├── adr_service.py        # NEW: ADRGenerationService
└── __init__.py           # MODIFY: Add new exports

tests/unit/audit/
├── test_adr_types.py     # NEW: Type tests
├── test_adr_memory_store.py  # NEW: Store tests
└── test_adr_service.py   # NEW: Service tests
```

### Testing Approach

Follow pattern from existing audit tests:
- Use pytest.mark.asyncio for async tests
- Create helper functions for test data (e.g., `_make_decision()`)
- Test both positive and negative scenarios
- Test thread safety for memory store

## Definition of Done

- [x] All acceptance criteria implemented and verified
- [x] Unit tests for all new modules with >90% coverage (97% achieved)
- [x] Type hints on all public functions (mypy passes)
- [x] Code formatted with ruff
- [x] Docstrings following Google style on all public APIs
- [x] Integration with existing audit module exports
- [x] No breaking changes to existing audit functionality (712 audit tests pass)

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

None - implementation completed without errors.

### Completion Notes List

1. **AutoADR dataclass** (adr_types.py): Created frozen dataclass with standard ADR fields (id, title, status, context, decision, consequences). Includes `to_dict()` for serialization and `__post_init__` validation with warnings for empty required fields.

2. **ADRStore Protocol** (adr_store.py): Created protocol defining the interface for ADR storage with methods for storing, retrieving by ID, querying by story, listing all, getting next number, and time range filtering.

3. **InMemoryADRStore** (adr_memory_store.py): Implemented thread-safe in-memory ADR storage following InMemoryDecisionStore pattern. Includes story index for fast lookup and sequential number generation.

4. **ADRGenerationService** (adr_service.py): Created service that generates ADRs from `architecture_choice` decisions. Includes content generation helpers for title, context, decision, and consequences sections. Added file export methods for AC4 compliance.

5. **AuditFilterService integration** (filter_service.py): Added ADRStore support to the unified filter service, enabling ADR filtering by story_id and time range. Updated `filter_all()` to include ADRs in results.

6. **Module exports** (__init__.py): Added all new ADR types and services to public API.

7. **Comprehensive tests**: Tests across 3 test files covering all acceptance criteria with 97% overall coverage.

### Code Review Fixes Applied

1. **H1: Added time range filtering** (adr_store.py, adr_memory_store.py): Added `get_adrs_by_time_range()` method to ADRStore protocol and InMemoryADRStore implementation for AC4 compliance. Supports filtering by start_time and/or end_time with inclusive bounds.

2. **H2: Added file system export** (adr_service.py): Added `export_adr_to_file()` and `export_all_adrs_to_directory()` methods with `_generate_adr_markdown()` helper for AC4 file system export requirement.

3. **H3: Integrated ADRs with AuditFilterService** (filter_service.py): Added ADRStore parameter to constructor, added `filter_adrs()` method, and updated `filter_all()` to include ADRs in results.

4. **M1: Fixed D413 docstring formatting** (adr_memory_store.py): Added blank lines after docstring sections per ruff D413 requirements.

5. **M2: Fixed TC001 import organization** (adr_memory_store.py): Moved AutoADR import to TYPE_CHECKING block with runtime import in `__init__`.

6. **Added tests for time range filtering** (test_adr_memory_store.py): Added TestInMemoryADRStoreTimeRange class with 7 tests covering time range queries.

### File List

**New Files:**
- `src/yolo_developer/audit/adr_types.py` - AutoADR dataclass and ADRStatus type
- `src/yolo_developer/audit/adr_store.py` - ADRStore protocol
- `src/yolo_developer/audit/adr_memory_store.py` - InMemoryADRStore implementation
- `src/yolo_developer/audit/adr_service.py` - ADRGenerationService class
- `tests/unit/audit/test_adr_types.py` - Tests for AutoADR type
- `tests/unit/audit/test_adr_memory_store.py` - Tests for InMemoryADRStore
- `tests/unit/audit/test_adr_service.py` - Tests for ADRGenerationService

**Modified Files:**
- `src/yolo_developer/audit/__init__.py` - Added ADR exports
- `src/yolo_developer/audit/filter_service.py` - Added ADRStore integration

## References

- Epic 11: Audit Trail & Observability requirements
- FR88: System can generate Architecture Decision Records automatically
- Story 7.3: ADR Generation (Architect Agent implementation)
- Story 11.1: Decision Logging (Decision types and store)
- Story 11.2: Requirement Traceability (artifact linking pattern)
- Story 11.6: Token/Cost Tracking (store pattern)
