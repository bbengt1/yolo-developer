# Story 2.4: Context Preservation Across Handoffs

Status: done

## Story

As a system user,
I want context preserved when agents hand off work,
So that subsequent agents have full understanding of prior decisions.

## Acceptance Criteria

1. **AC1: Previous Agent Outputs Available in State**
   - **Given** Agent A completes its work and produces output
   - **When** Agent B begins processing
   - **Then** all outputs from Agent A are available in the shared state
   - **And** outputs are structured according to YoloState TypedDict (ADR-001)

2. **AC2: Message Accumulation via LangGraph Reducers**
   - **Given** agents communicate via messages during processing
   - **When** multiple agents add messages to state
   - **Then** messages are accumulated using `Annotated[list[BaseMessage], add_messages]`
   - **And** full conversation history is preserved in order
   - **And** no messages are overwritten or lost

3. **AC3: Key Decisions Queryable from Memory Store**
   - **Given** an agent makes a significant decision during processing
   - **When** the decision is stored via memory store
   - **Then** subsequent agents can query for that decision
   - **And** decisions include source agent, timestamp, and rationale
   - **And** decisions are semantically searchable via vector embeddings

4. **AC4: Handoff Context Serialization**
   - **Given** an agent handoff occurs in the orchestrator
   - **When** state transitions from one agent node to another
   - **Then** a HandoffContext object is created with:
     - Source agent identifier
     - Target agent identifier
     - Key decisions summary
     - Relevant memory references (keys to stored embeddings/relationships)
   - **And** the HandoffContext is attached to state for the receiving agent

5. **AC5: Zero Context Loss During Handoffs**
   - **Given** state exists before a handoff
   - **When** the handoff completes
   - **Then** all state fields are preserved
   - **And** no data is corrupted or truncated
   - **And** state integrity can be verified via checksum or validation

## Tasks / Subtasks

- [x] Task 1: Define HandoffContext Data Structure (AC: 4)
  - [x] Create `src/yolo_developer/orchestrator/context.py` module
  - [x] Define `HandoffContext` dataclass with source, target, decisions, memory_refs
  - [x] Define `Decision` dataclass for structured decision capture
  - [x] Add timestamp and metadata fields for traceability
  - [x] Export from `orchestrator/__init__.py`

- [x] Task 2: Implement Message Accumulation (AC: 2)
  - [x] Update `src/yolo_developer/orchestrator/state.py` if needed
  - [x] Ensure `messages` field uses `Annotated[list[BaseMessage], add_messages]`
  - [x] Create helper function to append messages with agent attribution
  - [x] Write unit tests for message accumulation behavior

- [x] Task 3: Implement Decision Storage in Memory (AC: 3)
  - [x] Create `store_decision()` method in ChromaMemory
  - [x] Define decision metadata schema (agent, timestamp, rationale, related_artifacts)
  - [x] Create `query_decisions()` method for semantic search of decisions
  - [x] Store decisions with relationship to source agent in graph
  - [x] Write unit tests for decision storage and query

- [x] Task 4: Implement Handoff Context Creation (AC: 1, 4)
  - [x] Create `create_handoff_context()` function in context.py
  - [x] Gather decisions from current agent's processing
  - [x] Include memory references for relevant embeddings
  - [x] Attach HandoffContext to state for target agent
  - [x] Write unit tests for context creation

- [x] Task 5: Implement State Integrity Validation (AC: 5)
  - [x] Create `validate_state_integrity()` function
  - [x] Generate state checksum before handoff
  - [x] Verify checksum after handoff completes
  - [x] Log any integrity violations for debugging
  - [x] Write unit tests for integrity validation

- [x] Task 6: Integration with Orchestrator Graph (AC: 1, 2, 4, 5)
  - [x] Update node wrapper to create HandoffContext on exit
  - [x] Update edge handling to validate state integrity
  - [x] Ensure message accumulation works across all nodes
  - [x] Write integration tests for full handoff scenarios

- [x] Task 7: Write Unit Tests (AC: all)
  - [x] Create `tests/unit/orchestrator/test_context.py`
  - [x] Test: HandoffContext creation with all fields
  - [x] Test: Decision storage and retrieval
  - [x] Test: Message accumulation across agents
  - [x] Test: State integrity validation
  - [x] Test: No context loss simulation

- [x] Task 8: Write Integration Tests (AC: all)
  - [x] Add tests to `tests/integration/test_orchestrator.py`
  - [x] Test: End-to-end handoff between two mock agents
  - [x] Test: Multi-agent chain with context preservation
  - [x] Test: Context queries by receiving agent

## Dev Notes

### Critical Architecture Requirements

**From ADR-001 (State Management Pattern):**
- TypedDict for graph state with `add_messages` reducer
- State updates returned as dicts, never mutate state
- Pydantic at boundaries for validation

**From ADR-005 (Inter-Agent Communication):**
- LangGraph edges define explicit handoff conditions
- State machine transitions are predictable and replayable
- Message accumulation via reducers for audit trail

**From ADR-002 (Memory Persistence):**
- ChromaDB for vector storage of decisions
- JSONGraphStore for relationships between artifacts and decisions

**From Architecture Patterns:**
- Async-first design for all I/O operations
- Full type annotations on all functions
- Structured logging with logging module
- snake_case for all state dictionary keys

### Implementation Approach

**HandoffContext Structure:**

```python
# src/yolo_developer/orchestrator/context.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Decision:
    """A significant decision made by an agent during processing.

    Attributes:
        agent: The agent that made the decision (e.g., "analyst", "pm").
        summary: Brief description of the decision.
        rationale: Why this decision was made.
        timestamp: When the decision was made.
        related_artifacts: Keys of related embeddings/relationships.
    """
    agent: str
    summary: str
    rationale: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    related_artifacts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HandoffContext:
    """Context passed during agent handoffs.

    Contains decisions made by the source agent and references
    to relevant memory store entries for the target agent.

    Attributes:
        source_agent: Agent handing off work.
        target_agent: Agent receiving work.
        decisions: Key decisions from source agent processing.
        memory_refs: Keys to relevant embeddings in memory store.
        timestamp: When the handoff occurred.
    """
    source_agent: str
    target_agent: str
    decisions: tuple[Decision, ...] = field(default_factory=tuple)
    memory_refs: tuple[str, ...] = field(default_factory=tuple)
    timestamp: datetime = field(default_factory=datetime.utcnow)
```

**State Schema Update:**

```python
# src/yolo_developer/orchestrator/state.py
from typing import Annotated, TypedDict
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

from yolo_developer.orchestrator.context import HandoffContext, Decision


class YoloState(TypedDict):
    """Main state for YOLO Developer orchestration.

    Attributes:
        messages: Accumulated messages from all agents.
        current_agent: Currently executing agent identifier.
        handoff_context: Context from most recent handoff.
        decisions: All decisions made during sprint.
        # ... other fields per PRD requirements
    """
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: str
    handoff_context: HandoffContext | None
    decisions: list[Decision]
```

**Decision Storage Integration:**

```python
# In ChromaMemory or new ContextManager class

async def store_decision(
    self,
    decision: Decision,
    graph_store: JSONGraphStore | None = None,
) -> str:
    """Store a decision for later semantic retrieval.

    Args:
        decision: The decision to store.
        graph_store: Optional graph store for relationships.

    Returns:
        The key used to store the decision.
    """
    key = f"decision-{decision.agent}-{decision.timestamp.isoformat()}"

    # Store embedding for semantic search
    await self.store_embedding(
        key=key,
        content=f"{decision.summary}: {decision.rationale}",
        metadata={
            "type": "decision",
            "agent": decision.agent,
            "timestamp": decision.timestamp.isoformat(),
            "related_artifacts": decision.related_artifacts,
        },
    )

    # Store relationships in graph
    if graph_store:
        for artifact in decision.related_artifacts:
            await graph_store.store_relationship(
                source=key,
                target=artifact,
                relation="relates_to",
            )

    return key


async def query_decisions(
    self,
    query: str,
    agent: str | None = None,
    k: int = 5,
) -> list[MemoryResult]:
    """Query decisions semantically.

    Args:
        query: Semantic search query.
        agent: Optional filter by agent.
        k: Number of results to return.

    Returns:
        List of matching decisions.
    """
    results = await self.search_similar(query, k=k * 2)  # Over-fetch for filtering

    if agent:
        results = [r for r in results if r.metadata.get("agent") == agent]

    return results[:k]
```

**Handoff Context Creation:**

```python
# src/yolo_developer/orchestrator/context.py

def create_handoff_context(
    state: YoloState,
    source_agent: str,
    target_agent: str,
    decisions: list[Decision],
    memory_refs: list[str],
) -> dict:
    """Create a handoff context and return state update.

    Args:
        state: Current state (for reference).
        source_agent: Agent handing off.
        target_agent: Agent receiving.
        decisions: Decisions made during processing.
        memory_refs: Keys to relevant memory entries.

    Returns:
        State update dict with handoff_context.
    """
    context = HandoffContext(
        source_agent=source_agent,
        target_agent=target_agent,
        decisions=tuple(decisions),
        memory_refs=tuple(memory_refs),
    )

    return {
        "handoff_context": context,
        "current_agent": target_agent,
    }
```

**State Integrity Validation:**

```python
# src/yolo_developer/orchestrator/context.py
import hashlib
import json
from typing import Any


def compute_state_checksum(state: dict[str, Any], exclude_keys: set[str] | None = None) -> str:
    """Compute a checksum of state for integrity validation.

    Args:
        state: The state dict to checksum.
        exclude_keys: Keys to exclude (e.g., transient fields).

    Returns:
        SHA-256 hash of serialized state.
    """
    exclude = exclude_keys or {"handoff_context"}  # Context changes on handoff

    # Filter and sort for deterministic serialization
    filtered = {k: v for k, v in sorted(state.items()) if k not in exclude}

    # Serialize with custom encoder for non-JSON types
    serialized = json.dumps(filtered, default=str, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def validate_state_integrity(
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    exclude_keys: set[str] | None = None,
) -> bool:
    """Validate that state integrity was preserved during handoff.

    Args:
        before_state: State before handoff.
        after_state: State after handoff.
        exclude_keys: Keys to exclude from comparison.

    Returns:
        True if integrity preserved, False otherwise.
    """
    # Default: exclude keys that change during handoff
    exclude = exclude_keys or {"current_agent", "handoff_context", "messages"}

    before_checksum = compute_state_checksum(before_state, exclude)
    after_checksum = compute_state_checksum(after_state, exclude)

    return before_checksum == after_checksum
```

### Project Structure Notes

**New/Modified Module Locations:**
```
src/yolo_developer/orchestrator/
├── __init__.py      # Add context exports
├── context.py       # NEW: HandoffContext, Decision, create_handoff_context
├── state.py         # UPDATE: Add handoff_context to YoloState (if not exists)
└── ...

src/yolo_developer/memory/
├── vector.py        # UPDATE: Add store_decision, query_decisions methods
└── ...
```

**Test Location:**
```
tests/unit/orchestrator/
├── __init__.py
├── test_context.py  # NEW: HandoffContext, Decision, integrity tests
└── ...

tests/integration/
└── test_orchestrator.py  # ADD: Handoff integration tests
```

### Previous Story Learnings (from Story 2.3)

1. **Race conditions** - Ensure all shared state access uses asyncio.Lock
2. **Efficient data structures** - Use collections.deque for O(1) operations where needed
3. **Python 3 exception aliases** - IOError is OSError, don't duplicate
4. **Code review fixes expected** - Plan for race condition, logging, and efficiency fixes
5. **Protocol compliance** - Ensure methods match existing protocols exactly
6. **mypy validation** - Run mypy on both src and tests
7. **Frozen dataclasses** - Use frozen=True for immutable data
8. **Tenacity retry** - Apply retry decorator for file I/O operations

### Testing Approach

**Unit Tests (isolated components):**
- Test HandoffContext creation with all fields
- Test Decision dataclass immutability
- Test message accumulation via add_messages
- Test state checksum computation
- Test integrity validation

**Integration Tests (component interactions):**
- Test handoff between two mock agent nodes
- Test decision storage and retrieval across handoff
- Test multi-hop context preservation (A→B→C)
- Test message history preserved across agents

```python
# tests/unit/orchestrator/test_context.py
import pytest
from datetime import datetime

from yolo_developer.orchestrator.context import (
    Decision,
    HandoffContext,
    create_handoff_context,
    compute_state_checksum,
    validate_state_integrity,
)


class TestDecision:
    def test_decision_creation(self):
        """Decision should store all fields."""
        decision = Decision(
            agent="analyst",
            summary="Prioritized security over performance",
            rationale="User explicitly requested secure design",
        )
        assert decision.agent == "analyst"
        assert "security" in decision.summary

    def test_decision_is_frozen(self):
        """Decision should be immutable."""
        decision = Decision(agent="pm", summary="test", rationale="test")
        with pytest.raises(AttributeError):
            decision.agent = "architect"


class TestHandoffContext:
    def test_context_creation(self):
        """HandoffContext should store handoff details."""
        decision = Decision(agent="analyst", summary="test", rationale="test")
        context = HandoffContext(
            source_agent="analyst",
            target_agent="pm",
            decisions=(decision,),
            memory_refs=("req-001", "req-002"),
        )
        assert context.source_agent == "analyst"
        assert context.target_agent == "pm"
        assert len(context.decisions) == 1
        assert len(context.memory_refs) == 2
```

### References

- [Source: architecture.md#ADR-001] - State Management Pattern (TypedDict + reducers)
- [Source: architecture.md#ADR-005] - Inter-Agent Communication (state-based handoffs)
- [Source: architecture.md#ADR-002] - Memory Persistence Strategy
- [Source: 2-1-create-memory-store-protocol.md] - MemoryStore protocol definition
- [Source: 2-2-implement-chromadb-vector-storage.md] - ChromaMemory implementation
- [Source: 2-3-implement-json-graph-storage.md] - JSONGraphStore for relationships
