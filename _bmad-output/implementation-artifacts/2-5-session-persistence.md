# Story 2.5: Session Persistence

Status: done

## Story

As a developer,
I want work preserved across sessions,
So that I can resume where I left off after closing the tool.

## Acceptance Criteria

1. **AC1: Resume From Last Checkpoint**
   - **Given** I have an in-progress sprint
   - **When** I close and reopen YOLO Developer
   - **Then** I can resume from the last checkpoint
   - **And** the resume operation is initiated via CLI or SDK

2. **AC2: State Restoration Including Agent Positions**
   - **Given** a session was interrupted mid-sprint
   - **When** I resume the session
   - **Then** all state is restored including current_agent
   - **And** handoff_context is restored if present
   - **And** decisions list is restored with all prior decisions

3. **AC3: Memory Store Contents Persisted**
   - **Given** embeddings and relationships were stored during the session
   - **When** I resume the session
   - **Then** ChromaDB embeddings are available (via persist_directory)
   - **And** JSONGraphStore relationships are available (via persist_path)
   - **And** search operations return previously stored content

4. **AC4: Session Metadata Preserved**
   - **Given** a session has been running
   - **When** session state is persisted
   - **Then** session_id is stored for identification
   - **And** timestamps (created_at, last_checkpoint) are preserved
   - **And** sprint progress (current story, stories completed) is tracked
   - **And** the session can be listed and selected for resume

5. **AC5: Graceful Handling of Missing or Corrupted Sessions**
   - **Given** session persistence data may be incomplete
   - **When** attempting to load a session that is missing or corrupted
   - **Then** a clear error message is provided
   - **And** the system does not crash
   - **And** the user can start a fresh session instead

## Tasks / Subtasks

- [x] Task 1: Define Session Data Structures (AC: 4, 5)
  - [x] Create `src/yolo_developer/orchestrator/session.py` module
  - [x] Define `SessionMetadata` dataclass with session_id, timestamps, progress
  - [x] Define `SessionState` dataclass containing YoloState snapshot and metadata
  - [x] Add validation for session data integrity
  - [x] Export from `orchestrator/__init__.py`

- [x] Task 2: Implement Session Persistence Manager (AC: 1, 2, 4)
  - [x] Create `SessionManager` class in session.py
  - [x] Implement `save_session(state: YoloState) -> str` method returning session_id
  - [x] Implement `load_session(session_id: str) -> SessionState` method
  - [x] Implement `list_sessions() -> list[SessionMetadata]` method
  - [x] Store sessions as JSON files in `.yolo/sessions/` directory
  - [x] Use tenacity retry for file I/O operations

- [x] Task 3: Implement State Serialization (AC: 2, 3)
  - [x] Create `serialize_state(state: YoloState) -> dict` function
  - [x] Handle BaseMessage serialization via LangChain utilities
  - [x] Handle Decision and HandoffContext serialization
  - [x] Create `deserialize_state(data: dict) -> YoloState` function
  - [x] Handle None values and optional fields gracefully

- [x] Task 4: Implement Session Checkpointing Integration (AC: 1, 2)
  - [x] Add `Checkpointer` class with `checkpoint()` method
  - [x] Auto-checkpoint via `wrap_node` with checkpointer parameter
  - [x] Store checkpoint with current state and agent position
  - [x] Enable resume from most recent checkpoint

- [x] Task 5: Implement Error Handling for Corrupted Sessions (AC: 5)
  - [x] Add `SessionLoadError` exception class
  - [x] Add `SessionNotFoundError` exception class
  - [x] Validate JSON structure on load
  - [x] Validate required fields presence
  - [x] Log corruption details for debugging
  - [x] Provide recovery guidance in error messages

- [x] Task 6: Write Unit Tests (AC: all)
  - [x] Create `tests/unit/orchestrator/test_session.py`
  - [x] Test: SessionMetadata creation and validation
  - [x] Test: State serialization round-trip
  - [x] Test: BaseMessage serialization/deserialization
  - [x] Test: Decision and HandoffContext preservation
  - [x] Test: Session save and load operations
  - [x] Test: List sessions functionality
  - [x] Test: Error handling for missing sessions
  - [x] Test: Error handling for corrupted sessions
  - [x] Test: Checkpointer checkpoint and resume operations

- [x] Task 7: Write Integration Tests (AC: all)
  - [x] Add tests to `tests/integration/test_session.py`
  - [x] Test: Full session save/load with mock orchestrator state
  - [x] Test: Resume mid-sprint with agent position preserved
  - [x] Test: Handoff context preserved through session resume
  - [x] Test: Multiple sessions isolated correctly
  - [x] Test: Auto-checkpointing with wrap_node integration

## File List

### Created Files
- `src/yolo_developer/orchestrator/session.py` - Core session persistence: SessionManager, SessionMetadata, serialization functions
- `src/yolo_developer/orchestrator/graph.py` - Checkpointer class with wrap_node integration
- `src/yolo_developer/orchestrator/context.py` - Decision and HandoffContext dataclasses
- `src/yolo_developer/orchestrator/state.py` - YoloState TypedDict definition
- `tests/unit/orchestrator/test_session.py` - Unit tests for session persistence (56+ tests)
- `tests/unit/orchestrator/test_orchestrator.py` - Additional orchestrator unit tests
- `tests/integration/test_session.py` - Integration tests for session resume scenarios

### Modified Files
- `src/yolo_developer/orchestrator/__init__.py` - Added session module exports
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Updated story status

## Dev Notes

### Critical Architecture Requirements

**From ADR-001 (State Management Pattern):**
- TypedDict for graph state - must serialize/deserialize correctly
- State updates returned as dicts, never mutate state
- Pydantic at boundaries for validation

**From ADR-002 (Memory Persistence):**
- ChromaDB uses persist_directory for automatic persistence
- JSONGraphStore uses persist_path for relationship data
- Both are already persistent - session layer coordinates state

**From ADR-007 (Error Handling):**
- Tenacity retry for file I/O operations
- Clear error messages for recovery guidance

**From Architecture Patterns:**
- Async-first design for all I/O operations
- Full type annotations on all functions
- Structured logging with logging module
- snake_case for all state dictionary keys

### Implementation Approach

**Session Directory Structure:**
```
.yolo/
├── memory/           # ChromaDB persist_directory (existing)
│   └── chroma.sqlite3
├── graph.json        # JSONGraphStore persist_path (existing)
└── sessions/         # NEW: Session persistence
    ├── session-abc123.json
    ├── session-def456.json
    └── _active.json  # Pointer to most recent session
```

**SessionMetadata Structure:**
```python
# src/yolo_developer/orchestrator/session.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


@dataclass(frozen=True)
class SessionMetadata:
    """Metadata for a persisted session.

    Attributes:
        session_id: Unique identifier for the session.
        created_at: When the session was first created.
        last_checkpoint: When the session was last checkpointed.
        current_agent: The agent that was active at last checkpoint.
        stories_completed: Number of stories completed in sprint.
        stories_total: Total number of stories in sprint.
    """
    session_id: str
    created_at: datetime
    last_checkpoint: datetime
    current_agent: str
    stories_completed: int = 0
    stories_total: int = 0
```

**State Serialization:**
```python
from langchain_core.messages import messages_to_dict, messages_from_dict

def serialize_state(state: YoloState) -> dict[str, Any]:
    """Serialize YoloState to JSON-compatible dict.

    Handles special types:
    - BaseMessage list via LangChain utilities
    - Decision objects via dataclass asdict
    - HandoffContext via dataclass asdict
    - datetime via isoformat
    """
    result: dict[str, Any] = {}

    # Serialize messages using LangChain
    if state.get("messages"):
        result["messages"] = messages_to_dict(state["messages"])
    else:
        result["messages"] = []

    # Serialize decisions
    if state.get("decisions"):
        result["decisions"] = [
            {
                "agent": d.agent,
                "summary": d.summary,
                "rationale": d.rationale,
                "timestamp": d.timestamp.isoformat(),
                "related_artifacts": list(d.related_artifacts),
            }
            for d in state["decisions"]
        ]
    else:
        result["decisions"] = []

    # Serialize handoff_context
    if state.get("handoff_context"):
        ctx = state["handoff_context"]
        result["handoff_context"] = {
            "source_agent": ctx.source_agent,
            "target_agent": ctx.target_agent,
            "decisions": [
                {
                    "agent": d.agent,
                    "summary": d.summary,
                    "rationale": d.rationale,
                    "timestamp": d.timestamp.isoformat(),
                    "related_artifacts": list(d.related_artifacts),
                }
                for d in ctx.decisions
            ],
            "memory_refs": list(ctx.memory_refs),
            "timestamp": ctx.timestamp.isoformat(),
        }
    else:
        result["handoff_context"] = None

    # Copy simple fields
    result["current_agent"] = state.get("current_agent", "")

    return result


def deserialize_state(data: dict[str, Any]) -> YoloState:
    """Deserialize JSON dict back to YoloState.

    Reconstructs:
    - BaseMessage list via LangChain utilities
    - Decision objects from dicts
    - HandoffContext from dict
    - datetime from isoformat strings
    """
    from langchain_core.messages import messages_from_dict
    from yolo_developer.orchestrator.context import Decision, HandoffContext

    state: YoloState = {
        "messages": [],
        "current_agent": data.get("current_agent", ""),
        "handoff_context": None,
        "decisions": [],
    }

    # Deserialize messages
    if data.get("messages"):
        state["messages"] = messages_from_dict(data["messages"])

    # Deserialize decisions
    if data.get("decisions"):
        state["decisions"] = [
            Decision(
                agent=d["agent"],
                summary=d["summary"],
                rationale=d["rationale"],
                timestamp=datetime.fromisoformat(d["timestamp"]),
                related_artifacts=tuple(d.get("related_artifacts", [])),
            )
            for d in data["decisions"]
        ]

    # Deserialize handoff_context
    if data.get("handoff_context"):
        ctx_data = data["handoff_context"]
        decisions = tuple(
            Decision(
                agent=d["agent"],
                summary=d["summary"],
                rationale=d["rationale"],
                timestamp=datetime.fromisoformat(d["timestamp"]),
                related_artifacts=tuple(d.get("related_artifacts", [])),
            )
            for d in ctx_data.get("decisions", [])
        )
        state["handoff_context"] = HandoffContext(
            source_agent=ctx_data["source_agent"],
            target_agent=ctx_data["target_agent"],
            decisions=decisions,
            memory_refs=tuple(ctx_data.get("memory_refs", [])),
            timestamp=datetime.fromisoformat(ctx_data["timestamp"]),
        )

    return state
```

**SessionManager Class:**
```python
import json
import logging
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_TRANSIENT_EXCEPTIONS = (OSError, IOError)


class SessionLoadError(Exception):
    """Raised when a session cannot be loaded."""

    def __init__(self, message: str, session_id: str, cause: Exception | None = None):
        super().__init__(message)
        self.session_id = session_id
        self.cause = cause


class SessionNotFoundError(SessionLoadError):
    """Raised when a session does not exist."""
    pass


class SessionManager:
    """Manages session persistence for YOLO Developer.

    Sessions are stored as JSON files in the .yolo/sessions/ directory.
    Each session contains a snapshot of YoloState and metadata.
    """

    def __init__(self, sessions_dir: str | Path):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._active_file = self.sessions_dir / "_active.json"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=2.0),
        retry=retry_if_exception_type(_TRANSIENT_EXCEPTIONS),
        reraise=True,
    )
    def _write_json(self, path: Path, data: dict) -> None:
        """Write JSON with atomic rename for safety."""
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        temp_path.rename(path)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=2.0),
        retry=retry_if_exception_type(_TRANSIENT_EXCEPTIONS),
        reraise=True,
    )
    def _read_json(self, path: Path) -> dict:
        """Read JSON with retry for transient errors."""
        with open(path) as f:
            return json.load(f)

    async def save_session(
        self,
        state: YoloState,
        session_id: str | None = None,
    ) -> str:
        """Save session state to disk.

        Args:
            state: The YoloState to persist.
            session_id: Optional existing session ID (for updates).

        Returns:
            The session ID used for saving.
        """
        if session_id is None:
            session_id = f"session-{uuid.uuid4().hex[:12]}"

        now = datetime.now(timezone.utc)
        session_path = self.sessions_dir / f"{session_id}.json"

        # Build session data
        session_data = {
            "session_id": session_id,
            "created_at": now.isoformat(),
            "last_checkpoint": now.isoformat(),
            "state": serialize_state(state),
        }

        # If updating existing session, preserve created_at
        if session_path.exists():
            try:
                existing = self._read_json(session_path)
                session_data["created_at"] = existing.get("created_at", now.isoformat())
            except Exception:
                pass  # Use new timestamp if read fails

        self._write_json(session_path, session_data)

        # Update active session pointer
        self._write_json(self._active_file, {"session_id": session_id})

        logger.info("Session saved", extra={"session_id": session_id})
        return session_id

    async def load_session(self, session_id: str) -> tuple[YoloState, SessionMetadata]:
        """Load a session from disk.

        Args:
            session_id: The session ID to load.

        Returns:
            Tuple of (restored YoloState, SessionMetadata).

        Raises:
            SessionNotFoundError: If session file doesn't exist.
            SessionLoadError: If session data is corrupted.
        """
        session_path = self.sessions_dir / f"{session_id}.json"

        if not session_path.exists():
            raise SessionNotFoundError(
                f"Session not found: {session_id}",
                session_id=session_id,
            )

        try:
            data = self._read_json(session_path)
        except json.JSONDecodeError as e:
            raise SessionLoadError(
                f"Session file corrupted: {session_id}",
                session_id=session_id,
                cause=e,
            )

        # Validate required fields
        required = ["session_id", "state", "created_at", "last_checkpoint"]
        missing = [f for f in required if f not in data]
        if missing:
            raise SessionLoadError(
                f"Session missing fields {missing}: {session_id}",
                session_id=session_id,
            )

        state = deserialize_state(data["state"])
        metadata = SessionMetadata(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_checkpoint=datetime.fromisoformat(data["last_checkpoint"]),
            current_agent=state.get("current_agent", ""),
        )

        logger.info("Session loaded", extra={"session_id": session_id})
        return state, metadata

    async def list_sessions(self) -> list[SessionMetadata]:
        """List all available sessions.

        Returns:
            List of SessionMetadata sorted by last_checkpoint (newest first).
        """
        sessions: list[SessionMetadata] = []

        for path in self.sessions_dir.glob("session-*.json"):
            try:
                data = self._read_json(path)
                state_data = data.get("state", {})
                sessions.append(SessionMetadata(
                    session_id=data["session_id"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    last_checkpoint=datetime.fromisoformat(data["last_checkpoint"]),
                    current_agent=state_data.get("current_agent", ""),
                ))
            except Exception as e:
                logger.warning(
                    "Failed to read session",
                    extra={"path": str(path), "error": str(e)},
                )

        # Sort by last_checkpoint descending (newest first)
        sessions.sort(key=lambda s: s.last_checkpoint, reverse=True)
        return sessions

    async def get_active_session_id(self) -> str | None:
        """Get the most recently active session ID.

        Returns:
            Session ID or None if no active session.
        """
        if not self._active_file.exists():
            return None

        try:
            data = self._read_json(self._active_file)
            return data.get("session_id")
        except Exception:
            return None
```

### Project Structure Notes

**New/Modified Module Locations:**
```
src/yolo_developer/orchestrator/
├── __init__.py      # Add session exports
├── session.py       # NEW: SessionManager, SessionMetadata, serialization
├── context.py       # Existing: Decision, HandoffContext
├── state.py         # Existing: YoloState
└── graph.py         # Existing: wrap_node, validated_handoff
```

**Session Storage Location:**
```
.yolo/
└── sessions/
    ├── session-abc123.json
    └── _active.json
```

**Test Location:**
```
tests/unit/orchestrator/
├── test_session.py  # NEW: Unit tests for session persistence
└── ...

tests/integration/
└── test_session.py  # NEW: Integration tests for session resume
```

### Previous Story Learnings (from Story 2.4)

1. **asyncio.Lock** - Use locks for concurrent access to shared resources
2. **Type validation** - Validate types with isinstance before processing
3. **Consistent exclude keys** - Use named constants for default exclusions
4. **Over-fetch capping** - Cap multiplier-based queries to prevent abuse
5. **Import organization** - Move imports from TYPE_CHECKING to module level when runtime needed
6. **Tenacity retry** - Apply retry decorator for file I/O operations
7. **Frozen dataclasses** - Use frozen=True for immutable data
8. **mypy validation** - Run mypy on both src and tests

### Testing Approach

**Unit Tests (isolated components):**
- Test SessionMetadata creation with all fields
- Test serialize_state / deserialize_state round-trip
- Test BaseMessage serialization via LangChain utilities
- Test Decision and HandoffContext preservation through serialization
- Test SessionManager.save_session creates valid JSON
- Test SessionManager.load_session restores state correctly
- Test SessionManager.list_sessions returns sorted list
- Test error handling for non-existent sessions
- Test error handling for corrupted JSON

**Integration Tests (component interactions):**
- Test full save/load cycle with realistic YoloState
- Test resume with agent position and handoff_context preserved
- Test that ChromaDB memory store content persists (already handled by ChromaDB)
- Test multiple sessions don't interfere with each other

### References

- [Source: architecture.md#ADR-001] - State Management Pattern (TypedDict + reducers)
- [Source: architecture.md#ADR-002] - Memory Persistence Strategy (ChromaDB + JSON graph)
- [Source: architecture.md#ADR-007] - Error Handling Strategy (tenacity retry)
- [Source: epics.md#Story-2.5] - Session Persistence requirements
- [Source: 2-4-context-preservation-across-handoffs.md] - YoloState, Decision, HandoffContext structures
- [LangChain Messages Serialization](https://python.langchain.com/docs/how_to/serialization/) - messages_to_dict, messages_from_dict
