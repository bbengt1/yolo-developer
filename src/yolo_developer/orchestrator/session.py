"""Session persistence for YOLO Developer.

This module provides classes and functions for persisting and restoring
orchestrator state across sessions. Key concepts:

- **SessionMetadata**: Lightweight metadata about a persisted session.
- **SessionState**: Full session data including YoloState snapshot.
- **SessionManager**: Manages session file I/O with retry logic.
- **Serialization**: Functions to convert YoloState to/from JSON.

Example:
    >>> from yolo_developer.orchestrator.session import (
    ...     SessionManager,
    ...     SessionMetadata,
    ... )
    >>>
    >>> # Create session manager
    >>> manager = SessionManager(sessions_dir=".yolo/sessions")
    >>>
    >>> # Save current state
    >>> session_id = await manager.save_session(state)
    >>>
    >>> # List available sessions
    >>> sessions = await manager.list_sessions()
    >>> for s in sessions:
    ...     print(f"{s.session_id}: {s.current_agent}")
    >>>
    >>> # Resume a session
    >>> state, metadata = await manager.load_session(session_id)

Security Note:
    Session files are stored as JSON in the project's .yolo directory.
    Ensure the directory has appropriate permissions to prevent
    unauthorized access to session state.

Thread Safety:
    The SessionManager uses file-based locking via atomic rename
    operations to ensure safe concurrent writes.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    from yolo_developer.orchestrator.state import YoloState

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with timezone info.

    Using timezone-aware datetime for consistency and to avoid
    deprecation warnings from naive datetime comparisons.
    """
    return datetime.now(timezone.utc)


# Transient errors that should be retried (disk I/O issues)
_TRANSIENT_EXCEPTIONS = (OSError, TimeoutError, ConnectionError)


def _log_retry_attempt(retry_state: RetryCallState) -> None:
    """Log retry attempts for observability."""
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Session I/O retry attempt %d: %s",
        retry_state.attempt_number,
        str(exception) if exception else "unknown error",
        extra={
            "attempt": retry_state.attempt_number,
            "exception_type": type(exception).__name__ if exception else None,
        },
    )


# Retry decorator for file I/O operations: 3 attempts with exponential backoff
_session_io_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.1, min=0.1, max=2.0),
    retry=retry_if_exception_type(_TRANSIENT_EXCEPTIONS),
    before_sleep=_log_retry_attempt,
    reraise=True,
)


class SessionLoadError(Exception):
    """Raised when a session cannot be loaded.

    This exception provides context about the failure including
    the session ID and optional underlying cause.

    Attributes:
        session_id: The session that failed to load.
        cause: The underlying exception, if any.

    Example:
        >>> try:
        ...     state, metadata = await manager.load_session("invalid-id")
        ... except SessionLoadError as e:
        ...     print(f"Failed to load {e.session_id}: {e}")
    """

    def __init__(
        self,
        message: str,
        session_id: str,
        cause: Exception | None = None,
    ) -> None:
        """Initialize SessionLoadError with context.

        Args:
            message: Descriptive error message.
            session_id: The session that failed to load.
            cause: The underlying exception, if any.
        """
        super().__init__(message)
        self.session_id = session_id
        self.cause = cause


class SessionNotFoundError(SessionLoadError):
    """Raised when a session does not exist.

    This is a specific case of SessionLoadError for when the
    session file is not found on disk.

    Example:
        >>> try:
        ...     await manager.load_session("nonexistent-session")
        ... except SessionNotFoundError as e:
        ...     print(f"Session not found: {e.session_id}")
    """

    pass


@dataclass(frozen=True)
class SessionMetadata:
    """Metadata for a persisted session.

    Contains lightweight information about a session for listing
    and selection without loading the full state.

    Attributes:
        session_id: Unique identifier for the session.
        created_at: When the session was first created.
        last_checkpoint: When the session was last checkpointed.
        current_agent: The agent that was active at last checkpoint.
        stories_completed: Number of stories completed in sprint.
        stories_total: Total number of stories in sprint.

    Example:
        >>> metadata = SessionMetadata(
        ...     session_id="session-abc123",
        ...     created_at=datetime.now(timezone.utc),
        ...     last_checkpoint=datetime.now(timezone.utc),
        ...     current_agent="analyst",
        ...     stories_completed=2,
        ...     stories_total=5,
        ... )
        >>> metadata.session_id
        'session-abc123'
    """

    session_id: str
    created_at: datetime
    last_checkpoint: datetime
    current_agent: str
    stories_completed: int = 0
    stories_total: int = 0


@dataclass(frozen=True)
class SessionState:
    """Full session data including state snapshot and metadata.

    Contains the complete YoloState snapshot along with session
    metadata for restoration.

    Attributes:
        metadata: Session metadata (id, timestamps, progress).
        state_data: Serialized YoloState as a dictionary.

    Example:
        >>> session = SessionState(
        ...     metadata=metadata,
        ...     state_data=serialize_state(state),
        ... )
    """

    metadata: SessionMetadata
    state_data: dict[str, Any]


def serialize_state(state: YoloState) -> dict[str, Any]:
    """Serialize YoloState to JSON-compatible dict.

    Handles special types:
    - BaseMessage list via LangChain utilities
    - Decision objects via manual conversion
    - HandoffContext via manual conversion
    - datetime via isoformat

    Args:
        state: The YoloState to serialize.

    Returns:
        JSON-compatible dictionary representation.

    Example:
        >>> state = {"messages": [], "current_agent": "analyst", ...}
        >>> data = serialize_state(state)
        >>> json.dumps(data)  # Now JSON-serializable
    """
    from langchain_core.messages import messages_to_dict

    result: dict[str, Any] = {}

    # Serialize messages using LangChain utilities
    messages = state.get("messages")
    if messages:
        result["messages"] = messages_to_dict(messages)
    else:
        result["messages"] = []

    # Serialize decisions
    decisions = state.get("decisions")
    if decisions:
        result["decisions"] = [
            {
                "agent": d.agent,
                "summary": d.summary,
                "rationale": d.rationale,
                "timestamp": d.timestamp.isoformat(),
                "related_artifacts": list(d.related_artifacts),
            }
            for d in decisions
        ]
    else:
        result["decisions"] = []

    # Serialize handoff_context
    handoff_context = state.get("handoff_context")
    if handoff_context:
        ctx = handoff_context
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

    Args:
        data: JSON-compatible dictionary from serialize_state.

    Returns:
        Reconstructed YoloState.

    Raises:
        ValueError: If required fields are missing or malformed.

    Example:
        >>> data = {"messages": [], "current_agent": "analyst", ...}
        >>> state = deserialize_state(data)
        >>> state["current_agent"]
        'analyst'
    """
    from langchain_core.messages import messages_from_dict

    from yolo_developer.orchestrator.context import Decision, HandoffContext

    # Initialize with defaults
    state: dict[str, Any] = {
        "messages": [],
        "current_agent": data.get("current_agent", ""),
        "handoff_context": None,
        "decisions": [],
    }

    # Deserialize messages using LangChain utilities
    messages_data = data.get("messages")
    if messages_data:
        state["messages"] = messages_from_dict(messages_data)

    # Deserialize decisions
    decisions_data = data.get("decisions")
    if decisions_data:
        state["decisions"] = [
            Decision(
                agent=d["agent"],
                summary=d["summary"],
                rationale=d["rationale"],
                timestamp=datetime.fromisoformat(d["timestamp"]),
                related_artifacts=tuple(d.get("related_artifacts", [])),
            )
            for d in decisions_data
        ]

    # Deserialize handoff_context
    handoff_data = data.get("handoff_context")
    if handoff_data:
        decisions_tuple = tuple(
            Decision(
                agent=d["agent"],
                summary=d["summary"],
                rationale=d["rationale"],
                timestamp=datetime.fromisoformat(d["timestamp"]),
                related_artifacts=tuple(d.get("related_artifacts", [])),
            )
            for d in handoff_data.get("decisions", [])
        )
        state["handoff_context"] = HandoffContext(
            source_agent=handoff_data["source_agent"],
            target_agent=handoff_data["target_agent"],
            decisions=decisions_tuple,
            memory_refs=tuple(handoff_data.get("memory_refs", [])),
            timestamp=datetime.fromisoformat(handoff_data["timestamp"]),
        )

    # Type assertion - the dict matches YoloState structure
    return state  # type: ignore[return-value]


class SessionManager:
    """Manages session persistence for YOLO Developer.

    Sessions are stored as JSON files in the configured sessions directory.
    Each session contains a snapshot of YoloState and metadata for
    identification and progress tracking.

    Attributes:
        sessions_dir: Path to the sessions directory.

    Example:
        >>> manager = SessionManager(sessions_dir=".yolo/sessions")
        >>>
        >>> # Save a session
        >>> session_id = await manager.save_session(state)
        >>>
        >>> # Load it back
        >>> restored_state, metadata = await manager.load_session(session_id)
        >>>
        >>> # List all sessions
        >>> sessions = await manager.list_sessions()
    """

    def __init__(self, sessions_dir: str | Path) -> None:
        """Initialize SessionManager with storage directory.

        Args:
            sessions_dir: Path to the directory for session files.
                Will be created if it doesn't exist.
        """
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._active_file = self.sessions_dir / "_active.json"

    @_session_io_retry
    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        """Write JSON with atomic rename for safety.

        Uses a temporary file and atomic rename to prevent
        partial writes from corrupting data.

        Args:
            path: Target file path.
            data: Dictionary to serialize as JSON.
        """
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        temp_path.rename(path)

    @_session_io_retry
    def _read_json(self, path: Path) -> dict[str, Any]:
        """Read JSON with retry for transient errors.

        Args:
            path: File path to read.

        Returns:
            Parsed JSON as dictionary.

        Raises:
            FileNotFoundError: If file doesn't exist.
            json.JSONDecodeError: If file is not valid JSON.
        """
        with open(path, encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]

    async def save_session(
        self,
        state: YoloState,
        session_id: str | None = None,
        stories_completed: int = 0,
        stories_total: int = 0,
    ) -> str:
        """Save session state to disk.

        Creates a new session or updates an existing one. The session
        file contains the serialized state and metadata including
        timestamps.

        Args:
            state: The YoloState to persist.
            session_id: Optional existing session ID for updates.
                If None, a new session ID is generated.
            stories_completed: Number of stories completed in the sprint.
            stories_total: Total number of stories in the sprint.

        Returns:
            The session ID used for saving.

        Example:
            >>> # New session
            >>> session_id = await manager.save_session(state)
            >>>
            >>> # Update existing session with progress
            >>> await manager.save_session(
            ...     state,
            ...     session_id=session_id,
            ...     stories_completed=2,
            ...     stories_total=5,
            ... )
        """
        if session_id is None:
            session_id = f"session-{uuid.uuid4().hex[:12]}"

        now = _utcnow()
        session_path = self.sessions_dir / f"{session_id}.json"

        # Build session data
        session_data: dict[str, Any] = {
            "session_id": session_id,
            "created_at": now.isoformat(),
            "last_checkpoint": now.isoformat(),
            "state": serialize_state(state),
            "stories_completed": stories_completed,
            "stories_total": stories_total,
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

    async def load_session(
        self,
        session_id: str,
    ) -> tuple[YoloState, SessionMetadata]:
        """Load a session from disk.

        Restores the YoloState and creates metadata from the
        persisted session file.

        Args:
            session_id: The session ID to load.

        Returns:
            Tuple of (restored YoloState, SessionMetadata).

        Raises:
            SessionNotFoundError: If session file doesn't exist.
            SessionLoadError: If session data is corrupted or invalid.

        Example:
            >>> try:
            ...     state, metadata = await manager.load_session("session-abc123")
            ...     print(f"Restored at agent: {metadata.current_agent}")
            ... except SessionNotFoundError:
            ...     print("Session not found")
        """
        session_path = self.sessions_dir / f"{session_id}.json"

        if not session_path.exists():
            raise SessionNotFoundError(
                f"Session not found: {session_id}. "
                "You can start a fresh session or list available sessions.",
                session_id=session_id,
            )

        try:
            data = self._read_json(session_path)
        except json.JSONDecodeError as e:
            raise SessionLoadError(
                f"Session file corrupted: {session_id}. "
                "The session file contains invalid JSON and cannot be loaded.",
                session_id=session_id,
                cause=e,
            ) from e

        # Validate required fields
        required = ["session_id", "state", "created_at", "last_checkpoint"]
        missing = [f for f in required if f not in data]
        if missing:
            raise SessionLoadError(
                f"Session missing required fields {missing}: {session_id}. "
                "The session file may be incomplete or corrupted.",
                session_id=session_id,
            )

        try:
            state = deserialize_state(data["state"])
        except (KeyError, ValueError, TypeError) as e:
            raise SessionLoadError(
                f"Failed to deserialize session state: {session_id}. Error: {e}",
                session_id=session_id,
                cause=e,
            ) from e

        metadata = SessionMetadata(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_checkpoint=datetime.fromisoformat(data["last_checkpoint"]),
            current_agent=state.get("current_agent", ""),
            stories_completed=data.get("stories_completed", 0),
            stories_total=data.get("stories_total", 0),
        )

        logger.info("Session loaded", extra={"session_id": session_id})
        return state, metadata

    async def list_sessions(self) -> list[SessionMetadata]:
        """List all available sessions.

        Scans the sessions directory for valid session files and
        returns metadata for each, sorted by last checkpoint time.

        Returns:
            List of SessionMetadata sorted by last_checkpoint (newest first).

        Example:
            >>> sessions = await manager.list_sessions()
            >>> for s in sessions:
            ...     print(f"{s.session_id}: agent={s.current_agent}")
        """
        sessions: list[SessionMetadata] = []

        for path in self.sessions_dir.glob("session-*.json"):
            try:
                data = self._read_json(path)
                state_data = data.get("state", {})
                sessions.append(
                    SessionMetadata(
                        session_id=data["session_id"],
                        created_at=datetime.fromisoformat(data["created_at"]),
                        last_checkpoint=datetime.fromisoformat(data["last_checkpoint"]),
                        current_agent=state_data.get("current_agent", ""),
                        stories_completed=data.get("stories_completed", 0),
                        stories_total=data.get("stories_total", 0),
                    )
                )
            except Exception as e:
                logger.warning(
                    "Failed to read session file",
                    extra={"path": str(path), "error": str(e)},
                )

        # Sort by last_checkpoint descending (newest first)
        sessions.sort(key=lambda s: s.last_checkpoint, reverse=True)
        return sessions

    async def get_active_session_id(self) -> str | None:
        """Get the most recently active session ID.

        Returns the session ID from the _active.json pointer file,
        which tracks the most recently saved session.

        Returns:
            Session ID or None if no active session.

        Example:
            >>> active_id = await manager.get_active_session_id()
            >>> if active_id:
            ...     state, metadata = await manager.load_session(active_id)
        """
        if not self._active_file.exists():
            return None

        try:
            data = self._read_json(self._active_file)
            return data.get("session_id")
        except Exception:
            return None

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session from disk.

        Removes the session file. If this was the active session,
        the active pointer is also removed.

        Args:
            session_id: The session ID to delete.

        Returns:
            True if session was deleted, False if it didn't exist.

        Example:
            >>> deleted = await manager.delete_session("session-abc123")
            >>> if deleted:
            ...     print("Session deleted")
        """
        session_path = self.sessions_dir / f"{session_id}.json"

        if not session_path.exists():
            return False

        session_path.unlink()
        logger.info("Session deleted", extra={"session_id": session_id})

        # Clear active pointer if this was the active session
        if self._active_file.exists():
            try:
                data = self._read_json(self._active_file)
                if data.get("session_id") == session_id:
                    self._active_file.unlink()
            except Exception:
                pass

        return True
