"""Orchestrator module for YOLO Developer.

This module provides the orchestration layer for coordinating agent execution,
managing state transitions, preserving context across agent handoffs, and
session persistence for resumable workflows.

Exports:
    Decision: Dataclass for capturing agent decisions.
    HandoffContext: Dataclass for context passed during handoffs.
    create_handoff_context: Function to create handoff context and state update.
    compute_state_checksum: Function to compute state integrity checksum.
    validate_state_integrity: Function to validate state preservation.
    SessionMetadata: Lightweight metadata about a persisted session.
    SessionState: Full session data including YoloState snapshot.
    SessionManager: Manages session file I/O with retry logic.
    SessionLoadError: Exception for session load failures.
    SessionNotFoundError: Exception when session doesn't exist.
    serialize_state: Convert YoloState to JSON-compatible dict.
    deserialize_state: Restore YoloState from JSON dict.

Example:
    >>> from yolo_developer.orchestrator import (
    ...     Decision,
    ...     HandoffContext,
    ...     create_handoff_context,
    ...     SessionManager,
    ... )
    >>>
    >>> decision = Decision(
    ...     agent="analyst",
    ...     summary="Security prioritized",
    ...     rationale="User requirement",
    ... )
    >>> state_update = create_handoff_context(
    ...     source_agent="analyst",
    ...     target_agent="pm",
    ...     decisions=[decision],
    ... )
    >>>
    >>> # Session persistence
    >>> manager = SessionManager(".yolo/sessions")
    >>> session_id = await manager.save_session(state)
"""

from __future__ import annotations

from yolo_developer.orchestrator.context import (
    Decision,
    HandoffContext,
    compute_state_checksum,
    create_handoff_context,
    validate_state_integrity,
)
from yolo_developer.orchestrator.graph import (
    Checkpointer,
    validated_handoff,
    wrap_node,
)
from yolo_developer.orchestrator.session import (
    SessionLoadError,
    SessionManager,
    SessionMetadata,
    SessionNotFoundError,
    SessionState,
    deserialize_state,
    serialize_state,
)
from yolo_developer.orchestrator.state import (
    YoloState,
    create_agent_message,
    get_messages_reducer,
)

__all__ = [
    "Checkpointer",
    "Decision",
    "HandoffContext",
    "SessionLoadError",
    "SessionManager",
    "SessionMetadata",
    "SessionNotFoundError",
    "SessionState",
    "YoloState",
    "compute_state_checksum",
    "create_agent_message",
    "create_handoff_context",
    "deserialize_state",
    "get_messages_reducer",
    "serialize_state",
    "validate_state_integrity",
    "validated_handoff",
    "wrap_node",
]
