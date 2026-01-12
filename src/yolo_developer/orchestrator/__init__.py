"""Orchestrator module for YOLO Developer.

This module provides the orchestration layer for coordinating agent execution,
managing state transitions, preserving context across agent handoffs,
session persistence for resumable workflows, and LangGraph workflow execution.

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
    WorkflowConfig: Configuration for workflow behavior (Story 10.1).
    build_workflow: Build and compile the orchestration StateGraph (Story 10.1).
    run_workflow: Execute workflow and return final state (Story 10.1).
    stream_workflow: Stream workflow execution events (Story 10.1).
    create_initial_state: Create initial state for workflow (Story 10.1).
    create_workflow_with_checkpointing: Create workflow with checkpointer (Story 10.1).
    get_default_agent_nodes: Get default agent node registry (Story 10.1).
    route_after_analyst: Routing function after analyst (Story 10.1).
    route_after_pm: Routing function after PM (Story 10.1).
    route_after_architect: Routing function after architect (Story 10.1).
    route_after_dev: Routing function after dev (Story 10.1).
    route_after_tea: Routing function after TEA (Story 10.1).

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
from yolo_developer.orchestrator.workflow import (
    WorkflowConfig,
    build_workflow,
    create_initial_state,
    create_workflow_with_checkpointing,
    get_default_agent_nodes,
    route_after_analyst,
    route_after_architect,
    route_after_dev,
    route_after_pm,
    route_after_tea,
    run_workflow,
    stream_workflow,
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
    "WorkflowConfig",
    "YoloState",
    "build_workflow",
    "compute_state_checksum",
    "create_agent_message",
    "create_handoff_context",
    "create_initial_state",
    "create_workflow_with_checkpointing",
    "deserialize_state",
    "get_default_agent_nodes",
    "get_messages_reducer",
    "route_after_analyst",
    "route_after_architect",
    "route_after_dev",
    "route_after_pm",
    "route_after_tea",
    "run_workflow",
    "serialize_state",
    "stream_workflow",
    "validate_state_integrity",
    "validated_handoff",
    "wrap_node",
]
