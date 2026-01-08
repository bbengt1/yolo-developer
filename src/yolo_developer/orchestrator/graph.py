"""LangGraph integration for YOLO Developer orchestrator.

This module provides utilities for integrating agent nodes with LangGraph,
including node wrappers for automatic handoff context creation, state
integrity validation, and session checkpointing.

Key concepts:
- **wrap_node**: Decorator/wrapper to add handoff context creation to agent nodes
- **validated_handoff**: Utility to validate state integrity during handoffs
- **Checkpointer**: Class for session checkpointing during graph execution
- **AGENT_NODES**: Registry of available agent nodes for graph building

Available Agent Nodes (Story 5.1+):
    analyst_node: Requirement crystallization and gap analysis
    (More nodes will be added as agents are implemented)

Example:
    >>> from yolo_developer.orchestrator.graph import wrap_node, Checkpointer
    >>>
    >>> async def analyst_node(state: YoloState) -> dict:
    ...     # Agent logic
    ...     return {"messages": [...]}
    >>>
    >>> wrapped = wrap_node(analyst_node, agent_name="analyst", target_agent="pm")
    >>> # Use wrapped node in LangGraph
    >>>
    >>> # With checkpointing
    >>> checkpointer = Checkpointer(SessionManager(".yolo/sessions"))
    >>> wrapped = wrap_node(
    ...     analyst_node,
    ...     agent_name="analyst",
    ...     target_agent="pm",
    ...     checkpointer=checkpointer,
    ... )
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from yolo_developer.orchestrator.context import (
    Decision,
    HandoffContext,
    validate_state_integrity,
)
from yolo_developer.orchestrator.state import YoloState

if TYPE_CHECKING:
    from yolo_developer.orchestrator.session import SessionManager, SessionMetadata

from yolo_developer.orchestrator.session import SessionNotFoundError

logger = logging.getLogger(__name__)

# Type alias for agent node functions
AgentNode = Callable[[YoloState], Awaitable[dict[str, Any]]]


class Checkpointer:
    """Session checkpointer for graph execution.

    Wraps SessionManager to provide a simplified interface for
    checkpointing during graph node execution. Maintains the current
    session ID to enable incremental checkpoints.

    Attributes:
        session_manager: The underlying SessionManager for persistence.
        session_id: The current session ID (set after first checkpoint).

    Example:
        >>> from yolo_developer.orchestrator.session import SessionManager
        >>> from yolo_developer.orchestrator.graph import Checkpointer
        >>>
        >>> manager = SessionManager(".yolo/sessions")
        >>> checkpointer = Checkpointer(manager)
        >>>
        >>> # Checkpoint after each node
        >>> await checkpointer.checkpoint(state)
        >>>
        >>> # Resume from previous session
        >>> state, metadata = await checkpointer.resume()
    """

    def __init__(
        self,
        session_manager: SessionManager,
        session_id: str | None = None,
    ) -> None:
        """Initialize the checkpointer.

        Args:
            session_manager: SessionManager for persistence operations.
            session_id: Optional existing session ID for resuming.
        """
        self.session_manager = session_manager
        self.session_id = session_id

    async def checkpoint(self, state: YoloState) -> str:
        """Checkpoint the current state.

        Saves the state to the session file. If this is the first
        checkpoint, a new session ID is generated. Subsequent checkpoints
        update the existing session.

        Args:
            state: The current YoloState to persist.

        Returns:
            The session ID used for the checkpoint.
        """
        self.session_id = await self.session_manager.save_session(
            state,
            session_id=self.session_id,
        )
        logger.debug(
            "State checkpointed",
            extra={"session_id": self.session_id},
        )
        return self.session_id

    async def resume(self) -> tuple[YoloState, SessionMetadata]:
        """Resume from the current or most recent session.

        If session_id is set, loads that session. Otherwise, loads
        the most recently active session.

        Returns:
            Tuple of (restored YoloState, SessionMetadata).

        Raises:
            SessionNotFoundError: If no session exists to resume.
            SessionLoadError: If the session is corrupted.
        """
        # Use current session_id or get active session
        session_id = self.session_id
        if session_id is None:
            session_id = await self.session_manager.get_active_session_id()

        if session_id is None:
            raise SessionNotFoundError(
                "No active session to resume",
                session_id="",
            )

        state, metadata = await self.session_manager.load_session(session_id)
        self.session_id = session_id

        logger.info(
            "Session resumed",
            extra={
                "session_id": session_id,
                "current_agent": metadata.current_agent,
            },
        )

        return state, metadata


def wrap_node(
    node_fn: AgentNode,
    agent_name: str,
    target_agent: str,
    checkpointer: Checkpointer | None = None,
) -> AgentNode:
    """Wrap an agent node to add handoff context creation and optional checkpointing.

    This wrapper adds cross-cutting concerns to agent nodes:
    - Creates HandoffContext when the node completes
    - Updates current_agent to target_agent
    - Preserves all original node output
    - Optionally checkpoints state after node completion

    Args:
        node_fn: The async agent node function to wrap.
        agent_name: Name of this agent (source for handoff).
        target_agent: Name of the target agent for handoff.
        checkpointer: Optional Checkpointer for auto-checkpointing.

    Returns:
        Wrapped async function with same signature.

    Example:
        >>> async def analyst_node(state: YoloState) -> dict:
        ...     return {"messages": [msg]}
        >>>
        >>> wrapped = wrap_node(analyst_node, "analyst", "pm")
        >>> result = await wrapped(state)
        >>> result["handoff_context"].source_agent
        'analyst'
        >>>
        >>> # With auto-checkpointing
        >>> checkpointer = Checkpointer(SessionManager(".yolo/sessions"))
        >>> wrapped = wrap_node(analyst_node, "analyst", "pm", checkpointer)
    """

    async def wrapped(state: YoloState) -> dict[str, Any]:
        # Execute the original node
        result = await node_fn(state)

        # Extract and validate decisions from result (if any)
        raw_decisions = result.get("decisions", [])
        decisions: list[Decision] = []
        for item in raw_decisions:
            if isinstance(item, Decision):
                decisions.append(item)
            else:
                logger.warning(
                    "Invalid decision type in node output, skipping",
                    extra={
                        "agent": agent_name,
                        "item_type": type(item).__name__,
                    },
                )

        # Extract memory refs from result (if any)
        # Nodes can return memory_refs as a list of keys to stored embeddings
        raw_memory_refs = result.get("memory_refs", [])
        memory_refs: list[str] = [ref for ref in raw_memory_refs if isinstance(ref, str)]

        # Create handoff context
        handoff_context = HandoffContext(
            source_agent=agent_name,
            target_agent=target_agent,
            decisions=tuple(decisions),
            memory_refs=tuple(memory_refs),
        )

        # Add handoff context and update current agent
        result["handoff_context"] = handoff_context
        result["current_agent"] = target_agent

        logger.debug(
            "Node completed with handoff",
            extra={
                "source_agent": agent_name,
                "target_agent": target_agent,
                "decision_count": len(decisions),
            },
        )

        # Auto-checkpoint after node completion if checkpointer provided
        if checkpointer is not None:
            # Build updated state for checkpointing
            updated_state: YoloState = {
                "messages": state.get("messages", []) + result.get("messages", []),
                "current_agent": result["current_agent"],
                "handoff_context": result["handoff_context"],
                "decisions": list(state.get("decisions", [])) + decisions,
            }
            await checkpointer.checkpoint(updated_state)

        return result

    # Preserve function metadata
    wrapped.__name__ = f"wrapped_{node_fn.__name__}"
    wrapped.__doc__ = node_fn.__doc__

    return wrapped


def validated_handoff(
    before_state: dict[str, Any],
    after_state: dict[str, Any],
) -> None:
    """Validate state integrity during handoff and log violations.

    Checks that non-transient state fields are preserved during handoff.
    Logs a warning if integrity is violated but does not raise an exception
    (handoff continues to maintain system liveness).

    Args:
        before_state: State before the handoff.
        after_state: State after the handoff.

    Note:
        This function is intended to be called during edge transitions
        in the LangGraph orchestrator. It logs but does not block on
        violations to maintain system resilience.

    Example:
        >>> validated_handoff(before_state, after_state)
        # Logs warning if integrity violated
    """
    is_valid = validate_state_integrity(before_state, after_state)

    if not is_valid:
        logger.warning(
            "State integrity violation detected during handoff",
            extra={
                "before_keys": list(before_state.keys()),
                "after_keys": list(after_state.keys()),
            },
        )


# =============================================================================
# Agent Node Registry (Story 5.1+)
# =============================================================================
# Registry of available agent nodes for building the orchestration graph.
# Nodes are registered here as they are implemented across stories.
# The graph builder (Epic 10) will use this registry to construct the workflow.


def get_agent_nodes() -> dict[str, AgentNode]:
    """Get the registry of available agent nodes.

    Returns a dictionary mapping agent names to their node functions.
    Nodes are imported lazily to avoid circular dependencies.

    Returns:
        Dict mapping agent name to node function.

    Example:
        >>> nodes = get_agent_nodes()
        >>> analyst = nodes["analyst"]
        >>> result = await analyst(state)
    """
    from yolo_developer.agents import analyst_node

    return {
        "analyst": analyst_node,
        # Future agents will be added here:
        # "pm": pm_node,
        # "architect": architect_node,
        # "dev": dev_node,
        # "tea": tea_node,
        # "sm": sm_node,
    }
