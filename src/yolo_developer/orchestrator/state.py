"""State management for YOLO Developer orchestration.

This module defines the central state schema for the LangGraph workflow,
including message accumulation using reducers, agent tracking, and
handoff context preservation.

Key concepts:
- **YoloState**: TypedDict defining the shape of orchestration state
- **Message Accumulation**: Uses LangGraph's add_messages reducer to
  accumulate messages without overwriting
- **Agent Attribution**: Helper functions to create messages with
  agent metadata for audit trails

Example:
    >>> from yolo_developer.orchestrator.state import (
    ...     YoloState,
    ...     create_agent_message,
    ... )
    >>>
    >>> # Create initial state
    >>> state: YoloState = {
    ...     "messages": [],
    ...     "current_agent": "analyst",
    ...     "handoff_context": None,
    ...     "decisions": [],
    ... }
    >>>
    >>> # Create a message with agent attribution
    >>> msg = create_agent_message(
    ...     content="Analysis complete",
    ...     agent="analyst",
    ... )

Architecture Note:
    Per ADR-001, we use TypedDict for internal graph state with the
    add_messages reducer for message accumulation. Pydantic is used
    only at system boundaries.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph.message import add_messages

if TYPE_CHECKING:
    from yolo_developer.orchestrator.context import Decision, HandoffContext


class YoloState(TypedDict):
    """Main state for YOLO Developer orchestration.

    This TypedDict defines the shape of state passed through the LangGraph
    workflow. The messages field uses the add_messages reducer for
    accumulation behavior.

    Attributes:
        messages: Accumulated messages from all agents. Uses add_messages
            reducer to append rather than replace.
        current_agent: Currently executing agent identifier (e.g., "analyst").
        handoff_context: Context from most recent handoff, or None.
        decisions: All decisions made during the current sprint.

    Example:
        >>> state: YoloState = {
        ...     "messages": [],
        ...     "current_agent": "analyst",
        ...     "handoff_context": None,
        ...     "decisions": [],
        ... }

    Note:
        The Annotated type for messages enables LangGraph to use the
        add_messages reducer automatically when state updates are applied.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: str
    handoff_context: HandoffContext | None
    decisions: list[Decision]


def get_messages_reducer() -> Callable[[list[BaseMessage], list[BaseMessage]], list[BaseMessage]]:
    """Get the messages reducer function for manual use.

    Returns the add_messages reducer that LangGraph uses for the messages
    field. This is useful for testing or manual state manipulation.

    Returns:
        The add_messages reducer function.

    Example:
        >>> reducer = get_messages_reducer()
        >>> result = reducer([msg1], [msg2, msg3])
        >>> len(result)
        3
    """
    return add_messages  # type: ignore[return-value]


def create_agent_message(
    content: str,
    agent: str,
    metadata: dict[str, Any] | None = None,
) -> AIMessage:
    """Create an AIMessage with agent attribution in metadata.

    Helper function to create messages that include the agent identifier
    in additional_kwargs for audit trail purposes.

    Args:
        content: The message content.
        agent: The agent identifier (e.g., "analyst", "pm", "architect").
        metadata: Additional metadata to include in the message.

    Returns:
        AIMessage with agent and optional metadata in additional_kwargs.

    Example:
        >>> msg = create_agent_message(
        ...     content="Analysis complete",
        ...     agent="analyst",
        ...     metadata={"decision_count": 3},
        ... )
        >>> msg.additional_kwargs["agent"]
        'analyst'
    """
    additional_kwargs: dict[str, Any] = {"agent": agent}

    if metadata:
        additional_kwargs.update(metadata)

    return AIMessage(
        content=content,
        additional_kwargs=additional_kwargs,
    )
