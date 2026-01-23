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
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph.message import add_messages

# Import at runtime - LangGraph needs these types for StateGraph introspection
from yolo_developer.orchestrator.context import Decision, HandoffContext


class YoloState(TypedDict, total=False):
    """Main state for YOLO Developer orchestration.

    This TypedDict defines the shape of state passed through the LangGraph
    workflow. The messages field uses the add_messages reducer for
    accumulation behavior.

    Attributes:
        messages: Accumulated messages from all agents. Uses add_messages
            reducer to append rather than replace. Required.
        current_agent: Currently executing agent identifier (e.g., "analyst"). Required.
        handoff_context: Context from most recent handoff, or None.
        decisions: All decisions made during the current sprint. Required.
        tool_registry: Registry of external CLI tools (e.g., Claude Code).
            Optional - when present, agents can delegate tasks to external tools.
            Type is Any to avoid circular imports (actual type: ToolRegistry).
        analyst_output: Output from the analyst agent containing crystallized
            requirements for downstream agents (PM, Architect, etc.). Optional.
        pm_output: Output from the PM agent containing user stories with
            acceptance criteria for downstream agents (Architect, Dev). Optional.
        architect_output: Output from the architect agent containing design
            decisions for downstream agents (Dev, TEA). Optional.
        dev_output: Output from the dev agent containing implementation
            artifacts for downstream agents (TEA). Optional.

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

    # Required fields
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: str
    decisions: list[Decision]
    # Optional fields
    handoff_context: HandoffContext | None
    # Type is Any to avoid circular imports at runtime (LangGraph needs runtime access)
    # Actual type: yolo_developer.tools.ToolRegistry
    tool_registry: Any
    # Agent output fields for downstream agent consumption
    analyst_output: dict[str, Any] | None
    pm_output: dict[str, Any] | None
    architect_output: dict[str, Any] | None
    dev_output: dict[str, Any] | None


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
