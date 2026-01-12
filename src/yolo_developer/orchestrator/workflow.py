"""LangGraph workflow orchestration for YOLO Developer (Story 10.1).

This module provides the StateGraph-based workflow for orchestrating
agent execution, including:
- WorkflowConfig for configuring workflow behavior
- Agent node registration and graph construction
- Conditional routing between agents
- Checkpointing integration for recovery
- Workflow execution interface (run and stream)

Example:
    >>> from yolo_developer.orchestrator.workflow import (
    ...     build_workflow,
    ...     run_workflow,
    ...     create_initial_state,
    ... )
    >>>
    >>> # Build and run workflow
    >>> graph = build_workflow()
    >>> initial_state = create_initial_state()
    >>> result = await run_workflow(initial_state)

Architecture:
    Per ADR-005, this module implements explicit agent handoffs via
    LangGraph edges with conditional routing based on state flags.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from yolo_developer.orchestrator.state import YoloState

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

logger = structlog.get_logger()

# Type alias for agent node functions
AgentNode = Callable[[YoloState], Awaitable[dict[str, Any]]]


# =============================================================================
# Task 1: Workflow Module Structure
# =============================================================================


@dataclass(frozen=True)
class WorkflowConfig:
    """Configuration for workflow behavior.

    Attributes:
        entry_point: Starting agent for the workflow (default: "analyst").
        enable_checkpointing: Whether to enable state checkpointing (default: True).

    Example:
        >>> config = WorkflowConfig(entry_point="analyst")
        >>> graph = build_workflow(config=config)
    """

    entry_point: str = "analyst"
    enable_checkpointing: bool = True


def get_default_agent_nodes() -> dict[str, AgentNode]:
    """Get the default set of agent nodes for the workflow.

    Returns a dictionary mapping agent names to their node functions.
    All required agents (analyst, pm, architect, dev, tea) are included.

    Returns:
        Dict mapping agent name to async node function.

    Example:
        >>> nodes = get_default_agent_nodes()
        >>> assert "analyst" in nodes
        >>> assert callable(nodes["analyst"])
    """
    from yolo_developer.agents import (
        analyst_node,
        architect_node,
        dev_node,
        pm_node,
        tea_node,
    )

    return {
        "analyst": analyst_node,
        "pm": pm_node,
        "architect": architect_node,
        "dev": dev_node,
        "tea": tea_node,
    }


# =============================================================================
# Task 3: Conditional Routing Functions
# =============================================================================


def route_after_analyst(state: YoloState) -> str:
    """Route after analyst node execution.

    Routes to:
    - "escalate" if escalate_to_human flag is set
    - "pm" otherwise (normal flow)

    Args:
        state: Current workflow state.

    Returns:
        Next node name to execute.
    """
    # Check for escalation flag (uses .get() for optional fields)
    if state.get("escalate_to_human", False):
        logger.info("route_after_analyst", destination="escalate", reason="human_escalation")
        return "escalate"

    logger.debug("route_after_analyst", destination="pm")
    return "pm"


def route_after_pm(state: YoloState) -> str:
    """Route after PM node execution.

    Routes to:
    - "architect" if needs_architecture flag is set
    - "dev" otherwise (skip architecture for simple changes)

    Args:
        state: Current workflow state.

    Returns:
        Next node name to execute.
    """
    if state.get("needs_architecture", False):
        logger.debug("route_after_pm", destination="architect")
        return "architect"

    logger.debug("route_after_pm", destination="dev")
    return "dev"


def route_after_architect(state: YoloState) -> str:
    """Route after architect node execution.

    Always routes to dev - architect output feeds development.

    Args:
        state: Current workflow state.

    Returns:
        "dev" always.
    """
    logger.debug("route_after_architect", destination="dev")
    return "dev"


def route_after_dev(state: YoloState) -> str:
    """Route after dev node execution.

    Always routes to TEA for validation.

    Args:
        state: Current workflow state.

    Returns:
        "tea" always.
    """
    logger.debug("route_after_dev", destination="tea")
    return "tea"


def route_after_tea(state: YoloState) -> str:
    """Route after TEA node execution.

    Routes to:
    - "dev" if gate_blocked flag is set (needs fixes)
    - END otherwise (deployment ready or complete)

    Args:
        state: Current workflow state.

    Returns:
        Next node name or END.
    """
    if state.get("gate_blocked", False):
        logger.info("route_after_tea", destination="dev", reason="gate_blocked")
        return "dev"

    logger.debug("route_after_tea", destination="__end__")
    return END


# =============================================================================
# Task 2: StateGraph Construction
# =============================================================================


def build_workflow(
    config: WorkflowConfig | None = None,
    nodes: dict[str, AgentNode] | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Any:
    """Build and compile the orchestration workflow StateGraph.

    Constructs a LangGraph StateGraph with all agent nodes connected
    via conditional edges for intelligent routing.

    Args:
        config: Workflow configuration (uses defaults if None).
        nodes: Custom node functions (uses defaults if None).
        checkpointer: Optional checkpointer for state persistence.

    Returns:
        Compiled StateGraph ready for execution.

    Example:
        >>> graph = build_workflow()
        >>> result = await graph.ainvoke(initial_state)
    """
    if config is None:
        config = WorkflowConfig()

    if nodes is None:
        nodes = get_default_agent_nodes()

    # Create the StateGraph with YoloState as the state type (AC #3)
    builder: StateGraph[YoloState] = StateGraph(YoloState)

    # Add all agent nodes (AC #1)
    for agent_name, node_fn in nodes.items():
        builder.add_node(agent_name, node_fn)  # type: ignore[call-overload]

    # Add escalation node (placeholder that ends workflow)
    async def escalate_node(state: YoloState) -> dict[str, Any]:
        """Escalation placeholder - ends workflow for human intervention."""
        logger.warning("Workflow escalated to human")
        return {"messages": state.get("messages", [])}

    builder.add_node("escalate", escalate_node)

    # Set entry point (AC #1)
    builder.set_entry_point(config.entry_point)

    # Add conditional edges for routing (AC #2)
    builder.add_conditional_edges(
        "analyst",
        route_after_analyst,
        {"pm": "pm", "escalate": "escalate"},
    )

    builder.add_conditional_edges(
        "pm",
        route_after_pm,
        {"architect": "architect", "dev": "dev"},
    )

    builder.add_conditional_edges(
        "architect",
        route_after_architect,
        {"dev": "dev"},
    )

    builder.add_conditional_edges(
        "dev",
        route_after_dev,
        {"tea": "tea"},
    )

    builder.add_conditional_edges(
        "tea",
        route_after_tea,
        {"dev": "dev", END: END},
    )

    # Escalate always ends
    builder.add_edge("escalate", END)

    logger.info(
        "workflow_built",
        entry_point=config.entry_point,
        node_count=len(nodes) + 1,  # +1 for escalate
        checkpointing_enabled=checkpointer is not None,
    )

    # Compile with optional checkpointing (AC #4)
    if checkpointer is not None:
        return builder.compile(checkpointer=checkpointer)

    return builder.compile()


# =============================================================================
# Task 4: Checkpointing Integration
# =============================================================================


def create_workflow_with_checkpointing(
    checkpointer: BaseCheckpointSaver[Any],
    config: WorkflowConfig | None = None,
    nodes: dict[str, AgentNode] | None = None,
) -> Any:
    """Create a workflow with checkpointing enabled.

    Convenience function for creating checkpointed workflows.

    Args:
        checkpointer: The checkpointer to use (e.g., MemorySaver).
        config: Optional workflow configuration.
        nodes: Optional custom node functions.

    Returns:
        Compiled StateGraph with checkpointing enabled.

    Example:
        >>> from langgraph.checkpoint.memory import MemorySaver
        >>> checkpointer = MemorySaver()
        >>> graph = create_workflow_with_checkpointing(checkpointer)
    """
    return build_workflow(
        config=config,
        nodes=nodes,
        checkpointer=checkpointer,
    )


# =============================================================================
# Task 5: Workflow Execution Interface
# =============================================================================


def create_initial_state(
    starting_agent: str = "analyst",
    messages: list[BaseMessage] | None = None,
) -> YoloState:
    """Create an initial state for workflow execution.

    Helper function to create a properly structured YoloState
    for starting workflow execution.

    Args:
        starting_agent: The agent to start with (default: "analyst").
        messages: Optional initial messages to include.

    Returns:
        YoloState ready for workflow execution.

    Example:
        >>> state = create_initial_state()
        >>> assert state["current_agent"] == "analyst"
        >>>
        >>> from langchain_core.messages import HumanMessage
        >>> state = create_initial_state(messages=[HumanMessage(content="Build app")])
    """
    return YoloState(
        messages=messages or [],
        current_agent=starting_agent,
        handoff_context=None,
        decisions=[],
    )


async def run_workflow(
    initial_state: YoloState,
    config: WorkflowConfig | None = None,
    nodes: dict[str, AgentNode] | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    thread_id: str | None = None,
) -> YoloState:
    """Execute the workflow and return final state.

    Runs the complete workflow from initial state to completion,
    returning the final state after all agents have executed.

    Args:
        initial_state: Starting state for the workflow.
        config: Optional workflow configuration. If enable_checkpointing is True
            and no checkpointer is provided, a MemorySaver will be created.
        nodes: Optional custom node functions.
        checkpointer: Optional checkpointer for persistence.
        thread_id: Optional thread ID for checkpointing.

    Returns:
        Final YoloState after workflow completion.

    Example:
        >>> state = create_initial_state()
        >>> result = await run_workflow(state)
        >>> print(result["current_agent"])
    """
    if config is None:
        config = WorkflowConfig()

    # Auto-create MemorySaver if checkpointing enabled but no checkpointer provided
    if config.enable_checkpointing and checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()

    graph = build_workflow(config=config, nodes=nodes, checkpointer=checkpointer)

    logger.info(
        "workflow_started",
        current_agent=initial_state.get("current_agent"),
        message_count=len(initial_state.get("messages", [])),
    )

    # Build config for execution
    run_config: dict[str, Any] = {}
    if checkpointer is not None:
        # Checkpointing requires a thread_id - auto-generate if not provided
        effective_thread_id = thread_id or str(uuid.uuid4())
        run_config["configurable"] = {"thread_id": effective_thread_id}
    elif thread_id is not None:
        run_config["configurable"] = {"thread_id": thread_id}

    result = await graph.ainvoke(initial_state, run_config or None)

    logger.info(
        "workflow_completed",
        final_agent=result.get("current_agent"),
        decision_count=len(result.get("decisions", [])),
    )

    return result  # type: ignore[no-any-return]


async def stream_workflow(
    initial_state: YoloState,
    config: WorkflowConfig | None = None,
    nodes: dict[str, AgentNode] | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    thread_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream workflow execution events.

    Yields events as each node executes, allowing real-time
    monitoring of workflow progress.

    Args:
        initial_state: Starting state for the workflow.
        config: Optional workflow configuration. If enable_checkpointing is True
            and no checkpointer is provided, a MemorySaver will be created.
        nodes: Optional custom node functions.
        checkpointer: Optional checkpointer for persistence.
        thread_id: Optional thread ID for checkpointing.

    Yields:
        Dict events from each node execution.

    Example:
        >>> state = create_initial_state()
        >>> async for event in stream_workflow(state):
        ...     print(f"Event: {event}")
    """
    if config is None:
        config = WorkflowConfig()

    # Auto-create MemorySaver if checkpointing enabled but no checkpointer provided
    if config.enable_checkpointing and checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()

    graph = build_workflow(config=config, nodes=nodes, checkpointer=checkpointer)

    logger.info(
        "workflow_stream_started",
        current_agent=initial_state.get("current_agent"),
    )

    # Build config for execution
    run_config: dict[str, Any] = {}
    if checkpointer is not None:
        # Checkpointing requires a thread_id - auto-generate if not provided
        effective_thread_id = thread_id or str(uuid.uuid4())
        run_config["configurable"] = {"thread_id": effective_thread_id}
    elif thread_id is not None:
        run_config["configurable"] = {"thread_id": thread_id}

    async for event in graph.astream(initial_state, run_config or None):
        logger.debug("workflow_event", event_keys=list(event.keys()))
        yield event

    logger.info("workflow_stream_completed")
