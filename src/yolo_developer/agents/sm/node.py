"""SM agent node for LangGraph orchestration (Story 10.2).

This module provides the sm_node function that integrates with the
LangGraph orchestration workflow. The SM (Scrum Master) agent serves
as the control plane for orchestration decisions.

Key Concepts:
- **YoloState Input**: Receives state as TypedDict, not Pydantic
- **Immutable Updates**: Returns state update dict, never mutates input
- **Async I/O**: All operations use async/await
- **Structured Logging**: Uses structlog for audit trail
- **Routing Decisions**: Determines next agent based on state analysis
- **Circular Logic Detection**: Detects agent ping-pong patterns (>3 exchanges)
- **Escalation Handling**: Triggers human intervention when needed

Example:
    >>> from yolo_developer.agents.sm import sm_node
    >>> from yolo_developer.orchestrator.state import YoloState
    >>>
    >>> state: YoloState = {
    ...     "messages": [...],
    ...     "current_agent": "sm",
    ...     "handoff_context": None,
    ...     "decisions": [],
    ... }
    >>> result = await sm_node(state)
    >>> result["sm_output"]["routing_decision"]
    'analyst'

Architecture Note:
    Per ADR-005, this node follows the LangGraph pattern of receiving
    full state and returning only the updates to apply. Per ADR-007,
    SM is the control plane for orchestration decisions.
"""

from __future__ import annotations

from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.agents.sm.delegation import delegate_task, routing_to_task_type
from yolo_developer.agents.sm.health import monitor_health
from yolo_developer.agents.sm.health_types import HealthStatus
from yolo_developer.agents.sm.types import (
    CIRCULAR_LOGIC_THRESHOLD,
    NATURAL_SUCCESSOR,
    VALID_AGENTS,
    AgentExchange,
    EscalationReason,
    RoutingDecision,
    SMOutput,
)
from yolo_developer.gates import quality_gate
from yolo_developer.orchestrator.context import Decision
from yolo_developer.orchestrator.state import YoloState, create_agent_message

logger = structlog.get_logger(__name__)


# =============================================================================
# State Analysis Functions (Task 2)
# =============================================================================


def _analyze_current_state(state: YoloState) -> dict[str, Any]:
    """Analyze the current orchestration state.

    Evaluates workflow progress by examining state flags, agent history,
    and handoff context to inform routing decisions.

    Args:
        state: Current orchestration state.

    Returns:
        Dictionary containing state analysis results:
        - current_agent: The currently active agent
        - message_count: Number of messages in state
        - has_handoff_context: Whether handoff context is present
        - needs_architecture: Whether architecture work is needed
        - gate_blocked: Whether a gate is blocking progress
        - escalate_to_human: Whether human escalation is requested
        - decision_count: Number of decisions made

    Example:
        >>> analysis = _analyze_current_state(state)
        >>> analysis["needs_architecture"]
        True
    """
    return {
        "current_agent": state.get("current_agent", ""),
        "message_count": len(state.get("messages", [])),
        "has_handoff_context": state.get("handoff_context") is not None,
        "needs_architecture": state.get("needs_architecture", False),
        "gate_blocked": state.get("gate_blocked", False),
        "escalate_to_human": state.get("escalate_to_human", False),
        "decision_count": len(state.get("decisions", [])),
    }


def _count_recent_exchanges(state: YoloState) -> tuple[int, list[AgentExchange]]:
    """Count recent agent exchanges from message history.

    Analyzes messages to build a list of agent-to-agent exchanges
    for circular logic detection.

    Args:
        state: Current orchestration state.

    Returns:
        Tuple of (exchange_count, list of AgentExchange objects).

    Example:
        >>> count, exchanges = _count_recent_exchanges(state)
        >>> count
        2
    """
    messages = state.get("messages", [])
    exchanges: list[AgentExchange] = []
    skipped_count = 0

    prev_agent: str | None = None
    for msg in messages:
        # Extract agent from message metadata
        if hasattr(msg, "additional_kwargs"):
            agent = msg.additional_kwargs.get("agent")
            if agent and prev_agent and agent != prev_agent:
                exchange = AgentExchange(
                    source_agent=prev_agent,
                    target_agent=agent,
                    exchange_type="handoff",
                    topic="workflow_transition",
                )
                exchanges.append(exchange)
            prev_agent = agent
        else:
            skipped_count += 1

    if skipped_count > 0:
        logger.debug(
            "exchange_tracking_skipped_messages",
            skipped_count=skipped_count,
            total_messages=len(messages),
        )

    return len(exchanges), exchanges


def _detect_circular_pattern(exchanges: list[AgentExchange]) -> bool:
    """Detect circular logic pattern in exchanges.

    Checks if the same pair of agents have exchanged more than
    CIRCULAR_LOGIC_THRESHOLD times, indicating a ping-pong pattern.

    Args:
        exchanges: List of recent agent exchanges.

    Returns:
        True if circular pattern detected, False otherwise.

    Example:
        >>> _detect_circular_pattern(exchanges)
        True  # If analyst-pm exchanges happened 4+ times
    """
    if len(exchanges) < CIRCULAR_LOGIC_THRESHOLD:
        return False

    # Count exchanges between agent pairs
    pair_counts: dict[tuple[str, str], int] = {}
    for exchange in exchanges:
        # Normalize pair (order doesn't matter for circular detection)
        sorted_agents = sorted([exchange.source_agent, exchange.target_agent])
        pair: tuple[str, str] = (sorted_agents[0], sorted_agents[1])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    # Check if any pair exceeds threshold
    for pair, count in pair_counts.items():
        if count > CIRCULAR_LOGIC_THRESHOLD:
            logger.warning(
                "circular_logic_detected",
                agent_pair=pair,
                exchange_count=count,
                threshold=CIRCULAR_LOGIC_THRESHOLD,
            )
            return True

    return False


def _check_for_escalation(state: YoloState) -> tuple[bool, EscalationReason | None]:
    """Check if escalation to human is needed.

    Examines state flags and conditions that require human intervention.

    Args:
        state: Current orchestration state.

    Returns:
        Tuple of (should_escalate, escalation_reason).

    Example:
        >>> should_escalate, reason = _check_for_escalation(state)
        >>> should_escalate
        True
        >>> reason
        'human_requested'
    """
    # Priority 1: Explicit escalation flag
    if state.get("escalate_to_human", False):
        logger.info("escalation_check", result="human_requested")
        return True, "human_requested"

    # Priority 2: Agent failure flag
    if state.get("agent_failure", False):
        logger.info("escalation_check", result="agent_failure")
        return True, "agent_failure"

    return False, None


def _check_for_circular_logic(state: YoloState) -> tuple[bool, list[AgentExchange]]:
    """Check for circular logic pattern in state.

    Analyzes message history to detect agent ping-pong patterns
    that indicate circular logic (per FR12).

    Args:
        state: Current orchestration state.

    Returns:
        Tuple of (is_circular, exchanges).

    Example:
        >>> is_circular, exchanges = _check_for_circular_logic(state)
        >>> is_circular
        False
    """
    count, exchanges = _count_recent_exchanges(state)
    is_circular = _detect_circular_pattern(exchanges)

    logger.debug(
        "circular_logic_check",
        exchange_count=count,
        is_circular=is_circular,
    )

    return is_circular, exchanges


def _get_recovery_agent(state: YoloState) -> RoutingDecision:
    """Determine which agent should handle recovery when gate is blocked.

    Analyzes state to determine the appropriate agent to route to
    for fixing gate-blocked issues.

    Args:
        state: Current orchestration state with gate_blocked=True.

    Returns:
        Agent name to route to for recovery (guaranteed valid RoutingDecision).

    Example:
        >>> recovery_agent = _get_recovery_agent(state)
        >>> recovery_agent
        'dev'  # If TEA gate blocked, route back to dev
    """
    current = state.get("current_agent", "")

    # If TEA blocked, route back to dev for fixes
    if current == "tea":
        return "dev"

    # If architect blocked, route back to PM for re-scoping
    if current == "architect":
        return "pm"

    # Default: route to analyst for re-analysis
    return "analyst"


def _get_natural_successor(current_agent: str, state: YoloState) -> RoutingDecision:
    """Get the natural successor agent in the workflow.

    Uses standard workflow progression with state-aware overrides.

    Args:
        current_agent: Currently active agent.
        state: Current orchestration state.

    Returns:
        Next agent in natural workflow progression (guaranteed valid RoutingDecision).

    Example:
        >>> _get_natural_successor("analyst", state)
        'pm'
    """
    # Check for PM-specific routing (needs_architecture flag)
    if current_agent == "pm":
        if state.get("needs_architecture", False):
            return "architect"
        return "dev"  # Skip architect if not needed

    # Use standard succession - NATURAL_SUCCESSOR is typed as dict[str, RoutingDecision]
    return NATURAL_SUCCESSOR.get(current_agent, "analyst")


def _get_next_agent(state: YoloState) -> tuple[RoutingDecision, str]:
    """Determine the next agent based on state analysis.

    Priority-based routing decision:
    1. Check for explicit escalation
    2. Check for circular logic
    3. Check for blocked gates
    4. Normal flow based on current agent

    Args:
        state: Current orchestration state.

    Returns:
        Tuple of (routing_decision, rationale).

    Example:
        >>> decision, rationale = _get_next_agent(state)
        >>> decision
        'pm'
        >>> rationale
        'Natural workflow progression from analyst to pm'
    """
    # Priority 1: Check for escalation
    should_escalate, reason = _check_for_escalation(state)
    if should_escalate:
        return "escalate", f"Escalation triggered: {reason}"

    # Priority 2: Check for circular logic
    is_circular, _exchanges = _check_for_circular_logic(state)
    if is_circular:
        return "escalate", "Circular logic detected: agents ping-ponging on same issue"

    # Priority 3: Check for blocked gates
    if state.get("gate_blocked", False):
        recovery = _get_recovery_agent(state)
        return recovery, f"Gate blocked, routing to {recovery} for recovery"

    # Priority 4: Normal flow based on current agent
    current = state.get("current_agent", "")

    # Validate current agent
    if current not in VALID_AGENTS:
        logger.warning("invalid_current_agent", agent=current)
        return "analyst", f"Unknown agent '{current}', defaulting to analyst"

    # Get natural successor
    next_agent = _get_natural_successor(current, state)
    rationale = f"Natural workflow progression from {current} to {next_agent}"

    return next_agent, rationale


def _get_routing_rationale(
    state: YoloState,
    routing_decision: RoutingDecision,
    analysis: dict[str, Any],
) -> str:
    """Generate detailed rationale for routing decision.

    Creates a human-readable explanation of why the routing
    decision was made, including relevant state context.

    Args:
        state: Current orchestration state.
        routing_decision: The chosen routing target.
        analysis: State analysis results.

    Returns:
        Detailed rationale string.

    Example:
        >>> rationale = _get_routing_rationale(state, "pm", analysis)
        >>> rationale
        'Routing to pm: Natural workflow progression...'
    """
    current = analysis.get("current_agent", "unknown")
    msg_count = analysis.get("message_count", 0)
    decision_count = analysis.get("decision_count", 0)

    base_rationale = (
        f"Current agent: {current}, messages: {msg_count}, decisions: {decision_count}. "
    )

    if routing_decision == "escalate":
        if analysis.get("escalate_to_human"):
            return base_rationale + "Human escalation explicitly requested."
        return base_rationale + "Escalation triggered due to circular logic or unresolvable issue."

    if analysis.get("gate_blocked"):
        return base_rationale + f"Gate blocked, routing to {routing_decision} for recovery."

    return base_rationale + f"Natural workflow progression to {routing_decision}."


# =============================================================================
# SM Node Function (Task 3)
# =============================================================================


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
@quality_gate("sm_routing", blocking=False)
async def sm_node(state: YoloState) -> dict[str, Any]:
    """SM agent node for orchestration control plane.

    Receives orchestration state and makes routing decisions based on
    state analysis. Handles escalation, circular logic detection, and
    gate-blocked recovery.

    This function follows the LangGraph node pattern:
    - Receives full state as YoloState TypedDict
    - Returns only the state updates (not full state)
    - Never mutates the input state
    - Uses tenacity for retry with exponential backoff

    Args:
        state: Current orchestration state.

    Returns:
        State update dict with:
        - messages: List of new messages to append
        - decisions: List of new decisions to append
        - sm_output: Serialized SMOutput with routing details
        - routing_decision: Target agent for routing (convenience key)

    Example:
        >>> state: YoloState = {
        ...     "messages": [...],
        ...     "current_agent": "analyst",
        ...     "handoff_context": None,
        ...     "decisions": [],
        ... }
        >>> result = await sm_node(state)
        >>> result["sm_output"]["routing_decision"]
        'pm'
    """
    logger.info(
        "sm_node_start",
        current_agent=state.get("current_agent"),
        message_count=len(state.get("messages", [])),
    )

    # Step 1: Analyze current state (AC #2)
    analysis = _analyze_current_state(state)
    logger.debug("state_analysis", **analysis)

    # Step 2: Check for circular logic (AC #4)
    is_circular, exchanges = _check_for_circular_logic(state)
    exchange_count = len(exchanges)

    # Step 3: Check for escalation (AC #4)
    should_escalate, escalation_reason = _check_for_escalation(state)

    # Override escalation if circular logic detected
    if is_circular and not should_escalate:
        should_escalate = True
        escalation_reason = "circular_logic"

    # Step 4: Determine routing decision (AC #1, #2)
    routing_decision, base_rationale = _get_next_agent(state)
    routing_rationale = _get_routing_rationale(state, routing_decision, analysis)

    # Step 5: Handle gate-blocked recovery (AC #4)
    gate_blocked = analysis.get("gate_blocked", False)
    recovery_agent = _get_recovery_agent(state) if gate_blocked else None

    # Step 6: Delegate task to target agent (Story 10.4 - FR10)
    delegation_result = None
    handoff_context = None
    if routing_decision not in ("escalate", "sm"):
        # Convert routing decision to task type for delegation
        task_type = routing_to_task_type(routing_decision)
        if task_type is not None:
            delegation_result = await delegate_task(
                state=state,
                task_type=task_type,
                task_description=base_rationale,
            )
            if delegation_result.success:
                handoff_context = delegation_result.handoff_context

            logger.debug(
                "delegation_completed",
                target_agent=routing_decision,
                task_type=task_type,
                success=delegation_result.success,
                acknowledged=delegation_result.acknowledged,
            )

    # Step 6b: Monitor health (Story 10.5 - FR11, FR67)
    health_status: HealthStatus | None = None
    try:
        health_status = await monitor_health(state)
        if not health_status.is_healthy:
            logger.warning(
                "health_degraded",
                status=health_status.status,
                alert_count=len(health_status.alerts),
            )
    except Exception as e:
        # Health monitoring should never block the main workflow
        logger.error("health_monitoring_failed", error=str(e))

    # Step 7: Create processing notes
    processing_notes = (
        f"Analyzed state with {analysis['message_count']} messages, "
        f"{analysis['decision_count']} decisions. "
        f"Exchange count: {exchange_count}. "
        f"Circular: {is_circular}. "
        f"Gate blocked: {gate_blocked}."
    )
    if delegation_result:
        processing_notes += f" Delegation to {routing_decision}: {'success' if delegation_result.success else 'failed'}."
    if health_status:
        processing_notes += f" Health: {health_status.status} ({len(health_status.alerts)} alerts)."

    # Create SM output (AC #3 - structured format)
    output = SMOutput(
        routing_decision=routing_decision,
        routing_rationale=routing_rationale,
        circular_logic_detected=is_circular,
        escalation_triggered=should_escalate,
        escalation_reason=escalation_reason,
        exchange_count=exchange_count,
        recent_exchanges=tuple(exchanges[-10:]),  # Keep last 10 exchanges
        gate_blocked=gate_blocked,
        recovery_agent=recovery_agent,
        processing_notes=processing_notes,
        delegation_result=delegation_result.to_dict() if delegation_result else None,
        health_status=health_status.to_dict() if health_status else None,
    )

    # Create decision record (includes delegation info for audit trail - Task 4.2)
    delegation_summary = ""
    if delegation_result:
        delegation_summary = f" Delegated {delegation_result.request.task_type} task."
    decision = Decision(
        agent="sm",
        summary=f"Routing to {routing_decision}.{delegation_summary}",
        rationale=routing_rationale,
        related_artifacts=("delegation",) if delegation_result else (),
    )

    # Create output message
    message = create_agent_message(
        content=f"SM decision: route to {routing_decision}. {base_rationale}",
        agent="sm",
        metadata={"output": output.to_dict()},
    )

    logger.info(
        "sm_node_complete",
        routing_decision=routing_decision,
        circular_detected=is_circular,
        escalation_triggered=should_escalate,
        exchange_count=exchange_count,
        delegation_success=delegation_result.success if delegation_result else None,
        health_status=health_status.status if health_status else None,
    )

    # Return ONLY updates (AC #1, #2, #3, #4)
    # Includes handoff_context for state updates (Task 7.4)
    # Includes health_status for current snapshot (Story 10.5)
    # Includes health_history for trend analysis (Story 10.5 Task 7.4)
    # health_history accumulates snapshots over time for trend analysis
    health_snapshot = None
    if health_status:
        health_snapshot = {
            "status": health_status.status,
            "is_healthy": health_status.is_healthy,
            "alert_count": len(health_status.alerts),
            "overall_churn_rate": health_status.metrics.overall_churn_rate,
            "unproductive_churn_rate": health_status.metrics.unproductive_churn_rate,
            "cycle_time_percentiles": health_status.metrics.cycle_time_percentiles,
            "agent_idle_times": health_status.metrics.agent_idle_times,
            "evaluated_at": health_status.evaluated_at,
        }

    return {
        "messages": [message],
        "decisions": [decision],
        "sm_output": output.to_dict(),
        "routing_decision": routing_decision,  # Convenience key for routing
        "handoff_context": handoff_context,  # From delegation (Story 10.4)
        "health_status": health_status.to_dict() if health_status else None,  # Current snapshot (Story 10.5)
        "health_history": [health_snapshot] if health_snapshot else [],  # For trend analysis (Task 7.4)
    }
