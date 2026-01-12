"""Task delegation module for SM agent (Story 10.4).

This module provides the core delegation functionality for the SM agent:

- `delegate_task()`: Main async function to delegate tasks to specialized agents
- Task analysis and agent matching based on expertise
- Context preparation for handoffs (FR15, FR69)
- Implicit acknowledgment via LangGraph state updates (AC #4)

The delegation system enables the SM (Scrum Master) agent to coordinate
work distribution across specialized agents (Analyst, PM, Architect, Dev, TEA).

Acknowledgment Design:
    Acknowledgment is implicit in LangGraph-based orchestration. When SM
    returns a routing decision, the state update guarantees delivery to the
    target agent. The `DelegationResult.acknowledged=True` records the
    handoff commitment timestamp. For future distributed systems, explicit
    acknowledgment protocols can be added via `_verify_acknowledgment()`.

Example:
    >>> from yolo_developer.agents.sm.delegation import delegate_task
    >>> from yolo_developer.orchestrator.state import YoloState
    >>>
    >>> state: YoloState = {"current_agent": "sm", "messages": [], ...}
    >>> result = await delegate_task(
    ...     state=state,
    ...     task_type="implementation",
    ...     task_description="Implement feature X",
    ... )
    >>> result.target_agent
    'dev'

Architecture:
    Per ADR-005 and ADR-007, the delegation system:
    - Uses async/await for all I/O operations
    - Returns state update dicts (never mutates input)
    - Uses frozen dataclasses for immutable outputs
    - Logs all delegation events via structlog

References:
    - FR10: SM Agent can delegate tasks to appropriate specialized agents
    - FR15: System can handle agent handoffs with context preservation
    - FR68: SM Agent can trigger inter-agent sync protocols
    - FR69: SM Agent can inject context when agents lack information
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

import structlog

from yolo_developer.agents.sm.delegation_types import (
    AGENT_EXPERTISE,
    TASK_TO_AGENT,
    DelegationConfig,
    DelegationRequest,
    DelegationResult,
    Priority,
    TaskType,
)

if TYPE_CHECKING:
    from yolo_developer.orchestrator.state import YoloState

logger = structlog.get_logger(__name__)

# =============================================================================
# Task Analysis Functions (Task 2)
# =============================================================================


def _analyze_task(task_description: str) -> dict[str, Any]:
    """Analyze a task description to extract requirements.

    Examines the task description to identify key characteristics
    that help determine the appropriate agent and context needs.

    Args:
        task_description: Human-readable description of the task.

    Returns:
        Dictionary with analysis results:
        - keywords: Extracted keywords from description
        - estimated_complexity: low/medium/high
        - requires_context: List of context types needed

    Example:
        >>> analysis = _analyze_task("Implement user authentication")
        >>> "implement" in analysis["keywords"]
        True
    """
    # Extract keywords (simple word extraction for now)
    keywords = set(task_description.lower().split())

    # Estimate complexity based on keywords
    high_complexity_indicators = {"complex", "architecture", "refactor", "integrate"}
    low_complexity_indicators = {"simple", "fix", "update", "minor"}

    complexity: Literal["low", "medium", "high"] = "medium"
    if keywords & high_complexity_indicators:
        complexity = "high"
    elif keywords & low_complexity_indicators:
        complexity = "low"

    # Determine context requirements
    requires_context: list[str] = ["messages", "decisions"]
    if "test" in keywords or "validation" in keywords:
        requires_context.extend(["test_results", "coverage"])
    if "implement" in keywords or "code" in keywords:
        requires_context.extend(["current_story", "design"])
    if "requirement" in keywords or "analysis" in keywords:
        requires_context.append("seed_input")

    return {
        "keywords": keywords,
        "estimated_complexity": complexity,
        "requires_context": requires_context,
    }


def _match_agent(task_type: TaskType) -> str:
    """Determine the best agent for a task type based on expertise.

    Uses the TASK_TO_AGENT mapping to find the agent responsible
    for handling a specific type of task.

    Args:
        task_type: Type of task to delegate.

    Returns:
        Agent name (e.g., "dev", "analyst", "pm").

    Raises:
        ValueError: If task_type is not recognized.

    Example:
        >>> _match_agent("implementation")
        'dev'
        >>> _match_agent("requirement_analysis")
        'analyst'
    """
    if task_type not in TASK_TO_AGENT:
        raise ValueError(f"Unknown task type: {task_type}")

    agent = TASK_TO_AGENT[task_type]

    logger.debug(
        "agent_matched",
        task_type=task_type,
        matched_agent=agent,
    )

    return agent


async def _validate_agent_availability(
    target_agent: str,
    state: YoloState,
) -> bool:
    """Check if an agent is available to accept work.

    Examines the current state to determine if the target agent
    can accept a delegated task. Checks for blocking conditions
    like gate failures or escalations.

    Args:
        target_agent: Name of the agent to check.
        state: Current orchestration state.

    Returns:
        True if agent can accept work, False otherwise.

    Example:
        >>> await _validate_agent_availability("dev", state)
        True
    """
    # Check if agent is valid
    if target_agent not in AGENT_EXPERTISE:
        logger.warning(
            "invalid_agent",
            agent=target_agent,
        )
        return False

    # Check for blocking conditions
    if state.get("gate_blocked", False):
        current_agent = state.get("current_agent", "")
        # If gate is blocked and we're trying to route to same agent, not available
        if current_agent == target_agent:
            logger.debug(
                "agent_blocked_by_gate",
                agent=target_agent,
            )
            return False

    # Check for escalation state
    if state.get("escalate_to_human", False):
        logger.debug(
            "escalation_active",
            target_agent=target_agent,
        )
        # Allow delegation during escalation for recovery
        return True

    return True


def _get_agent_expertise(agent: str) -> tuple[TaskType, ...]:
    """Get the expertise areas for an agent.

    Args:
        agent: Agent name.

    Returns:
        Tuple of TaskType values the agent can handle.

    Example:
        >>> _get_agent_expertise("dev")
        ('implementation',)
    """
    return AGENT_EXPERTISE.get(agent, ())


# =============================================================================
# Context Preparation Functions (Task 3)
# =============================================================================


def _prepare_delegation_context(
    state: YoloState,
    task_type: TaskType,
    target_agent: str,
) -> dict[str, Any]:
    """Prepare context for delegation.

    Extracts relevant state information for the target agent
    based on task type and agent needs per FR15 and FR69.

    Args:
        state: Current orchestration state.
        task_type: Type of task being delegated.
        target_agent: Agent receiving the delegation.

    Returns:
        Dictionary of context data for the target agent.

    Example:
        >>> context = _prepare_delegation_context(state, "implementation", "dev")
        >>> "message_count" in context
        True
    """
    context: dict[str, Any] = {}

    # Always include core context
    context["message_count"] = len(state.get("messages", []))
    context["decision_count"] = len(state.get("decisions", []))
    context["source_agent"] = state.get("current_agent", "sm")

    # Task-specific context extraction
    if task_type == "requirement_analysis":
        context["seed_input"] = state.get("seed_input")
        context["sop_constraints"] = state.get("sop_constraints")
    elif task_type == "story_creation":
        context["requirements"] = state.get("requirements", [])
        context["priorities"] = state.get("priorities", [])
    elif task_type == "architecture_design":
        context["stories"] = state.get("stories", [])
        context["requirements"] = state.get("requirements", [])
        context["tech_stack"] = state.get("tech_stack")
    elif task_type == "implementation":
        context["current_story"] = state.get("current_story")
        context["design"] = state.get("design")
        context["patterns"] = state.get("patterns", [])
    elif task_type == "validation":
        context["implementation"] = state.get("implementation")
        context["test_results"] = state.get("test_results")
        context["coverage"] = state.get("coverage")
    elif task_type == "orchestration":
        context["sprint_plan"] = state.get("sprint_plan")
        context["health_metrics"] = state.get("health_metrics")

    # Include existing handoff context if present
    existing_handoff = state.get("handoff_context")
    if existing_handoff is not None:
        context["previous_handoff"] = existing_handoff

    logger.debug(
        "delegation_context_prepared",
        task_type=task_type,
        target_agent=target_agent,
        context_keys=list(context.keys()),
    )

    return context


def _get_relevant_state_keys(task_type: TaskType) -> tuple[str, ...]:
    """Get state keys relevant for a task type.

    Returns the state dictionary keys that should be preserved
    and passed during a handoff for the given task type.

    Args:
        task_type: Type of task being delegated.

    Returns:
        Tuple of state key names.

    Example:
        >>> keys = _get_relevant_state_keys("implementation")
        >>> "current_story" in keys
        True
    """
    base_keys = ("messages", "decisions", "current_agent")

    task_keys: dict[TaskType, tuple[str, ...]] = {
        "requirement_analysis": ("seed_input", "sop_constraints"),
        "story_creation": ("requirements", "priorities"),
        "architecture_design": ("stories", "requirements", "tech_stack"),
        "implementation": ("current_story", "design", "patterns"),
        "validation": ("implementation", "test_results", "coverage"),
        "orchestration": ("sprint_plan", "health_metrics"),
    }

    return base_keys + task_keys.get(task_type, ())


# =============================================================================
# Acknowledgment Verification Functions (Task 5)
# =============================================================================


async def _verify_acknowledgment(
    target_agent: str,
    config: DelegationConfig,
) -> tuple[bool, str | None]:
    """Verify that an agent acknowledged a delegation.

    Current Implementation (Implicit Acknowledgment):
        In LangGraph-based orchestration, acknowledgment is implicit via state
        update when the agent picks up the task. The routing mechanism guarantees
        delivery to the target agent, and the agent's execution constitutes
        acknowledgment. This satisfies AC #4 through the state machine guarantee.

    Why Implicit Works:
        - LangGraph ensures message delivery via state updates
        - Target agent will execute on next graph iteration
        - No network/async boundary that requires explicit ACK
        - The `DelegationResult.acknowledged=True` with timestamp records
          the moment SM committed to the handoff

    Future Enhancement:
        For distributed agent systems or async protocols, this function
        provides a hook for explicit acknowledgment with timeout handling.

    Args:
        target_agent: Agent that should acknowledge.
        config: Delegation configuration with timeout settings.

    Returns:
        Tuple of (acknowledged: bool, timestamp: str | None).
        Always returns (True, ISO timestamp) in current implementation.

    Example:
        >>> acknowledged, timestamp = await _verify_acknowledgment("dev", config)
        >>> acknowledged
        True
    """
    # Implicit acknowledgment: state update mechanism ensures delivery
    # The handoff is committed when SM returns the routing decision
    timestamp = datetime.now(timezone.utc).isoformat()

    logger.debug(
        "acknowledgment_verified",
        target_agent=target_agent,
        acknowledgment_type="implicit",
        timestamp=timestamp,
    )

    return True, timestamp


def _handle_unacknowledged_delegation(
    target_agent: str,
    task_type: TaskType,
    config: DelegationConfig,
) -> tuple[str, str]:
    """Handle case where delegation was not acknowledged.

    Determines the recovery action when an agent fails to acknowledge
    a delegated task within the timeout period.

    Args:
        target_agent: Agent that failed to acknowledge.
        task_type: Type of task that wasn't acknowledged.
        config: Delegation configuration.

    Returns:
        Tuple of (action: str, rationale: str).
        Action is one of: "retry", "escalate", "fallback".

    Example:
        >>> action, rationale = _handle_unacknowledged_delegation("dev", "implementation", config)
        >>> action in ("retry", "escalate", "fallback")
        True
    """
    # Determine recovery action based on retry attempts
    if config.max_retry_attempts > 0:
        return "retry", f"Will retry delegation to {target_agent}"

    # If retries exhausted, escalate
    return "escalate", f"Agent {target_agent} unresponsive after max retries"


# =============================================================================
# Main Delegation Function (Task 6)
# =============================================================================


async def delegate_task(
    state: YoloState,
    task_type: TaskType,
    task_description: str,
    priority: Priority = "normal",
    config: DelegationConfig | None = None,
) -> DelegationResult:
    """Delegate a task to the appropriate agent (FR10).

    Main entry point for task delegation. Orchestrates the full
    delegation flow: analysis -> matching -> context -> acknowledgment.

    Args:
        state: Current orchestration state.
        task_type: Type of task to delegate.
        task_description: Description of what needs to be done.
        priority: Task priority level.
        config: Delegation configuration (uses defaults if None).

    Returns:
        DelegationResult with delegation details and acknowledgment status.

    Example:
        >>> result = await delegate_task(
        ...     state=state,
        ...     task_type="implementation",
        ...     task_description="Implement user authentication",
        ...     priority="high",
        ... )
        >>> result.success
        True
        >>> result.request.target_agent
        'dev'
    """
    config = config or DelegationConfig()
    source_agent = state.get("current_agent", "sm")

    logger.info(
        "delegation_started",
        task_type=task_type,
        source_agent=source_agent,
        priority=priority,
    )

    # Step 1: Analyze task to understand requirements
    task_analysis = _analyze_task(task_description)

    logger.debug(
        "task_analyzed",
        keywords_count=len(task_analysis["keywords"]),
        complexity=task_analysis["estimated_complexity"],
        context_needs=task_analysis["requires_context"],
    )

    # Step 2: Match agent based on task type
    target_agent = _match_agent(task_type)

    # Step 3: Validate agent availability
    is_available = await _validate_agent_availability(target_agent, state)

    if not is_available:
        logger.warning(
            "delegation_failed_unavailable",
            target_agent=target_agent,
            task_type=task_type,
        )
        request = DelegationRequest(
            task_type=task_type,
            task_description=task_description,
            source_agent=source_agent,
            target_agent=target_agent,
            context={},
            priority=priority,
        )
        return DelegationResult(
            request=request,
            success=False,
            acknowledged=False,
            error_message=f"Agent {target_agent} is not available",
        )

    # Step 4: Prepare delegation context (enriched with task analysis)
    context = _prepare_delegation_context(state, task_type, target_agent)
    context["task_analysis"] = {
        "estimated_complexity": task_analysis["estimated_complexity"],
        "requires_context": task_analysis["requires_context"],
    }

    # Step 5: Create delegation request
    request = DelegationRequest(
        task_type=task_type,
        task_description=task_description,
        source_agent=source_agent,
        target_agent=target_agent,
        context=context,
        priority=priority,
    )

    # Step 6: Verify acknowledgment
    acknowledged, ack_timestamp = await _verify_acknowledgment(target_agent, config)

    # Step 7: Create handoff context for state updates
    relevant_keys = _get_relevant_state_keys(task_type)
    handoff_context = {
        "source_agent": source_agent,
        "target_agent": target_agent,
        "task_summary": task_description,
        "relevant_state_keys": relevant_keys,
        "instructions": f"Delegated {task_type} task with {priority} priority",
        "priority": priority,
    }

    logger.info(
        "delegation_complete",
        source_agent=source_agent,
        target_agent=target_agent,
        task_type=task_type,
        acknowledged=acknowledged,
    )

    return DelegationResult(
        request=request,
        success=True,
        acknowledged=acknowledged,
        acknowledgment_timestamp=ack_timestamp,
        handoff_context=handoff_context,
    )


# =============================================================================
# Utility Functions
# =============================================================================


def routing_to_task_type(routing_decision: str) -> TaskType | None:
    """Convert a routing decision to a task type.

    Maps agent names from routing decisions to the task type
    that agent handles.

    Args:
        routing_decision: Agent name from routing (e.g., "dev", "analyst").

    Returns:
        TaskType for that agent, or None if not applicable.

    Example:
        >>> routing_to_task_type("dev")
        'implementation'
        >>> routing_to_task_type("escalate")
        None
    """
    agent_to_task: dict[str, TaskType] = {
        "analyst": "requirement_analysis",
        "pm": "story_creation",
        "architect": "architecture_design",
        "dev": "implementation",
        "tea": "validation",
        "sm": "orchestration",
    }

    return agent_to_task.get(routing_decision)
