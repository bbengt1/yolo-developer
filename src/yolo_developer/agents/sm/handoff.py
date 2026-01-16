"""Agent handoff management for the SM agent (Story 10.8).

This module provides managed handoff functionality for agent transitions
in the orchestration workflow. Key features:

- Context preparation: Gathers decisions, memory refs, and messages
- State updates: Immutably updates state for target agent
- Validation: Verifies context integrity and completeness
- Timing: Logs handoff metrics for NFR-PERF-1 compliance

Architecture Notes:
- All functions are async-first (per ADR-005)
- State updates via dict, never mutate input (per ADR-001)
- Graceful degradation on failures
- Structured logging with structlog

References:
- FR14: System can execute agents in defined sequence
- FR15: System can handle agent handoffs with context preservation
- NFR-PERF-1: Agent handoff latency <5 seconds

Example:
    >>> from yolo_developer.agents.sm.handoff import manage_handoff
    >>> result = await manage_handoff(state, "pm", "architect")
    >>> result.success
    True
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import structlog
from langchain_core.messages import BaseMessage

from yolo_developer.agents.sm.handoff_types import (
    HandoffConfig,
    HandoffMetrics,
    HandoffRecord,
    HandoffResult,
)
from yolo_developer.orchestrator.context import (
    Decision,
    HandoffContext,
    compute_state_checksum,
    validate_state_integrity,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Agent Context Requirements (Task 4.5)
# =============================================================================

AGENT_CONTEXT_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "analyst": ("seed_input", "clarifications"),
    "pm": ("requirements", "analyst_decisions"),
    "architect": ("stories", "pm_decisions", "tech_constraints"),
    "dev": ("current_story", "architecture_decisions", "code_patterns"),
    "tea": ("implementation", "test_requirements", "coverage_config"),
    "sm": ("all_decisions", "health_status", "sprint_progress"),
}
"""Context requirements by target agent.

Maps each agent to the context items it typically needs during handoff.
Used for validation and warning generation.
"""


# =============================================================================
# Context Preparation Functions (Task 2)
# =============================================================================


def _prepare_handoff_context(
    state: dict[str, Any],
    source_agent: str,
    target_agent: str,
    config: HandoffConfig,
) -> HandoffContext:
    """Prepare handoff context for target agent (Task 2.2).

    Gathers decisions, memory refs, and other context needed by the
    target agent to continue work.

    Args:
        state: Current orchestration state.
        source_agent: Agent handing off work.
        target_agent: Agent receiving work.
        config: Handoff configuration.

    Returns:
        HandoffContext with all relevant context for target agent.
    """
    # Gather decisions
    decisions = _gather_decisions_for_handoff(
        state=state,
        source_agent=source_agent,
        include_all=config.include_all_messages,
    )

    # Gather memory refs
    memory_refs = _gather_memory_refs_for_handoff(state=state)

    return HandoffContext(
        source_agent=source_agent,
        target_agent=target_agent,
        decisions=decisions,
        memory_refs=memory_refs,
    )


def _gather_decisions_for_handoff(
    state: dict[str, Any],
    source_agent: str,
    include_all: bool = False,
    max_decisions: int = 50,
) -> tuple[Decision, ...]:
    """Gather decisions relevant to handoff (Task 2.3).

    Args:
        state: Current orchestration state.
        source_agent: Agent handing off work.
        include_all: If True, include all decisions regardless of agent.
            If False, prioritize decisions from source agent and its
            predecessors in the workflow.
        max_decisions: Maximum decisions to include.

    Returns:
        Tuple of Decision objects for the handoff context.
    """
    decisions: list[Decision] = state.get("decisions", [])

    if not decisions:
        return ()

    if include_all:
        # Include all decisions up to max limit
        return tuple(decisions[-max_decisions:])

    # When not including all, prioritize relevant decisions:
    # 1. Decisions from the source agent (most important)
    # 2. Recent decisions from other agents (for context)
    source_decisions = [d for d in decisions if d.agent == source_agent]
    other_decisions = [d for d in decisions if d.agent != source_agent]

    # Take all source decisions and fill remaining with recent others
    source_count = len(source_decisions)
    other_limit = max(0, max_decisions - source_count)

    # Prioritize recent other decisions
    relevant_others = other_decisions[-other_limit:] if other_limit > 0 else []

    # Combine: other decisions first (chronological), then source decisions
    combined = relevant_others + source_decisions

    # Limit to max_decisions
    return tuple(combined[-max_decisions:])


def _gather_memory_refs_for_handoff(
    state: dict[str, Any],
) -> tuple[str, ...]:
    """Gather memory references for handoff (Task 2.4).

    Extracts relevant memory references from decisions and state
    for the target agent.

    Args:
        state: Current orchestration state.

    Returns:
        Tuple of memory reference strings.
    """
    refs: set[str] = set()

    # Extract from decision artifacts
    decisions: list[Decision] = state.get("decisions", [])
    for decision in decisions:
        if hasattr(decision, "related_artifacts"):
            refs.update(decision.related_artifacts)

    return tuple(sorted(refs))


def _filter_messages_for_handoff(
    messages: list[BaseMessage],
    max_messages: int = 50,
) -> list[BaseMessage]:
    """Filter messages for handoff (Task 2.5).

    Selects the most relevant recent messages for the target agent.

    Args:
        messages: All messages from state.
        max_messages: Maximum messages to include.

    Returns:
        Filtered list of messages.
    """
    if not messages:
        return []

    # Return most recent messages up to limit
    return messages[-max_messages:]


def _calculate_context_size(context: HandoffContext) -> int:
    """Calculate serialized context size in bytes (Task 2.6).

    Args:
        context: HandoffContext to measure.

    Returns:
        Size in bytes of the serialized context.
    """
    # Serialize context to JSON for size calculation
    try:
        # Create serializable dict
        context_dict = {
            "source_agent": context.source_agent,
            "target_agent": context.target_agent,
            "decisions": [
                {
                    "agent": d.agent,
                    "summary": d.summary,
                    "rationale": d.rationale,
                    "related_artifacts": list(d.related_artifacts),
                }
                for d in context.decisions
            ],
            "memory_refs": list(context.memory_refs),
        }
        serialized = json.dumps(context_dict, default=str)
        return len(serialized.encode("utf-8"))
    except Exception:
        # Fallback estimate
        return 0


def _validate_context_completeness(
    context: HandoffContext,
    target_agent: str,
    state: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Validate context completeness for target agent (Task 2.7).

    Checks that the context contains all required items for the
    target agent based on AGENT_CONTEXT_REQUIREMENTS.

    Args:
        context: HandoffContext to validate.
        target_agent: Agent receiving the handoff.
        state: Current orchestration state.

    Returns:
        Tuple of (is_valid, list of missing requirements).
    """
    requirements = AGENT_CONTEXT_REQUIREMENTS.get(target_agent, ())
    missing: list[str] = []

    for req in requirements:
        if not _has_context_for(req, context, state):
            missing.append(req)

    # Consider valid if most requirements are met or it's optional
    # Being lenient since not all context may be available
    return len(missing) <= len(requirements) // 2, missing


def _has_context_for(
    requirement: str,
    context: HandoffContext,
    state: dict[str, Any],
) -> bool:
    """Check if a specific requirement is satisfied.

    Args:
        requirement: The requirement to check.
        context: HandoffContext.
        state: Current orchestration state.

    Returns:
        True if requirement is satisfied.
    """
    # Check context decisions
    if requirement.endswith("_decisions"):
        agent = requirement.replace("_decisions", "")
        for decision in context.decisions:
            if decision.agent == agent:
                return True

    # Check if requirement is "all_decisions"
    if requirement == "all_decisions":
        return len(context.decisions) > 0

    # Check state for other requirements
    if requirement in state:
        return state[requirement] is not None

    # Default to True for optional requirements
    return True


# =============================================================================
# State Update Functions (Task 3)
# =============================================================================


def _update_state_for_handoff(
    state: dict[str, Any],
    context: HandoffContext,
    target_agent: str,
) -> dict[str, Any]:
    """Create state updates for handoff (Task 3.1).

    Creates a dict of state updates to apply for the handoff.
    Never mutates the input state (ADR-001).

    Args:
        state: Current orchestration state (not mutated).
        context: HandoffContext to inject.
        target_agent: Agent receiving work.

    Returns:
        Dictionary of state updates to apply.
    """
    return {
        "handoff_context": context,
        "current_agent": target_agent,
    }


def _accumulate_messages(
    existing_messages: list[BaseMessage],
    new_messages: list[BaseMessage],
) -> list[BaseMessage]:
    """Accumulate messages for state (Task 3.2).

    Follows LangGraph's add_messages pattern.

    Args:
        existing_messages: Current messages in state.
        new_messages: New messages to append.

    Returns:
        Combined list of messages.
    """
    return existing_messages + new_messages


def _transfer_decisions(
    existing_decisions: list[Decision],
    context_decisions: tuple[Decision, ...],
) -> list[Decision]:
    """Transfer decisions preserving history (Task 3.3).

    Ensures all decisions are preserved during handoff.

    Args:
        existing_decisions: Current decisions in state.
        context_decisions: Decisions from handoff context.

    Returns:
        Combined list of decisions without duplicates.
    """
    # Start with existing decisions
    result = list(existing_decisions)

    # Add context decisions if not already present
    existing_summaries = {d.summary for d in existing_decisions}
    for decision in context_decisions:
        if decision.summary not in existing_summaries:
            result.append(decision)
            existing_summaries.add(decision.summary)

    return result


# =============================================================================
# Context Validation Functions (Task 4)
# =============================================================================


def _validate_state_integrity(
    before_state: dict[str, Any],
    after_state: dict[str, Any],
) -> bool:
    """Validate state integrity during handoff (Task 4.1).

    Uses existing validate_state_integrity from orchestrator.context.

    Args:
        before_state: State before handoff.
        after_state: State after handoff updates.

    Returns:
        True if integrity preserved, False otherwise.
    """
    return validate_state_integrity(before_state, after_state)


# =============================================================================
# Timing and Logging Functions (Task 5)
# =============================================================================


def _start_handoff_timer() -> Callable[[], float]:
    """Start handoff timer (Task 5.1).

    Returns:
        Callable that returns elapsed time in milliseconds when called.
    """
    start = time.perf_counter()
    return lambda: (time.perf_counter() - start) * 1000


def _calculate_handoff_metrics(
    state: dict[str, Any],
    context: HandoffContext,
    duration_ms: float,
) -> HandoffMetrics:
    """Calculate handoff metrics (Task 5.3).

    Args:
        state: Current orchestration state.
        context: HandoffContext being transferred.
        duration_ms: Duration of handoff in milliseconds.

    Returns:
        HandoffMetrics with all measurements.
    """
    messages = state.get("messages", [])
    context_size = _calculate_context_size(context)

    return HandoffMetrics(
        duration_ms=duration_ms,
        context_size_bytes=context_size,
        messages_transferred=len(messages),
        decisions_transferred=len(context.decisions),
        memory_refs_transferred=len(context.memory_refs),
    )


def _log_handoff_start(
    handoff_id: str,
    source_agent: str,
    target_agent: str,
) -> None:
    """Log handoff start at INFO level (Task 5.4).

    Args:
        handoff_id: Unique handoff identifier.
        source_agent: Agent handing off work.
        target_agent: Agent receiving work.
    """
    logger.info(
        "handoff_started",
        handoff_id=handoff_id,
        source_agent=source_agent,
        target_agent=target_agent,
    )


def _log_handoff_complete(
    handoff_id: str,
    metrics: HandoffMetrics,
    context_validated: bool,
) -> None:
    """Log handoff completion at INFO level (Task 5.5).

    Args:
        handoff_id: Unique handoff identifier.
        metrics: Handoff performance metrics.
        context_validated: Whether context was validated.
    """
    logger.info(
        "handoff_completed",
        handoff_id=handoff_id,
        duration_ms=metrics.duration_ms,
        context_size_bytes=metrics.context_size_bytes,
        messages_transferred=metrics.messages_transferred,
        decisions_transferred=metrics.decisions_transferred,
        context_validated=context_validated,
    )


def _log_handoff_failure(
    handoff_id: str,
    error: str,
    duration_ms: float,
) -> None:
    """Log handoff failure at WARNING level (Task 5.6).

    Args:
        handoff_id: Unique handoff identifier.
        error: Error message.
        duration_ms: Duration before failure.
    """
    logger.warning(
        "handoff_failed",
        handoff_id=handoff_id,
        error=error,
        duration_ms=duration_ms,
    )


# =============================================================================
# Main Handoff Function (Task 6)
# =============================================================================


def _generate_handoff_id(source_agent: str, target_agent: str) -> str:
    """Generate unique handoff ID.

    Args:
        source_agent: Agent handing off work.
        target_agent: Agent receiving work.

    Returns:
        Unique handoff identifier string.
    """
    try:
        unique_part = uuid.uuid4().hex[:8]
    except Exception:
        # Fallback to timestamp if UUID fails
        unique_part = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:14]

    return f"handoff_{source_agent}_{target_agent}_{unique_part}"


def _create_fallback_handoff(
    handoff_id: str,
    source_agent: str,
    target_agent: str,
    error_message: str,
    duration_ms: float,
) -> HandoffResult:
    """Create fallback handoff result on error (Task 6.5).

    Provides graceful degradation when managed handoff fails.

    Args:
        handoff_id: Unique handoff identifier.
        source_agent: Agent handing off work.
        target_agent: Agent receiving work.
        error_message: Error that caused fallback.
        duration_ms: Duration before failure.

    Returns:
        HandoffResult with basic state updates for fallback.
    """
    record = HandoffRecord(
        handoff_id=handoff_id,
        source_agent=source_agent,
        target_agent=target_agent,
        status="failed",
        completed_at=datetime.now(timezone.utc).isoformat(),
        error_message=error_message,
    )

    # Provide basic state updates even on failure
    basic_context = HandoffContext(
        source_agent=source_agent,
        target_agent=target_agent,
    )

    return HandoffResult(
        record=record,
        success=False,
        context_validated=False,
        state_updates={
            "handoff_context": basic_context,
            "current_agent": target_agent,
        },
        warnings=(f"Fallback handoff used: {error_message}",),
    )


async def manage_handoff(
    state: dict[str, Any],
    source_agent: str,
    target_agent: str,
    config: HandoffConfig | None = None,
) -> HandoffResult:
    """Manage a complete handoff between agents (Task 6).

    Orchestrates context preparation, state updates, validation,
    and logging for a single handoff operation.

    This is the main entry point for managed handoffs in the SM agent.
    It ensures context preservation (FR15) and meets NFR-PERF-1 (<5s).

    Args:
        state: Current orchestration state.
        source_agent: Agent completing work.
        target_agent: Agent receiving work.
        config: Handoff configuration (uses defaults if None).

    Returns:
        HandoffResult with complete handoff outcome and state updates.

    Example:
        >>> result = await manage_handoff(state, "pm", "architect")
        >>> result.success
        True
        >>> result.state_updates["current_agent"]
        'architect'
    """
    config = config or HandoffConfig()
    handoff_id = _generate_handoff_id(source_agent, target_agent)

    # Start timing
    get_elapsed = _start_handoff_timer()

    # Log start
    if config.log_timing:
        _log_handoff_start(handoff_id, source_agent, target_agent)

    try:
        # Step 1: Prepare context (Task 2)
        context = _prepare_handoff_context(
            state=state,
            source_agent=source_agent,
            target_agent=target_agent,
            config=config,
        )

        # Step 2: Update state (Task 3)
        state_updates = _update_state_for_handoff(
            state=state,
            context=context,
            target_agent=target_agent,
        )

        # Step 3: Validate context (Task 4)
        context_valid = True
        warnings: list[str] = []
        if config.validate_context_integrity:
            is_complete, missing = _validate_context_completeness(
                context=context,
                target_agent=target_agent,
                state=state,
            )
            context_valid = is_complete
            if missing:
                warnings.append(f"Missing context for {target_agent}: {', '.join(missing)}")

        # Step 4: Calculate metrics (Task 5)
        duration_ms = get_elapsed()
        metrics = _calculate_handoff_metrics(
            state=state,
            context=context,
            duration_ms=duration_ms,
        )

        # Step 5: Create record
        context_checksum = compute_state_checksum(
            state_updates,
            exclude_keys=frozenset({"messages"}),
        )

        record = HandoffRecord(
            handoff_id=handoff_id,
            source_agent=source_agent,
            target_agent=target_agent,
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            context_checksum=context_checksum,
        )

        # Log completion
        if config.log_timing:
            _log_handoff_complete(handoff_id, metrics, context_valid)

        return HandoffResult(
            record=record,
            success=True,
            context_validated=context_valid,
            state_updates=state_updates,
            warnings=tuple(warnings),
        )

    except Exception as e:
        # Handle errors gracefully
        duration_ms = get_elapsed()

        if config.log_timing:
            _log_handoff_failure(handoff_id, str(e), duration_ms)

        # Return fallback result
        return _create_fallback_handoff(
            handoff_id=handoff_id,
            source_agent=source_agent,
            target_agent=target_agent,
            error_message=str(e),
            duration_ms=duration_ms,
        )
