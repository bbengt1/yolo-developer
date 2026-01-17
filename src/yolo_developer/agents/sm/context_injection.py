"""Context injection functions for SM agent (Story 10.13).

This module provides functions for detecting context gaps and injecting
relevant context when agents lack information:

- detect_context_gap: Detect if an agent lacks sufficient context
- retrieve_relevant_context: Retrieve context from enabled sources
- inject_context: Inject retrieved context into state
- manage_context_injection: End-to-end context injection management

Example:
    >>> from yolo_developer.agents.sm.context_injection import (
    ...     detect_context_gap,
    ...     manage_context_injection,
    ... )
    >>> from yolo_developer.orchestrator.state import YoloState
    >>>
    >>> state: YoloState = {...}
    >>> gap = await detect_context_gap(state)
    >>> if gap:
    ...     result = await manage_context_injection(state)

References:
    - FR69: SM Agent can inject context when agents lack information
    - Story 10.5: Health Monitoring (cycle time metrics for gap detection)
    - Story 10.6: Circular Logic Detection (cycle analysis input)
    - Story 10.8: Handoff Management (context validation complements injection)
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import structlog

from yolo_developer.agents.sm.context_injection_types import (
    LONG_CYCLE_TIME_MULTIPLIER,
    ContextGap,
    GapReason,
    InjectionConfig,
    InjectionResult,
    RetrievedContext,
)

if TYPE_CHECKING:
    from yolo_developer.memory.protocol import MemoryStore
    from yolo_developer.orchestrator.state import YoloState

logger = structlog.get_logger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Keywords that indicate clarification is being requested
CLARIFICATION_KEYWORDS: frozenset[str] = frozenset(
    {
        "unclear",
        "need more info",
        "need clarification",
        "what do you mean",
        "can you clarify",
        "please explain",
        "i don't understand",
        "missing information",
        "need context",
        "not sure what",
    }
)

# Default injection target in state
DEFAULT_INJECTION_TARGET: str = "injected_context"


# =============================================================================
# Helper Functions
# =============================================================================


def _generate_gap_id() -> str:
    """Generate a unique gap identifier.

    Returns:
        Unique gap ID string.
    """
    return f"gap-{uuid.uuid4().hex[:12]}"


def _check_clarification_requested(state: YoloState) -> bool:
    """Check if recent messages indicate clarification is requested.

    Scans recent messages for clarification indicators like
    question marks, "unclear", "need more info" patterns.

    Args:
        state: Current orchestration state.

    Returns:
        True if clarification indicators found, False otherwise.
    """
    messages = state.get("messages", [])
    if not messages:
        return False

    # Check last 5 messages for clarification indicators
    recent_messages = messages[-5:] if len(messages) > 5 else messages

    for msg in recent_messages:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            content_lower = content.lower()
            # Check for question marks (multiple suggest confusion)
            if content_lower.count("?") >= 2:
                return True
            # Check for clarification keywords
            for keyword in CLARIFICATION_KEYWORDS:
                if keyword in content_lower:
                    return True

    return False


def _check_circular_logic_gap(state: YoloState) -> bool:
    """Check if circular logic indicates an information gap.

    Examines cycle_analysis for patterns that suggest agents
    are cycling due to missing information.

    Args:
        state: Current orchestration state.

    Returns:
        True if circular pattern detected, False otherwise.
    """
    cycle_analysis = state.get("cycle_analysis")
    if cycle_analysis is None:
        return False

    # Check if circular detected
    if isinstance(cycle_analysis, dict):
        return bool(cycle_analysis.get("circular_detected", False))

    # Handle if cycle_analysis has to_dict method (from CycleAnalysis)
    if hasattr(cycle_analysis, "circular_detected"):
        return bool(cycle_analysis.circular_detected)

    return False


def _check_long_cycle_time(state: YoloState) -> bool:
    """Check if current agent cycle time exceeds baseline.

    Compares current agent's cycle time to historical baseline
    from health monitoring. A cycle time >2x baseline suggests
    the agent may be struggling due to missing context.

    Args:
        state: Current orchestration state.

    Returns:
        True if cycle time exceeds threshold, False otherwise.
    """
    health_status = state.get("health_status")
    if health_status is None:
        return False

    current_agent = state.get("current_agent", "")
    if not current_agent:
        return False

    # Extract cycle times from health status
    if isinstance(health_status, dict):
        metrics = health_status.get("metrics", {})
        agent_cycle_times = metrics.get("agent_cycle_times", {})
        overall_cycle_time = metrics.get("overall_cycle_time", 0.0)

        # Get current agent's cycle time
        current_cycle_time = agent_cycle_times.get(current_agent, 0.0)

        # Compare to baseline
        if overall_cycle_time > 0 and current_cycle_time > 0:
            if current_cycle_time > overall_cycle_time * LONG_CYCLE_TIME_MULTIPLIER:
                return True

    return False


def _check_gate_failure(state: YoloState) -> bool:
    """Check if a gate failure indicates missing context.

    Examines gate_blocked and related state for indications
    that missing context caused the gate failure.

    Args:
        state: Current orchestration state.

    Returns:
        True if gate failure indicates context gap, False otherwise.
    """
    gate_blocked = state.get("gate_blocked", False)
    if not gate_blocked:
        return False

    # Check if gate failure reason mentions context
    gate_failure = state.get("gate_failure", "")
    if isinstance(gate_failure, str):
        failure_lower = gate_failure.lower()
        if any(
            keyword in failure_lower
            for keyword in ("missing", "context", "information", "unclear")
        ):
            return True

    return False


def _check_explicit_flag(state: YoloState) -> bool:
    """Check if agent explicitly flagged missing context.

    Looks for explicit missing_context flag in state or
    recent agent outputs.

    Args:
        state: Current orchestration state.

    Returns:
        True if explicit flag found, False otherwise.
    """
    # Check direct flag in state
    if state.get("missing_context", False):
        return True

    # Check handoff context for missing_context indicator
    handoff_context = state.get("handoff_context")
    if handoff_context is not None:
        if isinstance(handoff_context, dict):
            required_context = handoff_context.get("required_context", [])
            if required_context and len(required_context) > 0:
                # If there are required contexts listed, might indicate gaps
                return True

    return False


def _build_context_query(state: YoloState, gap_reason: GapReason) -> str:
    """Build a context query based on gap reason and state.

    Constructs a semantic query string that can be used to
    search memory for relevant context.

    Args:
        state: Current orchestration state.
        gap_reason: Reason the gap was detected.

    Returns:
        Query string for context retrieval.
    """
    current_agent = state.get("current_agent", "unknown")
    parts: list[str] = []

    # Include agent-specific context
    parts.append(f"Context for {current_agent} agent")

    # Include gap reason
    reason_queries = {
        "clarification_requested": "clarification on requirements or specifications",
        "circular_logic": "resolve conflicting information or decisions",
        "long_cycle_time": "technical guidance or implementation details",
        "gate_failure": "quality requirements or acceptance criteria",
        "explicit_flag": "missing information flagged by agent",
    }
    parts.append(reason_queries.get(gap_reason, "general context"))

    # Include recent message content for context
    messages = state.get("messages", [])
    if messages:
        recent = messages[-1] if len(messages) > 0 else None
        if recent and hasattr(recent, "content"):
            content = recent.content
            if isinstance(content, str) and len(content) > 20:
                # Truncate for query
                parts.append(content[:100])

    return " ".join(parts)


def _calculate_relevance_score(query: str, content: str) -> float:
    """Calculate simple relevance score between query and content.

    Uses keyword matching as a baseline. Could be enhanced with
    semantic similarity if embeddings are available.

    Args:
        query: The context query string.
        content: The content to score against.

    Returns:
        Relevance score between 0.0 and 1.0.
    """
    if not query or not content:
        return 0.0

    query_lower = query.lower()
    content_lower = content.lower()

    # Split into words
    query_words = set(query_lower.split())
    content_words = set(content_lower.split())

    # Remove common stop words
    stop_words = {"the", "a", "an", "is", "are", "for", "to", "of", "and", "or", "in"}
    query_words = query_words - stop_words
    content_words = content_words - stop_words

    if not query_words:
        return 0.0

    # Calculate Jaccard-like similarity
    intersection = query_words & content_words
    union = query_words | content_words

    if not union:
        return 0.0

    score = len(intersection) / len(union)

    # Boost if exact phrases match
    for word in query_words:
        if len(word) > 4 and word in content_lower:
            score = min(1.0, score + 0.1)

    return min(1.0, score)


# =============================================================================
# Context Gap Detection (Task 2)
# =============================================================================


async def detect_context_gap(state: YoloState) -> ContextGap | None:
    """Detect if current agent lacks sufficient context.

    Checks multiple indicators to determine if a context gap exists:
    1. Clarification requested in messages
    2. Circular logic patterns
    3. Long cycle times vs baseline
    4. Gate failures with missing context
    5. Explicit missing_context flags

    Args:
        state: Current orchestration state.

    Returns:
        ContextGap if gap detected, None otherwise.

    Example:
        >>> state = {"messages": [...], "current_agent": "architect", ...}
        >>> gap = await detect_context_gap(state)
        >>> if gap:
        ...     print(f"Gap detected: {gap.reason}")

    References:
        - FR69: SM Agent can inject context when agents lack information
    """
    logger.debug(
        "detecting_context_gap",
        current_agent=state.get("current_agent"),
        message_count=len(state.get("messages", [])),
    )

    current_agent = state.get("current_agent", "unknown")
    indicators: list[str] = []

    # Priority 1: Check for clarification requested (highest confidence)
    if _check_clarification_requested(state):
        logger.info(
            "context_gap_detected",
            reason="clarification_requested",
            agent=current_agent,
        )
        indicators.append("clarification_message_detected")
        return ContextGap(
            gap_id=_generate_gap_id(),
            agent=current_agent,
            reason="clarification_requested",
            context_query=_build_context_query(state, "clarification_requested"),
            confidence=0.9,
            indicators=tuple(indicators),
        )

    # Priority 2: Check for circular logic
    if _check_circular_logic_gap(state):
        logger.info(
            "context_gap_detected",
            reason="circular_logic",
            agent=current_agent,
        )
        indicators.append("circular_logic_detected")
        return ContextGap(
            gap_id=_generate_gap_id(),
            agent=current_agent,
            reason="circular_logic",
            context_query=_build_context_query(state, "circular_logic"),
            confidence=0.8,
            indicators=tuple(indicators),
        )

    # Priority 3: Check for gate failure
    if _check_gate_failure(state):
        logger.info(
            "context_gap_detected",
            reason="gate_failure",
            agent=current_agent,
        )
        indicators.append("gate_failure_missing_context")
        return ContextGap(
            gap_id=_generate_gap_id(),
            agent=current_agent,
            reason="gate_failure",
            context_query=_build_context_query(state, "gate_failure"),
            confidence=0.85,
            indicators=tuple(indicators),
        )

    # Priority 4: Check for explicit flag
    if _check_explicit_flag(state):
        logger.info(
            "context_gap_detected",
            reason="explicit_flag",
            agent=current_agent,
        )
        indicators.append("explicit_missing_context_flag")
        return ContextGap(
            gap_id=_generate_gap_id(),
            agent=current_agent,
            reason="explicit_flag",
            context_query=_build_context_query(state, "explicit_flag"),
            confidence=0.95,
            indicators=tuple(indicators),
        )

    # Priority 5: Check for long cycle time (lower confidence)
    if _check_long_cycle_time(state):
        logger.info(
            "context_gap_detected",
            reason="long_cycle_time",
            agent=current_agent,
        )
        indicators.append("cycle_time_exceeds_baseline")
        return ContextGap(
            gap_id=_generate_gap_id(),
            agent=current_agent,
            reason="long_cycle_time",
            context_query=_build_context_query(state, "long_cycle_time"),
            confidence=0.6,
            indicators=tuple(indicators),
        )

    logger.debug("no_context_gap_detected", agent=current_agent)
    return None


# =============================================================================
# Context Retrieval (Task 3)
# =============================================================================


async def _retrieve_from_memory(
    query: str,
    memory: MemoryStore,
    config: InjectionConfig,
) -> list[RetrievedContext]:
    """Retrieve relevant context from memory store.

    Uses semantic search to find relevant embeddings and
    decision history from the memory store.

    Args:
        query: Search query for context.
        memory: Memory store instance.
        config: Injection configuration.

    Returns:
        List of RetrievedContext from memory.
    """
    contexts: list[RetrievedContext] = []

    try:
        # Search for similar content
        results = await memory.search_similar(query, k=config.max_context_items)

        for result in results:
            # Convert MemoryResult to RetrievedContext
            # Estimate relevance from score if available
            relevance = result.score if hasattr(result, "score") else 0.5

            if relevance >= config.min_relevance_score:
                contexts.append(
                    RetrievedContext(
                        source="memory",
                        content=result.content,
                        relevance_score=relevance,
                        metadata=result.metadata if hasattr(result, "metadata") else {},
                    )
                )

        logger.debug(
            "memory_context_retrieved",
            query_length=len(query),
            results_count=len(results),
            contexts_added=len(contexts),
        )

    except Exception as e:
        logger.warning(
            "memory_retrieval_failed",
            error=str(e),
            query_length=len(query),
        )

    return contexts


def _retrieve_from_state(
    query: str,
    state: YoloState,
    config: InjectionConfig,
) -> list[RetrievedContext]:
    """Retrieve relevant context from orchestration state.

    Searches through state messages and decisions for
    content relevant to the query.

    Args:
        query: Search query for context.
        state: Current orchestration state.
        config: Injection configuration.

    Returns:
        List of RetrievedContext from state.
    """
    contexts: list[RetrievedContext] = []

    # Search messages
    messages = state.get("messages", [])
    for msg in messages[-20:]:  # Limit to recent messages
        content = getattr(msg, "content", "")
        if isinstance(content, str) and len(content) > 20:
            relevance = _calculate_relevance_score(query, content)
            if relevance >= config.min_relevance_score:
                agent = ""
                if hasattr(msg, "additional_kwargs"):
                    agent = msg.additional_kwargs.get("agent", "unknown")
                contexts.append(
                    RetrievedContext(
                        source="state",
                        content=content[:500],  # Truncate long content
                        relevance_score=relevance,
                        metadata={"type": "message", "agent": agent},
                    )
                )

    # Search decisions
    decisions = state.get("decisions", [])
    for decision in decisions[-10:]:  # Limit to recent decisions
        if hasattr(decision, "summary") and hasattr(decision, "rationale"):
            combined = f"{decision.summary} {decision.rationale}"
            relevance = _calculate_relevance_score(query, combined)
            if relevance >= config.min_relevance_score:
                contexts.append(
                    RetrievedContext(
                        source="state",
                        content=combined[:500],
                        relevance_score=relevance,
                        metadata={
                            "type": "decision",
                            "agent": getattr(decision, "agent", "unknown"),
                        },
                    )
                )

    # Include handoff context if present and relevant
    handoff_context = state.get("handoff_context")
    if handoff_context is not None and isinstance(handoff_context, dict):
        context_str = str(handoff_context)
        relevance = _calculate_relevance_score(query, context_str)
        if relevance >= config.min_relevance_score:
            contexts.append(
                RetrievedContext(
                    source="state",
                    content=context_str[:500],
                    relevance_score=relevance,
                    metadata={"type": "handoff_context"},
                )
            )

    logger.debug(
        "state_context_retrieved",
        query_length=len(query),
        messages_searched=min(len(messages), 20),
        decisions_searched=min(len(decisions), 10),
        contexts_found=len(contexts),
    )

    return contexts


def _retrieve_from_sprint(
    query: str,
    state: YoloState,
    config: InjectionConfig,
) -> list[RetrievedContext]:
    """Retrieve relevant context from sprint data.

    Extracts context from sprint_plan and sprint_progress
    in the state.

    Args:
        query: Search query for context.
        state: Current orchestration state.
        config: Injection configuration.

    Returns:
        List of RetrievedContext from sprint data.
    """
    contexts: list[RetrievedContext] = []

    # Check sprint plan
    sprint_plan = state.get("sprint_plan")
    if sprint_plan is not None and isinstance(sprint_plan, dict):
        plan_str = str(sprint_plan)
        relevance = _calculate_relevance_score(query, plan_str)
        if relevance >= config.min_relevance_score:
            contexts.append(
                RetrievedContext(
                    source="sprint",
                    content=plan_str[:500],
                    relevance_score=relevance,
                    metadata={"type": "sprint_plan"},
                )
            )

    # Check sprint progress
    sprint_progress = state.get("sprint_progress")
    if sprint_progress is not None and isinstance(sprint_progress, dict):
        progress_str = str(sprint_progress)
        relevance = _calculate_relevance_score(query, progress_str)
        if relevance >= config.min_relevance_score:
            contexts.append(
                RetrievedContext(
                    source="sprint",
                    content=progress_str[:500],
                    relevance_score=relevance,
                    metadata={"type": "sprint_progress"},
                )
            )

    logger.debug(
        "sprint_context_retrieved",
        has_plan=sprint_plan is not None,
        has_progress=sprint_progress is not None,
        contexts_found=len(contexts),
    )

    return contexts


async def retrieve_relevant_context(
    gap: ContextGap,
    memory: MemoryStore | None,
    state: YoloState,
    config: InjectionConfig | None = None,
) -> tuple[RetrievedContext, ...]:
    """Retrieve relevant context from enabled sources.

    Searches memory store, state history, and sprint context
    for information relevant to the detected gap.

    Args:
        gap: The detected context gap.
        memory: Memory store instance (optional).
        state: Current orchestration state.
        config: Injection configuration (uses defaults if None).

    Returns:
        Tuple of RetrievedContext sorted by relevance.

    Example:
        >>> gap = ContextGap(...)
        >>> contexts = await retrieve_relevant_context(gap, memory, state)
        >>> for ctx in contexts:
        ...     print(f"{ctx.source}: {ctx.relevance_score}")

    References:
        - FR69: SM Agent can inject context when agents lack information
    """
    if config is None:
        config = InjectionConfig()

    logger.debug(
        "retrieving_relevant_context",
        gap_id=gap.gap_id,
        query_length=len(gap.context_query),
        enabled_sources=config.enabled_sources,
    )

    contexts: list[RetrievedContext] = []

    # Retrieve from memory store
    if "memory" in config.enabled_sources and memory is not None:
        memory_contexts = await _retrieve_from_memory(
            gap.context_query, memory, config
        )
        contexts.extend(memory_contexts)

    # Retrieve from state
    if "state" in config.enabled_sources:
        state_contexts = _retrieve_from_state(gap.context_query, state, config)
        contexts.extend(state_contexts)

    # Retrieve from sprint
    if "sprint" in config.enabled_sources:
        sprint_contexts = _retrieve_from_sprint(gap.context_query, state, config)
        contexts.extend(sprint_contexts)

    # Sort by relevance and limit
    contexts.sort(key=lambda c: c.relevance_score, reverse=True)
    limited = contexts[: config.max_context_items]

    logger.info(
        "context_retrieval_complete",
        gap_id=gap.gap_id,
        total_retrieved=len(contexts),
        after_limit=len(limited),
        sources_searched=list(config.enabled_sources),
    )

    return tuple(limited)


# =============================================================================
# Context Injection (Task 4)
# =============================================================================


def _build_injection_payload(
    contexts: Sequence[RetrievedContext],
    config: InjectionConfig,
) -> dict[str, Any]:
    """Build injection payload from retrieved contexts.

    Formats contexts for agent consumption, including source
    attribution and respecting max size limits.

    Args:
        contexts: Retrieved contexts to include.
        config: Injection configuration.

    Returns:
        Formatted payload dictionary.
    """
    payload: dict[str, Any] = {
        "contexts": [],
        "source_summary": {},
        "total_items": len(contexts),
    }

    total_size = 0
    source_counts: dict[str, int] = {}

    for ctx in contexts:
        content_size = len(ctx.content.encode("utf-8"))

        # Respect max size
        if total_size + content_size > config.max_context_size_bytes:
            logger.debug(
                "injection_payload_truncated",
                current_size=total_size,
                max_size=config.max_context_size_bytes,
            )
            break

        payload["contexts"].append(
            {
                "source": ctx.source,
                "content": ctx.content,
                "relevance": ctx.relevance_score,
                "metadata": ctx.metadata,
            }
        )

        total_size += content_size
        source_counts[ctx.source] = source_counts.get(ctx.source, 0) + 1

    payload["source_summary"] = source_counts
    payload["total_size_bytes"] = total_size

    return payload


def _determine_injection_target(gap: ContextGap, state: YoloState) -> str:
    """Determine which state key to inject context into.

    Args:
        gap: The detected context gap.
        state: Current orchestration state.

    Returns:
        State key for injection.
    """
    # Could be enhanced to choose different targets based on gap reason
    # For now, use default injection target
    return DEFAULT_INJECTION_TARGET


async def inject_context(
    gap: ContextGap,
    contexts: Sequence[RetrievedContext],
    config: InjectionConfig | None = None,
) -> InjectionResult:
    """Inject retrieved context for agent consumption.

    Builds injection payload from retrieved contexts and
    prepares it for state injection.

    Args:
        gap: The detected context gap.
        contexts: Retrieved contexts to inject.
        config: Injection configuration (uses defaults if None).

    Returns:
        InjectionResult with injection details.

    Example:
        >>> result = await inject_context(gap, contexts)
        >>> if result.injected:
        ...     print(f"Injected {result.total_context_size} bytes")

    References:
        - FR69: SM Agent can inject context when agents lack information
    """
    if config is None:
        config = InjectionConfig()

    start_time = time.monotonic()

    # Build payload
    payload = _build_injection_payload(contexts, config)
    total_size = payload.get("total_size_bytes", 0)

    # Determine injection target
    injection_target = DEFAULT_INJECTION_TARGET

    # Determine if injection should happen
    injected = len(payload.get("contexts", [])) > 0

    duration_ms = (time.monotonic() - start_time) * 1000

    if config.log_injections:
        logger.info(
            "context_injection_complete",
            gap_id=gap.gap_id,
            injected=injected,
            injection_target=injection_target,
            total_context_size=total_size,
            contexts_count=len(contexts),
            duration_ms=duration_ms,
        )

    return InjectionResult(
        gap=gap,
        contexts_retrieved=tuple(contexts),
        injected=injected,
        injection_target=injection_target,
        total_context_size=total_size,
        duration_ms=duration_ms,
    )


# =============================================================================
# Main Orchestration Function (Task 5)
# =============================================================================


async def manage_context_injection(
    state: YoloState,
    memory: MemoryStore | None = None,
    config: InjectionConfig | None = None,
) -> tuple[InjectionResult | None, dict[str, Any] | None]:
    """Manage end-to-end context injection process.

    Detects context gaps, retrieves relevant context, and
    injects it into state for the struggling agent.

    Args:
        state: Current orchestration state.
        memory: Memory store instance (optional).
        config: Injection configuration (uses defaults if None).

    Returns:
        Tuple of (InjectionResult or None if no gap, payload or None).

    Example:
        >>> result, payload = await manage_context_injection(state, memory)
        >>> if result:
        ...     print(f"Gap: {result.gap.reason}, Injected: {result.injected}")
        ...     if payload:
        ...         # Update state with payload["injected_context"]
        ...         pass

    References:
        - FR69: SM Agent can inject context when agents lack information
    """
    if config is None:
        config = InjectionConfig()

    start_time = time.monotonic()

    logger.debug(
        "managing_context_injection",
        current_agent=state.get("current_agent"),
        memory_available=memory is not None,
    )

    # Step 1: Detect context gap
    gap = await detect_context_gap(state)

    if gap is None:
        logger.debug("no_injection_needed", reason="no_gap_detected")
        return None, None

    # Step 2: Retrieve relevant context
    contexts = await retrieve_relevant_context(gap, memory, state, config)

    if not contexts:
        logger.info(
            "no_context_retrieved",
            gap_id=gap.gap_id,
            reason=gap.reason,
        )
        # Still return result to indicate we tried
        result = InjectionResult(
            gap=gap,
            contexts_retrieved=(),
            injected=False,
            injection_target="",
            total_context_size=0,
            duration_ms=(time.monotonic() - start_time) * 1000,
        )
        return result, None

    # Step 3: Inject context
    result = await inject_context(gap, contexts, config)

    # Build state update payload
    payload: dict[str, Any] | None = None
    if result.injected:
        injection_payload = _build_injection_payload(contexts, config)
        payload = {
            result.injection_target: injection_payload,
        }

    total_duration_ms = (time.monotonic() - start_time) * 1000

    logger.info(
        "context_injection_managed",
        gap_id=gap.gap_id,
        gap_reason=gap.reason,
        gap_confidence=gap.confidence,
        contexts_retrieved=len(contexts),
        injected=result.injected,
        total_duration_ms=total_duration_ms,
    )

    return result, payload
