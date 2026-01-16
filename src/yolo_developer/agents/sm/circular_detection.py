"""Circular logic detection for SM agent (Story 10.6).

This module provides enhanced circular logic detection that tracks:
1. Topic-aware exchanges (same issue cycling)
2. Agent pair cycles (A->B->A)
3. Multi-agent cycles (A->B->C->A)

Key Functions:
    detect_circular_logic: Main entry point for circular logic detection
    _extract_exchange_topic: Extract semantic topic from exchange
    _detect_agent_pair_cycles: Detect A->B->A patterns
    _detect_multi_agent_cycles: Detect A->B->C->A patterns
    _detect_topic_cycles: Detect same-topic cycling

Example:
    >>> from yolo_developer.agents.sm.circular_detection import detect_circular_logic
    >>> from yolo_developer.orchestrator.state import YoloState
    >>>
    >>> state: YoloState = {...}
    >>> result = await detect_circular_logic(state)
    >>> result.circular_detected
    False

References:
    - FR12: SM Agent can detect circular logic between agents (>3 exchanges)
    - FR70: SM Agent can escalate to human when circular logic persists
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from yolo_developer.agents.sm.circular_detection_types import (
    CircularLogicConfig,
    CircularPattern,
    CycleAnalysis,
    CycleLog,
    CycleSeverity,
    InterventionStrategy,
)
from yolo_developer.agents.sm.types import AgentExchange

if TYPE_CHECKING:
    from yolo_developer.orchestrator.state import YoloState

logger = structlog.get_logger(__name__)

# =============================================================================
# Topic Extraction Functions (Task 2)
# =============================================================================


def _extract_exchange_topic(exchange: AgentExchange) -> str:
    """Extract semantic topic from an agent exchange.

    Args:
        exchange: The agent exchange to extract topic from.

    Returns:
        The topic string, or "workflow_transition" if no topic.
    """
    if exchange.topic:
        return exchange.topic
    return "workflow_transition"


def _group_exchanges_by_topic(
    exchanges: list[AgentExchange],
) -> dict[str, list[AgentExchange]]:
    """Group exchanges by their semantic topic.

    Args:
        exchanges: List of agent exchanges to group.

    Returns:
        Dictionary mapping topic to list of exchanges.
    """
    grouped: dict[str, list[AgentExchange]] = {}

    for exchange in exchanges:
        topic = _extract_exchange_topic(exchange)
        if topic not in grouped:
            grouped[topic] = []
        grouped[topic].append(exchange)

    return grouped


def _track_topic_exchanges(
    exchanges: list[AgentExchange],
) -> dict[str, list[str]]:
    """Track exchanges grouped by topic, returning exchange IDs.

    Args:
        exchanges: List of agent exchanges to track.

    Returns:
        Dictionary mapping topic to list of exchange IDs.
    """
    result: dict[str, list[str]] = {}

    for i, exchange in enumerate(exchanges):
        topic = _extract_exchange_topic(exchange)
        if topic not in result:
            result[topic] = []
        # Use index-based ID since AgentExchange doesn't have an ID field
        exchange_id = f"ex-{i}-{exchange.source_agent}-{exchange.target_agent}"
        result[topic].append(exchange_id)

    return result


def _extract_exchanges_from_state(state: YoloState) -> list[AgentExchange]:
    """Extract agent exchanges from orchestration state messages.

    Analyzes messages to build a list of agent-to-agent exchanges.

    Args:
        state: Current orchestration state.

    Returns:
        List of AgentExchange objects extracted from messages.
    """
    messages = state.get("messages", [])
    exchanges: list[AgentExchange] = []

    prev_agent: str | None = None
    prev_topic: str = "workflow_transition"

    for msg in messages:
        # Extract agent from message metadata
        if hasattr(msg, "additional_kwargs"):
            kwargs = msg.additional_kwargs
            agent = kwargs.get("agent")
            topic = kwargs.get("topic", "workflow_transition")

            if agent and prev_agent and agent != prev_agent:
                exchange = AgentExchange(
                    source_agent=prev_agent,
                    target_agent=agent,
                    exchange_type="handoff",
                    topic=topic or prev_topic,
                )
                exchanges.append(exchange)

            if agent:
                prev_agent = agent
            if topic:
                prev_topic = topic

    return exchanges


# =============================================================================
# Cycle Detection Functions (Task 3)
# =============================================================================


def _calculate_duration(exchanges: list[AgentExchange]) -> float:
    """Calculate duration in seconds between first and last exchange.

    Args:
        exchanges: List of exchanges to calculate duration for.

    Returns:
        Duration in seconds, or 0 if cannot be calculated.
    """
    if len(exchanges) < 2:
        return 0.0

    try:
        first_ts = exchanges[0].timestamp
        last_ts = exchanges[-1].timestamp

        # Parse ISO timestamps
        first_dt = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
        last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))

        return (last_dt - first_dt).total_seconds()
    except (ValueError, AttributeError):
        return 0.0


def _extract_common_topic(exchanges: list[AgentExchange]) -> str:
    """Extract the most common topic from a list of exchanges.

    Args:
        exchanges: List of exchanges to analyze.

    Returns:
        Most common topic, or "mixed" if no clear topic.
    """
    if not exchanges:
        return "unknown"

    topic_counts: dict[str, int] = {}
    for exchange in exchanges:
        topic = _extract_exchange_topic(exchange)
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    # Return most common topic
    most_common = max(topic_counts.items(), key=lambda x: x[1])
    return most_common[0]


def _calculate_severity(
    exchange_count: int,
    config: CircularLogicConfig,
) -> CycleSeverity:
    """Calculate severity based on exchange count and config thresholds.

    Args:
        exchange_count: Number of exchanges in the cycle.
        config: Configuration with severity thresholds.

    Returns:
        Severity level for the cycle.
    """
    thresholds = config.severity_thresholds

    if exchange_count >= thresholds.get("critical", 12):
        return "critical"
    if exchange_count >= thresholds.get("high", 8):
        return "high"
    if exchange_count >= thresholds.get("medium", 5):
        return "medium"
    return "low"


def _filter_by_time_window(
    exchanges: list[AgentExchange],
    config: CircularLogicConfig,
) -> list[AgentExchange]:
    """Filter exchanges to only include those within the configured time window.

    Args:
        exchanges: List of agent exchanges to filter.
        config: Configuration with time_window_seconds.

    Returns:
        Filtered list of exchanges within the time window.
    """
    if not exchanges:
        return []

    now = datetime.now(timezone.utc)

    filtered: list[AgentExchange] = []
    for exchange in exchanges:
        try:
            ts = exchange.timestamp
            exchange_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            age_seconds = (now - exchange_dt).total_seconds()

            if age_seconds <= config.time_window_seconds:
                filtered.append(exchange)
        except (ValueError, AttributeError):
            # If timestamp parsing fails, include the exchange to be safe
            filtered.append(exchange)

    return filtered


def _detect_agent_pair_cycles(
    exchanges: list[AgentExchange],
    config: CircularLogicConfig,
) -> list[CircularPattern]:
    """Detect A->B->A agent pair cycles.

    Counts exchanges between agent pairs and detects when
    the count exceeds the threshold.

    Args:
        exchanges: List of agent exchanges to analyze.
        config: Detection configuration.

    Returns:
        List of detected CircularPattern objects.
    """
    if len(exchanges) <= config.exchange_threshold:
        return []

    # Group by normalized agent pair
    pair_exchanges: dict[tuple[str, str], list[AgentExchange]] = {}

    for exchange in exchanges:
        # Normalize pair order for consistent counting
        sorted_agents = sorted([exchange.source_agent, exchange.target_agent])
        pair: tuple[str, str] = (sorted_agents[0], sorted_agents[1])
        if pair not in pair_exchanges:
            pair_exchanges[pair] = []
        pair_exchanges[pair].append(exchange)

    patterns: list[CircularPattern] = []

    for pair, pair_ex in pair_exchanges.items():
        if len(pair_ex) > config.exchange_threshold:
            severity = _calculate_severity(len(pair_ex), config)
            pattern = CircularPattern(
                pattern_type="agent_pair",
                agents_involved=pair,
                topic=_extract_common_topic(pair_ex),
                exchange_count=len(pair_ex),
                first_exchange_at=pair_ex[0].timestamp,
                last_exchange_at=pair_ex[-1].timestamp,
                duration_seconds=_calculate_duration(pair_ex),
                severity=severity,
            )
            patterns.append(pattern)

    return patterns


def _detect_multi_agent_cycles(
    exchanges: list[AgentExchange],
    config: CircularLogicConfig,
) -> list[CircularPattern]:
    """Detect A->B->C->A multi-agent cycles.

    Uses sliding window to detect sequences where agents
    return to a previous agent through intermediate agents.

    Args:
        exchanges: List of agent exchanges to analyze.
        config: Detection configuration.

    Returns:
        List of detected CircularPattern objects.
    """
    if len(exchanges) < 3:
        return []

    patterns: list[CircularPattern] = []

    # Build agent sequence from exchanges
    agent_sequence: list[str] = [exchanges[0].source_agent]
    for exchange in exchanges:
        agent_sequence.append(exchange.target_agent)

    # Track unique agent cycles we've seen
    seen_agent_sets: set[frozenset[str]] = set()

    # Look for repeating patterns involving 3+ agents
    # Count how many times each unique agent set appears in sequence
    for cycle_len in range(3, min(7, len(agent_sequence))):  # Cycles of 3-6 agents
        for start in range(len(agent_sequence) - cycle_len):
            window = agent_sequence[start : start + cycle_len + 1]
            unique_agents = set(window)

            # Need at least 3 unique agents and the sequence should return
            if len(unique_agents) >= 3 and window[0] == window[-1]:
                agent_set = frozenset(unique_agents)

                # Count full cycles with these agents in the entire sequence
                cycle_count = 0
                i = 0
                while i < len(agent_sequence) - len(unique_agents):
                    # Look for the same cycle pattern
                    sub_window = agent_sequence[i : i + cycle_len + 1]
                    if (
                        len(set(sub_window)) >= 3
                        and sub_window[0] == sub_window[-1]
                        and frozenset(sub_window) == agent_set
                    ):
                        cycle_count += 1
                        i += cycle_len
                    else:
                        i += 1

                # Only report if exceeds threshold and we haven't seen this set
                if cycle_count > 0 and agent_set not in seen_agent_sets:
                    # Count all exchanges involving these agents
                    relevant_exchanges = [
                        e
                        for e in exchanges
                        if e.source_agent in unique_agents
                        and e.target_agent in unique_agents
                    ]

                    if len(relevant_exchanges) > config.exchange_threshold:
                        seen_agent_sets.add(agent_set)
                        severity = _calculate_severity(
                            len(relevant_exchanges), config
                        )
                        pattern = CircularPattern(
                            pattern_type="multi_agent",
                            agents_involved=tuple(sorted(unique_agents)),
                            topic=_extract_common_topic(relevant_exchanges),
                            exchange_count=len(relevant_exchanges),
                            first_exchange_at=relevant_exchanges[0].timestamp,
                            last_exchange_at=relevant_exchanges[-1].timestamp,
                            duration_seconds=_calculate_duration(relevant_exchanges),
                            severity=severity,
                        )
                        patterns.append(pattern)

    return patterns


def _detect_topic_cycles(
    grouped_exchanges: dict[str, list[AgentExchange]],
    config: CircularLogicConfig,
) -> list[CircularPattern]:
    """Detect repeated exchanges on the same topic.

    Args:
        grouped_exchanges: Dictionary mapping topics to actual exchanges.
        config: Detection configuration.

    Returns:
        List of detected CircularPattern objects.
    """
    patterns: list[CircularPattern] = []

    for topic, topic_ex in grouped_exchanges.items():
        if len(topic_ex) > config.exchange_threshold:
            severity = _calculate_severity(len(topic_ex), config)

            # Extract unique agents involved in this topic cycle
            agents: set[str] = set()
            for ex in topic_ex:
                agents.add(ex.source_agent)
                agents.add(ex.target_agent)

            pattern = CircularPattern(
                pattern_type="topic_cycle",
                agents_involved=tuple(sorted(agents)),
                topic=topic,
                exchange_count=len(topic_ex),
                first_exchange_at=topic_ex[0].timestamp,
                last_exchange_at=topic_ex[-1].timestamp,
                duration_seconds=_calculate_duration(topic_ex),
                severity=severity,
            )
            patterns.append(pattern)

    return patterns


# =============================================================================
# Intervention Functions (Task 4)
# =============================================================================


def _get_max_severity(patterns: list[CircularPattern]) -> CycleSeverity | None:
    """Get the maximum severity from a list of patterns.

    Args:
        patterns: List of circular patterns.

    Returns:
        Maximum severity level, or None if no patterns.
    """
    if not patterns:
        return None

    severity_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    max_pattern = max(patterns, key=lambda p: severity_rank.get(p.severity, 0))
    return max_pattern.severity


def _severity_rank(severity: CycleSeverity) -> int:
    """Get numeric rank for severity comparison.

    Args:
        severity: Severity level.

    Returns:
        Numeric rank (0-3).
    """
    ranks = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    return ranks.get(severity, 0)


def _determine_intervention_strategy(
    patterns: list[CircularPattern],
) -> tuple[InterventionStrategy, str]:
    """Determine the appropriate intervention strategy.

    Strategies:
    - break_cycle: Force routing to a different agent (skip the cycle)
    - inject_context: Add clarifying context to help break the impasse
    - escalate_human: Escalate to human intervention
    - none: No intervention (monitoring only)

    Args:
        patterns: List of detected circular patterns.

    Returns:
        Tuple of (strategy, message).
    """
    if not patterns:
        return "none", ""

    max_severity = _get_max_severity(patterns)

    if max_severity == "critical":
        return (
            "escalate_human",
            "Critical circular logic detected - human intervention required",
        )

    if max_severity == "high":
        return (
            "inject_context",
            "High severity cycle detected - injecting additional context to break impasse",
        )

    # For medium/low, try to break the cycle
    most_severe = max(patterns, key=lambda p: _severity_rank(p.severity))
    agents = most_severe.agents_involved
    skip_agent = agents[-1] if agents else "unknown"

    return (
        "break_cycle",
        f"Breaking cycle by skipping {skip_agent} - routing to alternate path",
    )


def _generate_intervention_message(
    strategy: InterventionStrategy,
    agents_involved: tuple[str, ...],
) -> str:
    """Generate human-readable intervention message.

    Args:
        strategy: The intervention strategy.
        agents_involved: Agents involved in the cycle.

    Returns:
        Human-readable message describing the intervention.
    """
    agents_str = ", ".join(agents_involved) if agents_involved else "agents"

    if strategy == "escalate_human":
        return f"Escalating to human intervention due to circular logic between {agents_str}"
    if strategy == "inject_context":
        return f"Injecting additional context to help {agents_str} resolve their impasse"
    if strategy == "break_cycle":
        return f"Breaking cycle by routing away from {agents_str}"
    return "No intervention needed"


# =============================================================================
# Escalation Functions (Task 5)
# =============================================================================


def _should_escalate(
    patterns: list[CircularPattern],
    config: CircularLogicConfig,
) -> tuple[bool, str | None]:
    """Determine if escalation to human is needed.

    Args:
        patterns: List of detected patterns.
        config: Detection configuration.

    Returns:
        Tuple of (should_escalate, reason).
    """
    if not patterns:
        return False, None

    max_severity = _get_max_severity(patterns)

    if max_severity == config.auto_escalate_severity:
        return True, f"Circular logic severity reached {max_severity}"

    return False, None


# =============================================================================
# Logging Functions (Task 6)
# =============================================================================


def _create_cycle_log(
    patterns: list[CircularPattern],
    intervention: InterventionStrategy,
    escalation_triggered: bool,
) -> CycleLog:
    """Create an audit log entry for detected cycle.

    Args:
        patterns: Detected patterns.
        intervention: Intervention strategy used.
        escalation_triggered: Whether escalation was triggered.

    Returns:
        CycleLog entry for audit trail.
    """
    cycle_id = f"cycle-{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()

    # Generate resolution message
    if escalation_triggered:
        resolution = "Escalated to human intervention"
    elif intervention == "break_cycle":
        resolution = "Breaking cycle by routing to alternate path"
    elif intervention == "inject_context":
        resolution = "Injecting context to help resolve impasse"
    else:
        resolution = "No intervention taken"

    return CycleLog(
        cycle_id=cycle_id,
        detected_at=now,
        patterns=tuple(patterns),
        intervention_taken=intervention,
        escalation_triggered=escalation_triggered,
        resolution=resolution,
    )


def _log_cycle_detection(log: CycleLog) -> None:
    """Log cycle detection at appropriate level.

    Log levels:
    - INFO: Detection only (no intervention)
    - WARNING: Intervention taken
    - ERROR: Escalation triggered

    Args:
        log: The cycle log entry to log.
    """
    log_data = {
        "cycle_id": log.cycle_id,
        "pattern_count": len(log.patterns),
        "intervention": log.intervention_taken,
        "escalation": log.escalation_triggered,
        "resolution": log.resolution,
    }

    if log.escalation_triggered:
        logger.error("circular_logic_escalated", **log_data)
    elif log.intervention_taken != "none":
        logger.warning("circular_logic_intervention", **log_data)
    else:
        logger.info("circular_logic_detected", **log_data)


# =============================================================================
# Main Detection Function (Task 7)
# =============================================================================


async def detect_circular_logic(
    state: YoloState,
    config: CircularLogicConfig | None = None,
) -> CycleAnalysis:
    """Detect circular logic patterns in agent exchanges (FR12, FR70).

    Enhanced detection that tracks:
    1. Topic-aware exchanges (same issue cycling)
    2. Agent pair cycles (A->B->A)
    3. Multi-agent cycles (A->B->C->A)

    Args:
        state: Current orchestration state.
        config: Detection configuration (uses defaults if None).

    Returns:
        CycleAnalysis with detection results, intervention strategy, and logging.
    """
    config = config or CircularLogicConfig()

    logger.info(
        "circular_detection_started",
        current_agent=state.get("current_agent"),
    )

    # Step 1: Extract exchanges from state
    all_exchanges = _extract_exchanges_from_state(state)

    # Step 1b: Filter by time window (per config.time_window_seconds)
    exchanges = _filter_by_time_window(all_exchanges, config)

    # Step 2: Track topic exchanges for analysis
    topic_exchanges = _track_topic_exchanges(exchanges)

    # Step 3: Detect patterns
    patterns: list[CircularPattern] = []

    # 3a: Agent pair cycles (always enabled)
    pair_patterns = _detect_agent_pair_cycles(exchanges, config)
    patterns.extend(pair_patterns)

    # 3b: Multi-agent cycles (configurable)
    if config.enable_multi_agent_detection:
        multi_patterns = _detect_multi_agent_cycles(exchanges, config)
        patterns.extend(multi_patterns)

    # 3c: Topic-based cycles (configurable)
    if config.enable_topic_detection:
        # Group by topic for topic cycle detection
        grouped = _group_exchanges_by_topic(exchanges)
        topic_patterns = _detect_topic_cycles(grouped, config)
        patterns.extend(topic_patterns)

    # Step 4: Determine if circular logic detected
    circular_detected = len(patterns) > 0

    # Step 5: Determine intervention
    intervention_strategy: InterventionStrategy = "none"
    intervention_message = ""
    if circular_detected:
        intervention_strategy, intervention_message = _determine_intervention_strategy(
            patterns
        )

    # Step 6: Check for escalation
    escalation_triggered = False
    escalation_reason: str | None = None
    if circular_detected:
        escalation_triggered, escalation_reason = _should_escalate(patterns, config)

    # Step 7: Create cycle log
    cycle_log: CycleLog | None = None
    if circular_detected:
        cycle_log = _create_cycle_log(
            patterns, intervention_strategy, escalation_triggered
        )
        _log_cycle_detection(cycle_log)

    # Step 8: Build and return result
    result = CycleAnalysis(
        circular_detected=circular_detected,
        patterns_found=tuple(patterns),
        intervention_strategy=intervention_strategy,
        intervention_message=intervention_message,
        escalation_triggered=escalation_triggered,
        escalation_reason=escalation_reason,
        topic_exchanges=topic_exchanges,
        total_exchange_count=len(exchanges),
        cycle_log=cycle_log,
    )

    logger.info(
        "circular_detection_complete",
        circular_detected=result.circular_detected,
        pattern_count=len(result.patterns_found),
        intervention=result.intervention_strategy,
    )

    return result


__all__ = [
    # Internal functions exported for testing
    "_calculate_duration",
    "_calculate_severity",
    "_create_cycle_log",
    "_detect_agent_pair_cycles",
    "_detect_multi_agent_cycles",
    "_detect_topic_cycles",
    "_determine_intervention_strategy",
    "_extract_common_topic",
    "_extract_exchange_topic",
    "_extract_exchanges_from_state",
    "_filter_by_time_window",
    "_generate_intervention_message",
    "_get_max_severity",
    "_group_exchanges_by_topic",
    "_log_cycle_detection",
    "_severity_rank",
    "_should_escalate",
    "_track_topic_exchanges",
    "detect_circular_logic",
]
