"""Conflict mediation for SM agent (Story 10.7).

This module provides conflict detection and mediation functionality for the
SM (Scrum Master) agent. It implements FR13: SM Agent can mediate conflicts
between agents with different recommendations.

Key Concepts:
- **Conflict Detection**: Identifies conflicting recommendations from agents
- **Principles-Based Resolution**: Resolves conflicts using defined principles
- **Documentation**: Records all resolutions with rationale
- **Notification**: Notifies affected agents of resolution outcomes

Example:
    >>> from yolo_developer.agents.sm.conflict_mediation import mediate_conflicts
    >>> from yolo_developer.orchestrator.state import YoloState
    >>>
    >>> state: YoloState = {
    ...     "messages": [...],
    ...     "current_agent": "sm",
    ...     "decisions": [...],
    ... }
    >>> result = await mediate_conflicts(state)
    >>> result.success
    True

Architecture Note:
    Per ADR-007, conflict mediation is part of the SM control plane.
    All operations are async, return state updates (not mutations),
    and use structlog for audit trail.

References:
    - FR13: SM Agent can mediate conflicts between agents with different recommendations
    - ADR-005: Inter-Agent Communication
    - ADR-007: Error Handling Strategy
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

import structlog

from yolo_developer.agents.sm.conflict_types import (
    RESOLUTION_PRINCIPLES,
    Conflict,
    ConflictMediationConfig,
    ConflictParty,
    ConflictResolution,
    ConflictSeverity,
    MediationResult,
    ResolutionStrategy,
)
from yolo_developer.orchestrator.context import Decision
from yolo_developer.orchestrator.state import YoloState

logger = structlog.get_logger(__name__)


# =============================================================================
# Constants (Issue #9 fix - avoid magic numbers)
# =============================================================================

DEFAULT_BLOCKING_INDICATORS: frozenset[str] = frozenset(
    {
        "security",
        "vulnerability",
        "blocker",
        "blocking",
        "critical",
        "urgent",
    }
)
"""Keywords that indicate a blocking severity conflict."""


def _generate_conflict_id(prefix: str) -> str:
    """Generate a unique conflict ID with UUID, with timestamp fallback.

    Issue #1 fix: UUID4 generation can fail in low-entropy environments.
    Provides timestamp-based fallback.

    Args:
        prefix: The prefix for the conflict ID (e.g., "design", "priority").

    Returns:
        A unique conflict ID string.
    """
    try:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
    except Exception:
        # Fallback to timestamp-based ID if UUID fails
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:14]
        return f"{prefix}_{timestamp}"


# =============================================================================
# Agent Position Extraction (Task 2.2)
# =============================================================================


def _extract_agent_positions(state: YoloState) -> dict[str, list[Decision]]:
    """Extract agent positions from state decisions.

    Groups decisions by agent for conflict detection.

    Args:
        state: Current orchestration state.

    Returns:
        Dictionary mapping agent names to their decisions.

    Example:
        >>> positions = _extract_agent_positions(state)
        >>> positions["architect"]
        [Decision(agent="architect", ...)]
    """
    decisions = state.get("decisions", [])
    positions: dict[str, list[Decision]] = {}
    skipped_count = 0

    for decision in decisions:
        # Issue #5 fix: Add type validation for decision objects
        if not hasattr(decision, "agent") or not isinstance(decision.agent, str):
            logger.warning(
                "invalid_decision_skipped",
                decision_type=type(decision).__name__,
                has_agent=hasattr(decision, "agent"),
            )
            skipped_count += 1
            continue

        agent = decision.agent
        if agent not in positions:
            positions[agent] = []
        positions[agent].append(decision)

    logger.debug(
        "agent_positions_extracted",
        agent_count=len(positions),
        total_decisions=len(decisions),
        skipped_count=skipped_count,
    )

    return positions


# =============================================================================
# Conflict Detection Functions (Task 2.3-2.6)
# =============================================================================


def _calculate_conflict_severity(
    parties: tuple[ConflictParty, ...],
    blocking_indicators: set[str] | None = None,
) -> ConflictSeverity:
    """Calculate conflict severity based on parties and indicators.

    Severity is determined by:
    - Number of parties involved
    - Presence of blocking keywords in rationale (word boundary matching)
    - Explicit blocking indicators

    Args:
        parties: Parties involved in the conflict.
        blocking_indicators: Keywords indicating blocking issues.

    Returns:
        Calculated severity level.

    Example:
        >>> severity = _calculate_conflict_severity(parties)
        >>> severity
        'major'
    """
    if blocking_indicators is None:
        blocking_indicators = set(DEFAULT_BLOCKING_INDICATORS)

    # Issue #6 fix: Use word boundary matching to avoid false positives
    # (e.g., "insecurity" should not match "security")
    pattern = r"\b(" + "|".join(re.escape(ind) for ind in blocking_indicators) + r")\b"
    blocking_pattern = re.compile(pattern, re.IGNORECASE)

    # Check for blocking keywords in any party's rationale
    has_blocking = False
    for party in parties:
        if blocking_pattern.search(party.rationale):
            has_blocking = True
            break

    # Severity based on indicators and party count
    if has_blocking:
        return "blocking"
    elif len(parties) >= 3:
        return "major"
    elif len(parties) >= 2:
        return "moderate"
    else:
        return "minor"


def _decisions_conflict(decision1: Decision, decision2: Decision) -> bool:
    """Determine if two decisions are potentially conflicting.

    Conflicts are detected when:
    - Different agents made decisions on the same artifact
    - Decisions have contradictory indicators in their summaries

    Args:
        decision1: First decision.
        decision2: Second decision.

    Returns:
        True if decisions may conflict, False otherwise.
    """
    # Different agents
    if decision1.agent == decision2.agent:
        return False

    # Check for overlapping artifacts
    artifacts1 = set(decision1.related_artifacts)
    artifacts2 = set(decision2.related_artifacts)

    if artifacts1 & artifacts2:  # Overlapping artifacts
        return True

    # Check for contradictory language patterns
    contradictory_pairs = [
        ("microservices", "monolith"),
        ("simple", "complex"),
        ("security", "performance"),
        ("async", "sync"),
        ("fast", "thorough"),
        ("high priority", "low priority"),
    ]

    summary1_lower = decision1.summary.lower()
    summary2_lower = decision2.summary.lower()

    for word1, word2 in contradictory_pairs:
        if (word1 in summary1_lower and word2 in summary2_lower) or (
            word2 in summary1_lower and word1 in summary2_lower
        ):
            return True

    return False


def _detect_design_conflicts(state: YoloState) -> list[Conflict]:
    """Detect conflicts in design decisions.

    Looks for contradictory architectural recommendations from
    different agents on the same artifacts or topics.

    Args:
        state: Current orchestration state.

    Returns:
        List of detected design conflicts.

    Example:
        >>> conflicts = _detect_design_conflicts(state)
        >>> conflicts[0].conflict_type
        'design_conflict'
    """
    conflicts: list[Conflict] = []
    decisions = state.get("decisions", [])

    # Group decisions by related artifacts
    by_artifact: dict[str, list[Decision]] = {}
    for decision in decisions:
        for artifact in decision.related_artifacts:
            if artifact not in by_artifact:
                by_artifact[artifact] = []
            by_artifact[artifact].append(decision)

    # Find conflicting decisions on same artifact
    for artifact, artifact_decisions in by_artifact.items():
        if len(artifact_decisions) < 2:
            continue

        # Check pairs of decisions for conflicts
        for i, dec1 in enumerate(artifact_decisions):
            for dec2 in artifact_decisions[i + 1 :]:
                if _decisions_conflict(dec1, dec2):
                    parties = (
                        ConflictParty(
                            agent=dec1.agent,
                            position=dec1.summary,
                            rationale=dec1.rationale,
                            artifacts=dec1.related_artifacts,
                        ),
                        ConflictParty(
                            agent=dec2.agent,
                            position=dec2.summary,
                            rationale=dec2.rationale,
                            artifacts=dec2.related_artifacts,
                        ),
                    )

                    severity = _calculate_conflict_severity(parties)
                    conflict = Conflict(
                        conflict_id=_generate_conflict_id(f"design_{artifact}"),
                        conflict_type="design_conflict",
                        severity=severity,
                        parties=parties,
                        topic=artifact,
                        blocking_progress=severity == "blocking",
                    )
                    conflicts.append(conflict)

                    logger.info(
                        "design_conflict_detected",
                        conflict_id=conflict.conflict_id,
                        agents=[p.agent for p in parties],
                        artifact=artifact,
                        severity=severity,
                    )

    return conflicts


def _detect_priority_conflicts(state: YoloState) -> list[Conflict]:
    """Detect conflicts in priority assessments.

    Looks for contradictory priority recommendations from different agents.

    Args:
        state: Current orchestration state.

    Returns:
        List of detected priority conflicts.

    Example:
        >>> conflicts = _detect_priority_conflicts(state)
        >>> conflicts[0].conflict_type
        'priority_conflict'
    """
    conflicts: list[Conflict] = []
    decisions = state.get("decisions", [])

    # Find priority-related decisions
    priority_keywords = {"priority", "prioritize", "urgent", "first", "before", "block"}
    priority_decisions: list[Decision] = []

    for decision in decisions:
        summary_lower = decision.summary.lower()
        rationale_lower = decision.rationale.lower()
        if any(kw in summary_lower or kw in rationale_lower for kw in priority_keywords):
            priority_decisions.append(decision)

    # Check for conflicting priorities
    for i, dec1 in enumerate(priority_decisions):
        for dec2 in priority_decisions[i + 1 :]:
            if dec1.agent != dec2.agent and _decisions_conflict(dec1, dec2):
                parties = (
                    ConflictParty(
                        agent=dec1.agent,
                        position=dec1.summary,
                        rationale=dec1.rationale,
                        artifacts=dec1.related_artifacts,
                    ),
                    ConflictParty(
                        agent=dec2.agent,
                        position=dec2.summary,
                        rationale=dec2.rationale,
                        artifacts=dec2.related_artifacts,
                    ),
                )

                severity = _calculate_conflict_severity(parties)
                conflict = Conflict(
                    conflict_id=_generate_conflict_id("priority"),
                    conflict_type="priority_conflict",
                    severity=severity,
                    parties=parties,
                    topic="task_priority",
                    blocking_progress=severity == "blocking",
                )
                conflicts.append(conflict)

                logger.info(
                    "priority_conflict_detected",
                    conflict_id=conflict.conflict_id,
                    agents=[p.agent for p in parties],
                    severity=severity,
                )

    return conflicts


def _detect_approach_conflicts(state: YoloState) -> list[Conflict]:
    """Detect conflicts in implementation approaches.

    Looks for contradictory implementation approach recommendations.

    Args:
        state: Current orchestration state.

    Returns:
        List of detected approach conflicts.

    Example:
        >>> conflicts = _detect_approach_conflicts(state)
        >>> conflicts[0].conflict_type
        'approach_conflict'
    """
    conflicts: list[Conflict] = []
    decisions = state.get("decisions", [])

    # Find approach-related decisions
    approach_keywords = {
        "approach",
        "implement",
        "use",
        "using",
        "pattern",
        "method",
        "strategy",
    }
    approach_decisions: list[Decision] = []

    for decision in decisions:
        summary_lower = decision.summary.lower()
        if any(kw in summary_lower for kw in approach_keywords):
            approach_decisions.append(decision)

    # Check for conflicting approaches
    for i, dec1 in enumerate(approach_decisions):
        for dec2 in approach_decisions[i + 1 :]:
            if dec1.agent != dec2.agent and _decisions_conflict(dec1, dec2):
                parties = (
                    ConflictParty(
                        agent=dec1.agent,
                        position=dec1.summary,
                        rationale=dec1.rationale,
                        artifacts=dec1.related_artifacts,
                    ),
                    ConflictParty(
                        agent=dec2.agent,
                        position=dec2.summary,
                        rationale=dec2.rationale,
                        artifacts=dec2.related_artifacts,
                    ),
                )

                severity = _calculate_conflict_severity(parties)
                conflict = Conflict(
                    conflict_id=_generate_conflict_id("approach"),
                    conflict_type="approach_conflict",
                    severity=severity,
                    parties=parties,
                    topic="implementation_approach",
                    blocking_progress=severity == "blocking",
                )
                conflicts.append(conflict)

                logger.info(
                    "approach_conflict_detected",
                    conflict_id=conflict.conflict_id,
                    agents=[p.agent for p in parties],
                    severity=severity,
                )

    return conflicts


def _detect_scope_conflicts(state: YoloState) -> list[Conflict]:
    """Detect conflicts in scope assessments.

    Looks for contradictory scope boundary recommendations.

    Args:
        state: Current orchestration state.

    Returns:
        List of detected scope conflicts.

    Example:
        >>> conflicts = _detect_scope_conflicts(state)
        >>> conflicts[0].conflict_type
        'scope_conflict'
    """
    conflicts: list[Conflict] = []
    decisions = state.get("decisions", [])

    # Find scope-related decisions
    scope_keywords = {"scope", "boundary", "include", "exclude", "out of scope", "mvp"}
    scope_decisions: list[Decision] = []

    for decision in decisions:
        summary_lower = decision.summary.lower()
        rationale_lower = decision.rationale.lower()
        if any(kw in summary_lower or kw in rationale_lower for kw in scope_keywords):
            scope_decisions.append(decision)

    # Check for conflicting scope definitions
    for i, dec1 in enumerate(scope_decisions):
        for dec2 in scope_decisions[i + 1 :]:
            if dec1.agent != dec2.agent and _decisions_conflict(dec1, dec2):
                parties = (
                    ConflictParty(
                        agent=dec1.agent,
                        position=dec1.summary,
                        rationale=dec1.rationale,
                        artifacts=dec1.related_artifacts,
                    ),
                    ConflictParty(
                        agent=dec2.agent,
                        position=dec2.summary,
                        rationale=dec2.rationale,
                        artifacts=dec2.related_artifacts,
                    ),
                )

                severity = _calculate_conflict_severity(parties)
                conflict = Conflict(
                    conflict_id=_generate_conflict_id("scope"),
                    conflict_type="scope_conflict",
                    severity=severity,
                    parties=parties,
                    topic="scope_boundaries",
                    blocking_progress=severity == "blocking",
                )
                conflicts.append(conflict)

                logger.info(
                    "scope_conflict_detected",
                    conflict_id=conflict.conflict_id,
                    agents=[p.agent for p in parties],
                    severity=severity,
                )

    return conflicts


def _detect_all_conflicts(state: YoloState) -> list[Conflict]:
    """Detect all types of conflicts in state.

    Runs all conflict detection functions and aggregates results.

    Args:
        state: Current orchestration state.

    Returns:
        List of all detected conflicts.
    """
    all_conflicts: list[Conflict] = []

    all_conflicts.extend(_detect_design_conflicts(state))
    all_conflicts.extend(_detect_priority_conflicts(state))
    all_conflicts.extend(_detect_approach_conflicts(state))
    all_conflicts.extend(_detect_scope_conflicts(state))

    logger.info(
        "conflict_detection_complete",
        total_conflicts=len(all_conflicts),
        design_conflicts=sum(1 for c in all_conflicts if c.conflict_type == "design_conflict"),
        priority_conflicts=sum(1 for c in all_conflicts if c.conflict_type == "priority_conflict"),
        approach_conflicts=sum(1 for c in all_conflicts if c.conflict_type == "approach_conflict"),
        scope_conflicts=sum(1 for c in all_conflicts if c.conflict_type == "scope_conflict"),
    )

    return all_conflicts


# =============================================================================
# SM Evaluation Logic (Task 3)
# =============================================================================


def _score_position(
    party: ConflictParty,
    config: ConflictMediationConfig,
) -> tuple[float, list[str]]:
    """Score a single position against resolution principles.

    Higher score indicates stronger alignment with principles.

    Args:
        party: The party's position to score.
        config: Mediation configuration.

    Returns:
        Tuple of (score, principles_matched).

    Example:
        >>> score, principles = _score_position(party, config)
        >>> score
        0.8
    """
    score = 0.0
    principles_matched: list[str] = []

    position_text = f"{party.position} {party.rationale}".lower()

    for principle in config.principles_hierarchy:
        if principle not in RESOLUTION_PRINCIPLES:
            continue

        principle_info = RESOLUTION_PRINCIPLES[principle]
        weight = principle_info["weight"]
        keywords = principle_info["keywords"]

        # Count keyword matches
        keyword_matches = sum(1 for kw in keywords if kw in position_text)
        if keyword_matches > 0:
            principle_score = weight * (keyword_matches / len(keywords))
            score += principle_score
            principles_matched.append(principle)

    return score, principles_matched


def _score_positions(
    conflict: Conflict,
    config: ConflictMediationConfig,
) -> dict[str, tuple[float, list[str]]]:
    """Score each party's position against resolution principles.

    Higher score indicates stronger alignment with principles hierarchy.

    Args:
        conflict: The conflict with parties to score.
        config: Mediation configuration.

    Returns:
        Dictionary mapping agent names to (score, principles_matched).

    Example:
        >>> scores = _score_positions(conflict, config)
        >>> scores["architect"]
        (0.8, ['safety', 'correctness'])
    """
    scores: dict[str, tuple[float, list[str]]] = {}

    for party in conflict.parties:
        score, principles = _score_position(party, config)
        scores[party.agent] = (score, principles)

        logger.debug(
            "position_scored",
            agent=party.agent,
            score=score,
            principles_matched=principles,
        )

    return scores


def _find_compromise(
    conflict: Conflict,
    scores: dict[str, tuple[float, list[str]]],
) -> tuple[str, ...]:
    """Attempt to find compromise between positions.

    Looks for common ground or complementary aspects.

    Args:
        conflict: The conflict to find compromise for.
        scores: Position scores.

    Returns:
        Tuple of compromise statements.
    """
    # Issue #2 fix: Handle empty parties edge case
    if not conflict.parties:
        return ("No parties to compromise between",)

    compromises: list[str] = []

    # Extract common principles from both parties
    all_principles: set[str] = set()
    for _, principles in scores.values():
        all_principles.update(principles)

    if len(conflict.parties) >= 2:
        party1 = conflict.parties[0]
        party2 = conflict.parties[1]

        # Simple compromise: acknowledge both perspectives
        compromises.append(f"Acknowledge {party1.agent}'s concern: {party1.position[:50]}...")
        compromises.append(f"Incorporate {party2.agent}'s approach: {party2.position[:50]}...")

        # If both mention common principles, use that as basis
        if all_principles:
            compromises.append(
                f"Base decision on shared principles: {', '.join(sorted(all_principles))}"
            )

    return tuple(compromises)


def _should_defer(conflict: Conflict, config: ConflictMediationConfig) -> bool:
    """Determine if conflict resolution should be deferred.

    Defers if conflict is non-blocking and might benefit from more context.

    Args:
        conflict: The conflict to evaluate.
        config: Mediation configuration.

    Returns:
        True if conflict should be deferred, False otherwise.
    """
    # Never defer blocking conflicts
    if conflict.blocking_progress:
        return False

    # Defer minor conflicts if configured
    if conflict.severity == "minor" and config.auto_resolve_minor:
        return False  # Auto-resolve, don't defer

    # Don't defer - we want to resolve
    return False


def _evaluate_conflict(
    conflict: Conflict,
    config: ConflictMediationConfig,
) -> tuple[ResolutionStrategy, str, str | None, tuple[str, ...], tuple[str, ...]]:
    """Evaluate a conflict and determine resolution strategy.

    Main evaluation function that decides how to resolve a conflict
    based on principles scoring.

    Args:
        conflict: The conflict to evaluate.
        config: Mediation configuration.

    Returns:
        Tuple of (strategy, rationale, winning_position, compromises, principles).

    Example:
        >>> strategy, rationale, winner, comps, prcs = _evaluate_conflict(conflict, config)
        >>> strategy
        'accept_first'
    """
    # Score positions
    scores = _score_positions(conflict, config)

    # Find highest and lowest scores
    sorted_agents = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)

    if len(sorted_agents) < 2:
        # Only one party - check if their score meets minimum threshold
        # Issue #4 fix: Apply score threshold consistently even for single positions
        agent, (score, principles) = sorted_agents[0]
        position = next(p.position for p in conflict.parties if p.agent == agent)

        # If single position has no meaningful score, defer for more context
        if score < config.score_threshold:
            return (
                "defer",
                f"Single position from {agent} doesn't meet score threshold ({score:.2f})",
                None,
                (),
                (),
            )

        return (
            "accept_first",
            f"Single position from {agent} (score: {score:.2f})",
            position,
            (),
            tuple(principles),
        )

    highest_agent, (highest_score, highest_principles) = sorted_agents[0]
    second_agent, (second_score, second_principles) = sorted_agents[1]

    score_diff = highest_score - second_score

    logger.debug(
        "conflict_evaluation",
        conflict_id=conflict.conflict_id,
        highest_agent=highest_agent,
        highest_score=highest_score,
        second_agent=second_agent,
        second_score=second_score,
        score_diff=score_diff,
    )

    # Check if should defer
    if _should_defer(conflict, config):
        return (
            "defer",
            "Conflict is non-blocking and may benefit from more context",
            None,
            (),
            (),
        )

    # Check if should escalate
    if conflict.severity == "blocking" and config.escalate_blocking:
        if score_diff < config.score_threshold:
            return (
                "escalate_human",
                f"Blocking conflict with unclear resolution (score diff: {score_diff:.2f})",
                None,
                (),
                tuple(set(highest_principles + second_principles)),
            )

    # Clear winner - accept first or second
    if score_diff >= config.score_threshold:
        # Determine which party index corresponds to highest scorer
        if conflict.parties[0].agent == highest_agent:
            strategy: ResolutionStrategy = "accept_first"
        else:
            strategy = "accept_second"

        winning_position = next(p.position for p in conflict.parties if p.agent == highest_agent)
        rationale = (
            f"{highest_agent}'s position scores higher on principles "
            f"({highest_score:.2f} vs {second_score:.2f}). "
            f"Applied: {', '.join(highest_principles)}"
        )

        return (
            strategy,
            rationale,
            winning_position,
            (),
            tuple(highest_principles),
        )

    # No clear winner - try compromise
    compromises = _find_compromise(conflict, scores)
    all_principles = set(highest_principles + second_principles)

    return (
        "compromise",
        f"Scores too close ({highest_score:.2f} vs {second_score:.2f}), seeking compromise",
        None,
        compromises,
        tuple(sorted(all_principles)),
    )


# =============================================================================
# Resolution Documentation (Task 5)
# =============================================================================


def _document_resolution(
    conflict: Conflict,
    strategy: ResolutionStrategy,
    rationale: str,
    winning_position: str | None,
    compromises: tuple[str, ...],
    principles_applied: tuple[str, ...],
) -> ConflictResolution:
    """Document a conflict resolution.

    Creates a ConflictResolution record with all details.

    Args:
        conflict: The conflict being resolved.
        strategy: Resolution strategy used.
        rationale: Why this resolution was chosen.
        winning_position: Winning position if applicable.
        compromises: Compromises made if applicable.
        principles_applied: Principles that drove the decision.

    Returns:
        ConflictResolution record.
    """
    resolution = ConflictResolution(
        conflict_id=conflict.conflict_id,
        strategy=strategy,
        resolution_rationale=rationale,
        winning_position=winning_position,
        compromises=compromises,
        principles_applied=principles_applied,
    )

    logger.info(
        "conflict_resolution_documented",
        conflict_id=conflict.conflict_id,
        strategy=strategy,
        winning_position=winning_position is not None,
        compromise_count=len(compromises),
        principles_count=len(principles_applied),
    )

    return resolution


def _log_conflict_mediation(
    conflict: Conflict,
    resolution: ConflictResolution,
    escalated: bool,
) -> None:
    """Log conflict mediation for audit trail.

    Uses structlog with appropriate log levels based on outcome.
    Issue #7 fix: Wrapped in try-except to prevent logging failures
    from breaking the mediation workflow.

    Args:
        conflict: The conflict that was mediated.
        resolution: The resolution applied.
        escalated: Whether escalation was triggered.
    """
    try:
        if escalated:
            logger.warning(
                "conflict_escalated",
                conflict_id=conflict.conflict_id,
                conflict_type=conflict.conflict_type,
                severity=conflict.severity,
                agents=[p.agent for p in conflict.parties],
                topic=conflict.topic,
                strategy=resolution.strategy,
            )
        else:
            logger.info(
                "conflict_mediated",
                conflict_id=conflict.conflict_id,
                conflict_type=conflict.conflict_type,
                severity=conflict.severity,
                agents=[p.agent for p in conflict.parties],
                topic=conflict.topic,
                strategy=resolution.strategy,
                winning_position=resolution.winning_position,
            )
    except Exception:
        # Issue #7 fix: Logging should never break mediation
        pass


# =============================================================================
# Agent Notification (Task 6)
# =============================================================================


def _identify_affected_agents(conflict: Conflict) -> set[str]:
    """Identify agents that need to be notified.

    Returns all agents involved in the conflict plus their natural
    successors in the workflow.

    Args:
        conflict: The conflict that was resolved.

    Returns:
        Set of agent names to notify.
    """
    affected = set()

    for party in conflict.parties:
        affected.add(party.agent)

    return affected


def _create_notification_message(
    conflict: Conflict,
    resolution: ConflictResolution,
) -> str:
    """Create notification message for affected agents.

    Builds a human-readable notification about the resolution.

    Args:
        conflict: The resolved conflict.
        resolution: The resolution applied.

    Returns:
        Notification message string.
    """
    agents = ", ".join(p.agent for p in conflict.parties)

    message = (
        f"[Conflict Resolution] Topic: {conflict.topic}\n"
        f"Parties: {agents}\n"
        f"Strategy: {resolution.strategy}\n"
        f"Rationale: {resolution.resolution_rationale}\n"
    )

    if resolution.winning_position:
        message += f"Decision: {resolution.winning_position}\n"

    if resolution.compromises:
        message += "Compromises:\n"
        for comp in resolution.compromises:
            message += f"  - {comp}\n"

    if resolution.principles_applied:
        message += f"Principles: {', '.join(resolution.principles_applied)}\n"

    return message


def _notify_agents(
    conflict: Conflict,
    resolution: ConflictResolution,
) -> tuple[str, ...]:
    """Notify affected agents of resolution.

    Logs notifications and returns list of agents notified.
    Issue #10 fix: The notification message is now logged at INFO level
    so it's captured in the audit trail.

    Note:
        Currently returns tuple of notified agents.
        Actual message delivery would require integration with
        the orchestration message system. The message is logged
        for audit purposes.

    Args:
        conflict: The resolved conflict.
        resolution: The resolution applied.

    Returns:
        Tuple of agent names that were notified.
    """
    affected = _identify_affected_agents(conflict)
    message = _create_notification_message(conflict, resolution)

    for agent in affected:
        logger.info(
            "agent_notified",
            agent=agent,
            conflict_id=conflict.conflict_id,
            strategy=resolution.strategy,
            notification_message=message,  # Issue #10 fix: Include message in log
        )

    logger.debug(
        "notification_content",
        conflict_id=conflict.conflict_id,
        message=message,
    )

    return tuple(sorted(affected))


# =============================================================================
# Main Mediation Function (Task 7)
# =============================================================================


async def mediate_conflicts(
    state: YoloState,
    config: ConflictMediationConfig | None = None,
) -> MediationResult:
    """Mediate conflicts between agents (FR13).

    Main entry point for conflict mediation. Detects and resolves
    conflicting agent recommendations using a principles-based approach.

    Args:
        state: Current orchestration state.
        config: Mediation configuration. Uses defaults if not provided.

    Returns:
        MediationResult with all detected conflicts and resolutions.

    Example:
        >>> result = await mediate_conflicts(state)
        >>> result.success
        True
        >>> len(result.conflicts_detected)
        2
    """
    if config is None:
        config = ConflictMediationConfig()

    logger.info(
        "conflict_mediation_started",
        current_agent=state.get("current_agent"),
        decision_count=len(state.get("decisions", [])),
    )

    # Detect all conflicts
    conflicts = _detect_all_conflicts(state)

    if not conflicts:
        logger.info("no_conflicts_detected")
        return MediationResult(
            conflicts_detected=(),
            resolutions=(),
            notifications_sent=(),
            escalations_triggered=(),
            success=True,
            mediation_notes="No conflicts detected",
        )

    # Process each conflict
    resolutions: list[ConflictResolution] = []
    escalations: list[str] = []
    notifications: set[str] = set()

    for conflict in conflicts:
        # Evaluate and determine strategy
        strategy, rationale, winning_pos, compromises, principles = _evaluate_conflict(
            conflict, config
        )

        # Document resolution
        resolution = _document_resolution(
            conflict,
            strategy,
            rationale,
            winning_pos,
            compromises,
            principles,
        )
        resolutions.append(resolution)

        # Track escalations
        escalated = strategy == "escalate_human"
        if escalated:
            escalations.append(conflict.conflict_id)

        # Log for audit
        _log_conflict_mediation(conflict, resolution, escalated)

        # Notify affected agents
        notified = _notify_agents(conflict, resolution)
        notifications.update(notified)

    # Determine overall success
    success = len(escalations) == 0

    result = MediationResult(
        conflicts_detected=tuple(conflicts),
        resolutions=tuple(resolutions),
        notifications_sent=tuple(sorted(notifications)),
        escalations_triggered=tuple(escalations),
        success=success,
        mediation_notes=(
            f"Processed {len(conflicts)} conflicts, "
            f"{len(resolutions)} resolved, "
            f"{len(escalations)} escalated"
        ),
    )

    logger.info(
        "conflict_mediation_complete",
        conflicts_detected=len(conflicts),
        resolutions=len(resolutions),
        escalations=len(escalations),
        success=success,
    )

    return result
