"""PM agent node for LangGraph orchestration (Story 6.1, 6.2, 6.3, 6.4).

This module provides the pm_node function that integrates with the
LangGraph orchestration workflow. The PM agent transforms crystallized
requirements into user stories with acceptance criteria.

Key Concepts:
- **YoloState Input**: Receives state as TypedDict, not Pydantic
- **Immutable Updates**: Returns state update dict, never mutates input
- **Async I/O**: All LLM calls use async/await
- **Structured Logging**: Uses structlog for audit trail
- **LLM-Powered**: Uses LLM for story extraction and AC generation (Story 6.2)
- **Prioritization**: Stories are prioritized by value and dependencies (Story 6.4)

LLM Usage:
    Set _USE_LLM in llm.py to True to enable actual LLM calls.
    Set to False (default) to use stub implementations for testing.

Example:
    >>> from yolo_developer.agents.pm import pm_node
    >>> from yolo_developer.orchestrator.state import YoloState
    >>>
    >>> state: YoloState = {
    ...     "messages": [AIMessage(content="Analyst output...")],
    ...     "current_agent": "pm",
    ...     "handoff_context": None,
    ...     "decisions": [],
    ...     "analyst_output": {...},
    ... }
    >>> result = await pm_node(state)
    >>> result["messages"]  # New messages to append
    [AIMessage(...)]

Architecture Note:
    Per ADR-005, this node follows the LangGraph pattern of receiving
    full state and returning only the updates to apply.
"""

from __future__ import annotations

from typing import Any

import structlog

from yolo_developer.agents.pm.llm import (
    _estimate_complexity,
    _extract_story_components,
    _generate_acceptance_criteria_llm,
)
from yolo_developer.agents.pm.prioritization import prioritize_stories
from yolo_developer.agents.pm.testability import validate_story_testability
from yolo_developer.agents.pm.types import (
    AcceptanceCriterion,
    PMOutput,
    PrioritizationResult,
    Story,
    StoryPriority,
    StoryStatus,
)
from yolo_developer.gates import quality_gate
from yolo_developer.orchestrator.context import Decision
from yolo_developer.orchestrator.state import YoloState, create_agent_message

logger = structlog.get_logger(__name__)


def _determine_priority(category: str, requirement_text: str) -> StoryPriority:
    """Determine story priority based on requirement category and content.

    Args:
        category: Requirement category (functional, non_functional, constraint).
        requirement_text: The requirement text for content analysis.

    Returns:
        StoryPriority enum value.

    Example:
        >>> _determine_priority("functional", "User authentication")
        <StoryPriority.HIGH: 'high'>
    """
    text_lower = requirement_text.lower()

    # Critical indicators
    critical_keywords = ["security", "authentication", "authorization", "critical", "must"]
    if any(kw in text_lower for kw in critical_keywords):
        return StoryPriority.CRITICAL

    # High priority for functional requirements
    if category == "functional":
        return StoryPriority.HIGH

    # Medium for non-functional
    if category == "non_functional":
        return StoryPriority.MEDIUM

    # Low for everything else
    return StoryPriority.LOW


async def _generate_acceptance_criteria(
    requirement_id: str,
    requirement_text: str,
    story_components: dict[str, str],
) -> tuple[AcceptanceCriterion, ...]:
    """Generate acceptance criteria from requirement text using LLM.

    Creates Given/When/Then format acceptance criteria from the requirement
    using LLM-powered generation. Falls back to stub if LLM is disabled.

    Args:
        requirement_id: ID of the source requirement.
        requirement_text: Text of the requirement.
        story_components: Dict with role, action, benefit, title from extraction.

    Returns:
        Tuple of AcceptanceCriterion objects.

    Example:
        >>> acs = await _generate_acceptance_criteria(
        ...     "req-001",
        ...     "User can login",
        ...     {"role": "user", "action": "login", ...}
        ... )
        >>> acs[0].id
        'AC1'
    """
    ac_data = await _generate_acceptance_criteria_llm(
        requirement_id=requirement_id,
        requirement_text=requirement_text,
        story_components=story_components,
    )

    # Convert to AcceptanceCriterion objects
    acs = []
    for i, ac in enumerate(ac_data, start=1):
        acs.append(
            AcceptanceCriterion(
                id=f"AC{i}",
                given=ac.get("given", ""),
                when=ac.get("when", ""),
                then=ac.get("then", ""),
                and_clauses=tuple(ac.get("and_clauses", [])),
            )
        )

    return (
        tuple(acs)
        if acs
        else (
            AcceptanceCriterion(
                id="AC1",
                given=f"the system is ready to process {requirement_id}",
                when="the feature is used as specified",
                then=f"the requirement is satisfied: {requirement_text[:50]}...",
                and_clauses=(),
            ),
        )
    )


async def _transform_single_requirement(
    req: dict[str, Any],
    story_counter: int,
) -> Story | None:
    """Transform a single requirement into a user story.

    Args:
        req: Requirement dict from analyst output.
        story_counter: Counter for generating story IDs.

    Returns:
        Story object or None if requirement cannot be transformed.

    Example:
        >>> req = {"id": "req-001", "refined_text": "User can login", "category": "functional"}
        >>> story = await _transform_single_requirement(req, 1)
        >>> story.id
        'story-001'
    """
    req_id = req.get("id", f"req-{story_counter}")
    refined_text = req.get("refined_text", req.get("original_text", ""))
    category = req.get("category", "functional")

    logger.debug(
        "pm_transforming_requirement",
        requirement_id=req_id,
        category=category,
    )

    # Extract story components using LLM
    story_components = await _extract_story_components(
        requirement_id=req_id,
        requirement_text=refined_text,
        category=category,
    )

    # Generate acceptance criteria using LLM
    acs = await _generate_acceptance_criteria(
        requirement_id=req_id,
        requirement_text=refined_text,
        story_components=story_components,
    )

    # Determine priority
    priority = _determine_priority(category, refined_text)

    # Estimate complexity
    complexity = _estimate_complexity(refined_text, len(acs))

    # Create story
    story = Story(
        id=f"story-{story_counter:03d}",
        title=story_components.get(
            "title", refined_text[:50] if refined_text else f"Story for {req_id}"
        ),
        role=story_components.get("role", "user"),
        action=story_components.get("action", refined_text or f"complete requirement {req_id}"),
        benefit=story_components.get("benefit", "the system meets the specified requirement"),
        acceptance_criteria=acs,
        priority=priority,
        status=StoryStatus.DRAFT,
        source_requirements=(req_id,),
        dependencies=(),  # Dependency detection is Story 6.5
        estimated_complexity=complexity,
    )

    logger.info(
        "pm_story_created",
        story_id=story.id,
        requirement_id=req_id,
        role=story.role,
        ac_count=len(acs),
        complexity=complexity,
    )

    return story


async def _transform_requirements_to_stories(
    requirements: list[dict[str, Any]],
) -> tuple[tuple[Story, ...], tuple[str, ...]]:
    """Transform crystallized requirements into user stories.

    Maps requirements to story format with acceptance criteria using
    LLM-powered extraction and generation. Constraint requirements are
    filtered out and tracked as unprocessed.

    Args:
        requirements: List of crystallized requirement dicts from analyst.

    Returns:
        Tuple of (stories, unprocessed_requirement_ids).

    Example:
        >>> reqs = [{"id": "req-001", "refined_text": "User login", "category": "functional"}]
        >>> stories, unprocessed = await _transform_requirements_to_stories(reqs)
        >>> len(stories)
        1
    """
    stories: list[Story] = []
    unprocessed: list[str] = []
    story_counter = 0  # Separate counter for sequential story IDs

    for i, req in enumerate(requirements):
        req_id = req.get("id", f"req-{i}")
        category = req.get("category", "functional")

        # Skip constraint requirements for story transformation
        # They become constraints on functional stories
        if category == "constraint":
            logger.debug("pm_skipping_constraint", requirement_id=req_id)
            unprocessed.append(req_id)
            continue

        story_counter += 1

        try:
            story = await _transform_single_requirement(req, story_counter)
            if story:
                stories.append(story)
            else:
                unprocessed.append(req_id)
        except Exception as e:
            logger.warning(
                "pm_transformation_failed",
                requirement_id=req_id,
                error=str(e),
            )
            unprocessed.append(req_id)

    return tuple(stories), tuple(unprocessed)


# Note: ac_measurability gate evaluator registration is implemented in Story 6.3
# Until then, the gate will pass through (no evaluator = no blocking)
@quality_gate("ac_measurability", blocking=True)
async def pm_node(state: YoloState) -> dict[str, Any]:
    """Transform crystallized requirements into user stories.

    This LangGraph node receives analyst output from state and produces
    user stories with acceptance criteria. It is decorated with the
    ac_measurability quality gate to ensure AC are testable.

    Uses LLM-powered transformation (Story 6.2) when _USE_LLM is True,
    otherwise falls back to stub implementations for testing.

    Args:
        state: YoloState with analyst_output containing requirements.

    Returns:
        Dict with state updates including:
        - messages: AIMessage with processing summary
        - decisions: Decision records for audit trail
        - pm_output: PMOutput dict with stories

    Example:
        >>> result = await pm_node(state)
        >>> result["pm_output"]["story_count"]
        5
    """
    logger.info("pm_node_started")

    # Extract analyst output from state
    # Note: analyst_output is not in YoloState TypedDict (added dynamically by analyst node)
    # so we need to access it via dict-style access with type casting
    state_dict: dict[str, Any] = dict(state)
    analyst_output: dict[str, Any] = state_dict.get("analyst_output", {})
    requirements: list[dict[str, Any]] = analyst_output.get("requirements", [])
    escalations_from_analyst: list[dict[str, Any]] = analyst_output.get("escalations", [])
    # Extract gaps and contradictions for context
    # Note: Currently logged for visibility; used in Story 6.7 (PM escalation logic)
    gap_count = len(analyst_output.get("gaps", []))
    contradiction_count = len(analyst_output.get("contradictions", []))

    logger.info(
        "pm_node_received_input",
        requirement_count=len(requirements),
        escalation_count=len(escalations_from_analyst),
        gap_count=gap_count,
        contradiction_count=contradiction_count,
    )

    # Transform requirements to stories using LLM-powered transformation
    stories, unprocessed_reqs = await _transform_requirements_to_stories(requirements)

    # Validate each story's testability (Story 6.3)
    stories_with_issues = 0
    total_vague_terms = 0
    total_missing_edge_cases = 0

    for story in stories:
        result = validate_story_testability(story)
        if not result["is_valid"]:
            stories_with_issues += 1
            total_vague_terms += len(result["vague_terms_found"])
        total_missing_edge_cases += len(result["missing_edge_cases"])

    # Build validation summary for processing notes and decision rationale
    if stories_with_issues > 0:
        validation_summary = (
            f"Testability validation: {stories_with_issues}/{len(stories)} stories have issues "
            f"({total_vague_terms} vague terms, {total_missing_edge_cases} missing edge cases)"
        )
    else:
        validation_summary = "Testability validation: all stories passed"

    # Prioritize stories (Story 6.4)
    prioritization_result: PrioritizationResult = prioritize_stories(stories)

    # Build prioritization summary
    prioritization_summary = prioritization_result["analysis_notes"]
    if prioritization_result["quick_wins"]:
        prioritization_summary += f" (quick wins: {', '.join(prioritization_result['quick_wins'])})"
    if prioritization_result["dependency_cycles"]:
        prioritization_summary += (
            f" (warning: {len(prioritization_result['dependency_cycles'])} dependency cycles)"
        )

    logger.info(
        "pm_prioritization_complete",
        story_count=len(stories),
        quick_win_count=len(prioritization_result["quick_wins"]),
        cycle_count=len(prioritization_result["dependency_cycles"]),
        top_priority=prioritization_result["recommended_execution_order"][0]
        if prioritization_result["recommended_execution_order"]
        else None,
    )

    # Build processing notes with validation and prioritization summary
    processing_notes_parts = [
        f"Transformed {len(requirements)} requirements into {len(stories)} stories using LLM analysis",
        validation_summary,
        f"Prioritization: {prioritization_summary}",
    ]

    # Create PM output
    output = PMOutput(
        stories=stories,
        unprocessed_requirements=unprocessed_reqs,
        escalations_to_analyst=(),  # Escalation logic is Story 6.7
        processing_notes="; ".join(processing_notes_parts),
    )

    # Create decision record for audit trail with transformation details
    decision = Decision(
        agent="pm",
        summary=f"Created {output.story_count} stories from {len(requirements)} requirements",
        rationale=(
            f"Requirements transformed using LLM-powered story extraction. "
            f"Constraint requirements ({len(unprocessed_reqs)}) tracked separately. "
            f"Each story includes role, action, benefit, and acceptance criteria. "
            f"{validation_summary}. "
            f"Prioritization: {prioritization_summary}"
        ),
        related_artifacts=tuple(s.id for s in stories),
    )

    # Create summary message
    recommended_order = prioritization_result["recommended_execution_order"]
    if len(recommended_order) > 3:
        order_summary = f"{', '.join(recommended_order[:3])}..."
    else:
        order_summary = ", ".join(recommended_order) if recommended_order else "none"

    message = create_agent_message(
        content=(
            f"PM processing complete.\n"
            f"- Stories created: {output.story_count}\n"
            f"- Unprocessed requirements: {len(unprocessed_reqs)}\n"
            f"- Escalations to analyst: {len(output.escalations_to_analyst)}\n"
            f"- Quick wins: {len(prioritization_result['quick_wins'])}\n"
            f"- Recommended order: {order_summary}"
        ),
        agent="pm",
    )

    logger.info(
        "pm_node_completed",
        story_count=output.story_count,
        unprocessed_count=len(unprocessed_reqs),
        quick_win_count=len(prioritization_result["quick_wins"]),
    )

    return {
        "messages": [message],
        "decisions": [decision],
        "pm_output": output.to_dict(),
        "prioritization_result": prioritization_result,
    }
