"""PM agent node for LangGraph orchestration (Story 6.1).

This module provides the pm_node function that integrates with the
LangGraph orchestration workflow. The PM agent transforms crystallized
requirements into user stories with acceptance criteria.

Key Concepts:
- **YoloState Input**: Receives state as TypedDict, not Pydantic
- **Immutable Updates**: Returns state update dict, never mutates input
- **Async I/O**: All LLM calls use async/await
- **Structured Logging**: Uses structlog for audit trail

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

from yolo_developer.agents.pm.types import (
    AcceptanceCriterion,
    PMOutput,
    Story,
    StoryPriority,
    StoryStatus,
)
from yolo_developer.gates import quality_gate
from yolo_developer.orchestrator.context import Decision
from yolo_developer.orchestrator.state import YoloState, create_agent_message

logger = structlog.get_logger(__name__)


def _generate_acceptance_criteria(
    requirement_id: str,
    requirement_text: str,
) -> tuple[AcceptanceCriterion, ...]:
    """Generate acceptance criteria from requirement text.

    Creates a basic Given/When/Then format acceptance criterion from
    the requirement. This is a stub implementation for MVP; Story 6.2
    will add LLM-powered generation.

    Args:
        requirement_id: ID of the source requirement.
        requirement_text: Text of the requirement.

    Returns:
        Tuple containing a single AcceptanceCriterion.

    Example:
        >>> acs = _generate_acceptance_criteria("req-001", "User can login")
        >>> acs[0].id
        'AC1'
    """
    # Stub implementation - creates basic AC structure
    ac = AcceptanceCriterion(
        id="AC1",
        given=f"the system is ready to process {requirement_id}",
        when="the feature is used as specified",
        then=f"the requirement is satisfied: {requirement_text[:50]}...",
        and_clauses=(),
    )
    return (ac,)


def _transform_requirements_to_stories(
    requirements: list[dict[str, Any]],
) -> tuple[tuple[Story, ...], tuple[str, ...]]:
    """Transform crystallized requirements into user stories.

    Maps requirements to story format with acceptance criteria.
    This is a stub implementation for MVP; Story 6.2 will add
    LLM-powered transformation.

    Args:
        requirements: List of crystallized requirement dicts from analyst.

    Returns:
        Tuple of (stories, unprocessed_requirement_ids).

    Example:
        >>> reqs = [{"id": "req-001", "refined_text": "User login"}]
        >>> stories, unprocessed = _transform_requirements_to_stories(reqs)
        >>> len(stories)
        1
    """
    stories: list[Story] = []
    unprocessed: list[str] = []
    story_counter = 0  # Separate counter for sequential story IDs

    for i, req in enumerate(requirements):
        req_id = req.get("id", f"req-{i}")
        refined_text = req.get("refined_text", req.get("original_text", ""))
        category = req.get("category", "functional")

        # Skip constraint requirements for story transformation
        # They become constraints on functional stories
        if category == "constraint":
            unprocessed.append(req_id)
            continue

        story_counter += 1

        # Generate acceptance criteria
        acs = _generate_acceptance_criteria(req_id, refined_text)

        # Determine priority based on category
        if category == "functional":
            priority = StoryPriority.HIGH
        else:
            priority = StoryPriority.MEDIUM

        # Create story with stub data
        # Note: role is hardcoded to "user" in this stub; Story 6.2 will extract
        # specific roles from requirement context using LLM analysis
        story = Story(
            id=f"story-{story_counter:03d}",
            title=refined_text[:50] if refined_text else f"Story for {req_id}",
            role="user",
            action=refined_text or f"complete requirement {req_id}",
            benefit="the system meets the specified requirement",
            acceptance_criteria=acs,
            priority=priority,
            status=StoryStatus.DRAFT,
            source_requirements=(req_id,),
            dependencies=(),
            estimated_complexity="M",
        )
        stories.append(story)

    return tuple(stories), tuple(unprocessed)


# Note: ac_measurability gate evaluator registration is implemented in Story 6.3
# Until then, the gate will pass through (no evaluator = no blocking)
@quality_gate("ac_measurability", blocking=True)
async def pm_node(state: YoloState) -> dict[str, Any]:
    """Transform crystallized requirements into user stories.

    This LangGraph node receives analyst output from state and produces
    user stories with acceptance criteria. It is decorated with the
    ac_measurability quality gate to ensure AC are testable.

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
    # AC1: Extract gaps and contradictions for context (used in future story for decisions)
    gaps: list[dict[str, Any]] = analyst_output.get("gaps", [])
    contradictions: list[dict[str, Any]] = analyst_output.get("contradictions", [])

    # Note: Project configuration extraction deferred to Story 6.2 (LLM integration)
    # Config will be needed for LLM model selection and processing parameters

    logger.info(
        "pm_node_received_input",
        requirement_count=len(requirements),
        escalation_count=len(escalations_from_analyst),
        gap_count=len(gaps),
        contradiction_count=len(contradictions),
    )

    # Transform requirements to stories
    stories, unprocessed_reqs = _transform_requirements_to_stories(requirements)

    # Create PM output
    output = PMOutput(
        stories=stories,
        unprocessed_requirements=unprocessed_reqs,
        escalations_to_analyst=(),  # No escalations back in stub
        processing_notes=f"Transformed {len(requirements)} requirements into {len(stories)} stories",
    )

    # Create decision record for audit trail
    decision = Decision(
        agent="pm",
        summary=f"Created {output.story_count} stories from {len(requirements)} requirements",
        rationale="Requirements transformed using standard story template",
        related_artifacts=tuple(s.id for s in stories),
    )

    # Create summary message
    message = create_agent_message(
        content=(
            f"PM processing complete.\n"
            f"- Stories created: {output.story_count}\n"
            f"- Unprocessed requirements: {len(unprocessed_reqs)}\n"
            f"- Escalations to analyst: {len(output.escalations_to_analyst)}"
        ),
        agent="pm",
    )

    logger.info(
        "pm_node_completed",
        story_count=output.story_count,
        unprocessed_count=len(unprocessed_reqs),
    )

    return {
        "messages": [message],
        "decisions": [decision],
        "pm_output": output.to_dict(),
    }
