"""Architect agent node for LangGraph orchestration (Story 7.1).

This module provides the architect_node function that integrates with the
LangGraph orchestration workflow. The Architect agent produces design
decisions and Architecture Decision Records (ADRs) for stories.

Key Concepts:
- **YoloState Input**: Receives state as TypedDict, not Pydantic
- **Immutable Updates**: Returns state update dict, never mutates input
- **Async I/O**: All LLM calls use async/await
- **Structured Logging**: Uses structlog for audit trail

Example:
    >>> from yolo_developer.agents.architect import architect_node
    >>> from yolo_developer.orchestrator.state import YoloState
    >>>
    >>> state: YoloState = {
    ...     "messages": [...],
    ...     "current_agent": "architect",
    ...     "handoff_context": None,
    ...     "decisions": [],
    ... }
    >>> result = await architect_node(state)
    >>> result["messages"]  # New messages to append
    [AIMessage(...)]

Architecture Note:
    Per ADR-005, this node follows the LangGraph pattern of receiving
    full state and returning only the updates to apply.
"""

from __future__ import annotations

import time
from typing import Any, cast

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.agents.architect.twelve_factor import analyze_twelve_factor
from yolo_developer.agents.architect.types import (
    ADR,
    ArchitectOutput,
    DesignDecision,
    DesignDecisionType,
)
from yolo_developer.gates import quality_gate
from yolo_developer.orchestrator.context import Decision
from yolo_developer.orchestrator.state import YoloState, create_agent_message

logger = structlog.get_logger(__name__)


def _extract_stories_from_state(state: YoloState) -> list[dict[str, Any]]:
    """Extract stories from orchestration state.

    Stories can be present in state in two ways:
    1. Directly in pm_output key (preferred)
    2. In message metadata from PM agent messages

    Args:
        state: Current orchestration state.

    Returns:
        List of story dictionaries. Empty list if no stories found.

    Example:
        >>> state = {"pm_output": {"stories": [...]}}
        >>> stories = _extract_stories_from_state(state)
        >>> len(stories)
        2
    """
    # First, try to extract from pm_output (direct state key)
    pm_output = state.get("pm_output")
    if pm_output and isinstance(pm_output, dict):
        stories = pm_output.get("stories", [])
        if stories:
            story_list = cast(list[dict[str, Any]], stories)
            logger.info(
                "stories_extracted_from_pm_output",
                story_count=len(story_list),
                story_ids=[s.get("id") for s in story_list],
            )
            return story_list

    # Fallback: Extract from message metadata (find latest PM message)
    messages = state.get("messages", [])
    pm_messages = []

    for msg in messages:
        # Check if message has additional_kwargs with agent="pm"
        if hasattr(msg, "additional_kwargs"):
            kwargs = msg.additional_kwargs
            if kwargs.get("agent") == "pm":
                pm_messages.append(msg)

    # Get stories from the latest PM message
    if pm_messages:
        latest_pm_msg = pm_messages[-1]
        output = latest_pm_msg.additional_kwargs.get("output", {})
        stories = output.get("stories", [])
        if stories:
            story_list = cast(list[dict[str, Any]], stories)
            logger.info(
                "stories_extracted_from_message",
                story_count=len(story_list),
                story_ids=[s.get("id") for s in story_list],
            )
            return story_list

    logger.debug("no_stories_found_in_state")
    return []


# Counter for generating unique decision IDs within a session
# Note: Single-threaded execution assumed per LangGraph node invocation
_decision_counter = 0


async def _generate_design_decisions(
    stories: list[dict[str, Any]],
) -> tuple[list[DesignDecision], dict[str, Any]]:
    """Generate design decisions for stories with 12-Factor analysis.

    Creates a DesignDecision for each story that requires architectural
    consideration, including 12-Factor App compliance analysis in the
    rationale (Story 7.2).

    Args:
        stories: List of story dictionaries from PM output.

    Returns:
        Tuple of (list of DesignDecision objects, dict of twelve-factor analyses).

    Example:
        >>> stories = [{"id": "story-001", "title": "User Auth"}]
        >>> decisions, analyses = await _generate_design_decisions(stories)
        >>> len(decisions)
        1
    """
    global _decision_counter

    if not stories:
        logger.debug("no_stories_for_design_decisions")
        return [], {}

    decisions: list[DesignDecision] = []
    twelve_factor_analyses: dict[str, Any] = {}
    timestamp = int(time.time())

    for story in stories:
        _decision_counter += 1
        story_id = story.get("id", "unknown")
        story_title = story.get("title", "Untitled Story")

        # Perform 12-Factor compliance analysis (Story 7.2)
        twelve_factor_result = await analyze_twelve_factor(story)
        twelve_factor_analyses[story_id] = twelve_factor_result.to_dict()

        # Determine decision type based on story content
        decision_type: DesignDecisionType = _infer_decision_type(story)

        # Build rationale with 12-Factor compliance information
        compliance_pct = int(twelve_factor_result.overall_compliance * 100)
        recommendations = twelve_factor_result.recommendations
        recommendation_text = (
            f" Recommendations: {', '.join(recommendations[:2])}"
            if recommendations
            else ""
        )
        rationale = (
            f"12-Factor compliance: {compliance_pct}%.{recommendation_text}"
        )

        decision = DesignDecision(
            id=f"design-{timestamp}-{_decision_counter:03d}",
            story_id=story_id,
            decision_type=decision_type,
            description=f"Design approach for: {story_title}",
            rationale=rationale,
            alternatives_considered=("Alternative A", "Alternative B"),
        )
        decisions.append(decision)

        logger.debug(
            "design_decision_generated",
            decision_id=decision.id,
            story_id=story_id,
            decision_type=decision_type,
            twelve_factor_compliance=compliance_pct,
        )

    logger.info(
        "design_decisions_generated",
        count=len(decisions),
        story_ids=[d.story_id for d in decisions],
    )

    return decisions, twelve_factor_analyses


def _infer_decision_type(story: dict[str, Any]) -> DesignDecisionType:
    """Infer the type of design decision based on story content.

    Uses keyword-based heuristics to classify design decisions.
    The 12-Factor compliance analysis (Story 7.2) provides additional
    architectural guidance in the rationale.

    Args:
        story: Story dictionary with title and other fields.

    Returns:
        Inferred DesignDecisionType.
    """
    title = story.get("title", "").lower()
    description = story.get("description", "").lower()
    text = f"{title} {description}"

    # Simple keyword matching (stub - full LLM in Story 7.2)
    if any(kw in text for kw in ["security", "auth", "permission", "encrypt"]):
        return "security"
    if any(kw in text for kw in ["database", "storage", "data", "schema"]):
        return "data"
    if any(kw in text for kw in ["api", "integrate", "external", "service"]):
        return "integration"
    if any(kw in text for kw in ["deploy", "infra", "container", "cloud"]):
        return "infrastructure"
    if any(kw in text for kw in ["framework", "library", "stack", "tool"]):
        return "technology"

    # Default to pattern for general architectural decisions
    return "pattern"


# Counter for generating unique ADR IDs within a session
_adr_counter = 0


def _generate_adrs(decisions: list[DesignDecision]) -> list[ADR]:
    """Generate Architecture Decision Records from design decisions.

    Creates ADRs for significant decisions (technology and pattern types).
    This is a stub implementation - full ADR generation will be implemented
    in Story 7.3.

    Args:
        decisions: List of DesignDecision objects.

    Returns:
        List of ADR objects for significant decisions.

    Example:
        >>> decisions = [DesignDecision(id="d-1", story_id="s-1", ...)]
        >>> adrs = _generate_adrs(decisions)
        >>> len(adrs)
        1
    """
    global _adr_counter

    if not decisions:
        logger.debug("no_decisions_for_adrs")
        return []

    adrs: list[ADR] = []
    # Generate ADRs for significant decisions (technology and pattern types)
    significant_types = {"technology", "pattern"}

    for decision in decisions:
        if decision.decision_type in significant_types:
            _adr_counter += 1

            adr = ADR(
                id=f"ADR-{_adr_counter:03d}",
                title=f"ADR for {decision.description}",
                status="proposed",
                context=f"Decision required for story {decision.story_id}: {decision.description}",
                decision=f"Selected approach: {decision.rationale}",
                consequences="Stub implementation - full consequences analysis in Story 7.3",
                story_ids=(decision.story_id,),
            )
            adrs.append(adr)

            logger.debug(
                "adr_generated",
                adr_id=adr.id,
                story_id=decision.story_id,
                decision_type=decision.decision_type,
            )

    logger.info(
        "adrs_generated",
        count=len(adrs),
        adr_ids=[a.id for a in adrs],
    )

    return adrs


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
@quality_gate("architecture_validation", blocking=False)
async def architect_node(state: YoloState) -> dict[str, Any]:
    """Architect agent node for design decisions and ADR generation.

    Receives stories from state and produces design decisions and
    Architecture Decision Records for each story requiring architectural
    decisions.

    This function follows the LangGraph node pattern:
    - Receives full state as YoloState TypedDict
    - Returns only the state updates (not full state)
    - Never mutates the input state
    - Uses tenacity for retry with exponential backoff (AC5)

    Args:
        state: Current orchestration state with stories from PM.

    Returns:
        State update dict with:
        - messages: List of new messages to append
        - decisions: List of new decisions to append
        - architect_output: Serialized ArchitectOutput
        Never includes current_agent (handoff manages that).

    Example:
        >>> state: YoloState = {
        ...     "messages": [...],
        ...     "current_agent": "architect",
        ...     "handoff_context": None,
        ...     "decisions": [],
        ... }
        >>> result = await architect_node(state)
        >>> "messages" in result
        True
    """
    logger.info(
        "architect_node_start",
        current_agent=state.get("current_agent"),
        message_count=len(state.get("messages", [])),
    )

    # Extract stories from state (AC2)
    stories = _extract_stories_from_state(state)

    # Generate design decisions for stories with 12-Factor analysis (AC3, Story 7.2)
    design_decisions, twelve_factor_analyses = await _generate_design_decisions(stories)

    # Generate ADRs for significant decisions (AC4)
    adrs = _generate_adrs(design_decisions)

    # Build output with actual results including 12-Factor analyses
    output = ArchitectOutput(
        design_decisions=tuple(design_decisions),
        adrs=tuple(adrs),
        processing_notes=f"Processed {len(stories)} stories, "
        f"generated {len(design_decisions)} design decisions, "
        f"{len(adrs)} ADRs with 12-Factor compliance analysis",
        twelve_factor_analyses=twelve_factor_analyses,
    )

    # Create decision record with architect attribution
    decision = Decision(
        agent="architect",
        summary=f"Generated {len(design_decisions)} design decisions and {len(adrs)} ADRs",
        rationale=f"Processed {len(stories)} stories from PM. "
        "Story 7.1 provides stub implementation with keyword-based inference. "
        "Full LLM-powered design generation in Stories 7.2-7.8.",
        related_artifacts=tuple(d.id for d in design_decisions),
    )

    # Create output message with architect attribution
    message = create_agent_message(
        content=f"Architect processing complete: {len(design_decisions)} design decisions, {len(adrs)} ADRs generated.",
        agent="architect",
        metadata={"output": output.to_dict()},
    )

    logger.info(
        "architect_node_complete",
        story_count=len(stories),
        design_decision_count=len(output.design_decisions),
        adr_count=len(output.adrs),
    )

    # Return ONLY the updates, not full state
    return {
        "messages": [message],
        "decisions": [decision],
        "architect_output": output.to_dict(),
    }
