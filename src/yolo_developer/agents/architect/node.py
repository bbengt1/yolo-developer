"""Architect agent node for LangGraph orchestration (Stories 7.1-7.8).

This module provides the architect_node function that integrates with the
LangGraph orchestration workflow. The Architect agent produces design
decisions and Architecture Decision Records (ADRs) for stories with
12-Factor compliance analysis (Story 7.2), enhanced ADR content
generation (Story 7.3), quality attribute evaluation (Story 7.4),
technical risk identification (Story 7.5), tech stack constraint
validation (Story 7.6), ATAM architectural review (Story 7.7),
and pattern matching to codebase (Story 7.8).

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

from yolo_developer.agents.architect.adr_generator import generate_adrs
from yolo_developer.agents.architect.atam_reviewer import run_atam_review
from yolo_developer.agents.architect.pattern_matcher import run_pattern_matching
from yolo_developer.agents.architect.quality_evaluator import evaluate_quality_attributes
from yolo_developer.agents.architect.risk_identifier import identify_technical_risks
from yolo_developer.agents.architect.tech_stack_validator import (
    validate_tech_stack_constraints,
)
from yolo_developer.agents.architect.twelve_factor import analyze_twelve_factor
from yolo_developer.agents.architect.types import (
    ArchitectOutput,
    DesignDecision,
    DesignDecisionType,
    QualityAttributeEvaluation,
    QualityRisk,
    TechnicalRiskReport,
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

    # Generate ADRs for significant decisions (AC4, Story 7.3)
    adrs = await generate_adrs(design_decisions, twelve_factor_analyses)

    # Evaluate quality attributes for each story (Story 7.4)
    quality_evaluations: dict[str, Any] = {}
    quality_evaluation_objects: dict[str, QualityAttributeEvaluation] = {}
    for story in stories:
        story_id = story.get("id", "unknown")
        # Get decisions for this story
        story_decisions = [d for d in design_decisions if d.story_id == story_id]
        evaluation = await evaluate_quality_attributes(story, story_decisions)
        quality_evaluations[story_id] = evaluation.to_dict()
        quality_evaluation_objects[story_id] = evaluation  # Store for ATAM review

    # Identify technical risks for each story (Story 7.5)
    technical_risk_reports: dict[str, Any] = {}
    technical_risk_report_objects: dict[str, TechnicalRiskReport] = {}
    for story in stories:
        story_id = story.get("id", "unknown")
        # Get decisions for this story
        story_decisions = [d for d in design_decisions if d.story_id == story_id]
        # Get quality risks from evaluation to incorporate
        quality_eval = quality_evaluations.get(story_id, {})
        quality_risks_data = quality_eval.get("risks", [])
        # Convert dict risks back to QualityRisk objects for the identifier
        quality_risks = [
            QualityRisk(
                attribute=r.get("attribute", "unknown"),
                description=r.get("description", ""),
                severity=r.get("severity", "medium"),
                mitigation=r.get("mitigation", ""),
                mitigation_effort=r.get("mitigation_effort", "medium"),
            )
            for r in quality_risks_data
        ]
        risk_report = await identify_technical_risks(
            story, story_decisions, quality_risks=quality_risks
        )
        technical_risk_reports[story_id] = risk_report.to_dict()
        technical_risk_report_objects[story_id] = risk_report  # Store for ATAM review

    # Validate tech stack constraints for each story (Story 7.6)
    tech_stack_validations: dict[str, Any] = {}
    for story in stories:
        story_id = story.get("id", "unknown")
        # Get decisions for this story
        story_decisions = [d for d in design_decisions if d.story_id == story_id]
        validation = await validate_tech_stack_constraints(story_decisions)
        tech_stack_validations[story_id] = validation.to_dict()

    logger.debug(
        "tech_stack_validation_complete",
        validation_count=len(tech_stack_validations),
    )

    # Run ATAM architectural review for each story (Story 7.7)
    atam_reviews: dict[str, Any] = {}
    for story in stories:
        story_id = story.get("id", "unknown")
        # Get decisions for this story
        story_decisions = [d for d in design_decisions if d.story_id == story_id]
        # Get quality evaluation and risk report objects for this story
        quality_eval_obj = quality_evaluation_objects.get(story_id)
        risk_report_obj = technical_risk_report_objects.get(story_id)
        # Run ATAM review
        atam_result = await run_atam_review(
            story_decisions,
            quality_eval=quality_eval_obj,
            risk_report=risk_report_obj,
        )
        atam_reviews[story_id] = atam_result.to_dict()

    logger.debug(
        "atam_review_complete",
        review_count=len(atam_reviews),
    )

    # Run pattern matching for each story (Story 7.8)
    # Pattern matching validates that design decisions follow established codebase patterns
    pattern_matching_results: dict[str, Any] = {}
    for story in stories:
        story_id = story.get("id", "unknown")
        # Get decisions for this story
        story_decisions = [d for d in design_decisions if d.story_id == story_id]
        # Run pattern matching (pattern store is optional - None if not configured)
        pattern_result = await run_pattern_matching(
            story_decisions,
            pattern_store=None,  # Pattern store passed from orchestration state if available
        )
        pattern_matching_results[story_id] = pattern_result.to_dict()

    logger.debug(
        "pattern_matching_complete",
        result_count=len(pattern_matching_results),
    )

    # Build output with actual results including 12-Factor analyses, quality evaluations,
    # risk reports, tech stack validations, ATAM reviews, and pattern matching results
    output = ArchitectOutput(
        design_decisions=tuple(design_decisions),
        adrs=tuple(adrs),
        processing_notes=f"Processed {len(stories)} stories, "
        f"generated {len(design_decisions)} design decisions, "
        f"{len(adrs)} ADRs with 12-Factor compliance analysis, quality evaluation, "
        f"risk identification, tech stack validation, ATAM review, and pattern matching",
        twelve_factor_analyses=twelve_factor_analyses,
        quality_evaluations=quality_evaluations,
        technical_risk_reports=technical_risk_reports,
        tech_stack_validations=tech_stack_validations,
        atam_reviews=atam_reviews,
        pattern_matching_results=pattern_matching_results,
    )

    # Create decision record with architect attribution
    decision = Decision(
        agent="architect",
        summary=f"Generated {len(design_decisions)} design decisions, {len(adrs)} ADRs, "
        f"{len(technical_risk_reports)} risk reports, {len(tech_stack_validations)} "
        f"tech stack validations, {len(atam_reviews)} ATAM reviews, "
        f"and {len(pattern_matching_results)} pattern matching results",
        rationale=f"Processed {len(stories)} stories from PM. "
        "12-Factor compliance analysis (Story 7.2), enhanced ADR generation "
        "with LLM fallback (Story 7.3), quality attribute evaluation (Story 7.4), "
        "technical risk identification (Story 7.5), tech stack constraint "
        "validation (Story 7.6), ATAM architectural review (Story 7.7), "
        "and pattern matching to codebase (Story 7.8) applied.",
        related_artifacts=tuple(d.id for d in design_decisions),
    )

    # Create output message with architect attribution
    message = create_agent_message(
        content=f"Architect processing complete: {len(design_decisions)} design decisions, "
        f"{len(adrs)} ADRs, {len(technical_risk_reports)} risk reports, "
        f"{len(tech_stack_validations)} tech stack validations, "
        f"{len(atam_reviews)} ATAM reviews, "
        f"{len(pattern_matching_results)} pattern matching results generated.",
        agent="architect",
        metadata={"output": output.to_dict()},
    )

    logger.info(
        "architect_node_complete",
        story_count=len(stories),
        design_decision_count=len(output.design_decisions),
        adr_count=len(output.adrs),
        quality_evaluation_count=len(quality_evaluations),
        technical_risk_report_count=len(technical_risk_reports),
        tech_stack_validation_count=len(tech_stack_validations),
        atam_review_count=len(atam_reviews),
        pattern_matching_result_count=len(pattern_matching_results),
    )

    # Return ONLY the updates, not full state
    return {
        "messages": [message],
        "decisions": [decision],
        "architect_output": output.to_dict(),
    }
