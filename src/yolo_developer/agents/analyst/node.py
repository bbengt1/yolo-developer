"""Analyst agent node for LangGraph orchestration (Story 5.1, 5.2).

This module provides the analyst_node function that integrates with the
LangGraph orchestration workflow. The Analyst agent crystallizes requirements
from seed content, identifies gaps, and flags contradictions.

Key Concepts:
- **YoloState Input**: Receives state as TypedDict, not Pydantic
- **Immutable Updates**: Returns state update dict, never mutates input
- **Async I/O**: All LLM calls use async/await
- **Structured Logging**: Uses structlog for audit trail

Example:
    >>> from yolo_developer.agents.analyst import analyst_node
    >>> from yolo_developer.orchestrator.state import YoloState
    >>>
    >>> state: YoloState = {
    ...     "messages": [HumanMessage(content="Build a todo app")],
    ...     "current_agent": "analyst",
    ...     "handoff_context": None,
    ...     "decisions": [],
    ... }
    >>> result = await analyst_node(state)
    >>> result["messages"]  # New messages to append
    [AIMessage(...)]

Architecture Note:
    Per ADR-005, this node follows the LangGraph pattern of receiving
    full state and returning only the updates to apply.
"""

from __future__ import annotations

import json
import re
from typing import Any

import structlog
from langchain_core.messages import BaseMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.agents.analyst.types import AnalystOutput, CrystallizedRequirement
from yolo_developer.gates import quality_gate
from yolo_developer.orchestrator.context import Decision
from yolo_developer.orchestrator.state import YoloState, create_agent_message

logger = structlog.get_logger(__name__)

# Flag to enable/disable actual LLM calls (for testing)
_USE_LLM: bool = False

# Vague terms to detect in requirements (Story 5.2)
# These patterns indicate ambiguity that needs crystallization
VAGUE_TERMS: frozenset[str] = frozenset([
    # Quantifier vagueness
    "fast", "quick", "slow", "efficient", "performant",
    "scalable", "responsive", "real-time",
    # Ease vagueness
    "easy", "simple", "straightforward", "intuitive",
    "user-friendly", "seamless",
    # Certainty vagueness
    "should", "might", "could", "may", "possibly",
    "probably", "maybe", "sometimes",
    # Scope vagueness
    "etc", "and so on", "and more", "various", "multiple",
    "several", "many", "few", "some",
    # Quality vagueness
    "good", "better", "best", "nice", "beautiful",
    "clean", "modern", "robust",
])


def _detect_vague_terms(text: str) -> set[str]:
    """Detect vague terms in requirement text.

    Scans the input text for common vague terms that indicate
    ambiguity needing crystallization. Detection is case-insensitive.

    Args:
        text: The requirement text to analyze.

    Returns:
        Set of detected vague terms (lowercase). Empty set if none found.

    Example:
        >>> _detect_vague_terms("The system should be fast")
        {'should', 'fast'}
        >>> _detect_vague_terms("Response time < 200ms")
        set()
    """
    if not text:
        return set()

    text_lower = text.lower()
    detected: set[str] = set()

    for term in VAGUE_TERMS:
        # Check for the term as a word (not substring of another word)
        # Handle hyphenated terms specially
        if "-" in term:
            if term in text_lower:
                detected.add(term)
        else:
            # Use simple word boundary check
            # Check if term appears surrounded by non-alphanumeric chars
            pattern = rf"\b{re.escape(term)}\b"
            if re.search(pattern, text_lower):
                detected.add(term)

    return detected


@quality_gate("testability", blocking=True)
async def analyst_node(state: YoloState) -> dict[str, Any]:
    """Analyst agent node for requirement crystallization.

    Receives seed requirements from state messages and produces
    crystallized, categorized requirements with testability assessment.

    This function follows the LangGraph node pattern:
    - Receives full state as YoloState TypedDict
    - Returns only the state updates (not full state)
    - Never mutates the input state

    Args:
        state: Current orchestration state with accumulated messages.

    Returns:
        State update dict with:
        - messages: List of new messages to append
        - decisions: List of new decisions to append
        Never includes current_agent (handoff manages that).

    Example:
        >>> state: YoloState = {
        ...     "messages": [HumanMessage(content="Build an app")],
        ...     "current_agent": "analyst",
        ...     "handoff_context": None,
        ...     "decisions": [],
        ... }
        >>> result = await analyst_node(state)
        >>> "messages" in result
        True
    """
    logger.info(
        "analyst_node_start",
        current_agent=state.get("current_agent"),
        message_count=len(state.get("messages", [])),
    )

    # Extract seed content from messages
    seed_content = _extract_seed_from_messages(state.get("messages", []))

    # Process requirements using LLM
    output = await _crystallize_requirements(seed_content)

    # Create decision record
    decision = Decision(
        agent="analyst",
        summary=f"Crystallized {len(output.requirements)} requirements",
        rationale=(
            f"Analyzed seed content and extracted structured requirements. "
            f"Found {len(output.identified_gaps)} gaps and "
            f"{len(output.contradictions)} contradictions."
        ),
        related_artifacts=tuple(r.id for r in output.requirements),
    )

    # Create output message with analyst attribution
    message = create_agent_message(
        content=(
            f"Analysis complete: {len(output.requirements)} requirements crystallized. "
            f"Gaps identified: {len(output.identified_gaps)}. "
            f"Contradictions: {len(output.contradictions)}."
        ),
        agent="analyst",
        metadata={"output": output.to_dict()},
    )

    logger.info(
        "analyst_node_complete",
        requirement_count=len(output.requirements),
        gaps_count=len(output.identified_gaps),
        contradictions_count=len(output.contradictions),
    )

    # Return ONLY the updates, not full state
    return {
        "messages": [message],
        "decisions": [decision],
    }


def _extract_seed_from_messages(messages: list[BaseMessage]) -> str:
    """Extract seed content from accumulated messages.

    Looks through messages to find the seed document content that
    needs to be analyzed. Typically this is the first HumanMessage
    or content tagged as seed.

    Args:
        messages: List of accumulated messages in state.

    Returns:
        Concatenated seed content string.
    """
    if not messages:
        return ""

    # For now, concatenate all human message content as seed
    # In production, would look for specific seed tagging
    seed_parts: list[str] = []
    for msg in messages:
        # Check if it's a human message (has content attribute)
        if hasattr(msg, "content") and isinstance(msg.content, str):
            # Skip assistant/AI messages for seed extraction
            if msg.type == "human":
                seed_parts.append(msg.content)

    return "\n\n".join(seed_parts)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def _call_llm(prompt: str, system: str) -> str:
    """Call LLM with retry logic.

    Uses LiteLLM's async API for LLM calls with automatic retries
    on transient failures. Uses the cheap_model from config for
    routine analysis tasks.

    Args:
        prompt: The user prompt to send to the LLM.
        system: The system prompt defining the LLM's role.

    Returns:
        The LLM's response content as a string.

    Raises:
        Exception: If all retry attempts fail.
    """
    from litellm import acompletion

    from yolo_developer.config import load_config

    config = load_config()
    model = config.llm.cheap_model

    logger.info("calling_llm", model=model, prompt_length=len(prompt))

    response = await acompletion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content
    logger.debug("llm_response_received", response_length=len(content) if content else 0)

    return content or ""


def _parse_llm_response(response: str) -> AnalystOutput:
    """Parse LLM JSON response into AnalystOutput.

    Attempts to parse the LLM response as JSON and convert it to
    an AnalystOutput. Handles both legacy format (Story 5.1) and
    enhanced format with new fields (Story 5.2).

    Args:
        response: The raw LLM response string (expected to be JSON).

    Returns:
        AnalystOutput parsed from the response.
    """
    try:
        data = json.loads(response)

        requirements = tuple(
            CrystallizedRequirement(
                id=req.get("id", f"req-{i:03d}"),
                original_text=req.get("original_text", ""),
                refined_text=req.get("refined_text", ""),
                category=req.get("category", "functional"),
                testable=req.get("testable", True),
                # New fields from Story 5.2 (with backward-compatible defaults)
                scope_notes=req.get("scope_notes"),
                implementation_hints=tuple(req.get("implementation_hints", [])),
                confidence=float(req.get("confidence", 1.0)),
            )
            for i, req in enumerate(data.get("requirements", []), start=1)
        )

        return AnalystOutput(
            requirements=requirements,
            identified_gaps=tuple(data.get("identified_gaps", [])),
            contradictions=tuple(data.get("contradictions", [])),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("llm_response_parse_failed", error=str(e))
        return AnalystOutput(
            requirements=(),
            identified_gaps=("Failed to parse LLM response",),
            contradictions=(),
        )


async def _crystallize_requirements(seed_content: str) -> AnalystOutput:
    """Process seed content and extract crystallized requirements.

    When _USE_LLM is True, calls the LLM to analyze seed content
    and extract structured requirements. Otherwise, returns a
    placeholder output for testing that includes vague term detection.

    Args:
        seed_content: The raw seed document content.

    Returns:
        AnalystOutput with requirements, gaps, and contradictions.
    """
    logger.debug("crystallize_requirements_start", seed_length=len(seed_content))

    if not seed_content.strip():
        return AnalystOutput(
            requirements=(),
            identified_gaps=("No seed content provided for analysis",),
            contradictions=(),
        )

    # Detect vague terms for logging and placeholder generation
    vague_terms = _detect_vague_terms(seed_content)
    if vague_terms:
        logger.info(
            "vague_terms_detected",
            terms=list(vague_terms),
            count=len(vague_terms),
        )

    # Use LLM if enabled
    if _USE_LLM:
        from yolo_developer.agents.prompts.analyst import (
            ANALYST_SYSTEM_PROMPT,
            ANALYST_USER_PROMPT_TEMPLATE,
        )

        prompt = ANALYST_USER_PROMPT_TEMPLATE.format(seed_content=seed_content)
        response = await _call_llm(prompt, ANALYST_SYSTEM_PROMPT)
        output = _parse_llm_response(response)

        # Log transformation details for audit trail
        for req in output.requirements:
            logger.info(
                "requirement_crystallized",
                req_id=req.id,
                original_length=len(req.original_text),
                refined_length=len(req.refined_text),
                category=req.category,
                testable=req.testable,
                confidence=req.confidence,
                has_scope_notes=req.scope_notes is not None,
                hint_count=len(req.implementation_hints),
            )

        return output

    # Placeholder for testing (when LLM is disabled)
    # Generate confidence based on vague term detection
    confidence = 1.0 - (len(vague_terms) * 0.1) if vague_terms else 1.0
    confidence = max(0.3, min(1.0, confidence))  # Clamp to [0.3, 1.0]

    # Generate scope notes if vague terms detected
    scope_notes: str | None = None
    if vague_terms:
        scope_notes = f"Vague terms detected: {', '.join(sorted(vague_terms))}. Scope needs clarification."

    # Generate implementation hints based on content
    hints: tuple[str, ...] = ()
    seed_lower = seed_content.lower()
    if "api" in seed_lower or "endpoint" in seed_lower:
        hints = ("Consider async handlers for I/O operations",)
    elif "ui" in seed_lower or "interface" in seed_lower:
        hints = ("Follow component-based architecture",)
    elif "data" in seed_lower or "database" in seed_lower:
        hints = ("Use repository pattern for data access",)

    placeholder_req = CrystallizedRequirement(
        id="req-001",
        original_text=seed_content[:200] if len(seed_content) > 200 else seed_content,
        refined_text=f"Implement: {seed_content[:100]}..." if len(seed_content) > 100 else seed_content,
        category="functional",
        testable=True,
        scope_notes=scope_notes,
        implementation_hints=hints,
        confidence=confidence,
    )

    return AnalystOutput(
        requirements=(placeholder_req,),
        identified_gaps=(),
        contradictions=(),
    )
