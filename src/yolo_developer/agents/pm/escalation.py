"""Escalation to Analyst functionality for PM agent (Story 6.7).

This module provides the escalation capability that allows the PM agent
to escalate unclear requirements back to the Analyst for clarification.

Key Functions:
- _detect_ambiguity: Detect vague terms and missing criteria in requirements
- _generate_escalation_questions: Generate specific questions for ambiguities
- _create_escalation: Create an escalation object with full context
- check_for_escalation: Main function to check if a requirement needs escalation

Escalation is triggered when:
- Vague terms are detected (fast, easy, simple, intuitive, etc.)
- Missing success criteria for acceptance criteria generation
- Contradictory statements within a requirement
- Technical feasibility questions

Escalation does NOT trigger for:
- Requirements with clear scope and measurable criteria
- Minor clarifications that PM can infer

Example:
    >>> from yolo_developer.agents.pm.escalation import check_for_escalation
    >>>
    >>> req = {"id": "req-001", "refined_text": "The system should be fast", "category": "nf"}
    >>> escalation = check_for_escalation(req)
    >>> if escalation:
    ...     print(f"Escalation needed: {len(escalation['questions'])} questions")

Architecture Note:
    Per ADR-001, all data structures use TypedDict for internal state.
    Per ADR-005, escalation results are returned in node output, not mutated.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any

import structlog

from yolo_developer.agents.pm.types import (
    EscalationQuestion,
    EscalationReason,
    PMEscalation,
)

logger = structlog.get_logger(__name__)

# Counter for generating unique escalation IDs within a session.
# Note: This is module-level state. In concurrent scenarios, consider using
# threading.Lock or uuid4 for truly unique IDs. Current design assumes
# single-threaded execution per LangGraph node invocation.
_escalation_counter = 0

# =============================================================================
# Constants
# =============================================================================

# Vague terms that trigger escalation (configurable list per story Dev Notes)
VAGUE_TERMS: tuple[str, ...] = (
    # Speed/performance
    "fast",
    "quick",
    "rapid",
    # Ease of use
    "easy",
    "simple",
    "straightforward",
    # User experience
    "intuitive",
    "user-friendly",
    "clean",
    # Performance qualifiers
    "efficient",
    "performant",
    "optimized",
    # Architecture qualifiers
    "scalable",
    "flexible",
    "extensible",
    # Reliability qualifiers
    "robust",
    "reliable",
    "stable",
    # Generic qualifiers
    "good",
    "nice",
    "better",
)

# Contradictory patterns: pairs of terms that indicate conflicting requirements
CONTRADICTORY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("simple", "comprehensive"),
    ("fast", "thorough"),
    ("minimal", "complete"),
    ("lightweight", "feature-rich"),
    ("quick", "detailed"),
)

# Missing criteria indicators
MISSING_CRITERIA_PATTERNS: tuple[str, ...] = (
    "should be satisfied",
    "must be happy",
    "should be pleased",
    "users should like",
    "must be acceptable",
    "should work well",
    "must be good enough",
)

# Technical question patterns that need Analyst confirmation
TECHNICAL_QUESTION_PATTERNS: tuple[str, ...] = (
    "is it possible",
    "is this possible",
    "can we",
    "can the system",
    "technically feasible",
    "feasibility",
    "is there a way",
    "how would we",
    "how can we",
    "would it be possible",
)


# =============================================================================
# Task 2: Ambiguity Detection
# =============================================================================


def _detect_ambiguity(requirement: dict[str, Any]) -> list[str]:
    """Detect ambiguity in a requirement.

    Analyzes requirement text for vague terms, missing criteria, and
    contradictory statements that would prevent creating a clear story.

    Args:
        requirement: Dict with id, refined_text, and category fields.

    Returns:
        List of ambiguity descriptions. Empty list if no ambiguity detected.
        Each ambiguity is formatted as "type:detail" for processing.

    Example:
        >>> req = {"id": "req-001", "refined_text": "System should be fast"}
        >>> ambiguities = _detect_ambiguity(req)
        >>> ambiguities
        ['vague_term:fast']
    """
    ambiguities: list[str] = []
    text = requirement.get("refined_text", "").lower()

    if not text:
        logger.debug("escalation_empty_text", requirement_id=requirement.get("id"))
        return []

    # Check for vague terms
    for term in VAGUE_TERMS:
        # Match whole words only to avoid false positives
        # e.g., "fast" should not match "breakfast"
        if re.search(rf"\b{re.escape(term)}\b", text):
            ambiguities.append(f"vague_term:{term}")
            logger.debug(
                "escalation_vague_term_detected",
                requirement_id=requirement.get("id"),
                term=term,
            )

    # Check for contradictory patterns (e.g., "simple but comprehensive")
    for term_a, term_b in CONTRADICTORY_PATTERNS:
        if term_a in text and term_b in text:
            # Check for explicit "but" or "and" connecting them
            if " but " in text or " yet " in text or " and " in text:
                ambiguities.append(f"contradictory:{term_a} vs {term_b}")
                logger.debug(
                    "escalation_contradiction_detected",
                    requirement_id=requirement.get("id"),
                    term_a=term_a,
                    term_b=term_b,
                )

    # Check for missing measurable criteria
    for pattern in MISSING_CRITERIA_PATTERNS:
        if pattern in text:
            ambiguities.append("missing_criteria")
            logger.debug(
                "escalation_missing_criteria_detected",
                requirement_id=requirement.get("id"),
                pattern=pattern,
            )
            break  # Only flag once

    # Check for technical questions needing Analyst confirmation
    for pattern in TECHNICAL_QUESTION_PATTERNS:
        if pattern in text:
            ambiguities.append(f"technical_question:{pattern}")
            logger.debug(
                "escalation_technical_question_detected",
                requirement_id=requirement.get("id"),
                pattern=pattern,
            )
            break  # Only flag once per requirement

    logger.info(
        "escalation_ambiguity_detection_complete",
        requirement_id=requirement.get("id"),
        ambiguity_count=len(ambiguities),
    )

    return ambiguities


# =============================================================================
# Task 3: Question Generation
# =============================================================================


def _generate_escalation_questions(
    requirement: dict[str, Any],
    ambiguities: list[str],
) -> list[EscalationQuestion]:
    """Generate specific questions for each ambiguity.

    Creates actionable questions targeting the exact ambiguity to help
    the Analyst provide clarification.

    Args:
        requirement: Dict with id, refined_text, and category fields.
        ambiguities: List of ambiguity strings from _detect_ambiguity.

    Returns:
        List of EscalationQuestion objects with specific questions.

    Example:
        >>> req = {"id": "req-001", "refined_text": "System should be fast"}
        >>> questions = _generate_escalation_questions(req, ["vague_term:fast"])
        >>> questions[0]["question_text"]
        "What specific metric or behavior defines 'fast' for this requirement?"
    """
    questions: list[EscalationQuestion] = []
    req_id = requirement.get("id", "unknown")
    req_text = requirement.get("refined_text", "")

    for ambiguity in ambiguities:
        if ambiguity.startswith("vague_term:"):
            term = ambiguity.split(":", 1)[1]
            question: EscalationQuestion = {
                "question_text": f"What specific metric or behavior defines '{term}' for this requirement?",
                "source_requirement_id": req_id,
                "ambiguity_type": "vague_term",
                "context": f"Requirement mentions '{term}' without concrete definition: {req_text[:100]}...",
            }
            questions.append(question)

        elif ambiguity.startswith("contradictory:"):
            conflict = ambiguity.split(":", 1)[1]
            terms = conflict.split(" vs ")
            question = {
                "question_text": f"The requirement states both '{terms[0]}' and '{terms[1]}'. Which takes priority or how should they be balanced?",
                "source_requirement_id": req_id,
                "ambiguity_type": "contradictory",
                "context": f"Potentially conflicting terms in requirement: {req_text[:100]}...",
            }
            questions.append(question)

        elif ambiguity == "missing_criteria":
            question = {
                "question_text": "What are the concrete success criteria for this requirement? How can we measure when it's met?",
                "source_requirement_id": req_id,
                "ambiguity_type": "missing_criteria",
                "context": f"Requirement lacks measurable acceptance criteria: {req_text[:100]}...",
            }
            questions.append(question)

        elif ambiguity.startswith("technical_question:"):
            pattern = ambiguity.split(":", 1)[1]
            question = {
                "question_text": f"Is this technically feasible? The requirement asks '{pattern}' - please confirm approach or constraints.",
                "source_requirement_id": req_id,
                "ambiguity_type": "technical_question",
                "context": f"Requirement contains technical question needing confirmation: {req_text[:100]}...",
            }
            questions.append(question)

    logger.info(
        "escalation_questions_generated",
        requirement_id=req_id,
        question_count=len(questions),
    )

    return questions


# =============================================================================
# Task 4: Escalation Creation
# =============================================================================


def _create_escalation(
    requirement: dict[str, Any],
    questions: list[EscalationQuestion],
    reason: EscalationReason,
) -> PMEscalation:
    """Create an escalation object with full context.

    Args:
        requirement: Dict with id, refined_text, and category fields.
        questions: List of EscalationQuestion objects.
        reason: The primary escalation reason category.

    Returns:
        PMEscalation object with unique ID and full context.

    Example:
        >>> req = {"id": "req-001", "refined_text": "Test", "category": "functional"}
        >>> questions = [{"question_text": "Test?", ...}]
        >>> esc = _create_escalation(req, questions, "ambiguous_terms")
        >>> esc["id"]  # e.g., "esc-1704412345-001"
    """
    global _escalation_counter
    _escalation_counter += 1

    # Generate unique ID in format: esc-{timestamp}-{counter}
    timestamp = int(time.time())
    escalation_id = f"esc-{timestamp}-{_escalation_counter:03d}"

    escalation: PMEscalation = {
        "id": escalation_id,
        "source_agent": "pm",
        "target_agent": "analyst",
        "requirement_id": requirement.get("id", "unknown"),
        "questions": questions,
        # partial_work: Currently None as escalation check happens BEFORE story
        # transformation. Future enhancement: pass partial Story if escalation
        # detected mid-transformation. Per AC2, field exists for extensibility.
        "partial_work": None,
        "reason": reason,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "escalation_created",
        escalation_id=escalation_id,
        requirement_id=requirement.get("id"),
        reason=reason,
        question_count=len(questions),
    )

    return escalation


# =============================================================================
# Task 5: Main Escalation Check Function
# =============================================================================


def check_for_escalation(requirement: dict[str, Any]) -> PMEscalation | None:
    """Check if a requirement needs escalation to Analyst.

    This is the main entry point for escalation detection. It orchestrates
    ambiguity detection, question generation, and escalation creation.

    Args:
        requirement: Dict with id, refined_text, and category fields.

    Returns:
        PMEscalation if escalation is needed, None otherwise.

    Example:
        >>> req = {"id": "req-001", "refined_text": "System should be fast and easy"}
        >>> esc = check_for_escalation(req)
        >>> if esc:
        ...     print(f"Escalation: {len(esc['questions'])} questions")
    """
    req_id = requirement.get("id", "unknown")

    logger.debug(
        "escalation_check_started",
        requirement_id=req_id,
    )

    # Step 1: Detect ambiguities
    ambiguities = _detect_ambiguity(requirement)

    # Step 2: If no ambiguities, no escalation needed
    if not ambiguities:
        logger.debug(
            "escalation_not_needed",
            requirement_id=req_id,
        )
        return None

    # Step 3: Generate questions for each ambiguity
    questions = _generate_escalation_questions(requirement, ambiguities)

    # Step 4: Determine primary reason
    # Priority: contradictory > technical_question > missing_criteria > ambiguous_terms
    reason: EscalationReason = "ambiguous_terms"
    for amb in ambiguities:
        if amb.startswith("contradictory:"):
            reason = "contradictory"
            break
        elif amb.startswith("technical_question:"):
            reason = "technical_question"
        elif amb == "missing_criteria":
            reason = "missing_criteria"

    # Step 5: Create and return escalation
    escalation = _create_escalation(requirement, questions, reason)

    logger.info(
        "escalation_check_complete",
        requirement_id=req_id,
        escalation_needed=True,
        reason=reason,
        question_count=len(questions),
    )

    return escalation
