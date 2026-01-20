"""Ambiguity detection types and functions for seed documents (Story 4.3).

This module provides data models and functions for detecting ambiguities
in natural language seed documents and generating resolution prompts:

- AmbiguityType: Enum for ambiguity categories
- AmbiguitySeverity: Enum for severity levels (LOW, MEDIUM, HIGH)
- Ambiguity: Dataclass representing a detected ambiguity
- ResolutionPrompt: Dataclass for clarification questions
- AmbiguityResult: Complete result from ambiguity detection
- Resolution: User response to an ambiguity
- SeedContext: Context including original and clarified content
- calculate_ambiguity_confidence: Scoring function for ambiguity impact
- detect_ambiguities: Main async function for LLM-based detection

Example:
    >>> from yolo_developer.seed.ambiguity import (
    ...     detect_ambiguities,
    ...     AmbiguityType,
    ...     AmbiguitySeverity,
    ... )
    >>>
    >>> # Detect ambiguities in seed content
    >>> result = await detect_ambiguities("Build a fast, scalable app")
    >>> if result.has_ambiguities:
    ...     for amb in result.ambiguities:
    ...         print(f"{amb.ambiguity_type.value}: {amb.description}")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import litellm
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)


class AmbiguityType(Enum):
    """Type of ambiguity detected in seed document.

    Categorizes ambiguities by their nature to help with
    resolution and clarification.

    Values:
        SCOPE: Unclear boundaries (e.g., "handle all edge cases")
        TECHNICAL: Vague technical requirements (e.g., "fast", "scalable")
        PRIORITY: Unclear importance (e.g., "nice to have", "should")
        DEPENDENCY: Unclear relationships (e.g., "integrate with the system")
        UNDEFINED: Missing critical details (e.g., no auth specified)
    """

    SCOPE = "scope"
    TECHNICAL = "technical"
    PRIORITY = "priority"
    DEPENDENCY = "dependency"
    UNDEFINED = "undefined"


class AmbiguitySeverity(Enum):
    """Severity level of detected ambiguity.

    Indicates how much the ambiguity impacts implementation clarity.

    Values:
        LOW: Minor clarification needed, doesn't block implementation
        MEDIUM: Causes confusion, may lead to incorrect assumptions
        HIGH: Blocks implementation, critical clarification required
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AnswerFormat(str, Enum):
    """Expected format for user responses to clarification questions.

    Categorizes the type of response expected to help with validation
    and user guidance.

    Values:
        BOOLEAN: Yes/No questions
        NUMERIC: Number input (possibly with range)
        CHOICE: Pick from suggestions
        FREE_TEXT: Open-ended text response
        DATE: Date/time input (YYYY-MM-DD format)
        LIST: Multiple items (comma-separated)

    Example:
        >>> format_type = AnswerFormat.NUMERIC
        >>> print(format_type.value)  # "numeric"
    """

    BOOLEAN = "boolean"
    NUMERIC = "numeric"
    CHOICE = "choice"
    FREE_TEXT = "free_text"
    DATE = "date"
    LIST = "list"


@dataclass(frozen=True)
class Ambiguity:
    """A detected ambiguity in a seed document.

    Represents a specific phrase or concept that is unclear and
    may need clarification before implementation.

    Attributes:
        ambiguity_type: The category of ambiguity
        severity: How severe the ambiguity is
        source_text: The exact ambiguous phrase from the document
        location: Line number or section where found
        description: Why this is ambiguous

    Example:
        >>> ambiguity = Ambiguity(
        ...     ambiguity_type=AmbiguityType.TECHNICAL,
        ...     severity=AmbiguitySeverity.HIGH,
        ...     source_text="fast response times",
        ...     location="line 5",
        ...     description="No specific time threshold defined",
        ... )
    """

    ambiguity_type: AmbiguityType
    severity: AmbiguitySeverity
    source_text: str
    location: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields, enums as string values.
        """
        return {
            "ambiguity_type": self.ambiguity_type.value,
            "severity": self.severity.value,
            "source_text": self.source_text,
            "location": self.location,
            "description": self.description,
        }


@dataclass(frozen=True)
class ResolutionPrompt:
    """A clarification prompt for an ambiguity.

    Contains the question to ask the user and suggested answers
    to help resolve an ambiguity. Enhanced in Story 4.4 with
    answer format and validation support.

    Attributes:
        question: The clarification question to ask
        suggestions: Tuple of suggested answers
        default: Default answer if user doesn't provide one (optional)
        answer_format: Expected response format (default: FREE_TEXT)
        format_hint: Human-readable format guidance (optional)
        validation_pattern: Regex pattern for validation (optional)

    Example:
        >>> prompt = ResolutionPrompt(
        ...     question="What response time is acceptable?",
        ...     suggestions=("< 100ms", "< 500ms", "< 1 second"),
        ...     default="< 500ms",
        ...     answer_format=AnswerFormat.CHOICE,
        ...     format_hint="Choose from the options above",
        ... )
    """

    question: str
    suggestions: tuple[str, ...]
    default: str | None = None
    answer_format: AnswerFormat = AnswerFormat.FREE_TEXT
    format_hint: str | None = None
    validation_pattern: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields, suggestions as list,
            answer_format as string value.
        """
        return {
            "question": self.question,
            "suggestions": list(self.suggestions),
            "default": self.default,
            "answer_format": self.answer_format.value,
            "format_hint": self.format_hint,
            "validation_pattern": self.validation_pattern,
        }


@dataclass(frozen=True)
class AmbiguityResult:
    """Complete result from ambiguity detection.

    Contains all detected ambiguities, confidence score, and
    resolution prompts for user interaction. Enhanced in Story 4.4
    with prioritization support.

    Attributes:
        ambiguities: Tuple of detected ambiguities
        overall_confidence: Confidence score (0.0-1.0) reflecting ambiguity impact
        resolution_prompts: Tuple of resolution prompts matching ambiguities

    Example:
        >>> result = AmbiguityResult(
        ...     ambiguities=(),
        ...     overall_confidence=1.0,
        ...     resolution_prompts=(),
        ... )
        >>> print(result.has_ambiguities)
        False
        >>> # Get prioritized ambiguities (Story 4.4)
        >>> sorted_ambs = result.prioritized_ambiguities
    """

    ambiguities: tuple[Ambiguity, ...]
    overall_confidence: float
    resolution_prompts: tuple[ResolutionPrompt, ...]

    # Cache for prioritized ambiguities (Story 4.4)
    # Using __slots__ is not possible with frozen dataclass, so we use a workaround
    # by computing on access (Python's functools.cached_property would work but
    # doesn't play well with frozen dataclasses; we simply compute on demand)

    @property
    def has_ambiguities(self) -> bool:
        """Return True if any ambiguities were detected."""
        return len(self.ambiguities) > 0

    @property
    def prioritized_ambiguities(self) -> tuple[Ambiguity, ...]:
        """Return ambiguities sorted by priority (highest first).

        Uses the priority scoring algorithm from calculate_question_priority()
        with deterministic tie-breaking by source_text.

        Returns:
            Tuple of Ambiguity objects sorted by priority (highest first).

        Example:
            >>> result = AmbiguityResult(ambiguities=(amb1, amb2), ...)
            >>> for amb in result.prioritized_ambiguities:
            ...     print(f"{amb.source_text}: priority score")
        """
        # Import here to avoid circular import at module load time
        # (prioritize_questions is defined later in this module)
        from yolo_developer.seed.ambiguity import prioritize_questions

        return tuple(prioritize_questions(list(self.ambiguities)))

    def get_highest_priority_ambiguity(self) -> Ambiguity | None:
        """Return the highest priority ambiguity, if any.

        Convenience method for accessing the most critical ambiguity.

        Returns:
            The highest priority Ambiguity, or None if no ambiguities.

        Example:
            >>> result = AmbiguityResult(ambiguities=(amb1, amb2), ...)
            >>> top_amb = result.get_highest_priority_ambiguity()
            >>> if top_amb:
            ...     print(f"Most critical: {top_amb.source_text}")
        """
        if not self.ambiguities:
            return None
        return self.prioritized_ambiguities[0]

    def get_priority_score(self, ambiguity: Ambiguity) -> int:
        """Get the priority score for a specific ambiguity.

        Args:
            ambiguity: The Ambiguity to score.

        Returns:
            Priority score (higher = more important).

        Example:
            >>> result = AmbiguityResult(ambiguities=(amb,), ...)
            >>> score = result.get_priority_score(amb)
            >>> print(f"Priority score: {score}")
        """
        # Import here to avoid circular import
        from yolo_developer.seed.ambiguity import calculate_question_priority

        return calculate_question_priority(ambiguity)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Includes priority scores for each ambiguity (Story 4.4).

        Returns:
            Dictionary with all fields, nested objects serialized,
            and priority_scores list.
        """
        # Import here to avoid circular import
        from yolo_developer.seed.ambiguity import calculate_question_priority

        return {
            "ambiguities": [amb.to_dict() for amb in self.ambiguities],
            "overall_confidence": self.overall_confidence,
            "resolution_prompts": [prompt.to_dict() for prompt in self.resolution_prompts],
            "priority_scores": [calculate_question_priority(amb) for amb in self.ambiguities],
        }


@dataclass(frozen=True)
class Resolution:
    """A user's resolution to an ambiguity.

    Records the user's response to a clarification prompt.

    Attributes:
        ambiguity_id: Identifier for the ambiguity being resolved
        user_response: The user's clarification text
        timestamp: When the resolution was provided (ISO format)

    Example:
        >>> resolution = Resolution(
        ...     ambiguity_id="amb-001",
        ...     user_response="100 concurrent users",
        ...     timestamp="2026-01-08T10:00:00",
        ... )
    """

    ambiguity_id: str
    user_response: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields.
        """
        return {
            "ambiguity_id": self.ambiguity_id,
            "user_response": self.user_response,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class SeedContext:
    """Context for a seed document with resolutions applied.

    Tracks the original content, user resolutions, and the
    clarified content after resolutions are applied.

    Attributes:
        original_content: The original seed document content
        resolutions: Tuple of user resolutions
        clarified_content: Content after resolutions applied

    Example:
        >>> context = SeedContext(
        ...     original_content="Build a scalable app",
        ...     resolutions=(),
        ...     clarified_content="Build a scalable app",
        ... )
    """

    original_content: str
    resolutions: tuple[Resolution, ...]
    clarified_content: str


# =============================================================================
# Question Quality Validation (Story 4.4)
# =============================================================================

# Vague phrases that indicate non-actionable questions
_VAGUE_PHRASES: tuple[str, ...] = (
    "please clarify",
    "more information",
    "elaborate",
    "explain further",
    "tell me more",
    "what do you mean",
    "can you expand",
    "be more specific",
)

# Minimum question length (>10 characters)
_MIN_QUESTION_LENGTH = 10


def validate_question_quality(question: str) -> tuple[bool, list[str]]:
    """Validate that a clarification question is actionable and specific.

    Checks for:
    - Minimum length (>10 characters)
    - Absence of vague phrases that indicate non-actionable questions
    - Non-empty, non-whitespace content

    Args:
        question: The clarification question to validate.

    Returns:
        Tuple of (is_valid, suggestions) where:
        - is_valid: True if question passes all quality checks
        - suggestions: List of improvement suggestions (empty if valid)

    Example:
        >>> is_valid, suggestions = validate_question_quality(
        ...     "Please clarify what you mean."
        ... )
        >>> is_valid
        False
        >>> "vague phrase" in suggestions[0].lower()
        True
    """
    suggestions: list[str] = []
    question_stripped = question.strip()

    # Check for empty/whitespace-only
    if not question_stripped:
        suggestions.append("Question is empty. Provide a specific question.")
        return False, suggestions

    # Check minimum length
    if len(question_stripped) <= _MIN_QUESTION_LENGTH:
        suggestions.append(
            f"Question is too short (must be > {_MIN_QUESTION_LENGTH} characters). "
            "Make it more specific."
        )

    # Check for vague phrases (case-insensitive)
    question_lower = question_stripped.lower()
    found_vague_phrases = [phrase for phrase in _VAGUE_PHRASES if phrase in question_lower]
    if found_vague_phrases:
        phrases_str = ", ".join(f'"{p}"' for p in found_vague_phrases)
        suggestions.append(
            f"Contains vague phrase(s): {phrases_str}. Rewrite with specific, answerable questions."
        )

    is_valid = len(suggestions) == 0
    return is_valid, suggestions


# =============================================================================
# Question Prioritization (Story 4.4)
# =============================================================================

# Severity weights for priority scoring
_PRIORITY_SEVERITY_WEIGHTS: dict[AmbiguitySeverity, int] = {
    AmbiguitySeverity.HIGH: 30,
    AmbiguitySeverity.MEDIUM: 20,
    AmbiguitySeverity.LOW: 10,
}

# Type weights for priority scoring (blocking types rank higher)
_PRIORITY_TYPE_WEIGHTS: dict[AmbiguityType, int] = {
    AmbiguityType.UNDEFINED: 25,  # Missing info is most critical
    AmbiguityType.SCOPE: 20,  # Scope unclear blocks planning
    AmbiguityType.TECHNICAL: 15,  # Tech unclear affects design
    AmbiguityType.DEPENDENCY: 10,  # Dependencies can be clarified later
    AmbiguityType.PRIORITY: 5,  # Priority is lowest impact
}


def calculate_question_priority(ambiguity: Ambiguity) -> int:
    """Calculate priority score for question ordering.

    Higher score = higher priority (shown first).
    Score is based on severity weight + type weight.

    Args:
        ambiguity: The Ambiguity to score.

    Returns:
        Priority score (higher = more important).

    Example:
        >>> amb = Ambiguity(
        ...     ambiguity_type=AmbiguityType.UNDEFINED,
        ...     severity=AmbiguitySeverity.HIGH,
        ...     source_text="test",
        ...     location="line 1",
        ...     description="test",
        ... )
        >>> calculate_question_priority(amb)
        55
    """
    severity_weight = _PRIORITY_SEVERITY_WEIGHTS[ambiguity.severity]
    type_weight = _PRIORITY_TYPE_WEIGHTS[ambiguity.ambiguity_type]
    return severity_weight + type_weight


def prioritize_questions(ambiguities: list[Ambiguity]) -> list[Ambiguity]:
    """Sort ambiguities by priority (highest first).

    Uses deterministic tie-breaking by source_text for consistency.

    Args:
        ambiguities: List of Ambiguity objects to sort.

    Returns:
        New list sorted by priority (highest first), then source_text.

    Example:
        >>> from yolo_developer.seed.ambiguity import prioritize_questions
        >>> sorted_ambs = prioritize_questions([amb1, amb2, amb3])
        >>> # Highest priority ambiguity is first
    """
    if not ambiguities:
        return []

    # Sort by priority (descending), then source_text (ascending) for tie-breaking
    return sorted(
        ambiguities,
        key=lambda a: (-calculate_question_priority(a), a.source_text),
    )


# =============================================================================
# Confidence Scoring
# =============================================================================

# Confidence penalties per severity level
_SEVERITY_PENALTIES: dict[AmbiguitySeverity, float] = {
    AmbiguitySeverity.LOW: 0.05,
    AmbiguitySeverity.MEDIUM: 0.10,
    AmbiguitySeverity.HIGH: 0.15,
}

# Minimum confidence floor
_CONFIDENCE_FLOOR = 0.1


def calculate_ambiguity_confidence(
    ambiguities: Sequence[Ambiguity],
) -> float:
    """Calculate confidence score based on ambiguities.

    Starts at 1.0 (full confidence) and reduces based on
    the count and severity of ambiguities:
    - HIGH severity: -0.15 per ambiguity
    - MEDIUM severity: -0.10 per ambiguity
    - LOW severity: -0.05 per ambiguity

    The result is clamped to a minimum of 0.1 to avoid zero confidence.

    Args:
        ambiguities: Sequence of Ambiguity objects to score.

    Returns:
        Confidence score between 0.1 and 1.0.

    Example:
        >>> from yolo_developer.seed.ambiguity import (
        ...     Ambiguity,
        ...     AmbiguitySeverity,
        ...     AmbiguityType,
        ...     calculate_ambiguity_confidence,
        ... )
        >>> # No ambiguities = full confidence
        >>> calculate_ambiguity_confidence([])
        1.0
        >>> # One HIGH severity ambiguity
        >>> amb = Ambiguity(
        ...     ambiguity_type=AmbiguityType.SCOPE,
        ...     severity=AmbiguitySeverity.HIGH,
        ...     source_text="test",
        ...     location="line 1",
        ...     description="test",
        ... )
        >>> calculate_ambiguity_confidence([amb])
        0.85
    """
    if not ambiguities:
        return 1.0

    total_penalty = sum(_SEVERITY_PENALTIES[amb.severity] for amb in ambiguities)
    confidence = 1.0 - total_penalty

    # Apply floor
    return max(confidence, _CONFIDENCE_FLOOR)


# =============================================================================
# Ambiguity Detection Prompt
# =============================================================================

AMBIGUITY_DETECTION_PROMPT = """You are a requirements analyst. Analyze the following seed document for ambiguities that could cause implementation confusion.

Identify ambiguities in these categories:
1. **SCOPE**: Unclear boundaries (e.g., "handle all edge cases", "support many users")
2. **TECHNICAL**: Vague technical requirements (e.g., "fast", "scalable", "modern")
3. **PRIORITY**: Unclear importance (e.g., "nice to have", "should", "ideally")
4. **DEPENDENCY**: Unclear relationships (e.g., "integrate with the system", "work with existing")
5. **UNDEFINED**: Missing critical details (e.g., no error handling mentioned, no auth specified)

For each ambiguity, provide:
- type: One of SCOPE, TECHNICAL, PRIORITY, DEPENDENCY, UNDEFINED
- severity: HIGH (blocks implementation), MEDIUM (causes confusion), LOW (minor clarification)
- source_text: The exact ambiguous phrase from the document
- location: Line number or section where found
- description: Why this is ambiguous
- question: A SPECIFIC, ACTIONABLE question to clarify this ambiguity (avoid vague phrases like "please clarify" or "more information")
- suggestions: 2-3 possible interpretations or answers
- answer_format: One of BOOLEAN (yes/no), NUMERIC (number), CHOICE (pick from suggestions), FREE_TEXT (open text), DATE (date value), LIST (multiple items)

IMPORTANT: Questions must be actionable and answerable with a definitive response. Avoid open-ended questions without bounds.

Output as JSON:
{{
  "ambiguities": [
    {{
      "type": "SCOPE|TECHNICAL|PRIORITY|DEPENDENCY|UNDEFINED",
      "severity": "HIGH|MEDIUM|LOW",
      "source_text": "exact phrase",
      "location": "line X or section name",
      "description": "why this is ambiguous",
      "question": "specific actionable question",
      "suggestions": ["option 1", "option 2", "option 3"],
      "answer_format": "BOOLEAN|NUMERIC|CHOICE|FREE_TEXT|DATE|LIST"
    }}
  ]
}}

Seed Document:
---
{content}
---

Respond ONLY with the JSON object."""


# =============================================================================
# LLM Ambiguity Detection
# =============================================================================

# Default model for ambiguity detection
_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_TEMPERATURE = 0.1


# Default format hints for each answer format
_DEFAULT_FORMAT_HINTS: dict[AnswerFormat, str] = {
    AnswerFormat.BOOLEAN: "Answer yes or no",
    AnswerFormat.NUMERIC: "Enter a number (e.g., 100, 1000)",
    AnswerFormat.CHOICE: "Choose from the options above",
    AnswerFormat.FREE_TEXT: "Provide a brief description",
    AnswerFormat.DATE: "Enter a date (YYYY-MM-DD)",
    AnswerFormat.LIST: "Enter items separated by commas",
}


def _get_format_hint(answer_format: AnswerFormat) -> str:
    """Get the default format hint for an answer format.

    Args:
        answer_format: The AnswerFormat enum value.

    Returns:
        Human-readable format hint string.
    """
    return _DEFAULT_FORMAT_HINTS.get(answer_format, "Provide your response")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((json.JSONDecodeError, KeyError)),
    reraise=True,
)
async def _call_llm_for_ambiguities(
    content: str,
    model: str = _DEFAULT_MODEL,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> dict[str, Any]:
    """Call LLM to detect ambiguities in content.

    Args:
        content: The seed content to analyze.
        model: LiteLLM model identifier.
        temperature: Sampling temperature.

    Returns:
        Parsed JSON dictionary from LLM response.

    Raises:
        json.JSONDecodeError: If LLM response is not valid JSON.
    """
    logger.info("ambiguity_llm_call_started", model=model, content_length=len(content))

    prompt = AMBIGUITY_DETECTION_PROMPT.format(content=content)

    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )

    response_text = response.choices[0].message.content
    logger.debug("ambiguity_llm_response_received", response_length=len(response_text))

    # Extract JSON from response, handling markdown code blocks
    json_text = response_text.strip()
    if json_text.startswith("```"):
        # Remove markdown code block
        lines = json_text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            elif line.startswith("```") and in_block:
                break
            elif in_block:
                json_lines.append(line)
        json_text = "\n".join(json_lines)

    result: dict[str, Any] = json.loads(json_text)
    logger.info(
        "ambiguity_llm_call_completed",
        ambiguity_count=len(result.get("ambiguities", [])),
    )
    return result


def _parse_ambiguity_response(
    llm_output: dict[str, Any],
) -> tuple[tuple[Ambiguity, ...], tuple[ResolutionPrompt, ...]]:
    """Parse LLM response into Ambiguity and ResolutionPrompt objects.

    Args:
        llm_output: Parsed JSON from LLM response.

    Returns:
        Tuple of (ambiguities, resolution_prompts).
    """
    ambiguities: list[Ambiguity] = []
    prompts: list[ResolutionPrompt] = []

    raw_ambiguities = llm_output.get("ambiguities", [])

    # Map string types to enums
    type_map = {
        "SCOPE": AmbiguityType.SCOPE,
        "TECHNICAL": AmbiguityType.TECHNICAL,
        "PRIORITY": AmbiguityType.PRIORITY,
        "DEPENDENCY": AmbiguityType.DEPENDENCY,
        "UNDEFINED": AmbiguityType.UNDEFINED,
    }
    severity_map = {
        "LOW": AmbiguitySeverity.LOW,
        "MEDIUM": AmbiguitySeverity.MEDIUM,
        "HIGH": AmbiguitySeverity.HIGH,
    }
    # Map string formats to AnswerFormat enum (Story 4.4)
    format_map = {
        "BOOLEAN": AnswerFormat.BOOLEAN,
        "NUMERIC": AnswerFormat.NUMERIC,
        "CHOICE": AnswerFormat.CHOICE,
        "FREE_TEXT": AnswerFormat.FREE_TEXT,
        "DATE": AnswerFormat.DATE,
        "LIST": AnswerFormat.LIST,
    }

    for raw_amb in raw_ambiguities:
        try:
            if not isinstance(raw_amb, dict):
                logger.warning("ambiguity_parse_skipped", reason="not a dict")
                continue

            # Parse type and severity with fallbacks
            raw_type = str(raw_amb.get("type", "UNDEFINED")).upper()
            raw_severity = str(raw_amb.get("severity", "MEDIUM")).upper()

            amb_type = type_map.get(raw_type, AmbiguityType.UNDEFINED)
            severity = severity_map.get(raw_severity, AmbiguitySeverity.MEDIUM)

            ambiguity = Ambiguity(
                ambiguity_type=amb_type,
                severity=severity,
                source_text=str(raw_amb.get("source_text", "")),
                location=str(raw_amb.get("location", "unknown")),
                description=str(raw_amb.get("description", "")),
            )
            ambiguities.append(ambiguity)

            # Create resolution prompt from the question and suggestions
            question = raw_amb.get("question", "")
            suggestions = raw_amb.get("suggestions", [])
            if question:
                question_str = str(question)

                # Validate question quality (Story 4.4 - AC2)
                is_valid, quality_issues = validate_question_quality(question_str)
                if not is_valid:
                    logger.warning(
                        "question_quality_issues",
                        question=question_str[:50],
                        issues=quality_issues,
                    )

                # Parse answer_format with fallback to FREE_TEXT (Story 4.4)
                raw_format = str(raw_amb.get("answer_format", "FREE_TEXT")).upper()
                # Handle underscore variations
                raw_format = raw_format.replace(" ", "_")
                answer_format = format_map.get(raw_format, AnswerFormat.FREE_TEXT)
                format_hint = _get_format_hint(answer_format)

                prompt = ResolutionPrompt(
                    question=question_str,
                    suggestions=tuple(str(s) for s in suggestions) if suggestions else (),
                    default=None,
                    answer_format=answer_format,
                    format_hint=format_hint,
                )
                prompts.append(prompt)

            logger.debug(
                "ambiguity_parsed",
                type=amb_type.value,
                severity=severity.value,
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("ambiguity_parse_failed", error=str(e), raw=raw_amb)
            continue

    return tuple(ambiguities), tuple(prompts)


async def detect_ambiguities(
    content: str,
    model: str = _DEFAULT_MODEL,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> AmbiguityResult:
    """Detect ambiguities in seed document content using LLM.

    Analyzes the given content for vague requirements, unclear scope,
    missing details, and other ambiguities that could cause implementation
    confusion.

    Args:
        content: The seed document content to analyze.
        model: LiteLLM model identifier (default: gpt-4o-mini).
        temperature: Sampling temperature (default: 0.1).

    Returns:
        AmbiguityResult containing detected ambiguities, confidence score,
        and resolution prompts.

    Example:
        >>> result = await detect_ambiguities("Build a fast, scalable app")
        >>> if result.has_ambiguities:
        ...     for amb in result.ambiguities:
        ...         print(f"{amb.ambiguity_type.value}: {amb.description}")
    """
    logger.info("detect_ambiguities_started", content_length=len(content))

    try:
        llm_output = await _call_llm_for_ambiguities(content, model, temperature)
        ambiguities, prompts = _parse_ambiguity_response(llm_output)
        confidence = calculate_ambiguity_confidence(ambiguities)

        result = AmbiguityResult(
            ambiguities=ambiguities,
            overall_confidence=confidence,
            resolution_prompts=prompts,
        )

        logger.info(
            "detect_ambiguities_completed",
            ambiguity_count=len(ambiguities),
            confidence=confidence,
        )
        return result

    except Exception as e:
        logger.error("detect_ambiguities_failed", error=str(e))
        # Return empty result on error (graceful degradation)
        return AmbiguityResult(
            ambiguities=(),
            overall_confidence=1.0,
            resolution_prompts=(),
        )
