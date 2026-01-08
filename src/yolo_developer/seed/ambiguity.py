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
    to help resolve an ambiguity.

    Attributes:
        question: The clarification question to ask
        suggestions: Tuple of suggested answers
        default: Default answer if user doesn't provide one (optional)

    Example:
        >>> prompt = ResolutionPrompt(
        ...     question="What response time is acceptable?",
        ...     suggestions=("< 100ms", "< 500ms", "< 1 second"),
        ...     default="< 500ms",
        ... )
    """

    question: str
    suggestions: tuple[str, ...]
    default: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields, suggestions as list.
        """
        return {
            "question": self.question,
            "suggestions": list(self.suggestions),
            "default": self.default,
        }


@dataclass(frozen=True)
class AmbiguityResult:
    """Complete result from ambiguity detection.

    Contains all detected ambiguities, confidence score, and
    resolution prompts for user interaction.

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
    """

    ambiguities: tuple[Ambiguity, ...]
    overall_confidence: float
    resolution_prompts: tuple[ResolutionPrompt, ...]

    @property
    def has_ambiguities(self) -> bool:
        """Return True if any ambiguities were detected."""
        return len(self.ambiguities) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields, nested objects serialized.
        """
        return {
            "ambiguities": [amb.to_dict() for amb in self.ambiguities],
            "overall_confidence": self.overall_confidence,
            "resolution_prompts": [prompt.to_dict() for prompt in self.resolution_prompts],
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
- question: A specific question to clarify this ambiguity
- suggestions: 2-3 possible interpretations or answers

Output as JSON:
{{
  "ambiguities": [
    {{
      "type": "SCOPE|TECHNICAL|PRIORITY|DEPENDENCY|UNDEFINED",
      "severity": "HIGH|MEDIUM|LOW",
      "source_text": "exact phrase",
      "location": "line X or section name",
      "description": "why this is ambiguous",
      "question": "clarification question",
      "suggestions": ["option 1", "option 2", "option 3"]
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
                prompt = ResolutionPrompt(
                    question=str(question),
                    suggestions=tuple(str(s) for s in suggestions) if suggestions else (),
                    default=None,
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
