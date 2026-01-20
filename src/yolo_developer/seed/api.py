"""High-level seed parsing API (Story 4.1 - Task 9, Story 4.3, Story 4.5).

This module provides the main entry point for parsing seed documents.
It handles format detection, preprocessing, LLM-based extraction,
optional ambiguity detection, and optional SOP constraint validation.

Example:
    >>> from yolo_developer.seed.api import parse_seed
    >>> from yolo_developer.seed.types import SeedSource
    >>>
    >>> # Parse text directly
    >>> result = await parse_seed("Build an e-commerce platform with auth")
    >>> print(f"Found {result.goal_count} goals")
    >>>
    >>> # Parse from file
    >>> result = await parse_seed(content, filename="requirements.md")
    >>> print(f"Found {result.feature_count} features")
    >>>
    >>> # Parse with ambiguity detection
    >>> result = await parse_seed(content, detect_ambiguities=True)
    >>> if result.has_ambiguities:
    ...     print(f"Found {result.ambiguity_count} ambiguities")
    >>>
    >>> # Parse with SOP validation
    >>> from yolo_developer.seed.sop import InMemorySOPStore
    >>> store = InMemorySOPStore()
    >>> result = await parse_seed(content, validate_sop=True, sop_store=store)
    >>> if not result.sop_passed:
    ...     print(f"Found SOP conflicts")
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import structlog

from yolo_developer.seed.ambiguity import (
    detect_ambiguities as _detect_ambiguities,
)
from yolo_developer.seed.parser import (
    LLMSeedParser,
    _parse_markdown,
    _parse_plain_text,
    detect_source_format,
    normalize_content,
)
from yolo_developer.seed.sop import (
    validate_against_sop as _validate_against_sop,
)
from yolo_developer.seed.types import SeedParseResult, SeedSource

if TYPE_CHECKING:
    from yolo_developer.seed.sop import SOPStore

logger = structlog.get_logger(__name__)


async def parse_seed(
    content: str,
    source: SeedSource | None = None,
    filename: str | None = None,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
    preprocess: bool = True,
    detect_ambiguities: bool = False,
    validate_sop: bool = False,
    sop_store: SOPStore | None = None,
) -> SeedParseResult:
    """Parse a seed document into structured components.

    This is the main entry point for seed parsing. It:
    1. Detects the source format (file, text, url)
    2. Normalizes the content
    3. Applies format-specific preprocessing (optional)
    4. Invokes the LLM parser to extract goals, features, and constraints
    5. Optionally runs ambiguity detection on the content
    6. Optionally validates against SOP constraints

    Args:
        content: The raw seed document content.
        source: Optional source type override. If not provided, auto-detected.
        filename: Optional filename for format detection (e.g., "requirements.md").
        model: LLM model to use (default: gpt-4o-mini).
        temperature: LLM sampling temperature (default: 0.1).
        preprocess: Whether to apply format-specific preprocessing (default: True).
        detect_ambiguities: Whether to run ambiguity detection (default: False).
        validate_sop: Whether to run SOP constraint validation (default: False).
        sop_store: Store containing SOP constraints. Required if validate_sop is True.

    Returns:
        SeedParseResult containing extracted goals, features, constraints,
        and optionally ambiguity detection and SOP validation results.

    Raises:
        ValueError: If validate_sop is True but sop_store is None.

    Example:
        >>> # Simple text input
        >>> result = await parse_seed("Build a blog with user comments")
        >>> print(result.goal_count)
        1

        >>> # Markdown file
        >>> with open("requirements.md") as f:
        ...     content = f.read()
        >>> result = await parse_seed(content, filename="requirements.md")
        >>> for goal in result.goals:
        ...     print(f"- {goal.title}")

        >>> # With ambiguity detection
        >>> result = await parse_seed(content, detect_ambiguities=True)
        >>> if result.has_ambiguities:
        ...     for amb in result.ambiguities:
        ...         print(f"- {amb.description}")

        >>> # With SOP validation
        >>> result = await parse_seed(
        ...     content,
        ...     validate_sop=True,
        ...     sop_store=my_store,
        ... )
        >>> if not result.sop_passed:
        ...     print("SOP conflicts found!")
    """
    # Validate parameters
    if validate_sop and sop_store is None:
        raise ValueError("sop_store is required when validate_sop is True")

    logger.info(
        "parse_seed_started",
        content_length=len(content),
        source=source.value if source else "auto",
        filename=filename,
        detect_ambiguities=detect_ambiguities,
        validate_sop=validate_sop,
    )

    # Auto-detect source format if not provided
    if source is None:
        source = detect_source_format(content, filename=filename)
        logger.debug("source_auto_detected", source=source.value)

    # Normalize content
    normalized_content = normalize_content(content)

    # Apply format-specific preprocessing
    if preprocess:
        preprocessed_content = _preprocess_content(normalized_content, source, filename)
    else:
        preprocessed_content = normalized_content

    # Create parser and parse
    parser = LLMSeedParser(model=model, temperature=temperature)
    result = await parser.parse(preprocessed_content, source)

    # Replace raw_content with the original input content (not preprocessed)
    # The parser stores preprocessed content, but raw_content should be original
    result = replace(result, raw_content=content)

    # Optionally run ambiguity detection
    if detect_ambiguities:
        logger.info("running_ambiguity_detection")
        ambiguity_result = await _detect_ambiguities(
            content,
            model=model,
            temperature=temperature,
        )
        result = replace(
            result,
            ambiguities=ambiguity_result.ambiguities,
            ambiguity_confidence=ambiguity_result.overall_confidence,
        )
        logger.info(
            "ambiguity_detection_completed",
            ambiguity_count=len(ambiguity_result.ambiguities),
            ambiguity_confidence=ambiguity_result.overall_confidence,
        )

    # Optionally run SOP validation (after ambiguity detection)
    if validate_sop and sop_store is not None:
        logger.info("running_sop_validation")
        sop_result = await _validate_against_sop(
            content,
            sop_store,
            model=model,
        )
        result = replace(result, sop_validation=sop_result)
        logger.info(
            "sop_validation_completed",
            conflict_count=len(sop_result.conflicts),
            hard_conflicts=sop_result.hard_conflict_count,
            soft_conflicts=sop_result.soft_conflict_count,
            passed=sop_result.passed,
        )

    logger.info(
        "parse_seed_completed",
        goals=result.goal_count,
        features=result.feature_count,
        constraints=result.constraint_count,
        ambiguities=result.ambiguity_count if detect_ambiguities else 0,
        sop_conflicts=(
            result.sop_validation.hard_conflict_count + result.sop_validation.soft_conflict_count
            if result.sop_validation
            else 0
        ),
    )

    return result


def _preprocess_content(
    content: str,
    source: SeedSource,
    filename: str | None = None,
) -> str:
    """Apply format-specific preprocessing to content.

    Args:
        content: Normalized content to preprocess.
        source: Source type (affects preprocessing choice).
        filename: Optional filename for format detection.

    Returns:
        Preprocessed content with structure markers.
    """
    # Determine if content is markdown
    is_markdown = False
    if filename:
        lower_filename = filename.lower()
        is_markdown = lower_filename.endswith((".md", ".markdown"))
    elif source == SeedSource.FILE:
        # Check content for markdown indicators
        is_markdown = _looks_like_markdown(content)
    elif source == SeedSource.TEXT:
        # Text might still be markdown-formatted
        is_markdown = _looks_like_markdown(content)

    if is_markdown:
        logger.debug("applying_markdown_preprocessing")
        return _parse_markdown(content)
    else:
        logger.debug("applying_plain_text_preprocessing")
        return _parse_plain_text(content)


def _looks_like_markdown(content: str) -> bool:
    """Detect if content appears to be markdown-formatted.

    Checks for common markdown patterns:
    - Headings (# Heading)
    - Code blocks (```)
    - Links ([text](url))
    - Bold/italic (**text** or *text*)

    Args:
        content: The content to check.

    Returns:
        True if content appears to be markdown.
    """
    lines = content.split("\n")
    markdown_indicators = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            markdown_indicators += 1
        elif stripped.startswith("```"):
            markdown_indicators += 1
        elif "**" in stripped or "__" in stripped:
            markdown_indicators += 1
        elif stripped.startswith(">"):
            markdown_indicators += 1

    # If more than 2 markdown indicators, treat as markdown
    return markdown_indicators >= 2
