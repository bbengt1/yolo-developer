"""Seed parser protocol and infrastructure (Story 4.1 - Tasks 2-6).

This module provides the parsing protocol, LLM-based parser, and utility
functions for processing seed documents:

- SeedParser: Protocol defining the parser interface
- LLMSeedParser: LLM-based implementation for extracting components
- detect_source_format: Utility to detect input source type
- normalize_content: Utility to clean and standardize input

Example:
    >>> from yolo_developer.seed.parser import (
    ...     LLMSeedParser,
    ...     detect_source_format,
    ...     normalize_content,
    ... )
    >>> from yolo_developer.seed.types import SeedSource
    >>>
    >>> # Detect source format
    >>> source = detect_source_format("# My Seed", filename="seed.md")
    >>> assert source == SeedSource.FILE
    >>>
    >>> # Parse with LLM
    >>> parser = LLMSeedParser()
    >>> result = await parser.parse("Build an e-commerce app", SeedSource.TEXT)
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable

import litellm
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from yolo_developer.seed.types import (
    ConstraintCategory,
    SeedConstraint,
    SeedFeature,
    SeedGoal,
    SeedParseResult,
    SeedSource,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Constants and Prompts
# =============================================================================

SEED_ANALYSIS_PROMPT = """You are a requirements analyst. Parse the following seed document and extract structured information about the project.

Extract the following components:

1. **Goals**: High-level project objectives (what the project should achieve and why)
   - Each goal should have: title, description, priority (1-5, where 1 is highest), and optional rationale

2. **Features**: Discrete functional capabilities the system should have
   - Each feature should have: name, description, optional user_value, and optional related_goals (list of goal titles)

3. **Constraints**: Technical, business, timeline, resource, or compliance limitations
   - Each constraint should have: category (one of: technical, business, timeline, resource, compliance), description, optional impact, and optional related_items

Output your analysis as valid JSON with this exact structure:
{{
  "goals": [
    {{"title": "string", "description": "string", "priority": 1-5, "rationale": "string or null"}}
  ],
  "features": [
    {{"name": "string", "description": "string", "user_value": "string or null", "related_goals": ["goal title", ...]}}
  ],
  "constraints": [
    {{"category": "technical|business|timeline|resource|compliance", "description": "string", "impact": "string or null", "related_items": ["item name", ...]}}
  ]
}}

Important:
- Extract ALL goals, features, and constraints you can identify, including implicit ones
- If something is unclear, make a reasonable inference and include it
- Priority 1 means highest priority/most important
- Ensure the output is valid JSON that can be parsed

Seed Document:
---
{content}
---

Respond ONLY with the JSON object, no other text."""


@runtime_checkable
class SeedParser(Protocol):
    """Protocol for seed document parsers.

    Defines the interface that all seed parsers must implement.
    Parsers take raw content and return structured parse results.

    Example:
        >>> class MyParser:
        ...     async def parse(
        ...         self,
        ...         content: str,
        ...         source: SeedSource,
        ...     ) -> SeedParseResult:
        ...         # Implementation here
        ...         ...
        >>>
        >>> parser: SeedParser = MyParser()
    """

    async def parse(
        self,
        content: str,
        source: SeedSource,
    ) -> SeedParseResult:
        """Parse seed content into structured components.

        Args:
            content: The raw seed document content.
            source: The source type of the content.

        Returns:
            SeedParseResult containing extracted goals, features, and constraints.
        """
        ...


def detect_source_format(
    content: str,
    filename: str | None = None,
) -> SeedSource:
    """Detect the source format of seed content.

    Determines the source type based on filename extension or content analysis.
    If a filename is provided with a recognized extension, returns FILE.
    If content starts with a URL scheme, returns URL.
    Otherwise, returns TEXT.

    Args:
        content: The seed content to analyze.
        filename: Optional filename with extension.

    Returns:
        SeedSource indicating the detected source type.

    Example:
        >>> detect_source_format("Hello", filename="seed.txt")
        <SeedSource.FILE: 'file'>
        >>> detect_source_format("https://example.com/seed")
        <SeedSource.URL: 'url'>
        >>> detect_source_format("Build me an app")
        <SeedSource.TEXT: 'text'>
    """
    logger.debug(
        "detect_source_format",
        content_length=len(content),
        filename=filename,
    )

    # If filename is provided, check extension
    if filename:
        lower_filename = filename.lower()
        # Check for known file extensions
        if lower_filename.endswith((".txt", ".md", ".markdown", ".text")):
            logger.debug("detected_file_source", filename=filename)
            return SeedSource.FILE
        # Any other filename with extension is also a file
        if "." in lower_filename:
            logger.debug("detected_file_source_by_extension", filename=filename)
            return SeedSource.FILE

    # Check if content itself is a URL (starts with URL scheme)
    content_stripped = content.strip()
    if content_stripped.startswith(("http://", "https://")):
        # Only treat as URL if the entire content is a URL (no spaces after scheme)
        if " " not in content_stripped.split("\n")[0]:
            logger.debug("detected_url_source", url=content_stripped[:50])
            return SeedSource.URL

    # Default to text
    logger.debug("detected_text_source")
    return SeedSource.TEXT


def normalize_content(content: str) -> str:
    """Normalize seed content for consistent processing.

    Performs the following normalizations:
    - Strips leading and trailing whitespace
    - Converts all line endings to Unix-style (\\n)
    - Collapses multiple consecutive blank lines to a single blank line
    - Removes null bytes

    Args:
        content: The raw content to normalize.

    Returns:
        Normalized content string.

    Example:
        >>> normalize_content("  Hello\\r\\nWorld  ")
        'Hello\\nWorld'
        >>> normalize_content("A\\n\\n\\n\\nB")
        'A\\n\\nB'
    """
    logger.debug("normalize_content", original_length=len(content))

    # Remove null bytes
    result = content.replace("\x00", "")

    # Convert Windows (CRLF) and old Mac (CR) line endings to Unix (LF)
    result = result.replace("\r\n", "\n")
    result = result.replace("\r", "\n")

    # Collapse multiple consecutive blank lines to a single blank line
    # This regex matches 3+ newlines and replaces with 2 (one blank line)
    result = re.sub(r"\n{3,}", "\n\n", result)

    # Strip leading and trailing whitespace
    result = result.strip()

    logger.debug("normalize_content_complete", normalized_length=len(result))
    return result


# =============================================================================
# Extraction Helpers (Tasks 4-6)
# =============================================================================


def _extract_goals(llm_output: dict[str, Any]) -> list[SeedGoal]:
    """Extract goals from LLM output dictionary.

    Args:
        llm_output: Parsed JSON from LLM response.

    Returns:
        List of SeedGoal objects.
    """
    goals: list[SeedGoal] = []
    raw_goals = llm_output.get("goals", [])

    for raw_goal in raw_goals:
        try:
            # Skip non-dict entries
            if not isinstance(raw_goal, dict):
                logger.warning("goal_extraction_skipped", reason="not a dict")
                continue

            # Validate priority is in range
            priority = raw_goal.get("priority", 3)
            if not isinstance(priority, int) or priority < 1 or priority > 5:
                priority = 3  # Default to medium priority

            goal = SeedGoal(
                title=str(raw_goal.get("title", "Untitled Goal")),
                description=str(raw_goal.get("description", "")),
                priority=priority,
                rationale=raw_goal.get("rationale"),
            )
            goals.append(goal)
            logger.debug("extracted_goal", title=goal.title, priority=goal.priority)
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("goal_extraction_failed", error=str(e), raw_goal=raw_goal)
            continue

    return goals


def _extract_features(
    llm_output: dict[str, Any],
    goals: list[SeedGoal],
) -> list[SeedFeature]:
    """Extract features from LLM output dictionary.

    Args:
        llm_output: Parsed JSON from LLM response.
        goals: List of extracted goals for relationship validation.

    Returns:
        List of SeedFeature objects.
    """
    features: list[SeedFeature] = []
    raw_features = llm_output.get("features", [])
    goal_titles = {g.title for g in goals}

    for raw_feature in raw_features:
        try:
            # Skip non-dict entries
            if not isinstance(raw_feature, dict):
                logger.warning("feature_extraction_skipped", reason="not a dict")
                continue

            # Filter related_goals to only include valid goal titles
            raw_related = raw_feature.get("related_goals", [])
            if raw_related is None:
                raw_related = []
            related_goals = tuple(g for g in raw_related if g in goal_titles)

            feature = SeedFeature(
                name=str(raw_feature.get("name", "Untitled Feature")),
                description=str(raw_feature.get("description", "")),
                user_value=raw_feature.get("user_value"),
                related_goals=related_goals,
            )
            features.append(feature)
            logger.debug("extracted_feature", name=feature.name)
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("feature_extraction_failed", error=str(e), raw_feature=raw_feature)
            continue

    return features


def _extract_constraints(llm_output: dict[str, Any]) -> list[SeedConstraint]:
    """Extract constraints from LLM output dictionary.

    Args:
        llm_output: Parsed JSON from LLM response.

    Returns:
        List of SeedConstraint objects.
    """
    constraints: list[SeedConstraint] = []
    raw_constraints = llm_output.get("constraints", [])

    # Map string categories to enum values
    category_map = {
        "technical": ConstraintCategory.TECHNICAL,
        "business": ConstraintCategory.BUSINESS,
        "timeline": ConstraintCategory.TIMELINE,
        "resource": ConstraintCategory.RESOURCE,
        "compliance": ConstraintCategory.COMPLIANCE,
    }

    for raw_constraint in raw_constraints:
        try:
            # Skip non-dict entries
            if not isinstance(raw_constraint, dict):
                logger.warning("constraint_extraction_skipped", reason="not a dict")
                continue

            # Parse category with fallback to TECHNICAL
            raw_category = str(raw_constraint.get("category", "technical")).lower()
            category = category_map.get(raw_category, ConstraintCategory.TECHNICAL)

            # Handle related_items
            raw_related = raw_constraint.get("related_items", [])
            if raw_related is None:
                raw_related = []
            related_items = tuple(str(item) for item in raw_related)

            constraint = SeedConstraint(
                category=category,
                description=str(raw_constraint.get("description", "")),
                impact=raw_constraint.get("impact"),
                related_items=related_items,
            )
            constraints.append(constraint)
            logger.debug(
                "extracted_constraint",
                category=category.value,
                description=constraint.description[:50],
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(
                "constraint_extraction_failed",
                error=str(e),
                raw_constraint=raw_constraint,
            )
            continue

    return constraints


# =============================================================================
# Plain Text Preprocessor (Task 7)
# =============================================================================


def _parse_plain_text(content: str) -> str:
    """Preprocess plain text content for better LLM parsing.

    Enhances plain text by:
    - Adding structure markers for logical sections
    - Identifying list items for discrete requirement extraction
    - Adding line number annotations for traceability

    Args:
        content: The raw plain text content.

    Returns:
        Enhanced content with structure markers.

    Example:
        >>> result = _parse_plain_text("Build an app\\n- Feature 1\\n- Feature 2")
        >>> "Feature 1" in result
        True
    """
    logger.debug("parse_plain_text", content_length=len(content))

    lines = content.split("\n")
    enhanced_lines: list[str] = []
    current_section: list[str] = []
    in_list = False

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Detect list items (-, *, or numbered)
        is_list_item = bool(
            stripped
            and (
                stripped.startswith(("-", "*", "+"))
                or (len(stripped) > 1 and stripped[0].isdigit() and stripped[1] in ".)")
            )
        )

        # Track list context
        if is_list_item and not in_list:
            # Start of a new list - flush current section
            if current_section:
                enhanced_lines.extend(current_section)
                enhanced_lines.append("")
                current_section = []
            in_list = True
        elif not is_list_item and in_list and stripped:
            # End of list
            in_list = False

        # Add line with annotation for non-empty lines
        if stripped:
            enhanced_lines.append(f"[L{i}] {line}")
        else:
            enhanced_lines.append(line)

    logger.debug("parse_plain_text_complete", enhanced_lines=len(enhanced_lines))
    return "\n".join(enhanced_lines)


# =============================================================================
# Markdown Preprocessor (Task 8)
# =============================================================================


def _parse_markdown(content: str) -> str:
    """Preprocess markdown content for better LLM parsing.

    Enhances markdown by:
    - Extracting structure from headings (h1-h6)
    - Annotating list items for requirement extraction
    - Identifying code blocks as potential technical constraints
    - Marking emphasized text as potential priority signals

    Args:
        content: The raw markdown content.

    Returns:
        Enhanced content with structure markers.

    Example:
        >>> result = _parse_markdown("# Project\\n## Features\\n- Auth")
        >>> "Project" in result
        True
    """
    logger.debug("parse_markdown", content_length=len(content))

    lines = content.split("\n")
    enhanced_lines: list[str] = []
    in_code_block = False

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Track code blocks
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            if in_code_block:
                enhanced_lines.append(f"[L{i}][CODE_START] {line}")
            else:
                enhanced_lines.append(f"[L{i}][CODE_END] {line}")
            continue

        if in_code_block:
            # Preserve code block content as-is with line numbers
            enhanced_lines.append(f"[L{i}] {line}")
            continue

        # Detect headings
        if stripped.startswith("#"):
            # Count heading level
            heading_level = 0
            for char in stripped:
                if char == "#":
                    heading_level += 1
                else:
                    break
            enhanced_lines.append(f"[L{i}][H{heading_level}] {line}")
            continue

        # Detect list items
        is_list_item = bool(
            stripped
            and (
                stripped.startswith(("-", "*", "+"))
                or (len(stripped) > 1 and stripped[0].isdigit() and stripped[1] in ".)")
            )
        )

        if is_list_item:
            enhanced_lines.append(f"[L{i}][LIST] {line}")
            continue

        # Detect emphasis (potential priority signals)
        has_bold = "**" in stripped or "__" in stripped
        has_italic = "*" in stripped.replace("**", "") or "_" in stripped.replace("__", "")

        if has_bold:
            enhanced_lines.append(f"[L{i}][EMPHASIS] {line}")
        elif has_italic:
            enhanced_lines.append(f"[L{i}] {line}")
        elif stripped:
            enhanced_lines.append(f"[L{i}] {line}")
        else:
            enhanced_lines.append(line)

    logger.debug("parse_markdown_complete", enhanced_lines=len(enhanced_lines))
    return "\n".join(enhanced_lines)


# =============================================================================
# LLM-Based Parser (Task 3)
# =============================================================================


class LLMSeedParser:
    """LLM-based seed document parser.

    Uses LiteLLM to analyze seed documents and extract structured
    components (goals, features, constraints).

    Attributes:
        model: The LLM model to use (default: gpt-4o-mini for cost efficiency).
        temperature: Sampling temperature (default: 0.1 for consistency).

    Example:
        >>> parser = LLMSeedParser()
        >>> result = await parser.parse(
        ...     "Build an e-commerce platform with user auth",
        ...     SeedSource.TEXT,
        ... )
        >>> print(result.goal_count)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
    ) -> None:
        """Initialize the LLM seed parser.

        Args:
            model: LiteLLM model identifier (default: gpt-4o-mini).
            temperature: Sampling temperature for LLM (default: 0.1).
        """
        self.model = model
        self.temperature = temperature
        logger.info(
            "llm_seed_parser_initialized",
            model=model,
            temperature=temperature,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((json.JSONDecodeError, KeyError)),
        reraise=True,
    )
    async def _call_llm(self, content: str) -> dict[str, Any]:
        """Call LLM to analyze seed content.

        Args:
            content: The seed content to analyze.

        Returns:
            Parsed JSON dictionary from LLM response.

        Raises:
            json.JSONDecodeError: If LLM response is not valid JSON.
        """
        logger.info("llm_call_started", model=self.model, content_length=len(content))

        prompt = SEED_ANALYSIS_PROMPT.format(content=content)

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

        response_text = response.choices[0].message.content
        logger.debug("llm_response_received", response_length=len(response_text))

        # Try to extract JSON from response
        # Handle case where LLM wraps JSON in markdown code blocks
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
            "llm_call_completed",
            goals=len(result.get("goals", [])),
            features=len(result.get("features", [])),
            constraints=len(result.get("constraints", [])),
        )
        return result

    async def parse(
        self,
        content: str,
        source: SeedSource,
    ) -> SeedParseResult:
        """Parse seed content into structured components.

        Args:
            content: The raw seed document content.
            source: The source type of the content.

        Returns:
            SeedParseResult containing extracted goals, features, and constraints.
        """
        logger.info(
            "seed_parsing_started",
            source=source.value,
            content_length=len(content),
        )

        # Normalize content
        normalized = normalize_content(content)

        # Call LLM for analysis
        try:
            llm_output = await self._call_llm(normalized)
        except Exception as e:
            logger.error("seed_parsing_failed", error=str(e))
            # Return empty result on failure
            return SeedParseResult(
                goals=(),
                features=(),
                constraints=(),
                raw_content=content,
                source=source,
                metadata=(("error", str(e)),),
            )

        # Extract components
        goals = _extract_goals(llm_output)
        features = _extract_features(llm_output, goals)
        constraints = _extract_constraints(llm_output)

        result = SeedParseResult(
            goals=tuple(goals),
            features=tuple(features),
            constraints=tuple(constraints),
            raw_content=content,
            source=source,
            metadata=(("model", self.model),),
        )

        logger.info(
            "seed_parsing_completed",
            goals=result.goal_count,
            features=result.feature_count,
            constraints=result.constraint_count,
        )

        return result
