"""Seed parsing module for yolo-developer.

This module provides functionality for parsing natural language seed documents
and extracting structured components (goals, features, constraints).

Example:
    >>> from yolo_developer.seed import parse_seed, SeedSource
    >>>
    >>> # Parse a simple text seed
    >>> result = await parse_seed("Build an e-commerce platform with auth")
    >>> print(f"Found {result.goal_count} goals, {result.feature_count} features")
    >>>
    >>> # Parse from a file
    >>> with open("requirements.md") as f:
    ...     content = f.read()
    >>> result = await parse_seed(content, filename="requirements.md")
    >>>
    >>> # Parse with ambiguity detection (Story 4.3)
    >>> result = await parse_seed(content, detect_ambiguities=True)
    >>> if result.has_ambiguities:
    ...     for amb in result.ambiguities:
    ...         print(f"- {amb.description}")
"""

from __future__ import annotations

from yolo_developer.seed.ambiguity import (
    Ambiguity,
    AmbiguityResult,
    AmbiguitySeverity,
    AmbiguityType,
    Resolution,
    ResolutionPrompt,
    SeedContext,
    calculate_ambiguity_confidence,
    detect_ambiguities,
)
from yolo_developer.seed.api import parse_seed
from yolo_developer.seed.parser import (
    LLMSeedParser,
    SeedParser,
    detect_source_format,
    normalize_content,
)
from yolo_developer.seed.types import (
    ComponentType,
    ConstraintCategory,
    SeedComponent,
    SeedConstraint,
    SeedFeature,
    SeedGoal,
    SeedParseResult,
    SeedSource,
)

__all__ = [
    "Ambiguity",
    "AmbiguityResult",
    "AmbiguitySeverity",
    "AmbiguityType",
    "ComponentType",
    "ConstraintCategory",
    "LLMSeedParser",
    "Resolution",
    "ResolutionPrompt",
    "SeedComponent",
    "SeedConstraint",
    "SeedContext",
    "SeedFeature",
    "SeedGoal",
    "SeedParseResult",
    "SeedParser",
    "SeedSource",
    "calculate_ambiguity_confidence",
    "detect_ambiguities",
    "detect_source_format",
    "normalize_content",
    "parse_seed",
]
