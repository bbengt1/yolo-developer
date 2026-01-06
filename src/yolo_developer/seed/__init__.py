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
"""

from __future__ import annotations

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
    # Types
    "ComponentType",
    "ConstraintCategory",
    # Parser classes
    "LLMSeedParser",
    "SeedComponent",
    "SeedConstraint",
    "SeedFeature",
    "SeedGoal",
    "SeedParseResult",
    "SeedParser",
    "SeedSource",
    # Utilities
    "detect_source_format",
    "normalize_content",
    # Main API
    "parse_seed",
]
