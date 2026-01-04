"""Configuration module for YOLO Developer.

Exports the configuration schema classes and loading utilities for public API access.

Example:
    >>> from yolo_developer.config import load_config, YoloConfig
    >>> config = load_config()  # Loads from ./yolo.yaml if it exists
    >>> config.llm.cheap_model
    'gpt-4o-mini'
"""

from __future__ import annotations

from yolo_developer.config.loader import ConfigurationError, load_config
from yolo_developer.config.schema import (
    LLMConfig,
    MemoryConfig,
    QualityConfig,
    YoloConfig,
)
from yolo_developer.config.validators import (
    ValidationIssue,
    ValidationResult,
    validate_config,
)

__all__ = [
    "ConfigurationError",
    "LLMConfig",
    "MemoryConfig",
    "QualityConfig",
    "ValidationIssue",
    "ValidationResult",
    "YoloConfig",
    "load_config",
    "validate_config",
]
