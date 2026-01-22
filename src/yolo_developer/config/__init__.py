"""Configuration module for YOLO Developer.

Exports the configuration schema classes and loading utilities for public API access.

Example:
    >>> from yolo_developer.config import load_config, YoloConfig
    >>> config = load_config()  # Loads from ./yolo.yaml if it exists
    >>> config.llm.cheap_model
    'gpt-5.2-instant'

Export/Import Example:
    >>> from yolo_developer.config import export_config, import_config
    >>> export_config(config, Path("exported.yaml"))
    >>> import_config(Path("exported.yaml"), Path("yolo.yaml"))
"""

from __future__ import annotations

from yolo_developer.config.export import export_config, import_config
from yolo_developer.config.loader import ConfigurationError, load_config
from yolo_developer.config.schema import (
    GateThreshold,
    LLMConfig,
    LLMProvider,
    HybridConfig,
    HybridRoutingConfig,
    MemoryConfig,
    OpenAIConfig,
    QualityConfig,
    SeedThresholdConfig,
    YoloConfig,
    LLM_BEST_MODEL_DEFAULT,
    LLM_CHEAP_MODEL_DEFAULT,
    LLM_PREMIUM_MODEL_DEFAULT,
    OPENAI_CHEAP_MODEL_DEFAULT,
    OPENAI_CODE_MODEL_DEFAULT,
    OPENAI_PREMIUM_MODEL_DEFAULT,
    OPENAI_REASONING_MODEL_DEFAULT,
)
from yolo_developer.config.validators import (
    ValidationIssue,
    ValidationResult,
    validate_config,
)

__all__ = [
    "ConfigurationError",
    "GateThreshold",
    "LLMConfig",
    "LLMProvider",
    "LLM_BEST_MODEL_DEFAULT",
    "LLM_CHEAP_MODEL_DEFAULT",
    "LLM_PREMIUM_MODEL_DEFAULT",
    "HybridConfig",
    "HybridRoutingConfig",
    "MemoryConfig",
    "OpenAIConfig",
    "OPENAI_CHEAP_MODEL_DEFAULT",
    "OPENAI_CODE_MODEL_DEFAULT",
    "OPENAI_PREMIUM_MODEL_DEFAULT",
    "OPENAI_REASONING_MODEL_DEFAULT",
    "QualityConfig",
    "SeedThresholdConfig",
    "ValidationIssue",
    "ValidationResult",
    "YoloConfig",
    "export_config",
    "import_config",
    "load_config",
    "validate_config",
]
