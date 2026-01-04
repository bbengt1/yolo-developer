"""Configuration schema for YOLO Developer using Pydantic Settings.

This module defines the strongly-typed configuration schema per ADR-008.
Configuration is validated at system boundaries using Pydantic v2.

Environment Variable Pattern
-----------------------------
Environment variables use the ``YOLO_`` prefix with ``__`` as the nested delimiter:

- ``YOLO_PROJECT_NAME``: Project name (required)
- ``YOLO_LLM__CHEAP_MODEL``: LLM model for routine tasks
- ``YOLO_LLM__PREMIUM_MODEL``: LLM model for complex reasoning
- ``YOLO_LLM__BEST_MODEL``: LLM model for critical decisions
- ``YOLO_LLM__OPENAI_API_KEY``: OpenAI API key (secrets only via env vars)
- ``YOLO_LLM__ANTHROPIC_API_KEY``: Anthropic API key (secrets only via env vars)
- ``YOLO_QUALITY__TEST_COVERAGE_THRESHOLD``: Test coverage threshold (0.0-1.0)
- ``YOLO_QUALITY__CONFIDENCE_THRESHOLD``: Confidence threshold (0.0-1.0)
- ``YOLO_MEMORY__PERSIST_PATH``: Memory persistence directory
- ``YOLO_MEMORY__VECTOR_STORE_TYPE``: Vector store type (chromadb)
- ``YOLO_MEMORY__GRAPH_STORE_TYPE``: Graph store type (json, neo4j)

Configuration Priority Order
----------------------------
Configuration values are resolved in the following order (later overrides earlier):

1. Defaults (defined in schema)
2. YAML file (yolo.yaml)
3. Environment variables

API Key Security
----------------
API keys (openai_api_key, anthropic_api_key) are:

- Set via environment variables ONLY (never in YAML files)
- Stored as SecretStr for automatic masking in logs/repr
- Accessible via ``.get_secret_value()`` method when needed
- Never written to config file exports

Example Usage
-------------
>>> from yolo_developer.config import load_config
>>> config = load_config()
>>> config.llm.cheap_model
'gpt-4o-mini'
>>> if config.llm.openai_api_key:
...     api_key = config.llm.openai_api_key.get_secret_value()
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    """Configuration for LLM provider settings.

    Defines the model tiers used for different task complexities:
    - cheap_model: For routine, low-complexity tasks
    - premium_model: For complex reasoning tasks
    - best_model: For critical decisions requiring highest quality
    """

    cheap_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for routine, low-complexity tasks",
    )
    premium_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="LLM model for complex reasoning tasks",
    )
    best_model: str = Field(
        default="claude-opus-4-5-20251101",
        description="LLM model for critical decisions requiring highest quality",
    )

    # API keys - read from env only, masked in output (Story 1.6)
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key (set via YOLO_LLM__OPENAI_API_KEY env var)",
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Anthropic API key (set via YOLO_LLM__ANTHROPIC_API_KEY env var)",
    )


class QualityConfig(BaseModel):
    """Configuration for quality gate thresholds.

    Defines the minimum thresholds that must be met for quality gates
    to pass. All threshold values must be between 0.0 and 1.0.
    """

    test_coverage_threshold: float = Field(
        default=0.80,
        description="Minimum test coverage ratio (0.0-1.0) required for quality gates",
        ge=0.0,
        le=1.0,
    )
    confidence_threshold: float = Field(
        default=0.90,
        description="Minimum confidence score (0.0-1.0) for deployment approval",
        ge=0.0,
        le=1.0,
    )


class MemoryConfig(BaseModel):
    """Configuration for memory and storage settings.

    Defines how YOLO Developer persists memory, including vector embeddings
    and relationship graphs.
    """

    persist_path: str = Field(
        default=".yolo/memory",
        description="Directory path for persisting memory data",
    )
    vector_store_type: Literal["chromadb"] = Field(
        default="chromadb",
        description="Vector store backend type (chromadb supported)",
    )
    graph_store_type: Literal["json", "neo4j"] = Field(
        default="json",
        description="Graph store backend type (json for MVP, neo4j optional)",
    )


class YoloConfig(BaseSettings):
    """Main configuration class for YOLO Developer.

    This is the root configuration that composes all nested configurations.
    Inherits from pydantic_settings.BaseSettings for environment variable support.

    Environment Variable Mapping:
        - YOLO_PROJECT_NAME: Project name (required)
        - YOLO_LLM__CHEAP_MODEL: Override cheap_model
        - YOLO_LLM__PREMIUM_MODEL: Override premium_model
        - YOLO_LLM__BEST_MODEL: Override best_model
        - YOLO_LLM__OPENAI_API_KEY: OpenAI API key (secrets via env only)
        - YOLO_LLM__ANTHROPIC_API_KEY: Anthropic API key (secrets via env only)
        - YOLO_QUALITY__TEST_COVERAGE_THRESHOLD: Override test coverage threshold
        - YOLO_QUALITY__CONFIDENCE_THRESHOLD: Override confidence threshold
        - YOLO_MEMORY__PERSIST_PATH: Override memory persistence path
        - YOLO_MEMORY__VECTOR_STORE_TYPE: Override vector store type
        - YOLO_MEMORY__GRAPH_STORE_TYPE: Override graph store type

    Example:
        >>> config = YoloConfig(project_name="my-project")
        >>> config.llm.cheap_model
        'gpt-4o-mini'
        >>> warnings = config.validate_api_keys()
        >>> if warnings:
        ...     print("Warning: No API keys configured")
    """

    model_config = SettingsConfigDict(
        env_prefix="YOLO_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    # Required fields (no default)
    project_name: str = Field(
        description="Name of the project being developed",
    )

    # Nested configuration models with defaults
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM provider configuration",
    )
    quality: QualityConfig = Field(
        default_factory=QualityConfig,
        description="Quality gate threshold configuration",
    )
    memory: MemoryConfig = Field(
        default_factory=MemoryConfig,
        description="Memory and storage configuration",
    )

    def validate_api_keys(self) -> list[str]:
        """Validate API key configuration and return warnings.

        Checks if any API keys are configured. If no API keys are set,
        returns a warning message. This is a warning, not an error,
        because API keys may be set later via SDK or at runtime.

        Returns:
            List of warning messages. Empty list if at least one API key is configured.

        Example:
            >>> config = YoloConfig(project_name="my-project")
            >>> warnings = config.validate_api_keys()
            >>> if warnings:
            ...     for warning in warnings:
            ...         print(f"Warning: {warning}")
        """
        warnings: list[str] = []

        if self.llm.openai_api_key is None and self.llm.anthropic_api_key is None:
            warnings.append(
                "No API keys configured. Set YOLO_LLM__OPENAI_API_KEY or "
                "YOLO_LLM__ANTHROPIC_API_KEY environment variable for LLM operations."
            )

        return warnings
