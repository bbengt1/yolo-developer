"""Configuration schema for YOLO Developer using Pydantic Settings.

This module defines the strongly-typed configuration schema per ADR-008.
Configuration is validated at system boundaries using Pydantic v2.

Environment variables follow the pattern:
- YOLO_PROJECT_NAME for top-level fields
- YOLO_LLM__CHEAP_MODEL for nested fields (using __ delimiter)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
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
    - YOLO_QUALITY__TEST_COVERAGE_THRESHOLD: Override test coverage threshold
    - YOLO_MEMORY__PERSIST_PATH: Override memory persistence path

    Example:
        >>> config = YoloConfig(project_name="my-project")
        >>> config.llm.cheap_model
        'gpt-4o-mini'
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
