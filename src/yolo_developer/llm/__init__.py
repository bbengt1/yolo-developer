"""LLM abstraction layer for YOLO Developer (ADR-003).

This module provides the LLM router for multi-provider abstraction.

Example:
    >>> from yolo_developer.llm import LLMRouter
    >>> from yolo_developer.config.schema import LLMConfig
    >>>
    >>> config = LLMConfig()
    >>> router = LLMRouter(config)
    >>> response = await router.call(
    ...     messages=[{"role": "user", "content": "Hello"}],
    ...     tier="routine",
    ... )
"""

from __future__ import annotations

from yolo_developer.llm.router import (
    LLMConfigurationError,
    LLMProviderError,
    LLMRouter,
    LLMRouterError,
    ModelTier,
    TaskRouting,
    TaskType,
)

__all__ = [
    "LLMConfigurationError",
    "LLMProviderError",
    "LLMRouter",
    "LLMRouterError",
    "ModelTier",
    "TaskRouting",
    "TaskType",
]
