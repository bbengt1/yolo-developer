"""LLM Router for multi-provider abstraction (Story 8.2, ADR-003).

This module provides the LLMRouter class for routing LLM calls to the
appropriate provider and model based on task tier.

Model Tiers (per architecture):
    - "routine": cheap_model (gpt-4o-mini) for routine tasks
    - "complex": premium_model (claude-sonnet) for complex reasoning
    - "critical": best_model (claude-opus) for critical decisions

Example:
    >>> from yolo_developer.llm.router import LLMRouter
    >>> from yolo_developer.config.schema import LLMConfig
    >>>
    >>> config = LLMConfig()
    >>> router = LLMRouter(config)
    >>> response = await router.call(
    ...     messages=[{"role": "user", "content": "Hello"}],
    ...     tier="routine",
    ... )

Architecture Note:
    Per ADR-003, this uses LiteLLM for unified provider access with
    built-in cost tracking and token counting.
"""

from __future__ import annotations

from typing import Any, Literal

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from yolo_developer.config.schema import LLMConfig

logger = structlog.get_logger(__name__)

# Model tier type alias
ModelTier = Literal["routine", "complex", "critical"]


class LLMRouterError(Exception):
    """Base exception for LLM router errors."""

    pass


class LLMProviderError(LLMRouterError):
    """Raised when LLM provider returns an error."""

    pass


class LLMConfigurationError(LLMRouterError):
    """Raised when LLM configuration is invalid or missing."""

    pass


class LLMRouter:
    """Routes LLM calls to appropriate provider based on task tier.

    Provides a unified interface for calling different LLM models
    based on task complexity. Uses LiteLLM for provider abstraction.

    Attributes:
        config: LLM configuration with model names and API keys.
        model_map: Mapping from tier names to model identifiers.

    Example:
        >>> config = LLMConfig()
        >>> router = LLMRouter(config)
        >>> response = await router.call(
        ...     messages=[{"role": "user", "content": "Generate code"}],
        ...     tier="complex",
        ... )
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the LLM router with configuration.

        Args:
            config: LLM configuration with model names and API keys.

        Example:
            >>> config = LLMConfig(cheap_model="gpt-4o-mini")
            >>> router = LLMRouter(config)
        """
        self.config = config
        self.model_map: dict[ModelTier, str] = {
            "routine": config.cheap_model,
            "complex": config.premium_model,
            "critical": config.best_model,
        }

        logger.debug(
            "llm_router_initialized",
            routine_model=config.cheap_model,
            complex_model=config.premium_model,
            critical_model=config.best_model,
        )

    def get_model_for_tier(self, tier: ModelTier) -> str:
        """Get the model identifier for a given tier.

        Args:
            tier: Task complexity tier (routine, complex, critical).

        Returns:
            Model identifier string.

        Example:
            >>> router = LLMRouter(LLMConfig())
            >>> router.get_model_for_tier("complex")
            'claude-sonnet-4-20250514'
        """
        return self.model_map[tier]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((LLMProviderError,)),
        reraise=True,
    )
    async def call(
        self,
        messages: list[dict[str, str]],
        tier: ModelTier = "routine",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        """Call LLM with the specified tier and return response content.

        Uses tenacity retry with exponential backoff per ADR-007.

        Args:
            messages: List of message dicts with role and content.
            tier: Task complexity tier (routine, complex, critical).
                Defaults to "routine".
            temperature: Sampling temperature (0.0-2.0). Defaults to 0.7.
            max_tokens: Maximum tokens in response. Defaults to 4096.
            **kwargs: Additional arguments passed to LiteLLM.

        Returns:
            Response content string from the LLM.

        Raises:
            LLMProviderError: When provider returns an error (triggers retry).
            LLMConfigurationError: When configuration is invalid.

        Example:
            >>> response = await router.call(
            ...     messages=[{"role": "user", "content": "Hello"}],
            ...     tier="routine",
            ... )
        """
        model = self.get_model_for_tier(tier)

        logger.info(
            "llm_call_start",
            model=model,
            tier=tier,
            message_count=len(messages),
            max_tokens=max_tokens,
        )

        try:
            # Import litellm here to avoid import errors when not installed
            from litellm import acompletion

            response = await acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            content = response.choices[0].message.content or ""

            logger.info(
                "llm_call_complete",
                model=model,
                tier=tier,
                response_length=len(content),
            )

            return content

        except ImportError as e:
            logger.error("litellm_not_installed", error=str(e))
            raise LLMConfigurationError("LiteLLM is not installed. Run: uv add litellm") from e

        except Exception as e:
            logger.error(
                "llm_call_failed",
                model=model,
                tier=tier,
                error=str(e),
            )
            raise LLMProviderError(f"LLM call failed: {e}") from e

    async def call_with_fallback(
        self,
        messages: list[dict[str, str]],
        primary_tier: ModelTier = "complex",
        fallback_tier: ModelTier = "routine",
        **kwargs: Any,
    ) -> str:
        """Call LLM with fallback to a lower tier on failure.

        Attempts the primary tier first, then falls back to the fallback
        tier if all retries are exhausted.

        Args:
            messages: List of message dicts with role and content.
            primary_tier: Primary tier to try first. Defaults to "complex".
            fallback_tier: Fallback tier if primary fails. Defaults to "routine".
            **kwargs: Additional arguments passed to call().

        Returns:
            Response content string from the LLM.

        Raises:
            LLMProviderError: When both tiers fail.

        Example:
            >>> response = await router.call_with_fallback(
            ...     messages=[{"role": "user", "content": "Generate code"}],
            ...     primary_tier="complex",
            ...     fallback_tier="routine",
            ... )
        """
        try:
            return await self.call(messages, tier=primary_tier, **kwargs)
        except LLMProviderError:
            logger.warning(
                "llm_falling_back",
                primary_tier=primary_tier,
                fallback_tier=fallback_tier,
            )
            return await self.call(messages, tier=fallback_tier, **kwargs)
