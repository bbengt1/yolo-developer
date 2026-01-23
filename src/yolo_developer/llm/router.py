"""LLM Router for multi-provider abstraction (Story 8.2, ADR-003).

This module provides the LLMRouter class for routing LLM calls to the
appropriate provider and model based on task tier.

Model Tiers (per architecture):
    - "routine": cheap_model (gpt-5.2-instant) for routine tasks
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

from dataclasses import dataclass
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
TaskType = Literal[
    "code_generation",
    "code_review",
    "architecture",
    "analysis",
    "documentation",
    "testing",
]


@dataclass(frozen=True)
class TaskRouting:
    """Routing decision for a task-based LLM call."""

    task_type: TaskType
    provider: Literal["openai", "anthropic", "auto"]
    model: str
    tier: ModelTier


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
            >>> config = LLMConfig(cheap_model="gpt-5.2-instant")
            >>> router = LLMRouter(config)
        """
        self.config = config
        self.model_map: dict[ModelTier, str] = {
            "routine": config.cheap_model,
            "complex": config.premium_model,
            "critical": config.best_model,
        }
        self._usage_log: list[TaskRouting] = []

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

    def get_task_routing(self, task_type: TaskType) -> TaskRouting:
        """Resolve provider/model selection for a task type."""
        tier = _task_tier(task_type)
        provider = self._provider_for_task(task_type)

        if provider == "openai":
            model = _openai_model_for_task(self.config, task_type, tier)
        elif provider == "anthropic":
            model = self.get_model_for_tier(tier)
        else:
            model = self.get_model_for_tier(tier)
            provider = _provider_from_model(model)

        return TaskRouting(
            task_type=task_type,
            provider=provider,
            model=model,
            tier=tier,
        )

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
        provider = _provider_from_model(model)
        api_key = self._api_key_for_provider(provider, allow_missing=True)
        litellm_model = _litellm_model(provider, model)

        # Adjust temperature for models with constraints (e.g., gpt-5)
        adjusted_temperature = _adjust_temperature_for_model(model, temperature)

        logger.info(
            "llm_call_start",
            model=litellm_model,
            tier=tier,
            provider=provider,
            message_count=len(messages),
            max_tokens=max_tokens,
        )

        try:
            # Import litellm here to avoid import errors when not installed
            from litellm import acompletion

            response = await acompletion(
                model=litellm_model,
                messages=messages,
                temperature=adjusted_temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                **kwargs,
            )

            content = response.choices[0].message.content or ""

            logger.info(
                "llm_call_complete",
                model=model,
                tier=tier,
                provider=provider,
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

    async def call_task(
        self,
        messages: list[dict[str, str]],
        task_type: TaskType,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        """Call LLM using task-based routing."""
        routing = self.get_task_routing(task_type)
        resolved_provider = (
            _provider_from_model(routing.model)
            if routing.provider == "auto"
            else routing.provider
        )
        api_key = self._api_key_for_provider(resolved_provider, allow_missing=False)
        litellm_model = _litellm_model(resolved_provider, routing.model)

        # Adjust temperature for models with constraints (e.g., gpt-5)
        adjusted_temperature = _adjust_temperature_for_model(routing.model, temperature)

        logger.info(
            "llm_task_call_start",
            task_type=task_type,
            provider=resolved_provider,
            model=litellm_model,
            tier=routing.tier,
        )

        try:
            from litellm import acompletion

            response = await acompletion(
                model=litellm_model,
                messages=messages,
                temperature=adjusted_temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                **kwargs,
            )

            content = response.choices[0].message.content or ""

            logger.info(
                "llm_task_call_complete",
                task_type=task_type,
                provider=resolved_provider,
                model=routing.model,
                response_length=len(content),
            )

            self._usage_log.append(
                TaskRouting(
                    task_type=routing.task_type,
                    provider=resolved_provider,
                    model=routing.model,
                    tier=routing.tier,
                )
            )

            return content

        except ImportError as e:
            logger.error("litellm_not_installed", error=str(e))
            raise LLMConfigurationError("LiteLLM is not installed. Run: uv add litellm") from e

        except Exception as e:
            logger.error(
                "llm_task_call_failed",
                task_type=task_type,
                provider=resolved_provider,
                model=routing.model,
                error=str(e),
            )
            raise LLMProviderError(f"LLM call failed: {e}") from e

    def get_usage_log(self) -> tuple[TaskRouting, ...]:
        """Return a snapshot of task routing usage."""
        return tuple(self._usage_log)

    def clear_usage_log(self) -> None:
        """Clear stored task routing usage."""
        self._usage_log.clear()

    def _provider_for_task(self, task_type: TaskType) -> Literal["openai", "anthropic", "auto"]:
        if self.config.provider == "hybrid" or self.config.hybrid.enabled:
            routing = self.config.hybrid.routing
            return getattr(routing, task_type)
        if self.config.provider in ("openai", "anthropic"):
            return self.config.provider
        return "auto"

    def _api_key_for_provider(
        self,
        provider: Literal["openai", "anthropic", "auto"],
        *,
        allow_missing: bool,
    ) -> str | None:
        if provider == "openai":
            api_key = self.config.openai.api_key
            if api_key is None:
                api_key = self.config.openai_api_key
            if api_key is None and not allow_missing:
                raise LLMConfigurationError("Missing OpenAI API key (llm.openai.api_key)")
            return api_key.get_secret_value() if api_key else None
        if provider == "anthropic":
            api_key = self.config.anthropic_api_key
            if api_key is None and not allow_missing:
                raise LLMConfigurationError(
                    "Missing Anthropic API key (llm.anthropic_api_key)"
                )
            return api_key.get_secret_value() if api_key else None
        return None


def _task_tier(task_type: TaskType) -> ModelTier:
    tier_map: dict[TaskType, ModelTier] = {
        "code_generation": "complex",
        "code_review": "complex",
        "architecture": "complex",
        "analysis": "complex",
        "documentation": "routine",
        "testing": "complex",
    }
    return tier_map[task_type]


def _openai_model_for_task(config: LLMConfig, task_type: TaskType, tier: ModelTier) -> str:
    if task_type in ("code_generation", "code_review", "testing"):
        return config.openai.code_model
    if task_type == "documentation":
        return config.openai.cheap_model
    if task_type in ("analysis", "architecture"):
        return config.openai.reasoning_model or config.openai.premium_model
    return config.openai.premium_model


def _provider_from_model(model: str) -> Literal["openai", "anthropic", "auto"]:
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith(("gpt-", "o1-", "o3-")):
        return "openai"
    return "auto"


def _litellm_model(provider: Literal["openai", "anthropic", "auto"], model: str) -> str:
    if model.startswith(("openai/", "anthropic/")):
        return model
    if provider in ("openai", "anthropic"):
        return f"{provider}/{model}"
    return model


def _adjust_temperature_for_model(model: str, temperature: float) -> float:
    """Adjust temperature based on model constraints.

    Some models have restrictions on temperature values:
    - gpt-5 and gpt-5-mini/codex only support temperature=1
    - gpt-5.1+ support variable temperature with reasoning_effort='none'
    - o1/o3 reasoning models may have similar constraints

    Args:
        model: The model identifier.
        temperature: The requested temperature.

    Returns:
        The adjusted temperature value.
    """
    model_lower = model.lower()

    # gpt-5 base models (not gpt-5.1, gpt-5.2, etc.) only support temperature=1
    # Models like "gpt-5", "gpt-5-mini", "gpt-5-codex" need temperature=1
    # But "gpt-5.1-*", "gpt-5.2-*" support variable temperature
    if "gpt-5" in model_lower:
        # Check if it's a versioned model (gpt-5.1, gpt-5.2, etc.)
        import re

        # Match gpt-5 followed by a dot and number (e.g., gpt-5.1, gpt-5.2)
        if re.search(r"gpt-5\.\d", model_lower):
            # Versioned models support variable temperature
            return temperature
        # Base gpt-5 models only support temperature=1
        return 1.0

    return temperature
