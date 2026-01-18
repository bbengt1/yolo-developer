"""LiteLLM integration utilities for cost tracking (Story 11.6).

This module provides utilities for extracting token usage and cost data
from LiteLLM response objects.

Functions:
    extract_token_usage: Extract token counts from LiteLLM response
    extract_cost: Extract cost in USD from LiteLLM response
    calculate_cost_if_missing: Calculate cost when not in response

Example:
    >>> from yolo_developer.audit.cost_utils import extract_token_usage, extract_cost
    >>>
    >>> # After an LLM call
    >>> response = await acompletion(model="gpt-4o-mini", messages=[...])
    >>> token_usage = extract_token_usage(response)
    >>> cost_usd = extract_cost(response)
    >>>
    >>> # Or calculate if missing
    >>> cost_usd = calculate_cost_if_missing(response)

References:
    - FR86: System can track token usage and cost per operation
    - ADR-003: LiteLLM for unified provider access with built-in cost tracking
    - LiteLLM Token Usage: https://docs.litellm.ai/docs/completion/token_usage

LiteLLM Response Structure:
    response.usage.prompt_tokens: int - Number of prompt tokens
    response.usage.completion_tokens: int - Number of completion tokens
    response.usage.total_tokens: int - Total tokens
    response._hidden_params["response_cost"]: float - Cost in USD (if available)
"""

from __future__ import annotations

from typing import Any

import structlog

from yolo_developer.audit.cost_types import TokenUsage

logger = structlog.get_logger(__name__)


def extract_token_usage(response: Any) -> TokenUsage:
    """Extract token usage from LiteLLM response.

    Extracts prompt_tokens, completion_tokens, and total_tokens from
    a LiteLLM completion response object.

    Args:
        response: Response object from litellm.completion/acompletion.
            Expected to have a `usage` attribute with token counts.

    Returns:
        TokenUsage dataclass with prompt, completion, and total tokens.
        Returns zeros if usage data is unavailable.

    Example:
        >>> response = await acompletion(model="gpt-4o-mini", messages=[...])
        >>> usage = extract_token_usage(response)
        >>> print(f"Used {usage.total_tokens} tokens")
    """
    try:
        usage = response.usage
        prompt_tokens = usage.prompt_tokens or 0
        completion_tokens = usage.completion_tokens or 0
        total_tokens = usage.total_tokens or 0

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
    except (AttributeError, TypeError):
        logger.warning(
            "token_usage_extraction_failed",
            reason="Missing or invalid usage attribute",
        )
        return TokenUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )


def extract_cost(response: Any) -> float:
    """Extract cost in USD from LiteLLM response.

    Extracts the response cost from the _hidden_params dictionary
    that LiteLLM populates with cost information.

    Args:
        response: Response object from litellm.completion/acompletion.
            Expected to have `_hidden_params["response_cost"]`.

    Returns:
        Cost in USD as a float. Returns 0.0 if cost is not available.

    Note:
        Falls back to 0.0 if:
        - Response is None
        - _hidden_params is missing
        - response_cost key is missing
        - Value is not a valid number

    Example:
        >>> response = await acompletion(model="gpt-4o-mini", messages=[...])
        >>> cost = extract_cost(response)
        >>> print(f"Cost: ${cost:.4f}")
    """
    try:
        if response is None:
            return 0.0

        hidden_params = response._hidden_params
        cost = hidden_params.get("response_cost", 0.0)
        return float(cost)
    except (AttributeError, KeyError, TypeError, ValueError):
        return 0.0


def calculate_cost_if_missing(response: Any) -> float:
    """Calculate cost using LiteLLM when not available in response.

    First attempts to extract cost from response. If the cost is
    missing or zero, uses litellm.completion_cost() to calculate it.

    Args:
        response: Response object from litellm.completion/acompletion.

    Returns:
        Cost in USD as a float. Returns 0.0 if calculation fails.

    Note:
        Uses litellm.completion_cost() for calculation when needed.
        This function requires the litellm package to be installed.

    Example:
        >>> response = await acompletion(model="gpt-4o-mini", messages=[...])
        >>> cost = calculate_cost_if_missing(response)  # Calculates if needed
    """
    if response is None:
        return 0.0

    # First try to extract existing cost
    existing_cost = extract_cost(response)
    if existing_cost > 0.0:
        return existing_cost

    # Calculate using litellm
    try:
        from litellm import completion_cost

        calculated_cost = completion_cost(completion_response=response)
        logger.debug(
            "cost_calculated",
            cost_usd=calculated_cost,
            source="litellm.completion_cost",
        )
        return float(calculated_cost)
    except Exception as e:
        logger.warning(
            "cost_calculation_failed",
            error=str(e),
            error_type=type(e).__name__,
        )
        return 0.0
