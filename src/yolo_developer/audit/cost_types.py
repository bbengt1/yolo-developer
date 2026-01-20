"""Type definitions for token/cost tracking (Story 11.6).

This module provides the data types used for tracking LLM token usage and costs:

- CostGroupBy: Literal type for aggregation dimensions
- TokenUsage: Token counts for a single LLM call
- CostRecord: Complete cost record for an LLM call
- CostAggregation: Aggregated cost statistics

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.audit.cost_types import (
    ...     TokenUsage,
    ...     CostRecord,
    ...     CostAggregation,
    ... )
    >>>
    >>> usage = TokenUsage(
    ...     prompt_tokens=100,
    ...     completion_tokens=50,
    ...     total_tokens=150,
    ... )
    >>> usage.to_dict()
    {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150}

References:
    - FR86: System can track token usage and cost per operation
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
    - ADR-003: LiteLLM for unified provider access with built-in cost tracking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

# Use standard logging for dataclass validation (structlog not available at import time)
_logger = logging.getLogger(__name__)

# =============================================================================
# Literal Types (Subtask 1.1)
# =============================================================================

CostGroupBy = Literal["agent", "story", "sprint", "model", "tier"]
"""Dimension for grouping cost aggregations.

Values:
    agent: Group by agent_name
    story: Group by story_id
    sprint: Group by sprint_id
    model: Group by model name
    tier: Group by model tier (routine, complex, critical)
"""

# =============================================================================
# Constants (Subtask 1.5)
# =============================================================================

VALID_GROUPBY_VALUES: frozenset[str] = frozenset(
    {
        "agent",
        "story",
        "sprint",
        "model",
        "tier",
    }
)
"""Set of valid group-by dimension values."""

VALID_TIER_VALUES: frozenset[str] = frozenset(
    {
        "routine",
        "complex",
        "critical",
    }
)
"""Set of valid model tier values (per ADR-003)."""


# =============================================================================
# Data Classes (Subtasks 1.2, 1.3, 1.4)
# =============================================================================


@dataclass(frozen=True)
class TokenUsage:
    """Token usage for a single LLM call.

    Represents the token counts returned by LiteLLM's response.usage object.

    Attributes:
        prompt_tokens: Number of tokens in the input prompt
        completion_tokens: Number of tokens in the output completion
        total_tokens: Total tokens (prompt + completion)

    Example:
        >>> usage = TokenUsage(
        ...     prompt_tokens=100,
        ...     completion_tokens=50,
        ...     total_tokens=150,
        ... )
        >>> usage.total_tokens
        150
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        """Validate token usage data and log warnings for issues."""
        if self.prompt_tokens < 0:
            _logger.warning("TokenUsage prompt_tokens is negative: %d", self.prompt_tokens)
        if self.completion_tokens < 0:
            _logger.warning("TokenUsage completion_tokens is negative: %d", self.completion_tokens)
        if self.total_tokens < 0:
            _logger.warning("TokenUsage total_tokens is negative: %d", self.total_tokens)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation of the token usage.
        """
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class CostRecord:
    """Cost record for a single LLM call.

    Represents a complete record of token usage and cost for one LLM API call,
    including metadata for aggregation by agent, story, sprint, etc.

    Attributes:
        id: Unique identifier for the cost record (UUID)
        timestamp: ISO 8601 timestamp when the call was made
        model: LLM model identifier (e.g., "gpt-4o-mini")
        tier: Model tier per ADR-003 (routine, complex, critical)
        token_usage: Token counts for this call
        cost_usd: Cost in USD for this call
        agent_name: Name of the agent that made the call
        session_id: Session identifier for grouping
        story_id: Story identifier for per-story tracking (optional)
        sprint_id: Sprint identifier for per-sprint tracking (optional)
        metadata: Additional key-value data (optional)

    Example:
        >>> usage = TokenUsage(100, 50, 150)
        >>> record = CostRecord(
        ...     id="cost-001",
        ...     timestamp="2026-01-18T12:00:00Z",
        ...     model="gpt-4o-mini",
        ...     tier="routine",
        ...     token_usage=usage,
        ...     cost_usd=0.0015,
        ...     agent_name="analyst",
        ...     session_id="session-123",
        ... )
        >>> record.cost_usd
        0.0015
    """

    id: str
    timestamp: str
    model: str
    tier: str
    token_usage: TokenUsage
    cost_usd: float
    agent_name: str
    session_id: str
    story_id: str | None = None
    sprint_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate cost record data and log warnings for issues."""
        if not self.id:
            _logger.warning("CostRecord id is empty")
        if not self.timestamp:
            _logger.warning("CostRecord timestamp is empty for id=%s", self.id)
        if not self.model:
            _logger.warning("CostRecord model is empty for id=%s", self.id)
        if self.tier not in VALID_TIER_VALUES:
            _logger.warning(
                "CostRecord tier='%s' is not a valid tier for id=%s",
                self.tier,
                self.id,
            )
        if self.cost_usd < 0:
            _logger.warning(
                "CostRecord cost_usd is negative for id=%s: %f",
                self.id,
                self.cost_usd,
            )
        if not self.agent_name:
            _logger.warning("CostRecord agent_name is empty for id=%s", self.id)
        if not self.session_id:
            _logger.warning("CostRecord session_id is empty for id=%s", self.id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation with nested token_usage.
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "model": self.model,
            "tier": self.tier,
            "token_usage": self.token_usage.to_dict(),
            "cost_usd": self.cost_usd,
            "agent_name": self.agent_name,
            "session_id": self.session_id,
            "story_id": self.story_id,
            "sprint_id": self.sprint_id,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class CostAggregation:
    """Aggregated cost statistics.

    Represents aggregated token usage and cost data across multiple LLM calls,
    optionally filtered by time period or grouped by a dimension.

    Attributes:
        total_prompt_tokens: Sum of prompt tokens across all calls
        total_completion_tokens: Sum of completion tokens across all calls
        total_tokens: Sum of total tokens across all calls
        total_cost_usd: Sum of costs in USD across all calls
        call_count: Number of LLM calls in this aggregation
        models: Tuple of unique model names used
        period_start: Start of the aggregation period (optional)
        period_end: End of the aggregation period (optional)

    Example:
        >>> agg = CostAggregation(
        ...     total_prompt_tokens=1000,
        ...     total_completion_tokens=500,
        ...     total_tokens=1500,
        ...     total_cost_usd=0.015,
        ...     call_count=10,
        ...     models=("gpt-4o-mini",),
        ... )
        >>> agg.total_cost_usd
        0.015
    """

    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_cost_usd: float
    call_count: int
    models: tuple[str, ...]
    period_start: str | None = None
    period_end: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Returns:
            Dictionary representation of the cost aggregation.
        """
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "call_count": self.call_count,
            "models": list(self.models),
            "period_start": self.period_start,
            "period_end": self.period_end,
        }
