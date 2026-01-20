"""Tests for LiteLLM integration utilities (Story 11.6).

Tests for extract_token_usage, extract_cost, and calculate_cost_if_missing functions.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from yolo_developer.audit.cost_types import TokenUsage
from yolo_developer.audit.cost_utils import (
    calculate_cost_if_missing,
    extract_cost,
    extract_token_usage,
)


def _make_mock_response(
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    total_tokens: int | None = None,
    response_cost: float | None = 0.0015,
) -> MagicMock:
    """Create a mock LiteLLM response object.

    Args:
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        total_tokens: Total tokens (defaults to prompt + completion)
        response_cost: Cost in USD (None to simulate missing cost)

    Returns:
        Mock response object mimicking LiteLLM response structure
    """
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens

    mock = MagicMock()

    # Mock usage attribute
    mock.usage.prompt_tokens = prompt_tokens
    mock.usage.completion_tokens = completion_tokens
    mock.usage.total_tokens = total_tokens

    # Mock _hidden_params for cost
    if response_cost is not None:
        mock._hidden_params = {"response_cost": response_cost}
    else:
        mock._hidden_params = {}

    # Mock model for cost calculation
    mock.model = "gpt-4o-mini"

    return mock


class TestExtractTokenUsage:
    """Tests for extract_token_usage function."""

    def test_extract_token_usage_basic(self) -> None:
        """Test extracting token usage from a basic response."""
        response = _make_mock_response(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        result = extract_token_usage(response)

        assert isinstance(result, TokenUsage)
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150

    def test_extract_token_usage_zero_tokens(self) -> None:
        """Test extracting token usage with zero tokens."""
        response = _make_mock_response(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

        result = extract_token_usage(response)

        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0

    def test_extract_token_usage_large_values(self) -> None:
        """Test extracting token usage with large values."""
        response = _make_mock_response(
            prompt_tokens=100000,
            completion_tokens=50000,
            total_tokens=150000,
        )

        result = extract_token_usage(response)

        assert result.prompt_tokens == 100000
        assert result.completion_tokens == 50000
        assert result.total_tokens == 150000

    def test_extract_token_usage_missing_usage(self) -> None:
        """Test extracting from response with no usage attribute."""
        response = MagicMock(spec=[])  # No attributes
        del response.usage  # Ensure usage doesn't exist

        result = extract_token_usage(response)

        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0

    def test_extract_token_usage_none_values(self) -> None:
        """Test extracting when token values are None."""
        response = MagicMock()
        response.usage.prompt_tokens = None
        response.usage.completion_tokens = None
        response.usage.total_tokens = None

        result = extract_token_usage(response)

        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0


class TestExtractCost:
    """Tests for extract_cost function."""

    def test_extract_cost_basic(self) -> None:
        """Test extracting cost from response with cost."""
        response = _make_mock_response(response_cost=0.0015)

        result = extract_cost(response)

        assert result == pytest.approx(0.0015)

    def test_extract_cost_zero(self) -> None:
        """Test extracting zero cost."""
        response = _make_mock_response(response_cost=0.0)

        result = extract_cost(response)

        assert result == 0.0

    def test_extract_cost_large_value(self) -> None:
        """Test extracting large cost value."""
        response = _make_mock_response(response_cost=1.23456789)

        result = extract_cost(response)

        assert result == pytest.approx(1.23456789)

    def test_extract_cost_missing_hidden_params(self) -> None:
        """Test extracting cost when _hidden_params is missing."""
        response = MagicMock(spec=[])  # No attributes

        result = extract_cost(response)

        assert result == 0.0

    def test_extract_cost_missing_response_cost_key(self) -> None:
        """Test extracting cost when response_cost key is missing."""
        response = _make_mock_response(response_cost=None)

        result = extract_cost(response)

        assert result == 0.0

    def test_extract_cost_non_numeric(self) -> None:
        """Test extracting cost when value is non-numeric."""
        response = MagicMock()
        response._hidden_params = {"response_cost": "not a number"}

        result = extract_cost(response)

        assert result == 0.0

    def test_extract_cost_none_response(self) -> None:
        """Test extracting cost from None response."""
        result = extract_cost(None)

        assert result == 0.0


class TestCalculateCostIfMissing:
    """Tests for calculate_cost_if_missing function."""

    def test_calculate_cost_returns_existing_cost(self) -> None:
        """Test that existing cost is returned without calculation."""
        response = _make_mock_response(response_cost=0.0015)

        result = calculate_cost_if_missing(response)

        assert result == pytest.approx(0.0015)

    def test_calculate_cost_when_missing(self) -> None:
        """Test that cost is calculated when missing from response."""
        response = _make_mock_response(response_cost=None)

        with patch("litellm.completion_cost") as mock_cost:
            mock_cost.return_value = 0.002
            result = calculate_cost_if_missing(response)

        assert result == pytest.approx(0.002)
        mock_cost.assert_called_once_with(completion_response=response)

    def test_calculate_cost_zero_existing(self) -> None:
        """Test that zero cost triggers calculation."""
        response = _make_mock_response(response_cost=0.0)

        with patch("litellm.completion_cost") as mock_cost:
            mock_cost.return_value = 0.001
            result = calculate_cost_if_missing(response)

        assert result == pytest.approx(0.001)
        mock_cost.assert_called_once()

    def test_calculate_cost_litellm_error(self) -> None:
        """Test fallback when litellm.completion_cost fails."""
        response = _make_mock_response(response_cost=None)

        with patch("litellm.completion_cost") as mock_cost:
            mock_cost.side_effect = Exception("API error")
            result = calculate_cost_if_missing(response)

        assert result == 0.0

    def test_calculate_cost_none_response(self) -> None:
        """Test calculation with None response."""
        result = calculate_cost_if_missing(None)

        assert result == 0.0

    def test_calculate_cost_preserves_positive_value(self) -> None:
        """Test that positive cost is preserved without recalculation."""
        response = _make_mock_response(response_cost=0.05)

        with patch("litellm.completion_cost") as mock_cost:
            result = calculate_cost_if_missing(response)

        assert result == pytest.approx(0.05)
        mock_cost.assert_not_called()
