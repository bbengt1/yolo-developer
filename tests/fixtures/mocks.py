"""Mock objects for testing LLM and external services."""

from __future__ import annotations

from typing import Any  # noqa: F401


class MockLLMResponse:
    """Mock response from LLM calls."""

    def __init__(self, content: str) -> None:
        self.content = content


# Add more mocks as needed during implementation
