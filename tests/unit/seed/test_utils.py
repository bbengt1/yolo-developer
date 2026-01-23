"""Tests for seed utility helpers."""

from __future__ import annotations

from yolo_developer.config.schema import LLMConfig
from yolo_developer.seed.utils import get_api_key_for_model, resolve_litellm_model


def test_resolve_litellm_model_openai_prefix() -> None:
    assert resolve_litellm_model("gpt-5.2-instant") == "openai/gpt-5.2-instant"


def test_resolve_litellm_model_anthropic_prefix() -> None:
    assert resolve_litellm_model("claude-sonnet-4-20250514") == "anthropic/claude-sonnet-4-20250514"


def test_resolve_litellm_model_preserves_prefixed() -> None:
    assert resolve_litellm_model("openai/gpt-5.2-instant") == "openai/gpt-5.2-instant"


def test_get_api_key_for_model_openai() -> None:
    config = LLMConfig(openai_api_key="sk-test-openai")

    assert get_api_key_for_model("gpt-5.2-instant", config) == "sk-test-openai"


def test_get_api_key_for_model_anthropic() -> None:
    config = LLMConfig(anthropic_api_key="sk-test-anthropic")

    assert get_api_key_for_model("claude-sonnet-4-20250514", config) == "sk-test-anthropic"
