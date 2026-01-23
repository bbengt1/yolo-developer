"""Tests for seed utility helpers."""

from __future__ import annotations

from yolo_developer.seed.utils import resolve_litellm_model


def test_resolve_litellm_model_openai_prefix() -> None:
    assert resolve_litellm_model("gpt-5.2-instant") == "openai/gpt-5.2-instant"


def test_resolve_litellm_model_anthropic_prefix() -> None:
    assert resolve_litellm_model("claude-sonnet-4-20250514") == "anthropic/claude-sonnet-4-20250514"


def test_resolve_litellm_model_preserves_prefixed() -> None:
    assert resolve_litellm_model("openai/gpt-5.2-instant") == "openai/gpt-5.2-instant"
