"""Seed utilities for shared helpers."""

from __future__ import annotations


def resolve_litellm_model(model: str) -> str:
    """Prefix common model names with a LiteLLM provider."""
    if model.startswith(("openai/", "anthropic/")):
        return model
    if model.startswith("claude-"):
        return f"anthropic/{model}"
    if model.startswith(("gpt-", "o1-", "o3-")):
        return f"openai/{model}"
    return model
