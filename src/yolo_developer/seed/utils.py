"""Seed utilities for shared helpers."""

from __future__ import annotations

from typing import Literal

from yolo_developer.config.schema import LLMConfig

ProviderName = Literal["openai", "anthropic", "auto"]

def resolve_litellm_model(model: str) -> str:
    """Prefix common model names with a LiteLLM provider."""
    if model.startswith(("openai/", "anthropic/")):
        return model
    if model.startswith("claude-"):
        return f"anthropic/{model}"
    if model.startswith(("gpt-", "o1-", "o3-")):
        return f"openai/{model}"
    return model


def resolve_provider_from_model(model: str) -> ProviderName:
    if model.startswith("anthropic/") or model.startswith("claude-"):
        return "anthropic"
    if model.startswith("openai/") or model.startswith(("gpt-", "o1-", "o3-")):
        return "openai"
    return "auto"


def get_api_key_for_model(model: str, config: LLMConfig) -> str | None:
    provider = resolve_provider_from_model(model)
    if provider == "openai":
        api_key = config.openai.api_key or config.openai_api_key
        return api_key.get_secret_value() if api_key else None
    if provider == "anthropic":
        api_key = config.anthropic_api_key
        return api_key.get_secret_value() if api_key else None
    return None
