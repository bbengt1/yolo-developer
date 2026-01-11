"""Prompt templates for Dev agent code generation (Story 8.2).

This module provides structured prompt templates for LLM-powered code generation.
Templates include maintainability guidelines and project conventions.

Example:
    >>> from yolo_developer.agents.dev.prompts import build_code_generation_prompt
    >>>
    >>> prompt = build_code_generation_prompt(
    ...     story_title="User Authentication",
    ...     requirements="Implement login functionality",
    ...     acceptance_criteria=["Users can login with email/password"],
    ...     design_decisions={"pattern": "Repository"},
    ...     project_conventions={"naming": "snake_case"},
    ... )
"""

from __future__ import annotations

from yolo_developer.agents.dev.prompts.code_generation import (
    CODE_GENERATION_TEMPLATE,
    MAINTAINABILITY_GUIDELINES,
    PROJECT_CONVENTIONS,
    build_code_generation_prompt,
    build_retry_prompt,
)

__all__ = [
    "CODE_GENERATION_TEMPLATE",
    "MAINTAINABILITY_GUIDELINES",
    "PROJECT_CONVENTIONS",
    "build_code_generation_prompt",
    "build_retry_prompt",
]
