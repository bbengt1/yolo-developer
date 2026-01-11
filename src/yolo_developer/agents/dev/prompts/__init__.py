"""Prompt templates for Dev agent code and test generation (Story 8.2, 8.3, 8.4, 8.5).

This module provides structured prompt templates for LLM-powered code, test,
and documentation generation. Templates include maintainability guidelines,
project conventions, testing best practices, integration test guidance, and
documentation standards.

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
from yolo_developer.agents.dev.prompts.documentation_generation import (
    DOCUMENTATION_GENERATION_TEMPLATE,
    DOCUMENTATION_GUIDELINES,
    FUNCTION_DOCSTRING_TEMPLATE,
    MODULE_DOCSTRING_TEMPLATE,
    build_documentation_prompt,
    build_documentation_retry_prompt,
)
from yolo_developer.agents.dev.prompts.integration_test_generation import (
    INTEGRATION_TEST_TEMPLATE,
    INTEGRATION_TESTING_BEST_PRACTICES,
    PYTEST_INTEGRATION_CONVENTIONS,
    build_integration_test_prompt,
    build_integration_test_retry_prompt,
)
from yolo_developer.agents.dev.prompts.test_generation import (
    PYTEST_CONVENTIONS,
    TEST_GENERATION_TEMPLATE,
    TESTING_BEST_PRACTICES,
    build_test_generation_prompt,
    build_test_retry_prompt,
)

__all__ = [
    "CODE_GENERATION_TEMPLATE",
    "DOCUMENTATION_GENERATION_TEMPLATE",
    "DOCUMENTATION_GUIDELINES",
    "FUNCTION_DOCSTRING_TEMPLATE",
    "INTEGRATION_TESTING_BEST_PRACTICES",
    "INTEGRATION_TEST_TEMPLATE",
    "MAINTAINABILITY_GUIDELINES",
    "MODULE_DOCSTRING_TEMPLATE",
    "PROJECT_CONVENTIONS",
    "PYTEST_CONVENTIONS",
    "PYTEST_INTEGRATION_CONVENTIONS",
    "TESTING_BEST_PRACTICES",
    "TEST_GENERATION_TEMPLATE",
    "build_code_generation_prompt",
    "build_documentation_prompt",
    "build_documentation_retry_prompt",
    "build_integration_test_prompt",
    "build_integration_test_retry_prompt",
    "build_retry_prompt",
    "build_test_generation_prompt",
    "build_test_retry_prompt",
]
