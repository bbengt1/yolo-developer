"""Integration tests for Dev agent code generation (Story 8.2 - Task 10).

Tests the full code generation flow including prompt construction,
LLM integration (mocked), and syntax validation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yolo_developer.agents.dev.code_utils import (
    check_maintainability,
    extract_code_from_response,
    validate_python_syntax,
)
from yolo_developer.agents.dev.node import (
    _extract_project_context,
    _generate_code_with_llm,
    _generate_implementation,
)
from yolo_developer.agents.dev.prompts import build_code_generation_prompt
from yolo_developer.config.schema import LLMConfig
from yolo_developer.llm.router import LLMRouter


class TestPromptToCodeFlow:
    """Tests for the complete prompt to code generation flow."""

    def test_prompt_includes_all_story_elements(self) -> None:
        """Test that prompt includes all story information."""
        story_title = "User Authentication"
        requirements = "Implement login with email and password"
        acceptance_criteria = ["Users can login", "Invalid creds rejected"]
        design_decisions = {"pattern": "Repository", "db": "PostgreSQL"}

        prompt = build_code_generation_prompt(
            story_title=story_title,
            requirements=requirements,
            acceptance_criteria=acceptance_criteria,
            design_decisions=design_decisions,
        )

        assert story_title in prompt
        assert requirements in prompt
        assert "Users can login" in prompt
        assert "Repository" in prompt
        assert "50 lines" in prompt  # Maintainability guidelines

    def test_code_extraction_and_validation_flow(self) -> None:
        """Test code extraction and validation flow."""
        # Simulate LLM response
        llm_response = '''Here's the implementation:

```python
from __future__ import annotations


def authenticate_user(email: str, password: str) -> dict[str, str]:
    """Authenticate a user with email and password.

    Args:
        email: User's email address.
        password: User's password.

    Returns:
        Dict with authentication result.
    """
    if not email or not password:
        return {"status": "error", "message": "Missing credentials"}
    return {"status": "success", "user": email}
```

This function handles the basic authentication logic.
'''

        # Extract code
        code = extract_code_from_response(llm_response)
        assert "def authenticate_user" in code

        # Validate syntax
        is_valid, error = validate_python_syntax(code)
        assert is_valid is True
        assert error is None

        # Check maintainability
        report = check_maintainability(code)
        assert report.function_count == 1
        assert len(report.get_warnings_by_category("function_length")) == 0

    def test_maintainability_check_detects_issues(self) -> None:
        """Test that maintainability check detects code issues."""
        # Code with issues: long function, deep nesting
        problematic_code = '''
def process_everything(data):
    result = []
    if data:
        for item in data:
            if item:
                for subitem in item:
                    if subitem:
                        if subitem.value:
                            result.append(subitem.value)
    ''' + "\n".join([f"    x{i} = {i}" for i in range(55)]) + '''
    return result
'''
        report = check_maintainability(problematic_code)
        assert report.has_warnings() is True
        # Should have function length warning
        assert report.max_function_length > 50


class TestProjectContextExtraction:
    """Tests for project context extraction."""

    def test_extracts_architect_patterns(self) -> None:
        """Test extraction of patterns from architect_output."""
        state = {
            "architect_output": {
                "design_decisions": [
                    {"pattern": "Repository Pattern", "story_id": "s1"},
                    {"pattern": "Factory Pattern", "story_id": "s2"},
                    {"constraint": "Must use async I/O"},
                ]
            },
            "messages": [],
        }

        context = _extract_project_context(state)

        assert "Repository Pattern" in context["patterns"]
        assert "Factory Pattern" in context["patterns"]
        assert "Must use async I/O" in context["constraints"]

    def test_includes_default_conventions(self) -> None:
        """Test that default conventions are always included."""
        state = {"messages": []}

        context = _extract_project_context(state)

        assert "naming" in context["conventions"]
        assert "snake_case" in context["conventions"]["naming"]

    def test_extracts_memory_patterns(self) -> None:
        """Test extraction of patterns from memory context."""
        state = {
            "messages": [],
            "memory_context": {
                "patterns": ["Singleton Pattern", "Observer Pattern"],
            },
        }

        context = _extract_project_context(state)

        assert "Singleton Pattern" in context["patterns"]
        assert "Observer Pattern" in context["patterns"]


class TestLLMCodeGeneration:
    """Tests for LLM-powered code generation."""

    @pytest.fixture
    def mock_router(self) -> MagicMock:
        """Create mock LLM router."""
        router = MagicMock(spec=LLMRouter)
        router.call = AsyncMock()
        return router

    @pytest.mark.asyncio
    async def test_generates_valid_code(self, mock_router: MagicMock) -> None:
        """Test that valid code is generated and returned."""
        # Mock LLM response with valid Python code
        valid_response = '''```python
from __future__ import annotations


def process_data(items: list[str]) -> dict[str, int]:
    """Process items and count them."""
    return {"count": len(items)}
```'''
        mock_router.call.return_value = valid_response

        story = {"id": "test-001", "title": "Test Story"}
        context = {"patterns": [], "constraints": [], "conventions": {}}

        code, is_valid = await _generate_code_with_llm(story, context, mock_router)

        assert is_valid is True
        assert "def process_data" in code

    @pytest.mark.asyncio
    async def test_retries_on_syntax_error(self, mock_router: MagicMock) -> None:
        """Test that code generation retries on syntax error."""
        # First response has syntax error
        invalid_response = '''```python
def broken(
    pass
```'''
        # Second response is valid
        valid_response = '''```python
def fixed() -> None:
    pass
```'''
        mock_router.call.side_effect = [invalid_response, valid_response]

        story = {"id": "test-001", "title": "Test Story"}
        context = {"patterns": [], "constraints": [], "conventions": {}}

        code, is_valid = await _generate_code_with_llm(story, context, mock_router)

        # Should have called LLM twice (original + retry)
        assert mock_router.call.call_count == 2
        # Should return valid code from retry
        assert is_valid is True
        assert "def fixed" in code


class TestFullGenerationFlow:
    """Tests for the complete generation flow including fallback."""

    @pytest.mark.asyncio
    async def test_llm_generation_with_fallback(self) -> None:
        """Test LLM generation with fallback to stub."""
        story = {"id": "test-001", "title": "Test Story"}
        context = {"patterns": [], "constraints": [], "conventions": {}}

        # Without router, should use stub
        artifact = await _generate_implementation(story, context, router=None)

        assert artifact.story_id == "test-001"
        assert artifact.implementation_status == "completed"
        assert len(artifact.code_files) >= 1
        assert "stub" in artifact.notes.lower() or "LLM" not in artifact.notes

    @pytest.mark.asyncio
    async def test_llm_generation_success(self) -> None:
        """Test successful LLM code generation."""
        mock_router = MagicMock(spec=LLMRouter)
        valid_response = '''```python
from __future__ import annotations


def implement_test_001() -> dict[str, str]:
    """Implementation for test story."""
    return {"status": "implemented", "story_id": "test-001"}
```'''
        mock_router.call = AsyncMock(return_value=valid_response)

        story = {"id": "test-001", "title": "Test Story"}
        context = {"patterns": [], "constraints": [], "conventions": {}}

        artifact = await _generate_implementation(story, context, router=mock_router)

        assert artifact.story_id == "test-001"
        assert artifact.implementation_status == "completed"
        assert "LLM" in artifact.notes

    @pytest.mark.asyncio
    async def test_falls_back_on_llm_failure(self) -> None:
        """Test fallback to stub when LLM fails."""
        mock_router = MagicMock(spec=LLMRouter)
        # Always return invalid code to trigger fallback
        mock_router.call = AsyncMock(return_value="not valid python {{{")

        story = {"id": "test-001", "title": "Test Story"}
        context = {"patterns": [], "constraints": [], "conventions": {}}

        artifact = await _generate_implementation(story, context, router=mock_router)

        assert artifact.story_id == "test-001"
        assert artifact.implementation_status == "completed"
        # Should fall back to stub
        assert "fallback" in artifact.notes.lower() or "stub" in artifact.notes.lower()
