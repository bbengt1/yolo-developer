"""Tests for PM agent LLM integration (Story 6.2).

Tests cover:
- _call_llm function signature and retry behavior
- Prompt templates are well-formed
- Response parsing with valid and invalid JSON
- Vague term detection
- Complexity estimation
- Story component extraction (stub mode)
- AC generation (stub mode)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yolo_developer.agents.pm.llm import (
    AC_SYSTEM_PROMPT,
    AC_USER_PROMPT_TEMPLATE,
    PM_SYSTEM_PROMPT,
    PM_USER_PROMPT_TEMPLATE,
    VAGUE_TERMS,
    _USE_LLM,
    _call_llm,
    _contains_vague_terms,
    _estimate_complexity,
    _extract_story_components,
    _generate_acceptance_criteria_llm,
    _parse_ac_response,
    _parse_story_response,
)


class TestPromptTemplates:
    """Tests for prompt template constants."""

    def test_pm_system_prompt_contains_json_instruction(self) -> None:
        """PM system prompt should instruct JSON output."""
        assert "JSON" in PM_SYSTEM_PROMPT
        assert "role" in PM_SYSTEM_PROMPT
        assert "action" in PM_SYSTEM_PROMPT
        assert "benefit" in PM_SYSTEM_PROMPT
        assert "title" in PM_SYSTEM_PROMPT

    def test_pm_system_prompt_warns_against_default_user(self) -> None:
        """PM system prompt should warn against defaulting to 'user' role."""
        assert "user" in PM_SYSTEM_PROMPT.lower()
        assert "NOT default" in PM_SYSTEM_PROMPT or "do NOT" in PM_SYSTEM_PROMPT

    def test_pm_user_prompt_template_has_placeholders(self) -> None:
        """PM user prompt should have all required placeholders."""
        assert "{requirement_id}" in PM_USER_PROMPT_TEMPLATE
        assert "{requirement_text}" in PM_USER_PROMPT_TEMPLATE
        assert "{category}" in PM_USER_PROMPT_TEMPLATE

    def test_pm_user_prompt_template_can_format(self) -> None:
        """PM user prompt should format without errors."""
        formatted = PM_USER_PROMPT_TEMPLATE.format(
            requirement_id="req-001",
            requirement_text="User can login",
            category="functional",
        )
        assert "req-001" in formatted
        assert "User can login" in formatted
        assert "functional" in formatted

    def test_ac_system_prompt_contains_given_when_then(self) -> None:
        """AC system prompt should mention Given/When/Then format."""
        assert "Given" in AC_SYSTEM_PROMPT
        assert "When" in AC_SYSTEM_PROMPT
        assert "Then" in AC_SYSTEM_PROMPT

    def test_ac_system_prompt_warns_against_vague_terms(self) -> None:
        """AC system prompt should warn against vague terms."""
        assert "vague" in AC_SYSTEM_PROMPT.lower()
        # Should mention at least some vague terms as examples
        assert any(term in AC_SYSTEM_PROMPT.lower() for term in ["fast", "easy", "simple", "should"])

    def test_ac_user_prompt_template_has_placeholders(self) -> None:
        """AC user prompt should have all required placeholders."""
        assert "{title}" in AC_USER_PROMPT_TEMPLATE
        assert "{role}" in AC_USER_PROMPT_TEMPLATE
        assert "{action}" in AC_USER_PROMPT_TEMPLATE
        assert "{benefit}" in AC_USER_PROMPT_TEMPLATE
        assert "{requirement_text}" in AC_USER_PROMPT_TEMPLATE
        assert "{requirement_id}" in AC_USER_PROMPT_TEMPLATE

    def test_ac_user_prompt_template_can_format(self) -> None:
        """AC user prompt should format without errors."""
        formatted = AC_USER_PROMPT_TEMPLATE.format(
            title="User Login",
            role="user",
            action="login with email",
            benefit="access my account",
            requirement_text="User can login with email",
            requirement_id="req-001",
        )
        assert "User Login" in formatted
        assert "req-001" in formatted


class TestVagueTerms:
    """Tests for vague term detection."""

    def test_vague_terms_frozenset_not_empty(self) -> None:
        """VAGUE_TERMS should contain multiple terms."""
        assert len(VAGUE_TERMS) > 20

    def test_vague_terms_contains_quantifier_terms(self) -> None:
        """VAGUE_TERMS should contain quantifier vagueness terms."""
        assert "fast" in VAGUE_TERMS
        assert "slow" in VAGUE_TERMS
        assert "efficient" in VAGUE_TERMS
        assert "scalable" in VAGUE_TERMS

    def test_vague_terms_contains_ease_terms(self) -> None:
        """VAGUE_TERMS should contain ease vagueness terms."""
        assert "easy" in VAGUE_TERMS
        assert "simple" in VAGUE_TERMS
        assert "intuitive" in VAGUE_TERMS

    def test_vague_terms_contains_certainty_terms(self) -> None:
        """VAGUE_TERMS should contain certainty vagueness terms."""
        assert "should" in VAGUE_TERMS
        assert "might" in VAGUE_TERMS
        assert "could" in VAGUE_TERMS
        assert "maybe" in VAGUE_TERMS

    def test_contains_vague_terms_finds_single_term(self) -> None:
        """Should find a single vague term."""
        result = _contains_vague_terms("The system should be responsive")
        assert "should" in result
        assert "responsive" in result

    def test_contains_vague_terms_finds_multiple_terms(self) -> None:
        """Should find multiple vague terms."""
        result = _contains_vague_terms("It should be fast, easy, and simple")
        assert "should" in result
        assert "fast" in result
        assert "easy" in result
        assert "simple" in result

    def test_contains_vague_terms_returns_empty_for_clean_text(self) -> None:
        """Should return empty list for text without vague terms."""
        result = _contains_vague_terms("The user clicks the login button and enters credentials")
        assert result == []

    def test_contains_vague_terms_case_insensitive(self) -> None:
        """Should be case insensitive."""
        result = _contains_vague_terms("It SHOULD be FAST")
        assert "should" in result
        assert "fast" in result

    def test_contains_vague_terms_matches_whole_words(self) -> None:
        """Should match whole words only, not substrings."""
        result = _contains_vague_terms("The breakfast was served")
        # "fast" should not be found in "breakfast"
        assert "fast" not in result


class TestComplexityEstimation:
    """Tests for complexity estimation."""

    def test_estimate_complexity_simple(self) -> None:
        """Simple read/display operations should be S."""
        result = _estimate_complexity("Display the basic list", 1)
        assert result == "S"

        result = _estimate_complexity("Show simple view", 1)
        assert result == "S"

    def test_estimate_complexity_medium(self) -> None:
        """Standard operations should be M."""
        result = _estimate_complexity("User can update their profile information", 3)
        assert result == "M"

    def test_estimate_complexity_large_with_integrations(self) -> None:
        """Operations with integrations should be L."""
        result = _estimate_complexity("Send notification via external API service", 3)
        assert result == "L"

        result = _estimate_complexity("Process background queue tasks asynchronously", 3)
        assert result == "L"

    def test_estimate_complexity_xl_with_security(self) -> None:
        """Operations with security concerns should be XL."""
        result = _estimate_complexity("Implement authentication and authorization", 4)
        assert result == "XL"

        result = _estimate_complexity("Encrypt user data with AES encryption", 5)
        assert result == "XL"

    def test_estimate_complexity_xl_with_high_ac_count(self) -> None:
        """High AC count should push to XL."""
        result = _estimate_complexity("Simple feature", 5)
        assert result == "XL"

    def test_estimate_complexity_large_with_multiple_and(self) -> None:
        """Multiple 'and' conjunctions suggest complexity."""
        result = _estimate_complexity("User can create and update and delete items", 3)
        assert result == "L"


class TestParseStoryResponse:
    """Tests for story response parsing."""

    def test_parse_story_response_valid_json(self) -> None:
        """Should parse valid JSON response."""
        response = json.dumps({
            "role": "developer",
            "action": "deploy applications",
            "benefit": "reduce manual work",
            "title": "App Deployment",
        })
        result = _parse_story_response(response)
        assert result["role"] == "developer"
        assert result["action"] == "deploy applications"
        assert result["benefit"] == "reduce manual work"
        assert result["title"] == "App Deployment"

    def test_parse_story_response_json_with_markdown(self) -> None:
        """Should extract JSON from markdown-wrapped response."""
        response = """Here is the story:
```json
{"role": "admin", "action": "manage users", "benefit": "control access", "title": "User Management"}
```
"""
        result = _parse_story_response(response)
        assert result["role"] == "admin"
        assert result["action"] == "manage users"

    def test_parse_story_response_missing_fields(self) -> None:
        """Should return empty dict if required fields missing."""
        response = json.dumps({"role": "user"})  # Missing action, benefit, title
        result = _parse_story_response(response)
        assert result == {}

    def test_parse_story_response_invalid_json(self) -> None:
        """Should return empty dict for invalid JSON."""
        result = _parse_story_response("not valid json")
        assert result == {}

    def test_parse_story_response_empty_string(self) -> None:
        """Should return empty dict for empty string."""
        result = _parse_story_response("")
        assert result == {}


class TestParseAcResponse:
    """Tests for AC response parsing."""

    def test_parse_ac_response_valid_json_array(self) -> None:
        """Should parse valid JSON array response."""
        response = json.dumps([
            {
                "given": "a user is logged in",
                "when": "they click logout",
                "then": "they are redirected to login page",
                "and_clauses": ["session is invalidated"],
            },
            {
                "given": "a user is not logged in",
                "when": "they access protected page",
                "then": "they are redirected to login",
                "and_clauses": [],
            },
        ])
        result = _parse_ac_response(response)
        assert len(result) == 2
        assert result[0]["given"] == "a user is logged in"
        assert result[0]["and_clauses"] == ["session is invalidated"]
        assert result[1]["given"] == "a user is not logged in"

    def test_parse_ac_response_json_with_markdown(self) -> None:
        """Should extract JSON array from markdown-wrapped response."""
        response = """```json
[{"given": "x", "when": "y", "then": "z", "and_clauses": []}]
```"""
        result = _parse_ac_response(response)
        assert len(result) == 1
        assert result[0]["given"] == "x"

    def test_parse_ac_response_missing_required_fields(self) -> None:
        """Should skip ACs missing required fields."""
        response = json.dumps([
            {"given": "x", "when": "y"},  # Missing 'then'
            {"given": "a", "when": "b", "then": "c", "and_clauses": []},  # Valid
        ])
        result = _parse_ac_response(response)
        assert len(result) == 1
        assert result[0]["given"] == "a"

    def test_parse_ac_response_invalid_json(self) -> None:
        """Should return empty list for invalid JSON."""
        result = _parse_ac_response("not valid json")
        assert result == []

    def test_parse_ac_response_empty_array(self) -> None:
        """Should return empty list for empty array."""
        result = _parse_ac_response("[]")
        assert result == []

    def test_parse_ac_response_not_array(self) -> None:
        """Should return empty list if not an array."""
        result = _parse_ac_response('{"given": "x", "when": "y", "then": "z"}')
        assert result == []


class TestFeatureFlag:
    """Tests for _USE_LLM feature flag."""

    def test_use_llm_flag_default_false(self) -> None:
        """_USE_LLM should default to False for testing."""
        assert _USE_LLM is False


class TestExtractStoryComponentsStub:
    """Tests for story component extraction in stub mode."""

    @pytest.mark.asyncio
    async def test_extract_story_components_stub_mode(self) -> None:
        """Should return stub data when _USE_LLM is False."""
        result = await _extract_story_components(
            requirement_id="req-001",
            requirement_text="User can login with email",
            category="functional",
        )
        assert result["role"] == "user"  # No role keyword detected, defaults to user
        assert "login with email" in result["action"]
        assert result["benefit"] == "the system meets the specified requirement"
        assert len(result["title"]) <= 50

    @pytest.mark.asyncio
    async def test_extract_story_components_stub_empty_text(self) -> None:
        """Should handle empty requirement text."""
        result = await _extract_story_components(
            requirement_id="req-002",
            requirement_text="",
            category="functional",
        )
        assert result["role"] == "user"
        assert "req-002" in result["action"]
        assert "req-002" in result["title"]

    @pytest.mark.asyncio
    async def test_extract_story_components_extracts_developer_role(self) -> None:
        """Should extract developer role from requirement text."""
        result = await _extract_story_components(
            requirement_id="req-003",
            requirement_text="Developer can deploy applications via CLI",
            category="functional",
        )
        assert result["role"] == "developer"

    @pytest.mark.asyncio
    async def test_extract_story_components_extracts_admin_role(self) -> None:
        """Should extract admin role from requirement text."""
        result = await _extract_story_components(
            requirement_id="req-004",
            requirement_text="Admin can manage user permissions",
            category="functional",
        )
        assert result["role"] == "admin"


class TestGenerateAcceptanceCriteriaStub:
    """Tests for AC generation in stub mode."""

    @pytest.mark.asyncio
    async def test_generate_ac_stub_mode(self) -> None:
        """Should return 2 ACs when _USE_LLM is False (per AC2 requirement)."""
        story_components = {
            "role": "user",
            "action": "login with email",
            "benefit": "access my account",
            "title": "User Login",
        }
        result = await _generate_acceptance_criteria_llm(
            requirement_id="req-001",
            requirement_text="User can login with email",
            story_components=story_components,
        )
        assert len(result) == 2  # AC2 requires 2-5 ACs
        assert "req-001" in result[0]["given"]
        assert "when" in result[0]
        assert "then" in result[0]
        assert "and_clauses" in result[0]
        # Second AC should be error handling scenario
        assert "error" in result[1]["when"]

    @pytest.mark.asyncio
    async def test_generate_ac_stub_returns_list(self) -> None:
        """Should always return a list with 2+ ACs."""
        story_components = {
            "role": "admin",
            "action": "manage users",
            "benefit": "control access",
            "title": "User Management",
        }
        result = await _generate_acceptance_criteria_llm(
            requirement_id="req-002",
            requirement_text="Admin manages users",
            story_components=story_components,
        )
        assert isinstance(result, list)
        assert len(result) >= 2  # AC2 requires 2-5 ACs


class TestCallLlm:
    """Tests for _call_llm function."""

    @pytest.mark.asyncio
    async def test_call_llm_signature(self) -> None:
        """_call_llm should have correct async signature."""
        import inspect
        assert inspect.iscoroutinefunction(_call_llm)

    @pytest.mark.asyncio
    async def test_call_llm_with_mock(self) -> None:
        """Should call LiteLLM with correct parameters."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"test": "response"}'

        mock_config = MagicMock()
        mock_config.llm.cheap_model = "test-model"

        with (
            patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion,
            patch("yolo_developer.config.load_config", return_value=mock_config),
        ):
            mock_acompletion.return_value = mock_response

            result = await _call_llm("test prompt", "test system")

            mock_acompletion.assert_called_once()
            call_kwargs = mock_acompletion.call_args[1]
            assert call_kwargs["model"] == "test-model"
            assert call_kwargs["messages"][0]["role"] == "system"
            assert call_kwargs["messages"][0]["content"] == "test system"
            assert call_kwargs["messages"][1]["role"] == "user"
            assert call_kwargs["messages"][1]["content"] == "test prompt"
            assert result == '{"test": "response"}'

    @pytest.mark.asyncio
    async def test_call_llm_handles_none_content(self) -> None:
        """Should return empty string if content is None."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        mock_config = MagicMock()
        mock_config.llm.cheap_model = "test-model"

        with (
            patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion,
            patch("yolo_developer.config.load_config", return_value=mock_config),
        ):
            mock_acompletion.return_value = mock_response

            result = await _call_llm("prompt", "system")

            assert result == ""

    @pytest.mark.asyncio
    async def test_call_llm_retry_exhaustion(self) -> None:
        """Should raise exception after all retry attempts fail."""
        from tenacity import RetryError

        mock_config = MagicMock()
        mock_config.llm.cheap_model = "test-model"

        with (
            patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion,
            patch("yolo_developer.config.load_config", return_value=mock_config),
        ):
            # Simulate LLM failing every time
            mock_acompletion.side_effect = Exception("LLM unavailable")

            # Tenacity wraps the final exception in RetryError
            with pytest.raises(RetryError):
                await _call_llm("prompt", "system")

            # Should have tried 3 times (retry decorator)
            assert mock_acompletion.call_count == 3
