"""Unit tests for ambiguity detection types and functions (Story 4.3).

Tests cover:
- AmbiguityType and AmbiguitySeverity enums
- Ambiguity, ResolutionPrompt, AmbiguityResult dataclasses
- calculate_ambiguity_confidence() scoring function
- Resolution and SeedContext dataclasses
- apply_resolutions() content transformation
- detect_ambiguities() LLM-based detection
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAmbiguityTypes:
    """Tests for AmbiguityType enum."""

    def test_ambiguity_type_values(self) -> None:
        """Test AmbiguityType enum has all required values."""
        from yolo_developer.seed.ambiguity import AmbiguityType

        assert AmbiguityType.SCOPE.value == "scope"
        assert AmbiguityType.TECHNICAL.value == "technical"
        assert AmbiguityType.PRIORITY.value == "priority"
        assert AmbiguityType.DEPENDENCY.value == "dependency"
        assert AmbiguityType.UNDEFINED.value == "undefined"

    def test_ambiguity_type_from_string(self) -> None:
        """Test creating AmbiguityType from string value."""
        from yolo_developer.seed.ambiguity import AmbiguityType

        assert AmbiguityType("scope") == AmbiguityType.SCOPE
        assert AmbiguityType("technical") == AmbiguityType.TECHNICAL


class TestAmbiguitySeverity:
    """Tests for AmbiguitySeverity enum."""

    def test_severity_values(self) -> None:
        """Test AmbiguitySeverity enum has all required values."""
        from yolo_developer.seed.ambiguity import AmbiguitySeverity

        assert AmbiguitySeverity.LOW.value == "low"
        assert AmbiguitySeverity.MEDIUM.value == "medium"
        assert AmbiguitySeverity.HIGH.value == "high"

    def test_severity_ordering(self) -> None:
        """Test severity values can be compared by string."""
        from yolo_developer.seed.ambiguity import AmbiguitySeverity

        # All values exist
        assert len(AmbiguitySeverity) == 3


class TestAmbiguityDataclass:
    """Tests for Ambiguity dataclass."""

    def test_ambiguity_creation(self) -> None:
        """Test creating an Ambiguity instance."""
        from yolo_developer.seed.ambiguity import Ambiguity, AmbiguitySeverity, AmbiguityType

        ambiguity = Ambiguity(
            ambiguity_type=AmbiguityType.SCOPE,
            severity=AmbiguitySeverity.HIGH,
            source_text="handle all edge cases",
            location="line 5",
            description="Unclear what edge cases need to be handled",
        )

        assert ambiguity.ambiguity_type == AmbiguityType.SCOPE
        assert ambiguity.severity == AmbiguitySeverity.HIGH
        assert ambiguity.source_text == "handle all edge cases"
        assert ambiguity.location == "line 5"
        assert ambiguity.description == "Unclear what edge cases need to be handled"

    def test_ambiguity_is_frozen(self) -> None:
        """Test that Ambiguity is immutable (frozen)."""
        from yolo_developer.seed.ambiguity import Ambiguity, AmbiguitySeverity, AmbiguityType

        ambiguity = Ambiguity(
            ambiguity_type=AmbiguityType.SCOPE,
            severity=AmbiguitySeverity.HIGH,
            source_text="test",
            location="line 1",
            description="test description",
        )

        with pytest.raises(AttributeError):
            ambiguity.source_text = "modified"  # type: ignore[misc]

    def test_ambiguity_to_dict(self) -> None:
        """Test Ambiguity.to_dict() serialization."""
        from yolo_developer.seed.ambiguity import Ambiguity, AmbiguitySeverity, AmbiguityType

        ambiguity = Ambiguity(
            ambiguity_type=AmbiguityType.TECHNICAL,
            severity=AmbiguitySeverity.MEDIUM,
            source_text="fast response times",
            location="Requirements section",
            description="What constitutes 'fast'?",
        )

        result = ambiguity.to_dict()

        assert result["ambiguity_type"] == "technical"
        assert result["severity"] == "medium"
        assert result["source_text"] == "fast response times"
        assert result["location"] == "Requirements section"
        assert result["description"] == "What constitutes 'fast'?"


class TestResolutionPromptDataclass:
    """Tests for ResolutionPrompt dataclass."""

    def test_resolution_prompt_creation(self) -> None:
        """Test creating a ResolutionPrompt instance."""
        from yolo_developer.seed.ambiguity import ResolutionPrompt

        prompt = ResolutionPrompt(
            question="What response time is acceptable?",
            suggestions=("< 100ms", "< 500ms", "< 1 second"),
            default="< 500ms",
        )

        assert prompt.question == "What response time is acceptable?"
        assert prompt.suggestions == ("< 100ms", "< 500ms", "< 1 second")
        assert prompt.default == "< 500ms"

    def test_resolution_prompt_without_default(self) -> None:
        """Test ResolutionPrompt without default value."""
        from yolo_developer.seed.ambiguity import ResolutionPrompt

        prompt = ResolutionPrompt(
            question="What should happen on error?",
            suggestions=("Retry", "Fail silently", "Raise exception"),
        )

        assert prompt.default is None

    def test_resolution_prompt_to_dict(self) -> None:
        """Test ResolutionPrompt.to_dict() serialization."""
        from yolo_developer.seed.ambiguity import ResolutionPrompt

        prompt = ResolutionPrompt(
            question="How many users?",
            suggestions=("10", "100", "1000"),
            default="100",
        )

        result = prompt.to_dict()

        assert result["question"] == "How many users?"
        assert result["suggestions"] == ["10", "100", "1000"]
        assert result["default"] == "100"


class TestAmbiguityResultDataclass:
    """Tests for AmbiguityResult dataclass."""

    def test_ambiguity_result_creation(self) -> None:
        """Test creating an AmbiguityResult instance."""
        from yolo_developer.seed.ambiguity import (
            Ambiguity,
            AmbiguityResult,
            AmbiguitySeverity,
            AmbiguityType,
            ResolutionPrompt,
        )

        ambiguity = Ambiguity(
            ambiguity_type=AmbiguityType.SCOPE,
            severity=AmbiguitySeverity.HIGH,
            source_text="test",
            location="line 1",
            description="test",
        )
        prompt = ResolutionPrompt(
            question="Clarify?",
            suggestions=("A", "B"),
        )

        result = AmbiguityResult(
            ambiguities=(ambiguity,),
            overall_confidence=0.85,
            resolution_prompts=(prompt,),
        )

        assert len(result.ambiguities) == 1
        assert result.overall_confidence == 0.85
        assert len(result.resolution_prompts) == 1

    def test_ambiguity_result_empty(self) -> None:
        """Test AmbiguityResult with no ambiguities."""
        from yolo_developer.seed.ambiguity import AmbiguityResult

        result = AmbiguityResult(
            ambiguities=(),
            overall_confidence=1.0,
            resolution_prompts=(),
        )

        assert len(result.ambiguities) == 0
        assert result.overall_confidence == 1.0
        assert result.has_ambiguities is False

    def test_ambiguity_result_has_ambiguities_property(self) -> None:
        """Test has_ambiguities property returns True when ambiguities exist."""
        from yolo_developer.seed.ambiguity import (
            Ambiguity,
            AmbiguityResult,
            AmbiguitySeverity,
            AmbiguityType,
        )

        ambiguity = Ambiguity(
            ambiguity_type=AmbiguityType.SCOPE,
            severity=AmbiguitySeverity.LOW,
            source_text="test",
            location="line 1",
            description="test",
        )

        result = AmbiguityResult(
            ambiguities=(ambiguity,),
            overall_confidence=0.95,
            resolution_prompts=(),
        )

        assert result.has_ambiguities is True

    def test_ambiguity_result_to_dict(self) -> None:
        """Test AmbiguityResult.to_dict() serialization."""
        from yolo_developer.seed.ambiguity import (
            Ambiguity,
            AmbiguityResult,
            AmbiguitySeverity,
            AmbiguityType,
            ResolutionPrompt,
        )

        ambiguity = Ambiguity(
            ambiguity_type=AmbiguityType.TECHNICAL,
            severity=AmbiguitySeverity.MEDIUM,
            source_text="scalable",
            location="line 10",
            description="What scale?",
        )
        prompt = ResolutionPrompt(
            question="Define scale",
            suggestions=("100 users", "1000 users"),
        )

        result = AmbiguityResult(
            ambiguities=(ambiguity,),
            overall_confidence=0.9,
            resolution_prompts=(prompt,),
        )

        result_dict = result.to_dict()

        assert len(result_dict["ambiguities"]) == 1
        assert result_dict["overall_confidence"] == 0.9
        assert len(result_dict["resolution_prompts"]) == 1


class TestCalculateAmbiguityConfidence:
    """Tests for calculate_ambiguity_confidence() function."""

    def test_no_ambiguities_returns_full_confidence(self) -> None:
        """Test that no ambiguities returns 1.0 confidence."""
        from yolo_developer.seed.ambiguity import calculate_ambiguity_confidence

        confidence = calculate_ambiguity_confidence([])
        assert confidence == 1.0

    def test_single_low_severity_ambiguity(self) -> None:
        """Test LOW severity reduces confidence by 0.05."""
        from yolo_developer.seed.ambiguity import (
            Ambiguity,
            AmbiguitySeverity,
            AmbiguityType,
            calculate_ambiguity_confidence,
        )

        ambiguity = Ambiguity(
            ambiguity_type=AmbiguityType.SCOPE,
            severity=AmbiguitySeverity.LOW,
            source_text="test",
            location="line 1",
            description="test",
        )

        confidence = calculate_ambiguity_confidence([ambiguity])
        assert confidence == pytest.approx(0.95, abs=0.01)

    def test_single_medium_severity_ambiguity(self) -> None:
        """Test MEDIUM severity reduces confidence by 0.10."""
        from yolo_developer.seed.ambiguity import (
            Ambiguity,
            AmbiguitySeverity,
            AmbiguityType,
            calculate_ambiguity_confidence,
        )

        ambiguity = Ambiguity(
            ambiguity_type=AmbiguityType.TECHNICAL,
            severity=AmbiguitySeverity.MEDIUM,
            source_text="test",
            location="line 1",
            description="test",
        )

        confidence = calculate_ambiguity_confidence([ambiguity])
        assert confidence == pytest.approx(0.90, abs=0.01)

    def test_single_high_severity_ambiguity(self) -> None:
        """Test HIGH severity reduces confidence by 0.15."""
        from yolo_developer.seed.ambiguity import (
            Ambiguity,
            AmbiguitySeverity,
            AmbiguityType,
            calculate_ambiguity_confidence,
        )

        ambiguity = Ambiguity(
            ambiguity_type=AmbiguityType.UNDEFINED,
            severity=AmbiguitySeverity.HIGH,
            source_text="test",
            location="line 1",
            description="test",
        )

        confidence = calculate_ambiguity_confidence([ambiguity])
        assert confidence == pytest.approx(0.85, abs=0.01)

    def test_multiple_ambiguities_cumulative(self) -> None:
        """Test multiple ambiguities have cumulative effect."""
        from yolo_developer.seed.ambiguity import (
            Ambiguity,
            AmbiguitySeverity,
            AmbiguityType,
            calculate_ambiguity_confidence,
        )

        ambiguities = [
            Ambiguity(
                ambiguity_type=AmbiguityType.SCOPE,
                severity=AmbiguitySeverity.HIGH,  # -0.15
                source_text="test1",
                location="line 1",
                description="test",
            ),
            Ambiguity(
                ambiguity_type=AmbiguityType.TECHNICAL,
                severity=AmbiguitySeverity.MEDIUM,  # -0.10
                source_text="test2",
                location="line 2",
                description="test",
            ),
            Ambiguity(
                ambiguity_type=AmbiguityType.PRIORITY,
                severity=AmbiguitySeverity.LOW,  # -0.05
                source_text="test3",
                location="line 3",
                description="test",
            ),
        ]

        confidence = calculate_ambiguity_confidence(ambiguities)
        # 1.0 - 0.15 - 0.10 - 0.05 = 0.70
        assert confidence == pytest.approx(0.70, abs=0.01)

    def test_confidence_floor_at_minimum(self) -> None:
        """Test confidence doesn't go below 0.1."""
        from yolo_developer.seed.ambiguity import (
            Ambiguity,
            AmbiguitySeverity,
            AmbiguityType,
            calculate_ambiguity_confidence,
        )

        # Create 10 HIGH severity ambiguities (would be 1.0 - 10*0.15 = -0.5)
        ambiguities = [
            Ambiguity(
                ambiguity_type=AmbiguityType.UNDEFINED,
                severity=AmbiguitySeverity.HIGH,
                source_text=f"test{i}",
                location=f"line {i}",
                description="test",
            )
            for i in range(10)
        ]

        confidence = calculate_ambiguity_confidence(ambiguities)
        assert confidence == 0.1  # Floor value


class TestResolutionDataclass:
    """Tests for Resolution dataclass."""

    def test_resolution_creation(self) -> None:
        """Test creating a Resolution instance."""
        from yolo_developer.seed.ambiguity import Resolution

        resolution = Resolution(
            ambiguity_id="amb-001",
            user_response="100 concurrent users",
            timestamp="2026-01-08T10:00:00",
        )

        assert resolution.ambiguity_id == "amb-001"
        assert resolution.user_response == "100 concurrent users"
        assert resolution.timestamp == "2026-01-08T10:00:00"

    def test_resolution_to_dict(self) -> None:
        """Test Resolution.to_dict() serialization."""
        from yolo_developer.seed.ambiguity import Resolution

        resolution = Resolution(
            ambiguity_id="amb-002",
            user_response="Use JWT tokens",
            timestamp="2026-01-08T11:00:00",
        )

        result = resolution.to_dict()

        assert result["ambiguity_id"] == "amb-002"
        assert result["user_response"] == "Use JWT tokens"
        assert result["timestamp"] == "2026-01-08T11:00:00"


class TestSeedContextDataclass:
    """Tests for SeedContext dataclass."""

    def test_seed_context_creation(self) -> None:
        """Test creating a SeedContext instance."""
        from yolo_developer.seed.ambiguity import Resolution, SeedContext

        resolution = Resolution(
            ambiguity_id="amb-001",
            user_response="100 users",
            timestamp="2026-01-08T10:00:00",
        )

        context = SeedContext(
            original_content="Build a scalable app",
            resolutions=(resolution,),
            clarified_content="Build an app supporting 100 users",
        )

        assert context.original_content == "Build a scalable app"
        assert len(context.resolutions) == 1
        assert context.clarified_content == "Build an app supporting 100 users"

    def test_seed_context_empty_resolutions(self) -> None:
        """Test SeedContext with no resolutions."""
        from yolo_developer.seed.ambiguity import SeedContext

        context = SeedContext(
            original_content="Clear requirements",
            resolutions=(),
            clarified_content="Clear requirements",
        )

        assert len(context.resolutions) == 0
        assert context.original_content == context.clarified_content


class TestDetectAmbiguities:
    """Tests for detect_ambiguities() async function."""

    @pytest.mark.asyncio
    async def test_detect_ambiguities_with_ambiguous_content(self) -> None:
        """Test detect_ambiguities returns ambiguities for vague content."""
        from yolo_developer.seed.ambiguity import (
            AmbiguitySeverity,
            AmbiguityType,
            detect_ambiguities,
        )

        # Mock LLM response with ambiguities
        mock_llm_response: dict[str, Any] = {
            "ambiguities": [
                {
                    "type": "TECHNICAL",
                    "severity": "HIGH",
                    "source_text": "fast response",
                    "location": "line 1",
                    "description": "No specific time threshold",
                    "question": "What response time is acceptable?",
                    "suggestions": ["< 100ms", "< 500ms", "< 1 second"],
                }
            ]
        }

        with patch(
            "yolo_developer.seed.ambiguity.litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_llm:
            # Set up mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = str(mock_llm_response).replace("'", '"')
            mock_llm.return_value = mock_response

            result = await detect_ambiguities("Build an app with fast response")

            assert result.has_ambiguities is True
            assert len(result.ambiguities) == 1
            assert result.ambiguities[0].ambiguity_type == AmbiguityType.TECHNICAL
            assert result.ambiguities[0].severity == AmbiguitySeverity.HIGH
            assert len(result.resolution_prompts) == 1

    @pytest.mark.asyncio
    async def test_detect_ambiguities_no_ambiguities(self) -> None:
        """Test detect_ambiguities returns empty result for clear content."""
        from yolo_developer.seed.ambiguity import detect_ambiguities

        # Mock LLM response with no ambiguities
        mock_llm_response: dict[str, Any] = {"ambiguities": []}

        with patch(
            "yolo_developer.seed.ambiguity.litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = str(mock_llm_response).replace("'", '"')
            mock_llm.return_value = mock_response

            result = await detect_ambiguities("Create a login form with username and password")

            assert result.has_ambiguities is False
            assert len(result.ambiguities) == 0
            assert result.overall_confidence == 1.0

    @pytest.mark.asyncio
    async def test_detect_ambiguities_multiple_ambiguities(self) -> None:
        """Test detect_ambiguities handles multiple ambiguities."""
        from yolo_developer.seed.ambiguity import detect_ambiguities

        # Mock LLM response with multiple ambiguities
        mock_llm_response: dict[str, Any] = {
            "ambiguities": [
                {
                    "type": "SCOPE",
                    "severity": "HIGH",
                    "source_text": "handle all edge cases",
                    "location": "line 2",
                    "description": "Unclear scope",
                    "question": "What edge cases?",
                    "suggestions": ["Network errors", "Invalid input", "Timeouts"],
                },
                {
                    "type": "TECHNICAL",
                    "severity": "MEDIUM",
                    "source_text": "scalable",
                    "location": "line 3",
                    "description": "Unclear scale",
                    "question": "What scale?",
                    "suggestions": ["100 users", "1000 users"],
                },
            ]
        }

        with patch(
            "yolo_developer.seed.ambiguity.litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = str(mock_llm_response).replace("'", '"')
            mock_llm.return_value = mock_response

            result = await detect_ambiguities("Build a scalable app that handles all edge cases")

            assert result.has_ambiguities is True
            assert len(result.ambiguities) == 2
            assert len(result.resolution_prompts) == 2
            # Confidence: 1.0 - 0.15 (HIGH) - 0.10 (MEDIUM) = 0.75
            assert result.overall_confidence == pytest.approx(0.75, abs=0.01)

    @pytest.mark.asyncio
    async def test_detect_ambiguities_handles_llm_error(self) -> None:
        """Test detect_ambiguities handles LLM errors gracefully."""
        from yolo_developer.seed.ambiguity import detect_ambiguities

        with patch(
            "yolo_developer.seed.ambiguity.litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=Exception("LLM API error"),
        ):
            result = await detect_ambiguities("Some content")

            # Should return empty result on error
            assert result.has_ambiguities is False
            assert result.overall_confidence == 1.0

    @pytest.mark.asyncio
    async def test_detect_ambiguities_handles_json_wrapped_in_markdown(self) -> None:
        """Test detect_ambiguities handles JSON wrapped in markdown code blocks."""
        from yolo_developer.seed.ambiguity import detect_ambiguities

        # LLM sometimes wraps JSON in markdown
        markdown_wrapped = """```json
{"ambiguities": [{"type": "PRIORITY", "severity": "LOW", "source_text": "nice to have", "location": "line 5", "description": "Unclear priority", "question": "Is this required?", "suggestions": ["Required", "Optional"]}]}
```"""

        with patch(
            "yolo_developer.seed.ambiguity.litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = markdown_wrapped
            mock_llm.return_value = mock_response

            result = await detect_ambiguities("Add nice to have features")

            assert result.has_ambiguities is True
            assert len(result.ambiguities) == 1

    @pytest.mark.asyncio
    async def test_detect_ambiguities_empty_content(self) -> None:
        """Test detect_ambiguities with empty content."""
        from yolo_developer.seed.ambiguity import detect_ambiguities

        mock_llm_response: dict[str, Any] = {"ambiguities": []}

        with patch(
            "yolo_developer.seed.ambiguity.litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = str(mock_llm_response).replace("'", '"')
            mock_llm.return_value = mock_response

            result = await detect_ambiguities("")

            assert result.has_ambiguities is False
