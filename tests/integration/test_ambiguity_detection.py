"""Integration tests for ambiguity detection (Story 4.3).

Tests end-to-end ambiguity detection with mocked LLM responses,
including the CLI interactive mode and parse_seed integration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from yolo_developer.cli.main import app
from yolo_developer.seed.ambiguity import (
    Ambiguity,
    AmbiguityResult,
    AmbiguitySeverity,
    AmbiguityType,
    ResolutionPrompt,
    detect_ambiguities,
)
from yolo_developer.seed.api import parse_seed
from yolo_developer.seed.parser import LLMSeedParser

runner = CliRunner()

# Path to fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "seeds"


# =============================================================================
# Mock LLM Responses
# =============================================================================


def create_mock_ambiguity_response(
    ambiguity_count: int = 2,
) -> dict[str, Any]:
    """Create a mock LLM response for ambiguity detection."""
    ambiguities = []
    types = ["SCOPE", "TECHNICAL", "PRIORITY", "UNDEFINED"]
    severities = ["HIGH", "MEDIUM", "LOW"]

    for i in range(1, ambiguity_count + 1):
        ambiguities.append({
            "type": types[(i - 1) % len(types)],
            "severity": severities[(i - 1) % len(severities)],
            "source_text": f"ambiguous phrase {i}",
            "location": f"line {i * 5}",
            "description": f"This phrase is ambiguous because reason {i}",
            "question": f"What do you mean by 'ambiguous phrase {i}'?",
            "suggestions": [f"Option A{i}", f"Option B{i}", f"Option C{i}"],
        })

    return {"ambiguities": ambiguities}


def create_mock_parse_response(
    goals: int = 1,
    features: int = 2,
    constraints: int = 1,
) -> dict[str, Any]:
    """Create a mock LLM response for seed parsing."""
    return {
        "goals": [
            {
                "title": f"Goal {i}",
                "description": f"Description for goal {i}",
                "priority": i,
                "rationale": f"Rationale for goal {i}",
            }
            for i in range(1, goals + 1)
        ],
        "features": [
            {
                "name": f"Feature {i}",
                "description": f"Description for feature {i}",
                "user_value": f"User value for feature {i}",
                "related_goals": ["Goal 1"] if goals > 0 else [],
            }
            for i in range(1, features + 1)
        ],
        "constraints": [
            {
                "category": "technical",
                "description": f"Constraint {i}",
                "impact": f"Impact of constraint {i}",
                "related_items": [],
            }
            for i in range(1, constraints + 1)
        ],
    }


# =============================================================================
# Integration Tests: detect_ambiguities Function
# =============================================================================


class TestDetectAmbiguitiesIntegration:
    """Integration tests for the detect_ambiguities function."""

    @pytest.mark.asyncio
    async def test_detect_ambiguities_returns_result(self) -> None:
        """Test that detect_ambiguities returns AmbiguityResult."""
        content = "Build a fast, scalable web app with good UX"
        mock_response = create_mock_ambiguity_response(2)

        with patch(
            "yolo_developer.seed.ambiguity._call_llm_for_ambiguities",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await detect_ambiguities(content)

        assert isinstance(result, AmbiguityResult)
        assert result.has_ambiguities
        assert len(result.ambiguities) == 2

    @pytest.mark.asyncio
    async def test_detect_ambiguities_calculates_confidence(self) -> None:
        """Test that confidence score is calculated from ambiguities."""
        content = "Build something vague"
        mock_response = create_mock_ambiguity_response(3)

        with patch(
            "yolo_developer.seed.ambiguity._call_llm_for_ambiguities",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await detect_ambiguities(content)

        # With 3 ambiguities (HIGH, MEDIUM, LOW), confidence should be reduced
        # HIGH: -0.15, MEDIUM: -0.10, LOW: -0.05 = -0.30 total
        # 1.0 - 0.30 = 0.70
        assert result.overall_confidence == pytest.approx(0.70, abs=0.01)

    @pytest.mark.asyncio
    async def test_detect_ambiguities_creates_resolution_prompts(self) -> None:
        """Test that resolution prompts are created for each ambiguity."""
        content = "Build a modern app"
        mock_response = create_mock_ambiguity_response(2)

        with patch(
            "yolo_developer.seed.ambiguity._call_llm_for_ambiguities",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await detect_ambiguities(content)

        assert len(result.resolution_prompts) == 2
        for prompt in result.resolution_prompts:
            assert isinstance(prompt, ResolutionPrompt)
            assert prompt.question
            assert len(prompt.suggestions) > 0

    @pytest.mark.asyncio
    async def test_detect_ambiguities_handles_no_ambiguities(self) -> None:
        """Test handling when no ambiguities are detected."""
        content = "Clear requirements document"
        mock_response = {"ambiguities": []}

        with patch(
            "yolo_developer.seed.ambiguity._call_llm_for_ambiguities",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await detect_ambiguities(content)

        assert not result.has_ambiguities
        assert len(result.ambiguities) == 0
        assert result.overall_confidence == 1.0

    @pytest.mark.asyncio
    async def test_detect_ambiguities_handles_llm_error(self) -> None:
        """Test graceful handling of LLM errors."""
        content = "Some content"

        with patch(
            "yolo_developer.seed.ambiguity._call_llm_for_ambiguities",
            new_callable=AsyncMock,
            side_effect=Exception("LLM API error"),
        ):
            result = await detect_ambiguities(content)

        # Should return empty result on error
        assert not result.has_ambiguities
        assert result.overall_confidence == 1.0

    @pytest.mark.asyncio
    async def test_detect_ambiguities_parses_all_types(self) -> None:
        """Test that all ambiguity types are correctly parsed."""
        content = "Complex content"
        mock_response = {
            "ambiguities": [
                {
                    "type": "SCOPE",
                    "severity": "HIGH",
                    "source_text": "scope issue",
                    "location": "line 1",
                    "description": "Scope is unclear",
                    "question": "What scope?",
                    "suggestions": ["A", "B"],
                },
                {
                    "type": "TECHNICAL",
                    "severity": "MEDIUM",
                    "source_text": "tech issue",
                    "location": "line 2",
                    "description": "Tech is vague",
                    "question": "What tech?",
                    "suggestions": ["C", "D"],
                },
                {
                    "type": "PRIORITY",
                    "severity": "LOW",
                    "source_text": "priority issue",
                    "location": "line 3",
                    "description": "Priority unclear",
                    "question": "What priority?",
                    "suggestions": ["E", "F"],
                },
                {
                    "type": "DEPENDENCY",
                    "severity": "HIGH",
                    "source_text": "dependency issue",
                    "location": "line 4",
                    "description": "Dependency unclear",
                    "question": "What dependency?",
                    "suggestions": ["G", "H"],
                },
                {
                    "type": "UNDEFINED",
                    "severity": "MEDIUM",
                    "source_text": "undefined issue",
                    "location": "line 5",
                    "description": "Missing details",
                    "question": "What details?",
                    "suggestions": ["I", "J"],
                },
            ]
        }

        with patch(
            "yolo_developer.seed.ambiguity._call_llm_for_ambiguities",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await detect_ambiguities(content)

        assert len(result.ambiguities) == 5
        types = {amb.ambiguity_type for amb in result.ambiguities}
        assert AmbiguityType.SCOPE in types
        assert AmbiguityType.TECHNICAL in types
        assert AmbiguityType.PRIORITY in types
        assert AmbiguityType.DEPENDENCY in types
        assert AmbiguityType.UNDEFINED in types


# =============================================================================
# Integration Tests: parse_seed with detect_ambiguities
# =============================================================================


class TestParseSeedWithAmbiguities:
    """Integration tests for parse_seed with detect_ambiguities flag."""

    @pytest.mark.asyncio
    async def test_parse_seed_with_ambiguity_detection(self) -> None:
        """Test parse_seed with detect_ambiguities=True."""
        content = "Build a fast web app"
        mock_parse_response = create_mock_parse_response(1, 2, 1)

        with (
            patch.object(
                LLMSeedParser,
                "_call_llm",
                new_callable=AsyncMock,
                return_value=mock_parse_response,
            ),
            patch(
                "yolo_developer.seed.api._detect_ambiguities",
                new_callable=AsyncMock,
                return_value=AmbiguityResult(
                    ambiguities=(
                        Ambiguity(
                            ambiguity_type=AmbiguityType.TECHNICAL,
                            severity=AmbiguitySeverity.HIGH,
                            source_text="fast",
                            location="line 1",
                            description="No response time defined",
                        ),
                    ),
                    overall_confidence=0.85,
                    resolution_prompts=(
                        ResolutionPrompt(
                            question="What response time is acceptable?",
                            suggestions=("< 100ms", "< 500ms"),
                        ),
                    ),
                ),
            ),
        ):
            result = await parse_seed(content, detect_ambiguities=True)

        # Check parsing results
        assert result.goal_count == 1
        assert result.feature_count == 2
        assert result.constraint_count == 1

        # Check ambiguity results
        assert result.has_ambiguities
        assert result.ambiguity_count == 1
        assert result.ambiguity_confidence == 0.85

    @pytest.mark.asyncio
    async def test_parse_seed_without_ambiguity_detection(self) -> None:
        """Test parse_seed without detect_ambiguities flag."""
        content = "Build an app"
        mock_parse_response = create_mock_parse_response(1, 1, 0)

        with patch.object(
            LLMSeedParser,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=mock_parse_response,
        ):
            result = await parse_seed(content, detect_ambiguities=False)

        # Parsing should succeed
        assert result.goal_count == 1
        assert result.feature_count == 1

        # No ambiguity detection
        assert not result.has_ambiguities
        assert result.ambiguity_count == 0
        assert result.ambiguity_confidence == 1.0

    @pytest.mark.asyncio
    async def test_parse_seed_includes_ambiguities_in_json(self) -> None:
        """Test that ambiguities are included in to_dict() output."""
        content = "Build something"
        mock_parse_response = create_mock_parse_response(1, 1, 0)

        with (
            patch.object(
                LLMSeedParser,
                "_call_llm",
                new_callable=AsyncMock,
                return_value=mock_parse_response,
            ),
            patch(
                "yolo_developer.seed.api._detect_ambiguities",
                new_callable=AsyncMock,
                return_value=AmbiguityResult(
                    ambiguities=(
                        Ambiguity(
                            ambiguity_type=AmbiguityType.SCOPE,
                            severity=AmbiguitySeverity.MEDIUM,
                            source_text="something",
                            location="line 1",
                            description="Vague scope",
                        ),
                    ),
                    overall_confidence=0.90,
                    resolution_prompts=(),
                ),
            ),
        ):
            result = await parse_seed(content, detect_ambiguities=True)
            result_dict = result.to_dict()

        # Verify ambiguity data is serialized
        assert "ambiguities" in result_dict
        assert "ambiguity_confidence" in result_dict
        assert len(result_dict["ambiguities"]) == 1
        assert result_dict["ambiguity_confidence"] == 0.90

        # Verify ambiguity structure
        amb = result_dict["ambiguities"][0]
        assert amb["ambiguity_type"] == "scope"
        assert amb["severity"] == "medium"
        assert amb["source_text"] == "something"


# =============================================================================
# Integration Tests: CLI with Ambiguities
# =============================================================================


class TestCLIAmbiguityIntegration:
    """Integration tests for CLI with ambiguity features."""

    def test_cli_help_shows_interactive_flag(self) -> None:
        """Test that --interactive flag is documented in help."""
        result = runner.invoke(app, ["seed", "--help"])
        assert result.exit_code == 0
        assert "--interactive" in result.output
        assert "-i" in result.output

    def test_cli_json_output_includes_ambiguities(self) -> None:
        """Test that JSON output includes ambiguity data."""
        seed_file = FIXTURES_DIR / "simple_seed.txt"

        mock_parse_response = create_mock_parse_response(1, 2, 1)

        with (
            patch.object(
                LLMSeedParser,
                "_call_llm",
                new_callable=AsyncMock,
                return_value=mock_parse_response,
            ),
            patch(
                "yolo_developer.seed.api._detect_ambiguities",
                new_callable=AsyncMock,
                return_value=AmbiguityResult(
                    ambiguities=(),
                    overall_confidence=1.0,
                    resolution_prompts=(),
                ),
            ),
        ):
            result = runner.invoke(app, ["seed", str(seed_file), "--json"])

        assert result.exit_code == 0

        # Extract and verify JSON
        import re
        # Remove ANSI codes
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)

        # Find JSON in output
        json_match = re.search(r"\{[\s\S]*\}", clean_output)
        if json_match:
            data = json.loads(json_match.group())
            assert "ambiguities" in data
            assert "ambiguity_confidence" in data

    def test_cli_verbose_shows_ambiguity_summary(self) -> None:
        """Test that verbose mode mentions ambiguities."""
        seed_file = FIXTURES_DIR / "simple_seed.txt"
        mock_parse_response = create_mock_parse_response(1, 2, 1)

        with (
            patch.object(
                LLMSeedParser,
                "_call_llm",
                new_callable=AsyncMock,
                return_value=mock_parse_response,
            ),
            patch(
                "yolo_developer.seed.api._detect_ambiguities",
                new_callable=AsyncMock,
                return_value=AmbiguityResult(
                    ambiguities=(
                        Ambiguity(
                            ambiguity_type=AmbiguityType.TECHNICAL,
                            severity=AmbiguitySeverity.HIGH,
                            source_text="test",
                            location="line 1",
                            description="Test ambiguity",
                        ),
                    ),
                    overall_confidence=0.85,
                    resolution_prompts=(),
                ),
            ),
        ):
            result = runner.invoke(app, ["seed", str(seed_file), "--verbose"])

        assert result.exit_code == 0
        # Verbose mode should mention ambiguities
        assert "Ambiguities" in result.output or "ambiguit" in result.output.lower()

    def test_cli_interactive_flag_triggers_ambiguity_detection(self) -> None:
        """Test that --interactive flag triggers ambiguity detection."""
        seed_file = FIXTURES_DIR / "simple_seed.txt"
        mock_parse_response = create_mock_parse_response(1, 2, 1)
        mock_ambiguity_response = create_mock_ambiguity_response(0)

        with (
            patch.object(
                LLMSeedParser,
                "_call_llm",
                new_callable=AsyncMock,
                return_value=mock_parse_response,
            ),
            patch(
                "yolo_developer.seed.ambiguity._call_llm_for_ambiguities",
                new_callable=AsyncMock,
                return_value=mock_ambiguity_response,
            ) as mock_amb_call,
        ):
            result = runner.invoke(app, ["seed", str(seed_file), "--interactive"])

        assert result.exit_code == 0
        # Ambiguity detection should have been called
        mock_amb_call.assert_called_once()
        # Output should mention ambiguity detection
        assert "ambiguit" in result.output.lower() or "Detecting" in result.output

    def test_cli_interactive_displays_no_ambiguities_message(self) -> None:
        """Test that interactive mode shows message when no ambiguities."""
        seed_file = FIXTURES_DIR / "simple_seed.txt"
        mock_parse_response = create_mock_parse_response(1, 2, 1)
        mock_ambiguity_response = {"ambiguities": []}

        with (
            patch.object(
                LLMSeedParser,
                "_call_llm",
                new_callable=AsyncMock,
                return_value=mock_parse_response,
            ),
            patch(
                "yolo_developer.seed.ambiguity._call_llm_for_ambiguities",
                new_callable=AsyncMock,
                return_value=mock_ambiguity_response,
            ),
        ):
            result = runner.invoke(app, ["seed", str(seed_file), "-i"])

        assert result.exit_code == 0
        assert "No ambiguities" in result.output or "appears clear" in result.output


# =============================================================================
# Integration Tests: End-to-End Ambiguity Resolution Flow
# =============================================================================


class TestAmbiguityResolutionFlow:
    """Integration tests for the full ambiguity detection and resolution flow."""

    def test_interactive_displays_ambiguity_table(self) -> None:
        """Test that interactive mode displays ambiguity table."""
        seed_file = FIXTURES_DIR / "simple_seed.txt"
        mock_parse_response = create_mock_parse_response(1, 2, 1)
        mock_ambiguity_response = create_mock_ambiguity_response(2)

        with (
            patch.object(
                LLMSeedParser,
                "_call_llm",
                new_callable=AsyncMock,
                return_value=mock_parse_response,
            ),
            patch(
                "yolo_developer.seed.ambiguity._call_llm_for_ambiguities",
                new_callable=AsyncMock,
                return_value=mock_ambiguity_response,
            ),
        ):
            # Simulate user skipping all prompts
            result = runner.invoke(
                app,
                ["seed", str(seed_file), "--interactive"],
                input="s\ns\n",  # Skip both ambiguities
            )

        assert result.exit_code == 0
        # Should display ambiguity information
        assert "Ambiguities Detected" in result.output or "ambiguous phrase" in result.output

    def test_interactive_with_user_resolution(self) -> None:
        """Test interactive mode with user providing resolution."""
        seed_file = FIXTURES_DIR / "simple_seed.txt"
        mock_parse_response = create_mock_parse_response(1, 2, 1)
        mock_ambiguity_response = {
            "ambiguities": [
                {
                    "type": "TECHNICAL",
                    "severity": "HIGH",
                    "source_text": "fast",
                    "location": "line 1",
                    "description": "No performance threshold defined",
                    "question": "What response time is acceptable?",
                    "suggestions": ["< 100ms", "< 500ms", "< 1 second"],
                },
            ]
        }

        with (
            patch.object(
                LLMSeedParser,
                "_call_llm",
                new_callable=AsyncMock,
                return_value=mock_parse_response,
            ),
            patch(
                "yolo_developer.seed.ambiguity._call_llm_for_ambiguities",
                new_callable=AsyncMock,
                return_value=mock_ambiguity_response,
            ),
        ):
            # User selects option 1 (< 100ms)
            result = runner.invoke(
                app,
                ["seed", str(seed_file), "-i"],
                input="1\n",
            )

        assert result.exit_code == 0
        # Should show selection was made
        assert "Selected" in result.output or "< 100ms" in result.output or "clarification" in result.output.lower()

    def test_interactive_custom_resolution(self) -> None:
        """Test interactive mode with user providing custom resolution."""
        seed_file = FIXTURES_DIR / "simple_seed.txt"
        mock_parse_response = create_mock_parse_response(1, 2, 1)
        mock_ambiguity_response = {
            "ambiguities": [
                {
                    "type": "TECHNICAL",
                    "severity": "MEDIUM",
                    "source_text": "scalable",
                    "location": "line 2",
                    "description": "Scale not defined",
                    "question": "How many concurrent users?",
                    "suggestions": ["100", "1000", "10000"],
                },
            ]
        }

        with (
            patch.object(
                LLMSeedParser,
                "_call_llm",
                new_callable=AsyncMock,
                return_value=mock_parse_response,
            ),
            patch(
                "yolo_developer.seed.ambiguity._call_llm_for_ambiguities",
                new_callable=AsyncMock,
                return_value=mock_ambiguity_response,
            ),
        ):
            # User provides custom text
            result = runner.invoke(
                app,
                ["seed", str(seed_file), "-i"],
                input="5000 concurrent users\n",
            )

        assert result.exit_code == 0
        # Should apply the clarification
        assert "clarification" in result.output.lower() or "Re-parsing" in result.output


# =============================================================================
# Integration Tests: Fixture-Based Tests
# =============================================================================


class TestWithFixtureFiles:
    """Integration tests using fixture files."""

    @pytest.mark.asyncio
    async def test_complex_seed_with_ambiguity_detection(self) -> None:
        """Test ambiguity detection on complex seed fixture."""
        content = (FIXTURES_DIR / "complex_seed.md").read_text()
        mock_ambiguity_response = create_mock_ambiguity_response(3)

        with patch(
            "yolo_developer.seed.ambiguity._call_llm_for_ambiguities",
            new_callable=AsyncMock,
            return_value=mock_ambiguity_response,
        ):
            result = await detect_ambiguities(content)

        assert len(result.ambiguities) == 3
        assert result.overall_confidence < 1.0

    @pytest.mark.asyncio
    async def test_edge_case_seed_with_ambiguity_detection(self) -> None:
        """Test ambiguity detection on edge case seed fixture."""
        content = (FIXTURES_DIR / "edge_case_seed.txt").read_text()
        mock_ambiguity_response = create_mock_ambiguity_response(1)

        with patch(
            "yolo_developer.seed.ambiguity._call_llm_for_ambiguities",
            new_callable=AsyncMock,
            return_value=mock_ambiguity_response,
        ):
            result = await detect_ambiguities(content)

        # Should handle unicode content
        assert len(result.ambiguities) == 1


# =============================================================================
# Integration Tests: Error Handling
# =============================================================================


class TestAmbiguityErrorHandling:
    """Integration tests for error handling in ambiguity detection."""

    def test_cli_handles_ambiguity_detection_error(self) -> None:
        """Test CLI gracefully handles ambiguity detection errors."""
        seed_file = FIXTURES_DIR / "simple_seed.txt"
        mock_parse_response = create_mock_parse_response(1, 2, 1)

        with (
            patch.object(
                LLMSeedParser,
                "_call_llm",
                new_callable=AsyncMock,
                return_value=mock_parse_response,
            ),
            patch(
                "yolo_developer.seed.ambiguity._call_llm_for_ambiguities",
                new_callable=AsyncMock,
                side_effect=Exception("API timeout"),
            ),
        ):
            result = runner.invoke(app, ["seed", str(seed_file), "-i"])

        # Should still succeed with warning
        assert result.exit_code == 0
        assert "Warning" in result.output or "failed" in result.output.lower()
        # Parsing should still complete
        assert "Goals" in result.output or "complete" in result.output.lower()

    @pytest.mark.asyncio
    async def test_detect_ambiguities_handles_malformed_response(self) -> None:
        """Test handling of malformed LLM responses."""
        content = "Some content"

        # Missing required fields
        mock_response = {
            "ambiguities": [
                {"type": "INVALID_TYPE"},  # Missing fields
                "not a dict",  # Wrong type
            ]
        }

        with patch(
            "yolo_developer.seed.ambiguity._call_llm_for_ambiguities",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await detect_ambiguities(content)

        # Should handle gracefully, possibly with reduced results
        assert isinstance(result, AmbiguityResult)
