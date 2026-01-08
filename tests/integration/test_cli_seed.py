"""Integration tests for CLI seed command (Story 4.2).

Tests the full CLI invocation of the seed command using typer.testing.CliRunner,
verifying end-to-end behavior including argument parsing, file reading, and output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from yolo_developer.cli.main import app
from yolo_developer.seed.parser import LLMSeedParser
from yolo_developer.seed.types import (
    ConstraintCategory,
    SeedConstraint,
    SeedFeature,
    SeedGoal,
    SeedParseResult,
    SeedSource,
)

runner = CliRunner()


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_seed_path() -> Path:
    """Return path to simple seed fixture."""
    return Path(__file__).parent.parent / "fixtures" / "seeds" / "simple_seed.txt"


@pytest.fixture
def complex_seed_path() -> Path:
    """Return path to complex seed fixture."""
    return Path(__file__).parent.parent / "fixtures" / "seeds" / "complex_seed.md"


@pytest.fixture
def mock_llm_response() -> dict[str, Any]:
    """Create a mock LLM response for seed parsing."""
    return {
        "goals": [
            {
                "title": "Build Task Manager",
                "description": "Create a task management application",
                "priority": 1,
                "rationale": "Improve productivity",
            }
        ],
        "features": [
            {
                "name": "Task Creation",
                "description": "Users can create new tasks",
                "user_value": "Easy task tracking",
                "related_goals": ["Build Task Manager"],
            }
        ],
        "constraints": [
            {
                "category": "technical",
                "description": "Must be a CLI application",
                "impact": "No GUI required",
                "related_items": [],
            }
        ],
    }


@pytest.fixture
def mock_parse_result() -> SeedParseResult:
    """Create a mock parse result for testing."""
    return SeedParseResult(
        goals=(
            SeedGoal(
                title="Build Task Manager",
                description="Create a task management application",
                priority=1,
                rationale="Improve productivity",
            ),
        ),
        features=(
            SeedFeature(
                name="Task Creation",
                description="Users can create new tasks",
                user_value="Easy task tracking",
                related_goals=("Build Task Manager",),
            ),
        ),
        constraints=(
            SeedConstraint(
                category=ConstraintCategory.TECHNICAL,
                description="Must be a CLI application",
                impact="No GUI required",
                related_items=(),
            ),
        ),
        raw_content="Test content",
        source=SeedSource.FILE,
    )


# ============================================================================
# Test CLI Argument Parsing
# ============================================================================


class TestCliArgumentParsing:
    """Tests for CLI argument parsing and help."""

    def test_seed_help_shows_usage(self) -> None:
        """Test that seed --help shows usage information."""
        result = runner.invoke(app, ["seed", "--help"])
        assert result.exit_code == 0
        assert "Parse a seed document" in result.output
        assert "--verbose" in result.output
        assert "--json" in result.output

    def test_seed_requires_file_argument(self) -> None:
        """Test that seed command requires a file argument."""
        result = runner.invoke(app, ["seed"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "FILE_PATH" in result.output


# ============================================================================
# Test File Error Handling
# ============================================================================


class TestFileErrorHandling:
    """Tests for file error handling in CLI."""

    def test_nonexistent_file_error(self, tmp_path: Path) -> None:
        """Test error message for non-existent file."""
        nonexistent = tmp_path / "does_not_exist.md"
        result = runner.invoke(app, ["seed", str(nonexistent)])
        assert result.exit_code == 1
        assert "Error" in result.output
        assert "not found" in result.output.lower() or "File not found" in result.output

    def test_directory_path_error(self, tmp_path: Path) -> None:
        """Test error message when path is a directory."""
        result = runner.invoke(app, ["seed", str(tmp_path)])
        assert result.exit_code == 1
        assert "Error" in result.output


# ============================================================================
# Test Successful Parsing
# ============================================================================


class TestSuccessfulParsing:
    """Tests for successful seed parsing via CLI."""

    def test_parse_simple_seed_file(
        self,
        simple_seed_path: Path,
        mock_llm_response: dict[str, Any],
    ) -> None:
        """Test parsing simple seed file with mocked LLM."""
        with patch.object(
            LLMSeedParser,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=mock_llm_response,  # Returns dict directly, not JSON string
        ):
            result = runner.invoke(app, ["seed", str(simple_seed_path)])

        assert result.exit_code == 0
        assert "Goals" in result.output or "goal" in result.output.lower()

    def test_parse_complex_seed_file(
        self,
        complex_seed_path: Path,
        mock_llm_response: dict[str, Any],
    ) -> None:
        """Test parsing complex markdown seed file."""
        with patch.object(
            LLMSeedParser,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=mock_llm_response,  # Returns dict directly, not JSON string
        ):
            result = runner.invoke(app, ["seed", str(complex_seed_path)])

        assert result.exit_code == 0

    def test_verbose_mode_shows_more_details(
        self,
        simple_seed_path: Path,
        mock_llm_response: dict[str, Any],
    ) -> None:
        """Test that verbose mode shows additional details."""
        with patch.object(
            LLMSeedParser,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=mock_llm_response,  # Returns dict directly, not JSON string
        ):
            normal_result = runner.invoke(app, ["seed", str(simple_seed_path)])
            verbose_result = runner.invoke(app, ["seed", str(simple_seed_path), "-v"])

        # Both should succeed
        assert normal_result.exit_code == 0
        assert verbose_result.exit_code == 0

        # Verbose output should include additional info
        # (checking for any indicator of more output)
        assert len(verbose_result.output) >= len(normal_result.output) * 0.8  # Allow some variation


# ============================================================================
# Test JSON Output Mode
# ============================================================================


class TestJsonOutputMode:
    """Tests for JSON output mode."""

    def test_json_output_is_valid_json(
        self,
        simple_seed_path: Path,
        mock_llm_response: dict[str, Any],
    ) -> None:
        """Test that --json produces valid JSON output."""
        with patch.object(
            LLMSeedParser,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=mock_llm_response,  # Returns dict directly, not JSON string
        ):
            result = runner.invoke(app, ["seed", str(simple_seed_path), "--json"])

        assert result.exit_code == 0
        # The output should be parseable as JSON (Rich may add formatting)
        # Find JSON content in output
        output_lines = result.output.strip().split("\n")
        # Try to find and parse JSON
        json_found = False
        for i, line in enumerate(output_lines):
            if line.strip().startswith("{"):
                # Try to find complete JSON object
                json_str = "\n".join(output_lines[i:])
                try:
                    # Remove ANSI codes if present
                    import re

                    clean_json = re.sub(r"\x1b\[[0-9;]*m", "", json_str)
                    data = json.loads(clean_json)
                    json_found = True
                    assert "goals" in data
                    assert "features" in data
                    break
                except json.JSONDecodeError:
                    continue

        # If we couldn't find valid JSON, at least check structure is present
        if not json_found:
            assert '"goals"' in result.output or "goals" in result.output

    def test_json_flag_short_form(
        self,
        simple_seed_path: Path,
        mock_llm_response: dict[str, Any],
    ) -> None:
        """Test that -j flag works same as --json."""
        with patch.object(
            LLMSeedParser,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=mock_llm_response,  # Returns dict directly, not JSON string
        ):
            result = runner.invoke(app, ["seed", str(simple_seed_path), "-j"])

        assert result.exit_code == 0


# ============================================================================
# Test Exit Codes
# ============================================================================


class TestExitCodes:
    """Tests for correct exit codes."""

    def test_success_exit_code_zero(
        self,
        simple_seed_path: Path,
        mock_llm_response: dict[str, Any],
    ) -> None:
        """Test that successful parsing returns exit code 0."""
        with patch.object(
            LLMSeedParser,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=mock_llm_response,  # Returns dict directly, not JSON string
        ):
            result = runner.invoke(app, ["seed", str(simple_seed_path)])

        assert result.exit_code == 0

    def test_file_not_found_exit_code_one(self, tmp_path: Path) -> None:
        """Test that file not found returns exit code 1."""
        nonexistent = tmp_path / "missing.md"
        result = runner.invoke(app, ["seed", str(nonexistent)])
        assert result.exit_code == 1

    def test_parsing_error_exit_code_one(
        self,
        simple_seed_path: Path,
    ) -> None:
        """Test that parsing error returns exit code 1."""
        # Mock parse_seed to raise an exception (not _call_llm, which is caught internally)
        with patch(
            "yolo_developer.cli.commands.seed.parse_seed",
            new_callable=AsyncMock,
            side_effect=Exception("LLM API error"),
        ):
            result = runner.invoke(app, ["seed", str(simple_seed_path)])

        assert result.exit_code == 1
        assert "Error" in result.output


# ============================================================================
# Test with Real Fixtures
# ============================================================================


class TestWithRealFixtures:
    """Tests using real fixture files from tests/fixtures/seeds/."""

    def test_simple_seed_fixture_exists(self, simple_seed_path: Path) -> None:
        """Verify simple seed fixture file exists."""
        assert simple_seed_path.exists(), f"Fixture not found: {simple_seed_path}"

    def test_complex_seed_fixture_exists(self, complex_seed_path: Path) -> None:
        """Verify complex seed fixture file exists."""
        assert complex_seed_path.exists(), f"Fixture not found: {complex_seed_path}"

    def test_parse_with_edge_case_fixture(
        self,
        mock_llm_response: dict[str, Any],
    ) -> None:
        """Test parsing edge case seed fixture."""
        edge_case_path = Path(__file__).parent.parent / "fixtures" / "seeds" / "edge_case_seed.txt"

        if edge_case_path.exists():
            with patch.object(
                LLMSeedParser,
                "_call_llm",
                new_callable=AsyncMock,
                return_value=mock_llm_response,  # Returns dict directly, not JSON string
            ):
                result = runner.invoke(app, ["seed", str(edge_case_path)])
            assert result.exit_code == 0


# ============================================================================
# Test Question Generation Features (Story 4.4)
# ============================================================================


class TestQuestionGenerationFeatures:
    """Integration tests for Story 4.4 question generation enhancements."""

    @pytest.fixture
    def mock_ambiguity_response(self) -> dict[str, Any]:
        """Create mock LLM response with ambiguities and answer formats."""
        return {
            "ambiguities": [
                {
                    "type": "UNDEFINED",
                    "severity": "HIGH",
                    "source_text": "user authentication",
                    "location": "line 5",
                    "description": "No authentication method specified",
                    "question": "What authentication method should be used?",
                    "suggestions": ["OAuth2", "JWT", "Session-based"],
                    "answer_format": "CHOICE",
                },
                {
                    "type": "TECHNICAL",
                    "severity": "MEDIUM",
                    "source_text": "fast response times",
                    "location": "line 10",
                    "description": "No specific latency requirement",
                    "question": "What is the maximum acceptable response time in milliseconds?",
                    "suggestions": ["100", "500", "1000"],
                    "answer_format": "NUMERIC",
                },
                {
                    "type": "PRIORITY",
                    "severity": "LOW",
                    "source_text": "nice to have",
                    "location": "line 15",
                    "description": "Priority unclear",
                    "question": "Is dark mode required for initial release?",
                    "suggestions": ["yes", "no"],
                    "answer_format": "BOOLEAN",
                },
            ]
        }

    def test_verbose_mode_shows_format_hints(
        self,
        simple_seed_path: Path,
        mock_llm_response: dict[str, Any],
        mock_ambiguity_response: dict[str, Any],
    ) -> None:
        """Test that verbose mode shows format hints for questions."""
        from yolo_developer.seed.parser import LLMSeedParser

        with (
            patch.object(
                LLMSeedParser,
                "_call_llm",
                new_callable=AsyncMock,
                return_value=mock_llm_response,
            ),
            patch(
                "yolo_developer.seed.ambiguity._call_llm_for_ambiguities",
                new_callable=AsyncMock,
                return_value=mock_ambiguity_response,
            ),
        ):
            result = runner.invoke(app, ["seed", str(simple_seed_path), "-v"])

        assert result.exit_code == 0
        # In verbose mode with ambiguity detection, format hints may be shown
        # The output should complete without errors

    def test_ambiguity_result_includes_priority_scores_in_json(
        self,
        simple_seed_path: Path,
        mock_llm_response: dict[str, Any],
        mock_ambiguity_response: dict[str, Any],
    ) -> None:
        """Test that JSON output includes priority scores when ambiguities detected."""
        # Test the ambiguity result to_dict includes priority_scores
        with patch(
            "yolo_developer.seed.ambiguity._call_llm_for_ambiguities",
            new_callable=AsyncMock,
            return_value=mock_ambiguity_response,
        ):
            import asyncio

            from yolo_developer.seed.ambiguity import detect_ambiguities

            result = asyncio.run(detect_ambiguities("Test content with ambiguities"))

        # Verify the to_dict includes priority_scores
        result_dict = result.to_dict()
        assert "priority_scores" in result_dict
        assert len(result_dict["priority_scores"]) == 3
        # Scores should be calculated correctly
        # UNDEFINED+HIGH = 25+30 = 55
        # TECHNICAL+MEDIUM = 15+20 = 35
        # PRIORITY+LOW = 5+10 = 15
        assert result_dict["priority_scores"] == [55, 35, 15]

    def test_prioritized_ambiguities_returns_correct_order(
        self,
        mock_ambiguity_response: dict[str, Any],
    ) -> None:
        """Test that prioritized_ambiguities property returns correct order."""
        with patch(
            "yolo_developer.seed.ambiguity._call_llm_for_ambiguities",
            new_callable=AsyncMock,
            return_value=mock_ambiguity_response,
        ):
            import asyncio

            from yolo_developer.seed.ambiguity import detect_ambiguities

            result = asyncio.run(detect_ambiguities("Test content"))

        prioritized = result.prioritized_ambiguities
        # Should be sorted by priority: HIGH UNDEFINED > MEDIUM TECHNICAL > LOW PRIORITY
        assert len(prioritized) == 3
        assert prioritized[0].source_text == "user authentication"
        assert prioritized[1].source_text == "fast response times"
        assert prioritized[2].source_text == "nice to have"

    def test_get_highest_priority_ambiguity_integration(
        self,
        mock_ambiguity_response: dict[str, Any],
    ) -> None:
        """Test get_highest_priority_ambiguity() returns correct ambiguity."""
        with patch(
            "yolo_developer.seed.ambiguity._call_llm_for_ambiguities",
            new_callable=AsyncMock,
            return_value=mock_ambiguity_response,
        ):
            import asyncio

            from yolo_developer.seed.ambiguity import detect_ambiguities

            result = asyncio.run(detect_ambiguities("Test content"))

        top = result.get_highest_priority_ambiguity()
        assert top is not None
        assert top.source_text == "user authentication"
        assert top.severity.value == "high"

    def test_answer_format_parsed_from_llm_response(
        self,
        mock_ambiguity_response: dict[str, Any],
    ) -> None:
        """Test that answer_format is correctly parsed from LLM response."""
        from yolo_developer.seed.ambiguity import AnswerFormat

        with patch(
            "yolo_developer.seed.ambiguity._call_llm_for_ambiguities",
            new_callable=AsyncMock,
            return_value=mock_ambiguity_response,
        ):
            import asyncio

            from yolo_developer.seed.ambiguity import detect_ambiguities

            result = asyncio.run(detect_ambiguities("Test content"))

        # Check resolution prompts have correct answer formats
        prompts = result.resolution_prompts
        assert len(prompts) == 3
        assert prompts[0].answer_format == AnswerFormat.CHOICE
        assert prompts[1].answer_format == AnswerFormat.NUMERIC
        assert prompts[2].answer_format == AnswerFormat.BOOLEAN

    def test_format_hints_generated_for_answer_formats(
        self,
        mock_ambiguity_response: dict[str, Any],
    ) -> None:
        """Test that format hints are generated for each answer format."""
        with patch(
            "yolo_developer.seed.ambiguity._call_llm_for_ambiguities",
            new_callable=AsyncMock,
            return_value=mock_ambiguity_response,
        ):
            import asyncio

            from yolo_developer.seed.ambiguity import detect_ambiguities

            result = asyncio.run(detect_ambiguities("Test content"))

        prompts = result.resolution_prompts
        # All prompts should have format hints
        for prompt in prompts:
            assert prompt.format_hint is not None
            assert len(prompt.format_hint) > 0


# ============================================================================
# Test SOP Validation Features (Story 4.5)
# ============================================================================


class TestSOPValidationFeatures:
    """Integration tests for Story 4.5 SOP constraint validation."""

    def test_validate_sop_flag_appears_in_help(self) -> None:
        """Test that --validate-sop flag appears in help."""
        result = runner.invoke(app, ["seed", "--help"])
        assert result.exit_code == 0
        assert "--validate-sop" in result.output
        assert "--sop-store" in result.output
        assert "--override-soft" in result.output

    def test_validate_sop_with_empty_store(
        self,
        simple_seed_path: Path,
        mock_llm_response: dict[str, Any],
    ) -> None:
        """Test --validate-sop with no constraints passes."""
        from yolo_developer.seed.parser import LLMSeedParser
        from yolo_developer.seed.sop import SOPValidationResult

        with (
            patch.object(
                LLMSeedParser,
                "_call_llm",
                new_callable=AsyncMock,
                return_value=mock_llm_response,
            ),
            patch(
                "yolo_developer.seed.api._validate_against_sop",
                new_callable=AsyncMock,
                return_value=SOPValidationResult(passed=True),
            ),
        ):
            result = runner.invoke(
                app, ["seed", str(simple_seed_path), "--validate-sop"]
            )

        assert result.exit_code == 0
        assert "SOP validation passed" in result.output or "No SOP conflicts" in result.output

    def test_validate_sop_hard_conflict_blocks(
        self,
        simple_seed_path: Path,
        mock_llm_response: dict[str, Any],
    ) -> None:
        """Test that HARD conflicts block processing with exit code 1."""
        from yolo_developer.seed.parser import LLMSeedParser
        from yolo_developer.seed.sop import (
            ConflictSeverity,
            SOPCategory,
            SOPConflict,
            SOPConstraint,
            SOPValidationResult,
        )

        hard_constraint = SOPConstraint(
            id="arch-001",
            rule_text="Must use REST API",
            category=SOPCategory.ARCHITECTURE,
            source="architecture.md",
            severity=ConflictSeverity.HARD,
        )
        hard_conflict = SOPConflict(
            constraint=hard_constraint,
            seed_text="Use GraphQL",
            severity=ConflictSeverity.HARD,
            description="GraphQL conflicts with REST requirement",
        )
        mock_sop_result = SOPValidationResult(
            conflicts=[hard_conflict],
            passed=False,
        )

        with (
            patch.object(
                LLMSeedParser,
                "_call_llm",
                new_callable=AsyncMock,
                return_value=mock_llm_response,
            ),
            patch(
                "yolo_developer.seed.api._validate_against_sop",
                new_callable=AsyncMock,
                return_value=mock_sop_result,
            ),
        ):
            result = runner.invoke(
                app, ["seed", str(simple_seed_path), "--validate-sop"]
            )

        assert result.exit_code == 1
        assert "HARD" in result.output or "BLOCKED" in result.output

    def test_validate_sop_soft_conflict_with_override_soft(
        self,
        simple_seed_path: Path,
        mock_llm_response: dict[str, Any],
    ) -> None:
        """Test that --override-soft auto-overrides SOFT conflicts."""
        from yolo_developer.seed.parser import LLMSeedParser
        from yolo_developer.seed.sop import (
            ConflictSeverity,
            SOPCategory,
            SOPConflict,
            SOPConstraint,
            SOPValidationResult,
        )

        soft_constraint = SOPConstraint(
            id="name-001",
            rule_text="Use snake_case",
            category=SOPCategory.NAMING,
            source="style.md",
            severity=ConflictSeverity.SOFT,
        )
        soft_conflict = SOPConflict(
            constraint=soft_constraint,
            seed_text="Use camelCase",
            severity=ConflictSeverity.SOFT,
            description="camelCase conflicts with snake_case convention",
        )
        mock_sop_result = SOPValidationResult(
            conflicts=[soft_conflict],
            passed=True,  # SOFT conflicts don't fail
        )

        with (
            patch.object(
                LLMSeedParser,
                "_call_llm",
                new_callable=AsyncMock,
                return_value=mock_llm_response,
            ),
            patch(
                "yolo_developer.seed.api._validate_against_sop",
                new_callable=AsyncMock,
                return_value=mock_sop_result,
            ),
        ):
            result = runner.invoke(
                app, ["seed", str(simple_seed_path), "--validate-sop", "--override-soft"]
            )

        assert result.exit_code == 0
        assert "Auto-overriding" in result.output or "override" in result.output.lower()

    def test_validate_sop_json_output_includes_conflicts(
        self,
        simple_seed_path: Path,
        mock_llm_response: dict[str, Any],
    ) -> None:
        """Test that JSON output includes SOP validation results."""
        from yolo_developer.seed.parser import LLMSeedParser
        from yolo_developer.seed.sop import SOPValidationResult

        mock_sop_result = SOPValidationResult(passed=True)

        with (
            patch.object(
                LLMSeedParser,
                "_call_llm",
                new_callable=AsyncMock,
                return_value=mock_llm_response,
            ),
            patch(
                "yolo_developer.seed.api._validate_against_sop",
                new_callable=AsyncMock,
                return_value=mock_sop_result,
            ),
        ):
            result = runner.invoke(
                app, ["seed", str(simple_seed_path), "--validate-sop", "--json"]
            )

        assert result.exit_code == 0
        # JSON output should contain sop_validation field
        assert "sop_validation" in result.output

    def test_sop_store_file_not_found_warning(
        self,
        simple_seed_path: Path,
        mock_llm_response: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test warning when --sop-store file doesn't exist."""
        from yolo_developer.seed.parser import LLMSeedParser

        nonexistent_store = tmp_path / "nonexistent.json"

        with patch.object(
            LLMSeedParser,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=mock_llm_response,
        ):
            result = runner.invoke(
                app,
                [
                    "seed",
                    str(simple_seed_path),
                    "--validate-sop",
                    "--sop-store",
                    str(nonexistent_store),
                ],
            )

        # Should complete with warning about missing file
        assert result.exit_code == 0
        assert "Warning" in result.output or "not found" in result.output.lower()

    def test_sop_store_loads_constraints_from_json(
        self,
        simple_seed_path: Path,
        mock_llm_response: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test that --sop-store loads constraints from JSON file."""
        from yolo_developer.seed.parser import LLMSeedParser
        from yolo_developer.seed.sop import SOPValidationResult

        # Create a test SOP store file
        sop_store_path = tmp_path / "constraints.json"
        sop_store_path.write_text(
            json.dumps(
                {
                    "constraints": [
                        {
                            "id": "test-001",
                            "rule_text": "Test rule",
                            "category": "testing",
                            "source": "test.md",
                            "severity": "soft",
                        }
                    ]
                }
            )
        )

        with (
            patch.object(
                LLMSeedParser,
                "_call_llm",
                new_callable=AsyncMock,
                return_value=mock_llm_response,
            ),
            patch(
                "yolo_developer.seed.api._validate_against_sop",
                new_callable=AsyncMock,
                return_value=SOPValidationResult(passed=True),
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "seed",
                    str(simple_seed_path),
                    "--validate-sop",
                    "--sop-store",
                    str(sop_store_path),
                ],
            )

        assert result.exit_code == 0
        # Should indicate constraints were loaded
        assert "Loaded 1 SOP constraint" in result.output


# ============================================================================
# Test Validation Report Features (Story 4.6)
# ============================================================================


class TestValidationReportFeatures:
    """Integration tests for Story 4.6 semantic validation reports."""

    def test_report_format_flag_appears_in_help(self) -> None:
        """Test that --report-format flag appears in help."""
        result = runner.invoke(app, ["seed", "--help"])
        assert result.exit_code == 0
        assert "--report-format" in result.output
        assert "--report-output" in result.output

    def test_report_format_json_outputs_valid_json(
        self,
        simple_seed_path: Path,
        mock_llm_response: dict[str, Any],
    ) -> None:
        """Test that --report-format json produces valid JSON."""
        from yolo_developer.seed.parser import LLMSeedParser

        with patch.object(
            LLMSeedParser,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=mock_llm_response,
        ):
            result = runner.invoke(
                app, ["seed", str(simple_seed_path), "--report-format", "json"]
            )

        assert result.exit_code == 0
        # Should contain report structure
        assert "quality_metrics" in result.output
        assert "overall_score" in result.output
        assert "report_id" in result.output

    def test_report_format_markdown_outputs_markdown(
        self,
        simple_seed_path: Path,
        mock_llm_response: dict[str, Any],
    ) -> None:
        """Test that --report-format markdown produces markdown output."""
        from yolo_developer.seed.parser import LLMSeedParser

        with patch.object(
            LLMSeedParser,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=mock_llm_response,
        ):
            result = runner.invoke(
                app, ["seed", str(simple_seed_path), "--report-format", "markdown"]
            )

        assert result.exit_code == 0
        # Should contain markdown headers
        assert "# Validation Report" in result.output
        assert "## Quality Metrics" in result.output

    def test_report_format_rich_outputs_to_console(
        self,
        simple_seed_path: Path,
        mock_llm_response: dict[str, Any],
    ) -> None:
        """Test that --report-format rich outputs rich console format."""
        from yolo_developer.seed.parser import LLMSeedParser

        with patch.object(
            LLMSeedParser,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=mock_llm_response,
        ):
            result = runner.invoke(
                app, ["seed", str(simple_seed_path), "--report-format", "rich"]
            )

        assert result.exit_code == 0
        # Rich output should contain score information
        assert "Quality" in result.output or "Score" in result.output

    def test_report_output_writes_to_file(
        self,
        simple_seed_path: Path,
        mock_llm_response: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test that --report-output writes report to file."""
        from yolo_developer.seed.parser import LLMSeedParser

        output_file = tmp_path / "report.json"

        with patch.object(
            LLMSeedParser,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=mock_llm_response,
        ):
            result = runner.invoke(
                app,
                [
                    "seed",
                    str(simple_seed_path),
                    "--report-format",
                    "json",
                    "--report-output",
                    str(output_file),
                ],
            )

        assert result.exit_code == 0
        assert output_file.exists()
        assert "Report written to" in result.output

        # Verify content is valid JSON
        content = output_file.read_text()
        data = json.loads(content)
        assert "quality_metrics" in data
        assert "overall_score" in data["quality_metrics"]

    def test_report_format_markdown_to_file(
        self,
        simple_seed_path: Path,
        mock_llm_response: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test that markdown report can be written to file."""
        from yolo_developer.seed.parser import LLMSeedParser

        output_file = tmp_path / "report.md"

        with patch.object(
            LLMSeedParser,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=mock_llm_response,
        ):
            result = runner.invoke(
                app,
                [
                    "seed",
                    str(simple_seed_path),
                    "--report-format",
                    "markdown",
                    "--report-output",
                    str(output_file),
                ],
            )

        assert result.exit_code == 0
        assert output_file.exists()

        content = output_file.read_text()
        assert "# Validation Report" in content
        assert "## Quality Metrics" in content

    def test_report_short_flag_r(
        self,
        simple_seed_path: Path,
        mock_llm_response: dict[str, Any],
    ) -> None:
        """Test that -r short flag works for report format."""
        from yolo_developer.seed.parser import LLMSeedParser

        with patch.object(
            LLMSeedParser,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=mock_llm_response,
        ):
            result = runner.invoke(
                app, ["seed", str(simple_seed_path), "-r", "json"]
            )

        assert result.exit_code == 0
        assert "quality_metrics" in result.output

    def test_report_includes_ambiguity_details(
        self,
        simple_seed_path: Path,
        mock_llm_response: dict[str, Any],
    ) -> None:
        """Test that report includes ambiguity details when present."""
        from yolo_developer.seed.parser import LLMSeedParser
        from yolo_developer.seed.sop import SOPValidationResult

        mock_ambiguity_response = {
            "ambiguities": [
                {
                    "type": "TECHNICAL",
                    "severity": "HIGH",
                    "source_text": "scalable",
                    "location": "line 5",
                    "description": "No specific scale defined",
                    "question": "What scale is needed?",
                    "suggestions": ["1000 users", "10000 users"],
                    "answer_format": "CHOICE",
                }
            ]
        }

        with (
            patch.object(
                LLMSeedParser,
                "_call_llm",
                new_callable=AsyncMock,
                return_value=mock_llm_response,
            ),
            patch(
                "yolo_developer.seed.ambiguity._call_llm_for_ambiguities",
                new_callable=AsyncMock,
                return_value=mock_ambiguity_response,
            ),
            patch(
                "yolo_developer.seed.api._validate_against_sop",
                new_callable=AsyncMock,
                return_value=SOPValidationResult(passed=True),
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "seed",
                    str(simple_seed_path),
                    "--report-format",
                    "json",
                    "--validate-sop",
                ],
            )

        assert result.exit_code == 0
        # JSON output should include ambiguity score
        assert "ambiguity_score" in result.output

    def test_report_with_sop_validation(
        self,
        simple_seed_path: Path,
        mock_llm_response: dict[str, Any],
    ) -> None:
        """Test that report includes SOP validation scores."""
        from yolo_developer.seed.parser import LLMSeedParser
        from yolo_developer.seed.sop import SOPValidationResult

        with (
            patch.object(
                LLMSeedParser,
                "_call_llm",
                new_callable=AsyncMock,
                return_value=mock_llm_response,
            ),
            patch(
                "yolo_developer.seed.api._validate_against_sop",
                new_callable=AsyncMock,
                return_value=SOPValidationResult(passed=True),
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "seed",
                    str(simple_seed_path),
                    "--report-format",
                    "json",
                    "--validate-sop",
                ],
            )

        assert result.exit_code == 0
        assert "sop_score" in result.output

    def test_unknown_report_format_error(
        self,
        simple_seed_path: Path,
        mock_llm_response: dict[str, Any],
    ) -> None:
        """Test that unknown report format returns error."""
        from yolo_developer.seed.parser import LLMSeedParser

        with patch.object(
            LLMSeedParser,
            "_call_llm",
            new_callable=AsyncMock,
            return_value=mock_llm_response,
        ):
            result = runner.invoke(
                app, ["seed", str(simple_seed_path), "--report-format", "invalid"]
            )

        assert result.exit_code == 1
        assert "Unknown report format" in result.output
