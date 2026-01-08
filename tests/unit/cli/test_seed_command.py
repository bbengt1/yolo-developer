"""Unit tests for seed CLI command (Story 4.2, 4.3).

Tests the seed command implementation including:
- File reading and validation
- Parse result display formatting
- JSON output mode
- Error handling for various failure cases
- Ambiguity display and interactive resolution (Story 4.3)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from rich.console import Console

from yolo_developer.cli.commands.seed import (
    _apply_resolutions_to_content,
    _display_ambiguities,
    _display_parse_results,
    _output_json,
    _prompt_for_resolution,
    _read_seed_file,
    seed_command,
)
from yolo_developer.seed.ambiguity import (
    Ambiguity,
    AmbiguityResult,
    AmbiguitySeverity,
    AmbiguityType,
    Resolution,
    ResolutionPrompt,
)
from yolo_developer.seed.types import (
    ConstraintCategory,
    SeedConstraint,
    SeedFeature,
    SeedGoal,
    SeedParseResult,
    SeedSource,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_console() -> MagicMock:
    """Create a mock console for testing output."""
    return MagicMock(spec=Console)


@pytest.fixture
def sample_goal() -> SeedGoal:
    """Create a sample goal for testing."""
    return SeedGoal(
        title="Build E-commerce Platform",
        description="Create an online store for product sales",
        priority=1,
        rationale="Expand market reach",
    )


@pytest.fixture
def sample_feature() -> SeedFeature:
    """Create a sample feature for testing."""
    return SeedFeature(
        name="Shopping Cart",
        description="Users can add products and manage quantities",
        user_value="Convenient pre-checkout collection",
        related_goals=("Build E-commerce Platform",),
    )


@pytest.fixture
def sample_constraint() -> SeedConstraint:
    """Create a sample constraint for testing."""
    return SeedConstraint(
        category=ConstraintCategory.TECHNICAL,
        description="Must use Python 3.10+",
        impact="Limits deployment options",
        related_items=(),
    )


@pytest.fixture
def sample_parse_result(
    sample_goal: SeedGoal,
    sample_feature: SeedFeature,
    sample_constraint: SeedConstraint,
) -> SeedParseResult:
    """Create a sample parse result for testing."""
    return SeedParseResult(
        goals=(sample_goal,),
        features=(sample_feature,),
        constraints=(sample_constraint,),
        raw_content="Build an e-commerce platform with shopping cart",
        source=SeedSource.FILE,
        metadata=(("filename", "test.md"),),
    )


@pytest.fixture
def empty_parse_result() -> SeedParseResult:
    """Create an empty parse result for testing."""
    return SeedParseResult(
        goals=(),
        features=(),
        constraints=(),
        raw_content="Empty content",
        source=SeedSource.TEXT,
    )


@pytest.fixture
def temp_seed_file(tmp_path: Path) -> Path:
    """Create a temporary seed file for testing."""
    seed_file = tmp_path / "test_seed.md"
    seed_file.write_text("# Test Seed\n\nBuild something amazing.", encoding="utf-8")
    return seed_file


# ============================================================================
# Test _read_seed_file
# ============================================================================


class TestReadSeedFile:
    """Tests for _read_seed_file function."""

    def test_read_valid_file(self, temp_seed_file: Path) -> None:
        """Test reading a valid seed file."""
        content = _read_seed_file(temp_seed_file)
        assert "Test Seed" in content
        assert "Build something amazing" in content

    def test_read_nonexistent_file(self, tmp_path: Path) -> None:
        """Test reading a file that doesn't exist."""
        nonexistent = tmp_path / "nonexistent.md"
        with pytest.raises(typer.Exit) as exc_info:
            _read_seed_file(nonexistent)
        assert exc_info.value.exit_code == 1

    def test_read_directory_instead_of_file(self, tmp_path: Path) -> None:
        """Test error when path is a directory."""
        with pytest.raises(typer.Exit) as exc_info:
            _read_seed_file(tmp_path)
        assert exc_info.value.exit_code == 1

    def test_read_file_with_utf8_content(self, tmp_path: Path) -> None:
        """Test reading file with UTF-8 encoded content."""
        seed_file = tmp_path / "unicode.md"
        seed_file.write_text("# Unicode Test\n\nContains unicode: ", encoding="utf-8")
        content = _read_seed_file(seed_file)
        assert "" in content

    def test_read_empty_file(self, tmp_path: Path) -> None:
        """Test reading an empty file."""
        empty_file = tmp_path / "empty.md"
        empty_file.write_text("", encoding="utf-8")
        content = _read_seed_file(empty_file)
        assert content == ""

    @patch("yolo_developer.cli.commands.seed.console")
    def test_read_permission_error(self, mock_console: MagicMock, tmp_path: Path) -> None:
        """Test handling of permission errors."""
        seed_file = tmp_path / "no_access.md"
        seed_file.write_text("content", encoding="utf-8")

        with patch.object(Path, "read_text", side_effect=PermissionError("Access denied")):
            with pytest.raises(typer.Exit) as exc_info:
                _read_seed_file(seed_file)
            assert exc_info.value.exit_code == 1

    @patch("yolo_developer.cli.commands.seed.console")
    def test_read_unicode_decode_error(self, mock_console: MagicMock, tmp_path: Path) -> None:
        """Test handling of encoding errors."""
        seed_file = tmp_path / "bad_encoding.md"
        seed_file.write_bytes(b"\x80\x81\x82")  # Invalid UTF-8

        with pytest.raises(typer.Exit) as exc_info:
            _read_seed_file(seed_file)
        assert exc_info.value.exit_code == 1


# ============================================================================
# Test _display_parse_results
# ============================================================================


class TestDisplayParseResults:
    """Tests for _display_parse_results function."""

    @patch("yolo_developer.cli.commands.seed.console")
    def test_display_with_results(
        self,
        mock_console: MagicMock,
        sample_parse_result: SeedParseResult,
    ) -> None:
        """Test displaying parse results with data."""
        _display_parse_results(sample_parse_result, verbose=False)

        # Verify console.print was called multiple times
        assert mock_console.print.call_count >= 1

    @patch("yolo_developer.cli.commands.seed.console")
    def test_display_empty_results(
        self,
        mock_console: MagicMock,
        empty_parse_result: SeedParseResult,
    ) -> None:
        """Test displaying empty parse results."""
        _display_parse_results(empty_parse_result, verbose=False)

        # Should still print summary panel
        assert mock_console.print.call_count >= 1

    @patch("yolo_developer.cli.commands.seed.console")
    def test_display_verbose_mode(
        self,
        mock_console: MagicMock,
        sample_parse_result: SeedParseResult,
    ) -> None:
        """Test displaying results in verbose mode."""
        _display_parse_results(sample_parse_result, verbose=True)

        # Verbose mode should produce more output
        assert mock_console.print.call_count >= 2

    @patch("yolo_developer.cli.commands.seed.console")
    def test_display_truncates_long_descriptions(
        self,
        mock_console: MagicMock,
    ) -> None:
        """Test that long descriptions are truncated in tables."""
        long_feature = SeedFeature(
            name="Feature",
            description="A" * 100,  # 100 character description
            user_value="Value",
        )
        result = SeedParseResult(
            goals=(),
            features=(long_feature,),
            constraints=(),
            raw_content="content",
            source=SeedSource.TEXT,
        )
        _display_parse_results(result, verbose=False)

        # Should complete without error
        assert mock_console.print.call_count >= 1


# ============================================================================
# Test _output_json
# ============================================================================


class TestOutputJson:
    """Tests for _output_json function."""

    @patch("yolo_developer.cli.commands.seed.console")
    def test_json_output_structure(
        self,
        mock_console: MagicMock,
        sample_parse_result: SeedParseResult,
    ) -> None:
        """Test JSON output has correct structure."""
        _output_json(sample_parse_result)

        # Verify print_json was called
        mock_console.print_json.assert_called_once()
        json_str = mock_console.print_json.call_args[0][0]

        # Parse and verify structure
        data = json.loads(json_str)
        assert "goals" in data
        assert "features" in data
        assert "constraints" in data
        assert "raw_content" in data
        assert "source" in data

    @patch("yolo_developer.cli.commands.seed.console")
    def test_json_output_empty_result(
        self,
        mock_console: MagicMock,
        empty_parse_result: SeedParseResult,
    ) -> None:
        """Test JSON output with empty results."""
        _output_json(empty_parse_result)

        json_str = mock_console.print_json.call_args[0][0]
        data = json.loads(json_str)

        assert data["goals"] == []
        assert data["features"] == []
        assert data["constraints"] == []

    @patch("yolo_developer.cli.commands.seed.console")
    def test_json_output_contains_goal_data(
        self,
        mock_console: MagicMock,
        sample_parse_result: SeedParseResult,
    ) -> None:
        """Test JSON output contains correct goal data."""
        _output_json(sample_parse_result)

        json_str = mock_console.print_json.call_args[0][0]
        data = json.loads(json_str)

        assert len(data["goals"]) == 1
        assert data["goals"][0]["title"] == "Build E-commerce Platform"
        assert data["goals"][0]["priority"] == 1


# ============================================================================
# Test seed_command
# ============================================================================


class TestSeedCommand:
    """Tests for seed_command function."""

    @patch("yolo_developer.cli.commands.seed._display_parse_results")
    @patch("yolo_developer.cli.commands.seed.asyncio.run")
    @patch("yolo_developer.cli.commands.seed.console")
    def test_seed_command_success(
        self,
        mock_console: MagicMock,
        mock_asyncio_run: MagicMock,
        mock_display: MagicMock,
        temp_seed_file: Path,
        sample_parse_result: SeedParseResult,
    ) -> None:
        """Test successful seed command execution."""
        mock_asyncio_run.return_value = sample_parse_result

        seed_command(temp_seed_file, verbose=False, json_output=False)

        mock_asyncio_run.assert_called_once()
        mock_display.assert_called_once_with(sample_parse_result, verbose=False)

    @patch("yolo_developer.cli.commands.seed._output_json")
    @patch("yolo_developer.cli.commands.seed.asyncio.run")
    @patch("yolo_developer.cli.commands.seed.console")
    def test_seed_command_json_mode(
        self,
        mock_console: MagicMock,
        mock_asyncio_run: MagicMock,
        mock_output_json: MagicMock,
        temp_seed_file: Path,
        sample_parse_result: SeedParseResult,
    ) -> None:
        """Test seed command with JSON output mode."""
        mock_asyncio_run.return_value = sample_parse_result

        seed_command(temp_seed_file, verbose=False, json_output=True)

        mock_output_json.assert_called_once_with(sample_parse_result)

    @patch("yolo_developer.cli.commands.seed._display_parse_results")
    @patch("yolo_developer.cli.commands.seed.asyncio.run")
    @patch("yolo_developer.cli.commands.seed.console")
    def test_seed_command_verbose_mode(
        self,
        mock_console: MagicMock,
        mock_asyncio_run: MagicMock,
        mock_display: MagicMock,
        temp_seed_file: Path,
        sample_parse_result: SeedParseResult,
    ) -> None:
        """Test seed command with verbose mode."""
        mock_asyncio_run.return_value = sample_parse_result

        seed_command(temp_seed_file, verbose=True, json_output=False)

        mock_display.assert_called_once_with(sample_parse_result, verbose=True)

    @patch("yolo_developer.cli.commands.seed.console")
    def test_seed_command_file_not_found(
        self,
        mock_console: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test seed command with non-existent file."""
        nonexistent = tmp_path / "nonexistent.md"

        with pytest.raises(typer.Exit) as exc_info:
            seed_command(nonexistent, verbose=False, json_output=False)
        assert exc_info.value.exit_code == 1

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @patch("yolo_developer.cli.commands.seed.console")
    def test_seed_command_parsing_error(
        self,
        mock_console: MagicMock,
        temp_seed_file: Path,
    ) -> None:
        """Test seed command handles parsing errors.

        Note: The filterwarnings decorator suppresses RuntimeWarning about
        unawaited coroutines. This is a mock artifact - the actual coroutine
        IS awaited by asyncio.run(), but the mock module's internal handling
        triggers the warning. The test behavior is correct.
        """
        from unittest.mock import AsyncMock

        # Mock parse_seed as an AsyncMock that raises when awaited
        with patch(
            "yolo_developer.cli.commands.seed.parse_seed",
            new_callable=AsyncMock,
            side_effect=Exception("LLM API error"),
        ):
            with pytest.raises(typer.Exit) as exc_info:
                seed_command(temp_seed_file, verbose=False, json_output=False)
            assert exc_info.value.exit_code == 1


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @patch("yolo_developer.cli.commands.seed.console")
    def test_display_with_many_goals(self, mock_console: MagicMock) -> None:
        """Test display with multiple goals."""
        goals = tuple(
            SeedGoal(
                title=f"Goal {i}",
                description=f"Description {i}",
                priority=i % 5 + 1,
            )
            for i in range(10)
        )
        result = SeedParseResult(
            goals=goals,
            features=(),
            constraints=(),
            raw_content="content",
            source=SeedSource.TEXT,
        )
        _display_parse_results(result, verbose=False)
        assert mock_console.print.call_count >= 1

    @patch("yolo_developer.cli.commands.seed.console")
    def test_display_with_special_characters(self, mock_console: MagicMock) -> None:
        """Test display with special characters in content."""
        goal = SeedGoal(
            title='Goal with <special> & "characters"',
            description="Description with\nnewlines\tand\ttabs",
            priority=1,
            rationale="Contains ' quotes ' and backslash \\",
        )
        result = SeedParseResult(
            goals=(goal,),
            features=(),
            constraints=(),
            raw_content="content",
            source=SeedSource.TEXT,
        )
        _display_parse_results(result, verbose=True)
        assert mock_console.print.call_count >= 1

    @patch("yolo_developer.cli.commands.seed.console")
    def test_json_with_unicode(self, mock_console: MagicMock) -> None:
        """Test JSON output with unicode characters."""
        goal = SeedGoal(
            title="Japanese: ",
            description="Greek: ",
            priority=1,
        )
        result = SeedParseResult(
            goals=(goal,),
            features=(),
            constraints=(),
            raw_content="Unicode content: ",
            source=SeedSource.TEXT,
        )
        _output_json(result)

        json_str = mock_console.print_json.call_args[0][0]
        # Should be valid JSON
        data = json.loads(json_str)
        assert "" in data["goals"][0]["title"]


# ============================================================================
# Test Ambiguity Display (Story 4.3)
# ============================================================================


@pytest.fixture
def sample_ambiguity() -> Ambiguity:
    """Create a sample ambiguity for testing."""
    return Ambiguity(
        ambiguity_type=AmbiguityType.TECHNICAL,
        severity=AmbiguitySeverity.HIGH,
        source_text="fast response times",
        location="line 5",
        description="No specific time threshold defined",
    )


@pytest.fixture
def sample_resolution_prompt() -> ResolutionPrompt:
    """Create a sample resolution prompt for testing."""
    return ResolutionPrompt(
        question="What response time is acceptable?",
        suggestions=("< 100ms", "< 500ms", "< 1 second"),
        default="< 500ms",
    )


@pytest.fixture
def sample_ambiguity_result(
    sample_ambiguity: Ambiguity,
    sample_resolution_prompt: ResolutionPrompt,
) -> AmbiguityResult:
    """Create a sample ambiguity result for testing."""
    return AmbiguityResult(
        ambiguities=(sample_ambiguity,),
        overall_confidence=0.85,
        resolution_prompts=(sample_resolution_prompt,),
    )


@pytest.fixture
def empty_ambiguity_result() -> AmbiguityResult:
    """Create an empty ambiguity result for testing."""
    return AmbiguityResult(
        ambiguities=(),
        overall_confidence=1.0,
        resolution_prompts=(),
    )


class TestDisplayAmbiguities:
    """Tests for _display_ambiguities function."""

    @patch("yolo_developer.cli.commands.seed.console")
    def test_display_no_ambiguities(
        self,
        mock_console: MagicMock,
        empty_ambiguity_result: AmbiguityResult,
    ) -> None:
        """Test displaying when no ambiguities found."""
        _display_ambiguities(empty_ambiguity_result, verbose=False)

        # Should print "No ambiguities detected" message
        assert mock_console.print.call_count >= 1
        call_args = str(mock_console.print.call_args_list)
        assert "No ambiguities detected" in call_args

    @patch("yolo_developer.cli.commands.seed.console")
    def test_display_with_ambiguities(
        self,
        mock_console: MagicMock,
        sample_ambiguity_result: AmbiguityResult,
    ) -> None:
        """Test displaying ambiguities table."""
        _display_ambiguities(sample_ambiguity_result, verbose=False)

        # Should print table and confidence
        assert mock_console.print.call_count >= 2

    @patch("yolo_developer.cli.commands.seed.console")
    def test_display_verbose_shows_prompts(
        self,
        mock_console: MagicMock,
        sample_ambiguity_result: AmbiguityResult,
    ) -> None:
        """Test verbose mode shows resolution prompts."""
        _display_ambiguities(sample_ambiguity_result, verbose=True)

        # Verbose should produce more output
        assert mock_console.print.call_count >= 3

    @patch("yolo_developer.cli.commands.seed.console")
    def test_display_high_confidence(self, mock_console: MagicMock) -> None:
        """Test display with high confidence (>= 0.8)."""
        result = AmbiguityResult(
            ambiguities=(),
            overall_confidence=0.95,
            resolution_prompts=(),
        )
        _display_ambiguities(result, verbose=False)
        # Should show green styling for high confidence
        assert mock_console.print.call_count >= 1

    @patch("yolo_developer.cli.commands.seed.console")
    def test_display_multiple_ambiguities(self, mock_console: MagicMock) -> None:
        """Test displaying multiple ambiguities."""
        ambiguities = (
            Ambiguity(
                ambiguity_type=AmbiguityType.SCOPE,
                severity=AmbiguitySeverity.HIGH,
                source_text="handle all edge cases",
                location="line 2",
                description="Unclear scope",
            ),
            Ambiguity(
                ambiguity_type=AmbiguityType.TECHNICAL,
                severity=AmbiguitySeverity.MEDIUM,
                source_text="scalable",
                location="line 5",
                description="No scale defined",
            ),
            Ambiguity(
                ambiguity_type=AmbiguityType.PRIORITY,
                severity=AmbiguitySeverity.LOW,
                source_text="nice to have",
                location="line 8",
                description="Unclear priority",
            ),
        )
        result = AmbiguityResult(
            ambiguities=ambiguities,
            overall_confidence=0.70,
            resolution_prompts=(),
        )
        _display_ambiguities(result, verbose=False)
        assert mock_console.print.call_count >= 2


class TestPromptForResolution:
    """Tests for _prompt_for_resolution function."""

    @patch("yolo_developer.cli.commands.seed.Prompt.ask", return_value="s")
    @patch("yolo_developer.cli.commands.seed.console")
    def test_skip_resolution(
        self,
        mock_console: MagicMock,
        mock_prompt: MagicMock,
        sample_ambiguity: Ambiguity,
        sample_resolution_prompt: ResolutionPrompt,
    ) -> None:
        """Test skipping an ambiguity resolution."""
        result = _prompt_for_resolution(sample_ambiguity, sample_resolution_prompt, 1)
        assert result is None

    @patch("yolo_developer.cli.commands.seed.Prompt.ask", return_value="1")
    @patch("yolo_developer.cli.commands.seed.console")
    def test_select_numbered_option(
        self,
        mock_console: MagicMock,
        mock_prompt: MagicMock,
        sample_ambiguity: Ambiguity,
        sample_resolution_prompt: ResolutionPrompt,
    ) -> None:
        """Test selecting a numbered option."""
        result = _prompt_for_resolution(sample_ambiguity, sample_resolution_prompt, 1)
        assert result is not None
        assert result.user_response == "< 100ms"  # First suggestion
        assert result.ambiguity_id == "amb-1"

    @patch("yolo_developer.cli.commands.seed.Prompt.ask", return_value="custom answer")
    @patch("yolo_developer.cli.commands.seed.console")
    def test_custom_answer(
        self,
        mock_console: MagicMock,
        mock_prompt: MagicMock,
        sample_ambiguity: Ambiguity,
        sample_resolution_prompt: ResolutionPrompt,
    ) -> None:
        """Test providing a custom answer."""
        result = _prompt_for_resolution(sample_ambiguity, sample_resolution_prompt, 2)
        assert result is not None
        assert result.user_response == "custom answer"
        assert result.ambiguity_id == "amb-2"

    @patch("yolo_developer.cli.commands.seed.Prompt.ask", return_value="")
    @patch("yolo_developer.cli.commands.seed.console")
    def test_empty_input_skips(
        self,
        mock_console: MagicMock,
        mock_prompt: MagicMock,
        sample_ambiguity: Ambiguity,
        sample_resolution_prompt: ResolutionPrompt,
    ) -> None:
        """Test empty input is treated as skip."""
        result = _prompt_for_resolution(sample_ambiguity, sample_resolution_prompt, 1)
        assert result is None


class TestApplyResolutionsToContent:
    """Tests for _apply_resolutions_to_content function."""

    def test_no_resolutions(self, sample_ambiguity_result: AmbiguityResult) -> None:
        """Test with no resolutions returns original content."""
        content = "Original content"
        result = _apply_resolutions_to_content(content, sample_ambiguity_result, [])
        assert result == content

    def test_with_resolutions(self, sample_ambiguity_result: AmbiguityResult) -> None:
        """Test with resolutions appends clarifications."""
        content = "Original content"
        resolutions = [
            Resolution(
                ambiguity_id="amb-1",
                user_response="< 100ms response time",
                timestamp="2026-01-08T10:00:00",
            )
        ]
        result = _apply_resolutions_to_content(
            content, sample_ambiguity_result, resolutions
        )
        assert "## Clarifications (User-Provided)" in result
        assert "fast response times" in result
        assert "< 100ms response time" in result

    def test_multiple_resolutions(self) -> None:
        """Test with multiple resolutions."""
        ambiguities = (
            Ambiguity(
                ambiguity_type=AmbiguityType.SCOPE,
                severity=AmbiguitySeverity.HIGH,
                source_text="edge cases",
                location="line 2",
                description="Unclear scope",
            ),
            Ambiguity(
                ambiguity_type=AmbiguityType.TECHNICAL,
                severity=AmbiguitySeverity.MEDIUM,
                source_text="scalable",
                location="line 5",
                description="No scale defined",
            ),
        )
        ambiguity_result = AmbiguityResult(
            ambiguities=ambiguities,
            overall_confidence=0.75,
            resolution_prompts=(),
        )
        resolutions = [
            Resolution(
                ambiguity_id="amb-1",
                user_response="network errors and timeouts",
                timestamp="2026-01-08T10:00:00",
            ),
            Resolution(
                ambiguity_id="amb-2",
                user_response="1000 concurrent users",
                timestamp="2026-01-08T10:01:00",
            ),
        ]
        content = "Build a scalable app"
        result = _apply_resolutions_to_content(content, ambiguity_result, resolutions)
        assert "edge cases" in result
        assert "network errors" in result
        assert "scalable" in result
        assert "1000 concurrent users" in result


class TestSeedCommandInteractive:
    """Tests for seed_command with interactive mode."""

    @patch("yolo_developer.cli.commands.seed._display_parse_results")
    @patch("yolo_developer.cli.commands.seed.asyncio.run")
    @patch("yolo_developer.cli.commands.seed.console")
    def test_interactive_mode_no_ambiguities(
        self,
        mock_console: MagicMock,
        mock_asyncio_run: MagicMock,
        mock_display: MagicMock,
        temp_seed_file: Path,
        sample_parse_result: SeedParseResult,
        empty_ambiguity_result: AmbiguityResult,
    ) -> None:
        """Test interactive mode when no ambiguities found."""
        # First call is ambiguity detection, second is parsing
        mock_asyncio_run.side_effect = [empty_ambiguity_result, sample_parse_result]

        seed_command(
            temp_seed_file, verbose=False, json_output=False, interactive=True
        )

        # Should call asyncio.run twice (ambiguity detection + parsing)
        assert mock_asyncio_run.call_count == 2

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @patch("yolo_developer.cli.commands.seed._display_parse_results")
    @patch("yolo_developer.cli.commands.seed._prompt_for_resolution")
    @patch("yolo_developer.cli.commands.seed.asyncio.run")
    @patch("yolo_developer.cli.commands.seed.console")
    def test_interactive_mode_with_skip_all(
        self,
        mock_console: MagicMock,
        mock_asyncio_run: MagicMock,
        mock_prompt: MagicMock,
        mock_display: MagicMock,
        temp_seed_file: Path,
        sample_parse_result: SeedParseResult,
        sample_ambiguity_result: AmbiguityResult,
    ) -> None:
        """Test interactive mode with all ambiguities skipped."""
        mock_asyncio_run.side_effect = [sample_ambiguity_result, sample_parse_result]
        mock_prompt.return_value = None  # Skip all

        seed_command(
            temp_seed_file, verbose=False, json_output=False, interactive=True
        )

        assert mock_asyncio_run.call_count == 2


# ============================================================================
# Test Report Format CLI Options (Story 4.6)
# ============================================================================


class TestSeedCommandReportFormat:
    """Tests for seed_command with report format options (Story 4.6)."""

    @patch("yolo_developer.cli.commands.seed._display_parse_results")
    @patch("yolo_developer.cli.commands.seed.asyncio.run")
    @patch("yolo_developer.cli.commands.seed.console")
    def test_report_format_none_uses_default_display(
        self,
        mock_console: MagicMock,
        mock_asyncio_run: MagicMock,
        mock_display: MagicMock,
        temp_seed_file: Path,
        sample_parse_result: SeedParseResult,
    ) -> None:
        """Test no report format uses default display."""
        mock_asyncio_run.return_value = sample_parse_result

        seed_command(temp_seed_file, verbose=False, json_output=False)

        mock_display.assert_called_once_with(sample_parse_result, verbose=False)

    @patch("yolo_developer.cli.commands.seed.format_report_json")
    @patch("yolo_developer.cli.commands.seed.generate_validation_report")
    @patch("yolo_developer.cli.commands.seed.asyncio.run")
    @patch("yolo_developer.cli.commands.seed.console")
    def test_report_format_json(
        self,
        mock_console: MagicMock,
        mock_asyncio_run: MagicMock,
        mock_generate_report: MagicMock,
        mock_format_json: MagicMock,
        temp_seed_file: Path,
        sample_parse_result: SeedParseResult,
    ) -> None:
        """Test --report-format json outputs JSON validation report."""
        mock_asyncio_run.return_value = sample_parse_result
        mock_format_json.return_value = '{"test": "json"}'

        seed_command(
            temp_seed_file,
            verbose=False,
            json_output=False,
            report_format="json",
        )

        mock_generate_report.assert_called_once()
        mock_format_json.assert_called_once()

    @patch("yolo_developer.cli.commands.seed.format_report_markdown")
    @patch("yolo_developer.cli.commands.seed.generate_validation_report")
    @patch("yolo_developer.cli.commands.seed.asyncio.run")
    @patch("yolo_developer.cli.commands.seed.console")
    def test_report_format_markdown(
        self,
        mock_console: MagicMock,
        mock_asyncio_run: MagicMock,
        mock_generate_report: MagicMock,
        mock_format_markdown: MagicMock,
        temp_seed_file: Path,
        sample_parse_result: SeedParseResult,
    ) -> None:
        """Test --report-format markdown outputs markdown report."""
        mock_asyncio_run.return_value = sample_parse_result
        mock_format_markdown.return_value = "# Report"

        seed_command(
            temp_seed_file,
            verbose=False,
            json_output=False,
            report_format="markdown",
        )

        mock_generate_report.assert_called_once()
        mock_format_markdown.assert_called_once()

    @patch("yolo_developer.cli.commands.seed.format_report_rich")
    @patch("yolo_developer.cli.commands.seed.generate_validation_report")
    @patch("yolo_developer.cli.commands.seed.asyncio.run")
    @patch("yolo_developer.cli.commands.seed.console")
    def test_report_format_rich(
        self,
        mock_console: MagicMock,
        mock_asyncio_run: MagicMock,
        mock_generate_report: MagicMock,
        mock_format_rich: MagicMock,
        temp_seed_file: Path,
        sample_parse_result: SeedParseResult,
    ) -> None:
        """Test --report-format rich outputs rich console report."""
        mock_asyncio_run.return_value = sample_parse_result

        seed_command(
            temp_seed_file,
            verbose=False,
            json_output=False,
            report_format="rich",
        )

        mock_generate_report.assert_called_once()
        mock_format_rich.assert_called_once()

    @patch("yolo_developer.cli.commands.seed.generate_validation_report")
    @patch("yolo_developer.cli.commands.seed.asyncio.run")
    @patch("yolo_developer.cli.commands.seed.console")
    def test_report_output_file(
        self,
        mock_console: MagicMock,
        mock_asyncio_run: MagicMock,
        mock_generate_report: MagicMock,
        temp_seed_file: Path,
        sample_parse_result: SeedParseResult,
        tmp_path: Path,
    ) -> None:
        """Test --report-output writes report to file."""
        from yolo_developer.seed.report import QualityMetrics, ValidationReport

        mock_asyncio_run.return_value = sample_parse_result
        mock_report = ValidationReport(
            parse_result=sample_parse_result,
            quality_metrics=QualityMetrics(
                ambiguity_score=1.0,
                sop_score=1.0,
                extraction_score=1.0,
                overall_score=1.0,
            ),
            report_id="test-123",
            source_file=str(temp_seed_file),
        )
        mock_generate_report.return_value = mock_report

        output_file = tmp_path / "report.json"
        seed_command(
            temp_seed_file,
            verbose=False,
            json_output=False,
            report_format="json",
            report_output=output_file,
        )

        assert output_file.exists()
        content = output_file.read_text()
        assert "test-123" in content
