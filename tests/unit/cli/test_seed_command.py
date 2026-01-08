"""Unit tests for seed CLI command (Story 4.2).

Tests the seed command implementation including:
- File reading and validation
- Parse result display formatting
- JSON output mode
- Error handling for various failure cases
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from rich.console import Console

from yolo_developer.cli.commands.seed import (
    _display_parse_results,
    _output_json,
    _read_seed_file,
    seed_command,
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
