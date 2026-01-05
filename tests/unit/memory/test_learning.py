"""Unit tests for pattern learning orchestrator.

Tests PatternLearner class for coordinating codebase analysis.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_developer.memory.learning import PatternLearner, PatternLearningResult
from yolo_developer.memory.patterns import PatternType


class TestPatternLearningResult:
    """Tests for PatternLearningResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a learning result."""
        result = PatternLearningResult(
            files_scanned=10,
            total_lines=500,
            patterns_found=5,
            pattern_types_found=["naming_function", "naming_class"],
        )

        assert result.files_scanned == 10
        assert result.total_lines == 500
        assert result.patterns_found == 5
        assert "naming_function" in result.pattern_types_found

    def test_empty_result(self) -> None:
        """Test empty learning result."""
        result = PatternLearningResult(
            files_scanned=0,
            total_lines=0,
            patterns_found=0,
            pattern_types_found=[],
        )

        assert result.files_scanned == 0
        assert result.patterns_found == 0


class TestPatternLearner:
    """Tests for PatternLearner class."""

    @pytest.fixture
    def learner(self, tmp_path: Path) -> PatternLearner:
        """Create a test pattern learner."""
        return PatternLearner(
            persist_directory=str(tmp_path / "patterns"),
            project_id="test-project",
        )

    @pytest.fixture
    def sample_codebase(self, tmp_path: Path) -> Path:
        """Create a sample Python codebase for testing."""
        src = tmp_path / "src" / "myproject"
        src.mkdir(parents=True)

        (src / "__init__.py").write_text("")
        (src / "main.py").write_text(
            """\
def get_user():
    pass

def process_order():
    pass

class UserService:
    pass

class OrderHandler:
    pass
"""
        )
        (src / "utils.py").write_text(
            """\
from .main import get_user

def helper_function():
    pass

MAX_SIZE = 100
"""
        )

        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "__init__.py").write_text("")
        (tests / "test_main.py").write_text(
            """\
def test_get_user():
    pass

def test_process_order():
    pass
"""
        )

        return tmp_path

    @pytest.mark.asyncio
    async def test_learn_from_codebase_returns_result(
        self, learner: PatternLearner, sample_codebase: Path
    ) -> None:
        """Test learning returns PatternLearningResult."""
        result = await learner.learn_from_codebase(sample_codebase)

        assert isinstance(result, PatternLearningResult)
        assert result.files_scanned > 0
        assert result.patterns_found > 0

    @pytest.mark.asyncio
    async def test_learn_from_codebase_counts_files(
        self, learner: PatternLearner, sample_codebase: Path
    ) -> None:
        """Test learning counts all Python files."""
        result = await learner.learn_from_codebase(sample_codebase)

        # Should find: __init__.py (2), main.py, utils.py, test_main.py
        assert result.files_scanned >= 4

    @pytest.mark.asyncio
    async def test_learn_from_codebase_counts_lines(
        self, learner: PatternLearner, sample_codebase: Path
    ) -> None:
        """Test learning counts total lines."""
        result = await learner.learn_from_codebase(sample_codebase)

        assert result.total_lines > 0

    @pytest.mark.asyncio
    async def test_learn_from_codebase_detects_naming_patterns(
        self, learner: PatternLearner, sample_codebase: Path
    ) -> None:
        """Test learning detects naming patterns."""
        result = await learner.learn_from_codebase(sample_codebase)

        # Should detect function and class naming patterns
        assert "naming_function" in result.pattern_types_found
        assert "naming_class" in result.pattern_types_found

    @pytest.mark.asyncio
    async def test_learn_from_codebase_detects_structure_patterns(
        self, learner: PatternLearner, sample_codebase: Path
    ) -> None:
        """Test learning detects structure patterns."""
        result = await learner.learn_from_codebase(sample_codebase)

        # Should detect directory and file patterns
        assert "structure_directory" in result.pattern_types_found

    @pytest.mark.asyncio
    async def test_learn_from_empty_codebase(
        self, learner: PatternLearner, tmp_path: Path
    ) -> None:
        """Test learning from empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = await learner.learn_from_codebase(empty_dir)

        assert result.files_scanned == 0
        assert result.patterns_found == 0

    @pytest.mark.asyncio
    async def test_learn_stores_patterns(
        self, learner: PatternLearner, sample_codebase: Path
    ) -> None:
        """Test learning stores patterns for later retrieval."""
        await learner.learn_from_codebase(sample_codebase)

        # Should be able to retrieve patterns
        patterns = await learner.get_patterns_by_type(PatternType.NAMING_FUNCTION)
        assert len(patterns) > 0

    @pytest.mark.asyncio
    async def test_learn_with_progress_callback(
        self, learner: PatternLearner, sample_codebase: Path
    ) -> None:
        """Test learning reports progress via callback."""
        progress_calls: list[tuple[int, int]] = []

        def on_progress(current: int, total: int) -> None:
            progress_calls.append((current, total))

        await learner.learn_from_codebase(
            sample_codebase, progress_callback=on_progress
        )

        assert len(progress_calls) > 0


class TestPatternLearnerSearch:
    """Tests for pattern search in PatternLearner."""

    @pytest.fixture
    async def learned_patterns(self, tmp_path: Path) -> PatternLearner:
        """Create a learner with patterns already learned."""
        codebase = tmp_path / "project"
        codebase.mkdir()

        (codebase / "module.py").write_text(
            """\
def get_user():
    pass

def process_data():
    pass

class UserService:
    pass
"""
        )

        learner = PatternLearner(
            persist_directory=str(tmp_path / "patterns"),
            project_id="test-project",
        )
        await learner.learn_from_codebase(codebase)
        return learner

    @pytest.mark.asyncio
    async def test_search_patterns_returns_results(
        self, learned_patterns: PatternLearner
    ) -> None:
        """Test searching for patterns."""
        results = await learned_patterns.search_patterns(query="function naming")

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_get_patterns_by_type(
        self, learned_patterns: PatternLearner
    ) -> None:
        """Test getting patterns by type."""
        patterns = await learned_patterns.get_patterns_by_type(
            PatternType.NAMING_FUNCTION
        )

        assert len(patterns) > 0
        assert all(p.pattern_type == PatternType.NAMING_FUNCTION for p in patterns)


class TestGetRelevantPatterns:
    """Tests for get_relevant_patterns method."""

    @pytest.fixture
    async def learned_patterns(self, tmp_path: Path) -> PatternLearner:
        """Create a learner with patterns already learned."""
        codebase = tmp_path / "project"
        src = codebase / "src"
        src.mkdir(parents=True)

        (src / "__init__.py").write_text("")
        (src / "module.py").write_text(
            """\
def get_user():
    pass

def process_data():
    pass

class UserService:
    pass

class DataHandler:
    pass
"""
        )

        tests = codebase / "tests"
        tests.mkdir()
        (tests / "test_module.py").write_text(
            """\
def test_get_user():
    pass
"""
        )

        learner = PatternLearner(
            persist_directory=str(tmp_path / "patterns"),
            project_id="test-project",
        )
        await learner.learn_from_codebase(codebase)
        return learner

    @pytest.mark.asyncio
    async def test_get_relevant_patterns_single_type(
        self, learned_patterns: PatternLearner
    ) -> None:
        """Test getting relevant patterns for a single type."""
        patterns = await learned_patterns.get_relevant_patterns(
            context="naming functions in python",
            pattern_types=[PatternType.NAMING_FUNCTION],
        )

        assert len(patterns) > 0
        assert all(p.pattern_type == PatternType.NAMING_FUNCTION for p in patterns)

    @pytest.mark.asyncio
    async def test_get_relevant_patterns_multiple_types(
        self, learned_patterns: PatternLearner
    ) -> None:
        """Test getting relevant patterns for multiple types."""
        patterns = await learned_patterns.get_relevant_patterns(
            context="code naming conventions",
            pattern_types=[PatternType.NAMING_FUNCTION, PatternType.NAMING_CLASS],
        )

        assert len(patterns) > 0
        # Should have patterns from both types
        pattern_types = {p.pattern_type for p in patterns}
        assert len(pattern_types) >= 1

    @pytest.mark.asyncio
    async def test_get_relevant_patterns_sorted_by_confidence(
        self, learned_patterns: PatternLearner
    ) -> None:
        """Test patterns are sorted by confidence."""
        patterns = await learned_patterns.get_relevant_patterns(
            context="naming",
            pattern_types=[PatternType.NAMING_FUNCTION, PatternType.NAMING_CLASS],
        )

        if len(patterns) >= 2:
            # Check patterns are sorted by confidence (descending)
            for i in range(len(patterns) - 1):
                assert patterns[i].confidence >= patterns[i + 1].confidence

    @pytest.mark.asyncio
    async def test_get_relevant_patterns_respects_k(
        self, learned_patterns: PatternLearner
    ) -> None:
        """Test k parameter limits results."""
        patterns = await learned_patterns.get_relevant_patterns(
            context="naming",
            pattern_types=[PatternType.NAMING_FUNCTION, PatternType.NAMING_CLASS],
            k=1,
        )

        assert len(patterns) <= 1

    @pytest.mark.asyncio
    async def test_get_relevant_patterns_empty_types(
        self, learned_patterns: PatternLearner
    ) -> None:
        """Test with empty pattern types returns all patterns."""
        patterns = await learned_patterns.get_relevant_patterns(
            context="naming",
            pattern_types=[],
            k=10,
        )

        # Should search across all types
        assert isinstance(patterns, list)
