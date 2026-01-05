"""Unit tests for pattern storage.

Tests PatternStore protocol and ChromaPatternStore implementation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_developer.memory.pattern_store import ChromaPatternStore
from yolo_developer.memory.patterns import CodePattern, PatternResult, PatternType


class TestChromaPatternStoreInit:
    """Tests for ChromaPatternStore initialization."""

    def test_creates_with_persist_directory(self, tmp_path: Path) -> None:
        """Test store creates with persist directory."""
        store = ChromaPatternStore(
            persist_directory=str(tmp_path / "patterns"),
            project_id="test-project",
        )
        assert store is not None

    def test_creates_with_project_id(self, tmp_path: Path) -> None:
        """Test store creates with project ID for isolation."""
        store = ChromaPatternStore(
            persist_directory=str(tmp_path / "patterns"),
            project_id="my-project",
        )
        assert store.project_id == "my-project"

    def test_default_project_id(self, tmp_path: Path) -> None:
        """Test store uses default project ID if not provided."""
        store = ChromaPatternStore(
            persist_directory=str(tmp_path / "patterns"),
        )
        assert store.project_id == "default"


class TestStorePattern:
    """Tests for storing patterns."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> ChromaPatternStore:
        """Create a test pattern store."""
        return ChromaPatternStore(
            persist_directory=str(tmp_path / "patterns"),
            project_id="test-project",
        )

    @pytest.fixture
    def sample_pattern(self) -> CodePattern:
        """Create a sample pattern for testing."""
        return CodePattern(
            pattern_type=PatternType.NAMING_FUNCTION,
            name="function_naming",
            value="snake_case",
            confidence=0.95,
            examples=("get_user", "process_order", "validate_input"),
            source_files=("main.py", "utils.py"),
        )

    @pytest.mark.asyncio
    async def test_store_pattern_returns_id(
        self, store: ChromaPatternStore, sample_pattern: CodePattern
    ) -> None:
        """Test storing a pattern returns an ID."""
        pattern_id = await store.store_pattern(sample_pattern)

        assert pattern_id is not None
        assert isinstance(pattern_id, str)
        assert len(pattern_id) > 0

    @pytest.mark.asyncio
    async def test_store_pattern_id_contains_type(
        self, store: ChromaPatternStore, sample_pattern: CodePattern
    ) -> None:
        """Test pattern ID contains pattern type."""
        pattern_id = await store.store_pattern(sample_pattern)

        assert "naming_function" in pattern_id

    @pytest.mark.asyncio
    async def test_store_pattern_upserts(
        self, store: ChromaPatternStore, sample_pattern: CodePattern
    ) -> None:
        """Test storing same pattern twice upserts."""
        id1 = await store.store_pattern(sample_pattern)
        id2 = await store.store_pattern(sample_pattern)

        assert id1 == id2

    @pytest.mark.asyncio
    async def test_store_multiple_patterns(self, store: ChromaPatternStore) -> None:
        """Test storing multiple different patterns."""
        pattern1 = CodePattern(
            pattern_type=PatternType.NAMING_FUNCTION,
            name="function_naming",
            value="snake_case",
            confidence=0.95,
        )
        pattern2 = CodePattern(
            pattern_type=PatternType.NAMING_CLASS,
            name="class_naming",
            value="PascalCase",
            confidence=0.90,
        )

        id1 = await store.store_pattern(pattern1)
        id2 = await store.store_pattern(pattern2)

        assert id1 != id2


class TestSearchPatterns:
    """Tests for searching patterns."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> ChromaPatternStore:
        """Create a test pattern store."""
        return ChromaPatternStore(
            persist_directory=str(tmp_path / "patterns"),
            project_id="test-project",
        )

    @pytest.fixture
    async def populated_store(self, store: ChromaPatternStore) -> ChromaPatternStore:
        """Create a store with some patterns."""
        patterns = [
            CodePattern(
                pattern_type=PatternType.NAMING_FUNCTION,
                name="function_naming",
                value="snake_case",
                confidence=0.95,
                examples=("get_user", "process_order"),
            ),
            CodePattern(
                pattern_type=PatternType.NAMING_CLASS,
                name="class_naming",
                value="PascalCase",
                confidence=0.90,
                examples=("UserService", "OrderHandler"),
            ),
            CodePattern(
                pattern_type=PatternType.IMPORT_STYLE,
                name="import_style",
                value="absolute",
                confidence=0.85,
                examples=("from mypackage import x",),
            ),
        ]
        for pattern in patterns:
            await store.store_pattern(pattern)
        return store

    @pytest.mark.asyncio
    async def test_search_returns_pattern_results(
        self, populated_store: ChromaPatternStore
    ) -> None:
        """Test search returns PatternResult instances."""
        results = await populated_store.search_patterns(query="function naming")

        assert len(results) > 0
        for result in results:
            assert isinstance(result, PatternResult)
            assert isinstance(result.pattern, CodePattern)
            assert isinstance(result.similarity, float)

    @pytest.mark.asyncio
    async def test_search_filters_by_type(self, populated_store: ChromaPatternStore) -> None:
        """Test search can filter by pattern type."""
        results = await populated_store.search_patterns(
            pattern_type=PatternType.NAMING_FUNCTION,
            query="naming",
        )

        assert len(results) > 0
        for result in results:
            assert result.pattern.pattern_type == PatternType.NAMING_FUNCTION

    @pytest.mark.asyncio
    async def test_search_respects_k_parameter(self, populated_store: ChromaPatternStore) -> None:
        """Test search respects k parameter."""
        results = await populated_store.search_patterns(query="naming", k=1)

        assert len(results) <= 1

    @pytest.mark.asyncio
    async def test_search_returns_empty_list_no_matches(self, store: ChromaPatternStore) -> None:
        """Test search returns empty list when no patterns stored."""
        results = await store.search_patterns(query="anything")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_similarity_in_valid_range(
        self, populated_store: ChromaPatternStore
    ) -> None:
        """Test similarity scores are in valid range."""
        results = await populated_store.search_patterns(query="naming")

        for result in results:
            assert 0.0 <= result.similarity <= 1.0


class TestGetPatternsByType:
    """Tests for getting patterns by type."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> ChromaPatternStore:
        """Create a test pattern store."""
        return ChromaPatternStore(
            persist_directory=str(tmp_path / "patterns"),
            project_id="test-project",
        )

    @pytest.fixture
    async def populated_store(self, store: ChromaPatternStore) -> ChromaPatternStore:
        """Create a store with some patterns."""
        patterns = [
            CodePattern(
                pattern_type=PatternType.NAMING_FUNCTION,
                name="function_naming",
                value="snake_case",
                confidence=0.95,
            ),
            CodePattern(
                pattern_type=PatternType.NAMING_FUNCTION,
                name="private_function_naming",
                value="_prefixed",
                confidence=0.80,
            ),
            CodePattern(
                pattern_type=PatternType.NAMING_CLASS,
                name="class_naming",
                value="PascalCase",
                confidence=0.90,
            ),
        ]
        for pattern in patterns:
            await store.store_pattern(pattern)
        return store

    @pytest.mark.asyncio
    async def test_get_patterns_by_type_returns_matching(
        self, populated_store: ChromaPatternStore
    ) -> None:
        """Test getting patterns by type returns matching patterns."""
        results = await populated_store.get_patterns_by_type(PatternType.NAMING_FUNCTION)

        assert len(results) == 2
        for pattern in results:
            assert pattern.pattern_type == PatternType.NAMING_FUNCTION

    @pytest.mark.asyncio
    async def test_get_patterns_by_type_returns_empty_no_matches(
        self, populated_store: ChromaPatternStore
    ) -> None:
        """Test getting patterns returns empty list when no matches."""
        results = await populated_store.get_patterns_by_type(PatternType.STRUCTURE_DIRECTORY)

        assert results == []


class TestPatternStorePersistence:
    """Tests for pattern persistence."""

    @pytest.mark.asyncio
    async def test_patterns_persist_across_instances(self, tmp_path: Path) -> None:
        """Test patterns persist when creating new store instance."""
        persist_dir = str(tmp_path / "patterns")
        project_id = "test-project"

        # Create and store patterns
        store1 = ChromaPatternStore(
            persist_directory=persist_dir,
            project_id=project_id,
        )
        pattern = CodePattern(
            pattern_type=PatternType.NAMING_FUNCTION,
            name="function_naming",
            value="snake_case",
            confidence=0.95,
        )
        await store1.store_pattern(pattern)

        # Create new instance and verify patterns are there
        store2 = ChromaPatternStore(
            persist_directory=persist_dir,
            project_id=project_id,
        )
        results = await store2.get_patterns_by_type(PatternType.NAMING_FUNCTION)

        assert len(results) == 1
        assert results[0].value == "snake_case"


class TestProjectIsolation:
    """Tests for project isolation."""

    @pytest.mark.asyncio
    async def test_patterns_isolated_by_project_id(self, tmp_path: Path) -> None:
        """Test patterns are isolated between projects."""
        persist_dir = str(tmp_path / "patterns")

        # Store pattern in project1
        store1 = ChromaPatternStore(
            persist_directory=persist_dir,
            project_id="project1",
        )
        pattern1 = CodePattern(
            pattern_type=PatternType.NAMING_FUNCTION,
            name="function_naming",
            value="snake_case",
            confidence=0.95,
        )
        await store1.store_pattern(pattern1)

        # Store pattern in project2
        store2 = ChromaPatternStore(
            persist_directory=persist_dir,
            project_id="project2",
        )
        pattern2 = CodePattern(
            pattern_type=PatternType.NAMING_FUNCTION,
            name="function_naming",
            value="camelCase",
            confidence=0.90,
        )
        await store2.store_pattern(pattern2)

        # Verify isolation
        project1_patterns = await store1.get_patterns_by_type(PatternType.NAMING_FUNCTION)
        project2_patterns = await store2.get_patterns_by_type(PatternType.NAMING_FUNCTION)

        assert len(project1_patterns) == 1
        assert project1_patterns[0].value == "snake_case"

        assert len(project2_patterns) == 1
        assert project2_patterns[0].value == "camelCase"
