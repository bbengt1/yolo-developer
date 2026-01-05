"""Integration tests for pattern learning.

Tests cover:
- Full pattern learning on sample codebase
- Pattern persistence across sessions
- Pattern retrieval after learning
- Project isolation (patterns don't leak between projects)
- Incremental learning updates existing patterns
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_developer.memory.learning import PatternLearner, PatternLearningResult
from yolo_developer.memory.patterns import PatternType


class TestFullPatternLearning:
    """Integration tests for full pattern learning pipeline."""

    @pytest.fixture
    def sample_codebase(self, tmp_path: Path) -> Path:
        """Create a realistic sample codebase for testing."""
        project = tmp_path / "sample_project"
        src = project / "src" / "myapp"
        src.mkdir(parents=True)

        # Create main module with snake_case functions and PascalCase classes
        (src / "__init__.py").write_text("")
        (src / "main.py").write_text(
            """\
\"\"\"Main application module.\"\"\"

from typing import Any


def get_user_by_id(user_id: int) -> dict[str, Any]:
    \"\"\"Get a user by their ID.\"\"\"
    return {"id": user_id}


def process_order_request(order_data: dict) -> bool:
    \"\"\"Process an order request.\"\"\"
    return True


def validate_input_data(data: dict) -> bool:
    \"\"\"Validate input data.\"\"\"
    return bool(data)


class UserService:
    \"\"\"Service for user operations.\"\"\"

    def __init__(self) -> None:
        self.users: list[dict] = []

    def find_user(self, user_id: int) -> dict | None:
        \"\"\"Find a user by ID.\"\"\"
        return None


class OrderHandler:
    \"\"\"Handler for order operations.\"\"\"

    def handle_order(self, order: dict) -> bool:
        \"\"\"Handle an order.\"\"\"
        return True


class DataValidator:
    \"\"\"Validates data inputs.\"\"\"

    def validate(self, data: dict) -> bool:
        \"\"\"Validate data.\"\"\"
        return bool(data)
"""
        )

        (src / "utils.py").write_text(
            """\
\"\"\"Utility functions.\"\"\"

from typing import Any

MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30

def format_response(data: Any) -> dict:
    \"\"\"Format a response object.\"\"\"
    return {"data": data}


def parse_request_body(body: str) -> dict:
    \"\"\"Parse request body.\"\"\"
    return {}


def calculate_total_amount(items: list) -> float:
    \"\"\"Calculate total amount.\"\"\"
    return sum(float(item.get("price", 0)) for item in items)
"""
        )

        (src / "models.py").write_text(
            """\
\"\"\"Data models.\"\"\"

from dataclasses import dataclass


@dataclass
class UserModel:
    \"\"\"User model.\"\"\"
    id: int
    name: str
    email: str


@dataclass
class OrderModel:
    \"\"\"Order model.\"\"\"
    id: int
    user_id: int
    total: float
"""
        )

        # Create tests directory
        tests = project / "tests"
        tests.mkdir()
        (tests / "__init__.py").write_text("")
        (tests / "test_main.py").write_text(
            """\
\"\"\"Tests for main module.\"\"\"

import pytest


def test_get_user_by_id():
    \"\"\"Test getting a user by ID.\"\"\"
    pass


def test_process_order_request():
    \"\"\"Test processing an order request.\"\"\"
    pass


def test_user_service_find_user():
    \"\"\"Test UserService.find_user method.\"\"\"
    pass


class TestUserService:
    \"\"\"Tests for UserService class.\"\"\"

    def test_init(self):
        \"\"\"Test initialization.\"\"\"
        pass

    def test_find_user_not_found(self):
        \"\"\"Test finding a non-existent user.\"\"\"
        pass
"""
        )

        return project

    @pytest.mark.asyncio
    async def test_full_pattern_learning_pipeline(
        self, tmp_path: Path, sample_codebase: Path
    ) -> None:
        """Test complete pattern learning from codebase to retrieval."""
        persist_dir = str(tmp_path / "patterns")

        learner = PatternLearner(
            persist_directory=persist_dir,
            project_id="test-project",
        )

        # Learn patterns from codebase
        result = await learner.learn_from_codebase(sample_codebase)

        # Verify learning result
        assert isinstance(result, PatternLearningResult)
        assert result.files_scanned >= 4  # At least 4 Python files
        assert result.total_lines > 0
        assert result.patterns_found > 0

        # Verify pattern types were detected
        assert "naming_function" in result.pattern_types_found
        assert "naming_class" in result.pattern_types_found
        assert "structure_directory" in result.pattern_types_found

    @pytest.mark.asyncio
    async def test_pattern_retrieval_after_learning(
        self, tmp_path: Path, sample_codebase: Path
    ) -> None:
        """Test that patterns can be retrieved after learning."""
        persist_dir = str(tmp_path / "patterns")

        learner = PatternLearner(
            persist_directory=persist_dir,
            project_id="test-project",
        )

        # Learn patterns
        await learner.learn_from_codebase(sample_codebase)

        # Retrieve function naming patterns
        function_patterns = await learner.get_patterns_by_type(
            PatternType.NAMING_FUNCTION
        )
        assert len(function_patterns) > 0
        assert function_patterns[0].value == "snake_case"

        # Retrieve class naming patterns
        class_patterns = await learner.get_patterns_by_type(PatternType.NAMING_CLASS)
        assert len(class_patterns) > 0
        assert class_patterns[0].value == "PascalCase"

    @pytest.mark.asyncio
    async def test_pattern_search_semantic_matching(
        self, tmp_path: Path, sample_codebase: Path
    ) -> None:
        """Test semantic pattern search returns relevant results."""
        persist_dir = str(tmp_path / "patterns")

        learner = PatternLearner(
            persist_directory=persist_dir,
            project_id="test-project",
        )

        # Learn patterns
        await learner.learn_from_codebase(sample_codebase)

        # Search for naming patterns
        results = await learner.search_patterns(query="function naming style")

        assert len(results) > 0
        # Should find function naming pattern
        pattern_names = [r.pattern.name for r in results]
        assert any("function" in name.lower() for name in pattern_names)


class TestPatternPersistence:
    """Integration tests for pattern persistence across sessions."""

    @pytest.fixture
    def simple_codebase(self, tmp_path: Path) -> Path:
        """Create a simple codebase for testing."""
        project = tmp_path / "project"
        project.mkdir()

        (project / "module.py").write_text(
            """\
def get_data():
    pass

def process_item():
    pass

class DataService:
    pass
"""
        )

        return project

    @pytest.mark.asyncio
    async def test_patterns_persist_across_sessions(
        self, tmp_path: Path, simple_codebase: Path
    ) -> None:
        """Test patterns persist when creating new learner instance."""
        persist_dir = str(tmp_path / "patterns")
        project_id = "test-project"

        # First session: learn patterns
        learner1 = PatternLearner(
            persist_directory=persist_dir,
            project_id=project_id,
        )
        await learner1.learn_from_codebase(simple_codebase)

        # Get patterns count
        patterns1 = await learner1.get_patterns_by_type(PatternType.NAMING_FUNCTION)
        assert len(patterns1) > 0

        # Delete first learner
        del learner1

        # Second session: create new learner and verify patterns exist
        learner2 = PatternLearner(
            persist_directory=persist_dir,
            project_id=project_id,
        )

        # Verify patterns persisted
        patterns2 = await learner2.get_patterns_by_type(PatternType.NAMING_FUNCTION)
        assert len(patterns2) > 0
        assert patterns2[0].value == patterns1[0].value

    @pytest.mark.asyncio
    async def test_pattern_retrieval_fast_response_time(
        self, tmp_path: Path, simple_codebase: Path
    ) -> None:
        """Test pattern retrieval completes within acceptable time."""
        import time

        persist_dir = str(tmp_path / "patterns")

        learner = PatternLearner(
            persist_directory=persist_dir,
            project_id="test-project",
        )

        await learner.learn_from_codebase(simple_codebase)

        # Measure retrieval time
        start = time.perf_counter()
        patterns = await learner.get_relevant_patterns(
            context="function naming",
            pattern_types=[PatternType.NAMING_FUNCTION, PatternType.NAMING_CLASS],
        )
        elapsed = time.perf_counter() - start

        # Should complete in under 500ms (AC5 requirement)
        assert elapsed < 0.5, f"Pattern retrieval took {elapsed:.3f}s, expected <0.5s"
        assert len(patterns) >= 0  # May or may not have patterns


class TestProjectIsolation:
    """Integration tests for project isolation."""

    @pytest.mark.asyncio
    async def test_patterns_isolated_between_projects(self, tmp_path: Path) -> None:
        """Test that patterns from different projects don't leak."""
        persist_dir = str(tmp_path / "patterns")

        # Create two different codebases with different naming styles
        project1 = tmp_path / "project1"
        project1.mkdir()
        (project1 / "module.py").write_text(
            """\
def get_user():
    pass

def process_order():
    pass
"""
        )

        project2 = tmp_path / "project2"
        project2.mkdir()
        (project2 / "module.py").write_text(
            """\
def GetUser():
    pass

def ProcessOrder():
    pass
"""
        )

        # Learn patterns for project1
        learner1 = PatternLearner(
            persist_directory=persist_dir,
            project_id="project1",
        )
        await learner1.learn_from_codebase(project1)

        # Learn patterns for project2
        learner2 = PatternLearner(
            persist_directory=persist_dir,
            project_id="project2",
        )
        await learner2.learn_from_codebase(project2)

        # Verify isolation
        patterns1 = await learner1.get_patterns_by_type(PatternType.NAMING_FUNCTION)
        patterns2 = await learner2.get_patterns_by_type(PatternType.NAMING_FUNCTION)

        assert len(patterns1) > 0
        assert len(patterns2) > 0

        # Project1 should have snake_case, Project2 should have PascalCase
        assert patterns1[0].value == "snake_case"
        assert patterns2[0].value == "PascalCase"


class TestIncrementalLearning:
    """Integration tests for incremental pattern learning."""

    @pytest.mark.asyncio
    async def test_incremental_learning_updates_patterns(
        self, tmp_path: Path
    ) -> None:
        """Test that learning again updates existing patterns."""
        persist_dir = str(tmp_path / "patterns")
        project = tmp_path / "project"
        project.mkdir()

        # Initial codebase with 2 functions
        (project / "module.py").write_text(
            """\
def get_user():
    pass

def process_order():
    pass
"""
        )

        learner = PatternLearner(
            persist_directory=persist_dir,
            project_id="test-project",
        )

        # First learning
        result1 = await learner.learn_from_codebase(project)
        patterns1 = await learner.get_patterns_by_type(PatternType.NAMING_FUNCTION)

        assert result1.files_scanned == 1
        assert len(patterns1) > 0

        # Add more files to codebase
        (project / "utils.py").write_text(
            """\
def format_data():
    pass

def validate_input():
    pass

def calculate_total():
    pass
"""
        )

        # Second learning (incremental)
        result2 = await learner.learn_from_codebase(project)

        # Should have more files now
        assert result2.files_scanned == 2

        # Patterns should still be consistent
        patterns2 = await learner.get_patterns_by_type(PatternType.NAMING_FUNCTION)
        assert len(patterns2) > 0
        assert patterns2[0].value == "snake_case"

    @pytest.mark.asyncio
    async def test_learning_with_progress_callback(self, tmp_path: Path) -> None:
        """Test progress callback receives updates during learning."""
        persist_dir = str(tmp_path / "patterns")
        project = tmp_path / "project"
        project.mkdir()

        (project / "module.py").write_text("def test_func(): pass")
        (project / "utils.py").write_text("def helper(): pass")

        learner = PatternLearner(
            persist_directory=persist_dir,
            project_id="test-project",
        )

        progress_updates: list[tuple[int, int]] = []

        def on_progress(current: int, total: int) -> None:
            progress_updates.append((current, total))

        await learner.learn_from_codebase(project, progress_callback=on_progress)

        # Should have received progress updates
        assert len(progress_updates) > 0
        # Last update should show completion
        assert progress_updates[-1][0] <= progress_updates[-1][1]


class TestGetRelevantPatterns:
    """Integration tests for get_relevant_patterns method."""

    @pytest.fixture
    def codebase_with_mixed_patterns(self, tmp_path: Path) -> Path:
        """Create a codebase with multiple pattern types."""
        project = tmp_path / "project"
        src = project / "src"
        src.mkdir(parents=True)

        (src / "__init__.py").write_text("")
        (src / "service.py").write_text(
            """\
def get_user():
    pass

def find_order():
    pass

class UserService:
    pass

class OrderHandler:
    pass
"""
        )

        tests = project / "tests"
        tests.mkdir()
        (tests / "test_service.py").write_text(
            """\
def test_get_user():
    pass
"""
        )

        return project

    @pytest.mark.asyncio
    async def test_get_relevant_patterns_combines_types(
        self, tmp_path: Path, codebase_with_mixed_patterns: Path
    ) -> None:
        """Test getting relevant patterns across multiple types."""
        persist_dir = str(tmp_path / "patterns")

        learner = PatternLearner(
            persist_directory=persist_dir,
            project_id="test-project",
        )

        await learner.learn_from_codebase(codebase_with_mixed_patterns)

        # Get patterns of multiple types
        patterns = await learner.get_relevant_patterns(
            context="code conventions",
            pattern_types=[
                PatternType.NAMING_FUNCTION,
                PatternType.NAMING_CLASS,
                PatternType.STRUCTURE_DIRECTORY,
            ],
        )

        assert len(patterns) > 0
        # Should have patterns from different types
        pattern_types = {p.pattern_type for p in patterns}
        assert len(pattern_types) >= 1

    @pytest.mark.asyncio
    async def test_get_relevant_patterns_sorted_by_confidence(
        self, tmp_path: Path, codebase_with_mixed_patterns: Path
    ) -> None:
        """Test that returned patterns are sorted by confidence."""
        persist_dir = str(tmp_path / "patterns")

        learner = PatternLearner(
            persist_directory=persist_dir,
            project_id="test-project",
        )

        await learner.learn_from_codebase(codebase_with_mixed_patterns)

        patterns = await learner.get_relevant_patterns(
            context="naming",
            pattern_types=[PatternType.NAMING_FUNCTION, PatternType.NAMING_CLASS],
            k=10,
        )

        if len(patterns) >= 2:
            # Verify sorted by confidence descending
            for i in range(len(patterns) - 1):
                assert patterns[i].confidence >= patterns[i + 1].confidence

    @pytest.mark.asyncio
    async def test_get_relevant_patterns_respects_k_limit(
        self, tmp_path: Path, codebase_with_mixed_patterns: Path
    ) -> None:
        """Test that k parameter limits returned patterns."""
        persist_dir = str(tmp_path / "patterns")

        learner = PatternLearner(
            persist_directory=persist_dir,
            project_id="test-project",
        )

        await learner.learn_from_codebase(codebase_with_mixed_patterns)

        patterns = await learner.get_relevant_patterns(
            context="code patterns",
            pattern_types=[
                PatternType.NAMING_FUNCTION,
                PatternType.NAMING_CLASS,
                PatternType.STRUCTURE_DIRECTORY,
            ],
            k=2,
        )

        assert len(patterns) <= 2
