"""Unit tests for pattern data structures.

Tests CodePattern, PatternType, and PatternResult dataclasses.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from yolo_developer.memory.patterns import CodePattern, PatternResult, PatternType


class TestPatternType:
    """Tests for PatternType enum."""

    def test_naming_function_value(self) -> None:
        """Test NAMING_FUNCTION has correct value."""
        assert PatternType.NAMING_FUNCTION.value == "naming_function"

    def test_naming_class_value(self) -> None:
        """Test NAMING_CLASS has correct value."""
        assert PatternType.NAMING_CLASS.value == "naming_class"

    def test_naming_variable_value(self) -> None:
        """Test NAMING_VARIABLE has correct value."""
        assert PatternType.NAMING_VARIABLE.value == "naming_variable"

    def test_naming_module_value(self) -> None:
        """Test NAMING_MODULE has correct value."""
        assert PatternType.NAMING_MODULE.value == "naming_module"

    def test_structure_directory_value(self) -> None:
        """Test STRUCTURE_DIRECTORY has correct value."""
        assert PatternType.STRUCTURE_DIRECTORY.value == "structure_directory"

    def test_structure_file_value(self) -> None:
        """Test STRUCTURE_FILE has correct value."""
        assert PatternType.STRUCTURE_FILE.value == "structure_file"

    def test_import_style_value(self) -> None:
        """Test IMPORT_STYLE has correct value."""
        assert PatternType.IMPORT_STYLE.value == "import_style"

    def test_design_pattern_value(self) -> None:
        """Test DESIGN_PATTERN has correct value."""
        assert PatternType.DESIGN_PATTERN.value == "design_pattern"

    def test_all_pattern_types_defined(self) -> None:
        """Verify all expected pattern types exist."""
        expected = {
            "NAMING_FUNCTION",
            "NAMING_CLASS",
            "NAMING_VARIABLE",
            "NAMING_MODULE",
            "STRUCTURE_DIRECTORY",
            "STRUCTURE_FILE",
            "IMPORT_STYLE",
            "DESIGN_PATTERN",
        }
        actual = {pt.name for pt in PatternType}
        assert actual == expected


class TestCodePattern:
    """Tests for CodePattern dataclass."""

    def test_create_minimal_pattern(self) -> None:
        """Test creating pattern with required fields only."""
        pattern = CodePattern(
            pattern_type=PatternType.NAMING_FUNCTION,
            name="function_naming",
            value="snake_case",
            confidence=0.95,
        )

        assert pattern.pattern_type == PatternType.NAMING_FUNCTION
        assert pattern.name == "function_naming"
        assert pattern.value == "snake_case"
        assert pattern.confidence == 0.95
        assert pattern.examples == ()
        assert pattern.source_files == ()
        assert isinstance(pattern.created_at, datetime)

    def test_create_pattern_with_all_fields(self) -> None:
        """Test creating pattern with all fields specified."""
        created = datetime(2026, 1, 4, 12, 0, 0, tzinfo=timezone.utc)
        pattern = CodePattern(
            pattern_type=PatternType.NAMING_CLASS,
            name="class_naming",
            value="PascalCase",
            confidence=0.88,
            examples=("UserService", "OrderHandler", "PaymentProcessor"),
            source_files=("src/services/user.py", "src/handlers/order.py"),
            created_at=created,
        )

        assert pattern.pattern_type == PatternType.NAMING_CLASS
        assert pattern.name == "class_naming"
        assert pattern.value == "PascalCase"
        assert pattern.confidence == 0.88
        assert pattern.examples == ("UserService", "OrderHandler", "PaymentProcessor")
        assert pattern.source_files == ("src/services/user.py", "src/handlers/order.py")
        assert pattern.created_at == created

    def test_pattern_is_frozen(self) -> None:
        """Test that CodePattern is immutable."""
        pattern = CodePattern(
            pattern_type=PatternType.NAMING_FUNCTION,
            name="function_naming",
            value="snake_case",
            confidence=0.95,
        )

        with pytest.raises(AttributeError):
            pattern.value = "camelCase"  # type: ignore[misc]

    def test_to_embedding_text_basic(self) -> None:
        """Test embedding text generation with no examples."""
        pattern = CodePattern(
            pattern_type=PatternType.NAMING_FUNCTION,
            name="function_naming",
            value="snake_case",
            confidence=0.95,
        )

        text = pattern.to_embedding_text()

        assert "naming_function" in text
        assert "function_naming" in text
        assert "snake_case" in text

    def test_to_embedding_text_with_examples(self) -> None:
        """Test embedding text includes examples."""
        pattern = CodePattern(
            pattern_type=PatternType.NAMING_FUNCTION,
            name="function_naming",
            value="snake_case",
            confidence=0.95,
            examples=("get_user", "process_order", "validate_input"),
        )

        text = pattern.to_embedding_text()

        assert "get_user" in text
        assert "process_order" in text
        assert "validate_input" in text

    def test_to_embedding_text_limits_examples(self) -> None:
        """Test embedding text limits to 5 examples."""
        many_examples = tuple(f"func_{i}" for i in range(10))
        pattern = CodePattern(
            pattern_type=PatternType.NAMING_FUNCTION,
            name="function_naming",
            value="snake_case",
            confidence=0.95,
            examples=many_examples,
        )

        text = pattern.to_embedding_text()

        # First 5 should be present
        assert "func_0" in text
        assert "func_4" in text
        # 6th and beyond should not
        assert "func_5" not in text
        assert "func_9" not in text

    def test_confidence_range_valid(self) -> None:
        """Test pattern accepts valid confidence values."""
        low = CodePattern(
            pattern_type=PatternType.NAMING_FUNCTION,
            name="test",
            value="snake_case",
            confidence=0.0,
        )
        high = CodePattern(
            pattern_type=PatternType.NAMING_FUNCTION,
            name="test",
            value="snake_case",
            confidence=1.0,
        )

        assert low.confidence == 0.0
        assert high.confidence == 1.0

    def test_pattern_equality(self) -> None:
        """Test pattern equality comparison."""
        created = datetime(2026, 1, 4, 12, 0, 0, tzinfo=timezone.utc)
        pattern1 = CodePattern(
            pattern_type=PatternType.NAMING_FUNCTION,
            name="function_naming",
            value="snake_case",
            confidence=0.95,
            created_at=created,
        )
        pattern2 = CodePattern(
            pattern_type=PatternType.NAMING_FUNCTION,
            name="function_naming",
            value="snake_case",
            confidence=0.95,
            created_at=created,
        )

        assert pattern1 == pattern2

    def test_pattern_hash(self) -> None:
        """Test pattern can be used as dict key or in set."""
        created = datetime(2026, 1, 4, 12, 0, 0, tzinfo=timezone.utc)
        pattern = CodePattern(
            pattern_type=PatternType.NAMING_FUNCTION,
            name="function_naming",
            value="snake_case",
            confidence=0.95,
            created_at=created,
        )

        # Should be hashable
        pattern_set = {pattern}
        assert pattern in pattern_set

        pattern_dict = {pattern: "value"}
        assert pattern_dict[pattern] == "value"


class TestPatternResult:
    """Tests for PatternResult dataclass."""

    def test_create_pattern_result(self) -> None:
        """Test creating a pattern result."""
        pattern = CodePattern(
            pattern_type=PatternType.NAMING_FUNCTION,
            name="function_naming",
            value="snake_case",
            confidence=0.95,
        )
        result = PatternResult(
            pattern=pattern,
            similarity=0.87,
        )

        assert result.pattern == pattern
        assert result.similarity == 0.87

    def test_pattern_result_is_frozen(self) -> None:
        """Test that PatternResult is immutable."""
        pattern = CodePattern(
            pattern_type=PatternType.NAMING_FUNCTION,
            name="function_naming",
            value="snake_case",
            confidence=0.95,
        )
        result = PatternResult(pattern=pattern, similarity=0.87)

        with pytest.raises(AttributeError):
            result.similarity = 0.5  # type: ignore[misc]

    def test_pattern_result_equality(self) -> None:
        """Test pattern result equality comparison."""
        pattern = CodePattern(
            pattern_type=PatternType.NAMING_FUNCTION,
            name="function_naming",
            value="snake_case",
            confidence=0.95,
        )
        result1 = PatternResult(pattern=pattern, similarity=0.87)
        result2 = PatternResult(pattern=pattern, similarity=0.87)

        assert result1 == result2


class TestPatternSerialization:
    """Tests for pattern serialization support."""

    def test_pattern_to_dict(self) -> None:
        """Test pattern can be converted to dict for JSON serialization."""
        from dataclasses import asdict

        created = datetime(2026, 1, 4, 12, 0, 0, tzinfo=timezone.utc)
        pattern = CodePattern(
            pattern_type=PatternType.NAMING_FUNCTION,
            name="function_naming",
            value="snake_case",
            confidence=0.95,
            examples=("get_user", "process_order"),
            source_files=("src/main.py",),
            created_at=created,
        )

        data = asdict(pattern)

        assert data["pattern_type"] == PatternType.NAMING_FUNCTION
        assert data["name"] == "function_naming"
        assert data["value"] == "snake_case"
        assert data["confidence"] == 0.95
        assert data["examples"] == ("get_user", "process_order")
        assert data["source_files"] == ("src/main.py",)
        assert data["created_at"] == created
