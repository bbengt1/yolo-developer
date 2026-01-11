"""Unit tests for pattern following utilities (Story 8.7).

Tests for pattern types (Task 10), pattern query functions (Task 11),
naming analysis (Task 12), error handling analysis (Task 13),
style analysis (Task 14), and pattern validation (Task 15).

Example:
    >>> pytest tests/unit/agents/dev/test_pattern_utils.py -v
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from yolo_developer.agents.dev.pattern_utils import clear_pattern_cache
from yolo_developer.memory.patterns import CodePattern, PatternType


@pytest.fixture(autouse=True)
def clear_cache_before_test() -> None:
    """Clear pattern cache before each test to ensure isolation."""
    clear_pattern_cache()


# =============================================================================
# Task 10: Tests for Pattern Types
# =============================================================================


class TestPatternDeviation:
    """Tests for PatternDeviation dataclass (AC: 4, 6, 7)."""

    def test_pattern_deviation_construction(self) -> None:
        """Test PatternDeviation dataclass construction with all fields."""
        from yolo_developer.agents.dev.pattern_utils import PatternDeviation

        deviation = PatternDeviation(
            pattern_type=PatternType.NAMING_FUNCTION,
            pattern_name="function_naming",
            expected_value="snake_case",
            actual_value="camelCase",
            severity="high",
            justification="Legacy code compatibility",
            location="line 42: myFunction",
        )

        assert deviation.pattern_type == PatternType.NAMING_FUNCTION
        assert deviation.pattern_name == "function_naming"
        assert deviation.expected_value == "snake_case"
        assert deviation.actual_value == "camelCase"
        assert deviation.severity == "high"
        assert deviation.justification == "Legacy code compatibility"
        assert deviation.location == "line 42: myFunction"

    def test_pattern_deviation_minimal_construction(self) -> None:
        """Test PatternDeviation construction with required fields only."""
        from yolo_developer.agents.dev.pattern_utils import PatternDeviation

        deviation = PatternDeviation(
            pattern_type=PatternType.NAMING_CLASS,
            pattern_name="class_naming",
            expected_value="PascalCase",
            actual_value="snake_case",
            severity="medium",
        )

        assert deviation.pattern_type == PatternType.NAMING_CLASS
        assert deviation.justification is None
        assert deviation.location is None

    def test_pattern_deviation_frozen(self) -> None:
        """Test PatternDeviation is immutable (frozen)."""
        from yolo_developer.agents.dev.pattern_utils import PatternDeviation

        deviation = PatternDeviation(
            pattern_type=PatternType.NAMING_FUNCTION,
            pattern_name="function_naming",
            expected_value="snake_case",
            actual_value="camelCase",
            severity="high",
        )

        with pytest.raises(AttributeError):
            deviation.severity = "low"  # type: ignore[misc]

    def test_pattern_deviation_severity_values(self) -> None:
        """Test PatternDeviation accepts valid severity values."""
        from yolo_developer.agents.dev.pattern_utils import PatternDeviation

        for severity in ["high", "medium", "low"]:
            deviation = PatternDeviation(
                pattern_type=PatternType.NAMING_FUNCTION,
                pattern_name="test",
                expected_value="expected",
                actual_value="actual",
                severity=severity,  # type: ignore[arg-type]
            )
            assert deviation.severity == severity


class TestPatternValidationResult:
    """Tests for PatternValidationResult dataclass (AC: 4, 5)."""

    def test_pattern_validation_result_construction(self) -> None:
        """Test PatternValidationResult dataclass construction."""
        from yolo_developer.agents.dev.pattern_utils import PatternValidationResult

        result = PatternValidationResult(
            score=85,
            passed=True,
            threshold=70,
            patterns_checked=10,
            adherence_percentage=85.0,
        )

        assert result.score == 85
        assert result.passed is True
        assert result.threshold == 70
        assert result.patterns_checked == 10
        assert result.adherence_percentage == 85.0
        assert result.deviations == []

    def test_pattern_validation_result_defaults(self) -> None:
        """Test PatternValidationResult has correct defaults."""
        from yolo_developer.agents.dev.pattern_utils import PatternValidationResult

        result = PatternValidationResult()

        assert result.score == 100
        assert result.passed is True
        assert result.threshold == 70
        assert result.patterns_checked == 0
        assert result.adherence_percentage == 100.0
        assert result.deviations == []

    def test_pattern_validation_result_to_dict(self) -> None:
        """Test PatternValidationResult to_dict() serialization."""
        from yolo_developer.agents.dev.pattern_utils import (
            PatternDeviation,
            PatternValidationResult,
        )

        deviation = PatternDeviation(
            pattern_type=PatternType.NAMING_FUNCTION,
            pattern_name="function_naming",
            expected_value="snake_case",
            actual_value="camelCase",
            severity="high",
        )

        result = PatternValidationResult(
            score=80,
            passed=True,
            threshold=70,
            patterns_checked=5,
            adherence_percentage=80.0,
            deviations=[deviation],
        )

        result_dict = result.to_dict()

        assert result_dict["score"] == 80
        assert result_dict["passed"] is True
        assert result_dict["threshold"] == 70
        assert result_dict["patterns_checked"] == 5
        assert result_dict["adherence_percentage"] == 80.0
        assert result_dict["deviation_count"] == 1
        assert len(result_dict["deviations"]) == 1
        assert result_dict["deviations"][0]["pattern_type"] == "naming_function"
        assert result_dict["deviations"][0]["severity"] == "high"

    def test_pattern_validation_result_mutable(self) -> None:
        """Test PatternValidationResult is mutable (not frozen)."""
        from yolo_developer.agents.dev.pattern_utils import (
            PatternDeviation,
            PatternValidationResult,
        )

        result = PatternValidationResult()

        # Should be able to modify
        result.score = 50
        result.passed = False

        deviation = PatternDeviation(
            pattern_type=PatternType.NAMING_FUNCTION,
            pattern_name="test",
            expected_value="expected",
            actual_value="actual",
            severity="medium",
        )
        result.deviations.append(deviation)

        assert result.score == 50
        assert result.passed is False
        assert len(result.deviations) == 1


class TestErrorHandlingPattern:
    """Tests for ErrorHandlingPattern dataclass (AC: 2, 6)."""

    def test_error_handling_pattern_construction(self) -> None:
        """Test ErrorHandlingPattern dataclass construction."""
        from yolo_developer.agents.dev.pattern_utils import ErrorHandlingPattern

        pattern = ErrorHandlingPattern(
            pattern_name="specific_exceptions",
            exception_types=("ValueError", "TypeError", "KeyError"),
            handling_style="specific exceptions with context",
            examples=("try:\n    ...\nexcept ValueError as e:\n    ...",),
        )

        assert pattern.pattern_name == "specific_exceptions"
        assert pattern.exception_types == ("ValueError", "TypeError", "KeyError")
        assert pattern.handling_style == "specific exceptions with context"
        assert len(pattern.examples) == 1

    def test_error_handling_pattern_frozen(self) -> None:
        """Test ErrorHandlingPattern is immutable (frozen)."""
        from yolo_developer.agents.dev.pattern_utils import ErrorHandlingPattern

        pattern = ErrorHandlingPattern(
            pattern_name="specific_exceptions",
            exception_types=("ValueError",),
            handling_style="specific exceptions",
        )

        with pytest.raises(AttributeError):
            pattern.handling_style = "generic"  # type: ignore[misc]

    def test_error_handling_pattern_defaults(self) -> None:
        """Test ErrorHandlingPattern has correct defaults for examples."""
        from yolo_developer.agents.dev.pattern_utils import ErrorHandlingPattern

        pattern = ErrorHandlingPattern(
            pattern_name="test",
            exception_types=("Exception",),
            handling_style="generic",
        )

        assert pattern.examples == ()


class TestStylePattern:
    """Tests for StylePattern dataclass (AC: 3, 6)."""

    def test_style_pattern_construction(self) -> None:
        """Test StylePattern dataclass construction."""
        from yolo_developer.agents.dev.pattern_utils import StylePattern

        pattern = StylePattern(
            pattern_name="import_ordering",
            category="import_style",
            value="stdlib, third_party, local",
            examples=("import os\n\nimport pytest\n\nfrom mypackage import foo",),
        )

        assert pattern.pattern_name == "import_ordering"
        assert pattern.category == "import_style"
        assert pattern.value == "stdlib, third_party, local"
        assert len(pattern.examples) == 1

    def test_style_pattern_categories(self) -> None:
        """Test StylePattern accepts valid category values."""
        from yolo_developer.agents.dev.pattern_utils import StylePattern

        for category in ["import_style", "docstring_format", "type_hint_style"]:
            pattern = StylePattern(
                pattern_name="test",
                category=category,  # type: ignore[arg-type]
                value="test_value",
            )
            assert pattern.category == category

    def test_style_pattern_frozen(self) -> None:
        """Test StylePattern is immutable (frozen)."""
        from yolo_developer.agents.dev.pattern_utils import StylePattern

        pattern = StylePattern(
            pattern_name="test",
            category="import_style",
            value="test_value",
        )

        with pytest.raises(AttributeError):
            pattern.value = "new_value"  # type: ignore[misc]


# =============================================================================
# Task 11: Tests for Pattern Query Functions
# =============================================================================


class TestGetNamingPatterns:
    """Tests for get_naming_patterns() function (AC: 6)."""

    def test_get_naming_patterns_with_memory_context(self) -> None:
        """Test get_naming_patterns() with memory_context in state."""
        from yolo_developer.agents.dev.pattern_utils import get_naming_patterns

        # Create mock PatternLearner
        mock_learner = MagicMock()
        mock_learner.get_patterns_by_type = MagicMock(
            return_value=[
                CodePattern(
                    pattern_type=PatternType.NAMING_FUNCTION,
                    name="function_naming",
                    value="snake_case",
                    confidence=0.95,
                    examples=("get_user", "process_order"),
                ),
            ]
        )

        state: dict[str, Any] = {
            "memory_context": {
                "pattern_learner": mock_learner,
            }
        }

        patterns = get_naming_patterns(state)

        assert len(patterns) >= 1
        assert any(p.pattern_type == PatternType.NAMING_FUNCTION for p in patterns)

    def test_get_naming_patterns_without_memory_context(self) -> None:
        """Test get_naming_patterns() returns defaults without memory_context."""
        from yolo_developer.agents.dev.pattern_utils import get_naming_patterns

        state: dict[str, Any] = {}

        patterns = get_naming_patterns(state)

        # Should return default patterns
        assert len(patterns) >= 1
        # Check for default naming patterns
        assert any(p.value == "snake_case" for p in patterns)

    def test_get_naming_patterns_with_empty_memory_context(self) -> None:
        """Test get_naming_patterns() with empty memory_context."""
        from yolo_developer.agents.dev.pattern_utils import get_naming_patterns

        state: dict[str, Any] = {"memory_context": {}}

        patterns = get_naming_patterns(state)

        # Should return default patterns
        assert len(patterns) >= 1


class TestGetErrorPatterns:
    """Tests for get_error_patterns() function (AC: 6)."""

    def test_get_error_patterns_with_memory_context(self) -> None:
        """Test get_error_patterns() with memory_context in state."""
        from yolo_developer.agents.dev.pattern_utils import (
            ErrorHandlingPattern,
            get_error_patterns,
        )

        state: dict[str, Any] = {
            "memory_context": {
                "error_patterns": [
                    ErrorHandlingPattern(
                        pattern_name="specific_exceptions",
                        exception_types=("ValueError",),
                        handling_style="specific exceptions",
                    ),
                ],
            }
        }

        patterns = get_error_patterns(state)

        assert len(patterns) >= 1
        assert all(isinstance(p, ErrorHandlingPattern) for p in patterns)

    def test_get_error_patterns_without_memory_context(self) -> None:
        """Test get_error_patterns() returns defaults without memory_context."""
        from yolo_developer.agents.dev.pattern_utils import (
            ErrorHandlingPattern,
            get_error_patterns,
        )

        state: dict[str, Any] = {}

        patterns = get_error_patterns(state)

        # Should return default patterns
        assert len(patterns) >= 1
        assert all(isinstance(p, ErrorHandlingPattern) for p in patterns)


class TestGetStylePatterns:
    """Tests for get_style_patterns() function (AC: 6)."""

    def test_get_style_patterns_with_memory_context(self) -> None:
        """Test get_style_patterns() with memory_context in state."""
        from yolo_developer.agents.dev.pattern_utils import (
            StylePattern,
            get_style_patterns,
        )

        state: dict[str, Any] = {
            "memory_context": {
                "style_patterns": [
                    StylePattern(
                        pattern_name="import_ordering",
                        category="import_style",
                        value="stdlib, third_party, local",
                    ),
                ],
            }
        }

        patterns = get_style_patterns(state)

        assert len(patterns) >= 1
        assert all(isinstance(p, StylePattern) for p in patterns)

    def test_get_style_patterns_without_memory_context(self) -> None:
        """Test get_style_patterns() returns defaults without memory_context."""
        from yolo_developer.agents.dev.pattern_utils import (
            StylePattern,
            get_style_patterns,
        )

        state: dict[str, Any] = {}

        patterns = get_style_patterns(state)

        # Should return default patterns
        assert len(patterns) >= 1
        assert all(isinstance(p, StylePattern) for p in patterns)


# =============================================================================
# Task 12: Tests for Naming Pattern Analysis
# =============================================================================


class TestAnalyzeNamingPatterns:
    """Tests for analyze_naming_patterns() function (AC: 1)."""

    def test_analyze_naming_patterns_snake_case_functions(self) -> None:
        """Test analyze_naming_patterns() with valid snake_case functions."""
        from yolo_developer.agents.dev.pattern_utils import analyze_naming_patterns

        code = '''
def get_user():
    pass

def process_order():
    pass

def validate_input():
    pass
'''

        patterns = [
            CodePattern(
                pattern_type=PatternType.NAMING_FUNCTION,
                name="function_naming",
                value="snake_case",
                confidence=1.0,
            ),
        ]

        deviations = analyze_naming_patterns(code, patterns)

        # No deviations expected for valid snake_case
        assert len(deviations) == 0

    def test_analyze_naming_patterns_pascal_case_classes(self) -> None:
        """Test analyze_naming_patterns() with valid PascalCase classes."""
        from yolo_developer.agents.dev.pattern_utils import analyze_naming_patterns

        code = '''
class UserService:
    pass

class OrderProcessor:
    pass

class DataValidator:
    pass
'''

        patterns = [
            CodePattern(
                pattern_type=PatternType.NAMING_CLASS,
                name="class_naming",
                value="PascalCase",
                confidence=1.0,
            ),
        ]

        deviations = analyze_naming_patterns(code, patterns)

        # No deviations expected for valid PascalCase
        assert len(deviations) == 0

    def test_analyze_naming_patterns_camel_case_deviation(self) -> None:
        """Test analyze_naming_patterns() detecting camelCase deviation."""
        from yolo_developer.agents.dev.pattern_utils import analyze_naming_patterns

        code = '''
def getUser():
    pass

def processOrder():
    pass
'''

        patterns = [
            CodePattern(
                pattern_type=PatternType.NAMING_FUNCTION,
                name="function_naming",
                value="snake_case",
                confidence=1.0,
            ),
        ]

        deviations = analyze_naming_patterns(code, patterns)

        # Should detect camelCase deviations
        assert len(deviations) >= 2
        assert all(d.pattern_type == PatternType.NAMING_FUNCTION for d in deviations)
        assert all(d.expected_value == "snake_case" for d in deviations)

    def test_analyze_naming_patterns_empty_patterns(self) -> None:
        """Test analyze_naming_patterns() with no patterns (empty list)."""
        from yolo_developer.agents.dev.pattern_utils import analyze_naming_patterns

        code = '''
def getUser():
    pass
'''

        patterns: list[CodePattern] = []

        deviations = analyze_naming_patterns(code, patterns)

        # No deviations when no patterns to check against
        assert len(deviations) == 0

    def test_analyze_naming_patterns_invalid_syntax(self) -> None:
        """Test analyze_naming_patterns() with invalid Python syntax."""
        from yolo_developer.agents.dev.pattern_utils import analyze_naming_patterns

        code = "def invalid syntax("

        patterns = [
            CodePattern(
                pattern_type=PatternType.NAMING_FUNCTION,
                name="function_naming",
                value="snake_case",
                confidence=1.0,
            ),
        ]

        deviations = analyze_naming_patterns(code, patterns)

        # Should return empty list on syntax error
        assert len(deviations) == 0


# =============================================================================
# Task 13: Tests for Error Handling Pattern Analysis
# =============================================================================


class TestAnalyzeErrorHandlingPatterns:
    """Tests for analyze_error_handling_patterns() function (AC: 2)."""

    def test_analyze_error_handling_matching_patterns(self) -> None:
        """Test analyze_error_handling_patterns() with matching patterns."""
        from yolo_developer.agents.dev.pattern_utils import (
            ErrorHandlingPattern,
            analyze_error_handling_patterns,
        )

        code = '''
try:
    result = process_data()
except ValueError as e:
    logger.error("Invalid value", error=str(e))
    raise
'''

        patterns = [
            ErrorHandlingPattern(
                pattern_name="specific_exceptions",
                exception_types=("ValueError", "TypeError"),
                handling_style="specific exceptions with context",
            ),
        ]

        deviations = analyze_error_handling_patterns(code, patterns)

        # No deviations for matching patterns
        assert len(deviations) == 0

    def test_analyze_error_handling_generic_exception(self) -> None:
        """Test analyze_error_handling_patterns() with generic Exception usage."""
        from yolo_developer.agents.dev.pattern_utils import (
            ErrorHandlingPattern,
            analyze_error_handling_patterns,
        )

        code = '''
try:
    result = process_data()
except Exception as e:
    pass
'''

        patterns = [
            ErrorHandlingPattern(
                pattern_name="specific_exceptions",
                exception_types=("ValueError", "TypeError"),
                handling_style="specific exceptions with context",
            ),
        ]

        deviations = analyze_error_handling_patterns(code, patterns)

        # Should flag generic Exception usage
        assert len(deviations) >= 1
        assert any("Exception" in d.actual_value for d in deviations)

    def test_analyze_error_handling_bare_except(self) -> None:
        """Test analyze_error_handling_patterns() detecting bare except."""
        from yolo_developer.agents.dev.pattern_utils import (
            ErrorHandlingPattern,
            analyze_error_handling_patterns,
        )

        code = '''
try:
    result = process_data()
except:
    pass
'''

        patterns = [
            ErrorHandlingPattern(
                pattern_name="specific_exceptions",
                exception_types=("ValueError",),
                handling_style="specific exceptions",
            ),
        ]

        deviations = analyze_error_handling_patterns(code, patterns)

        # Should flag bare except
        assert len(deviations) >= 1
        assert any("bare except" in d.actual_value.lower() for d in deviations)

    def test_analyze_error_handling_custom_exceptions(self) -> None:
        """Test analyze_error_handling_patterns() with custom exceptions."""
        from yolo_developer.agents.dev.pattern_utils import (
            ErrorHandlingPattern,
            analyze_error_handling_patterns,
        )

        code = '''
try:
    result = process_data()
except ConfigurationError as e:
    logger.error("Config error", error=str(e))
    raise
'''

        patterns = [
            ErrorHandlingPattern(
                pattern_name="project_exceptions",
                exception_types=("ConfigurationError", "ValidationError"),
                handling_style="project custom exceptions",
            ),
        ]

        deviations = analyze_error_handling_patterns(code, patterns)

        # Should not flag custom project exceptions that match patterns
        assert len(deviations) == 0


# =============================================================================
# Task 14: Tests for Style Pattern Analysis
# =============================================================================


class TestAnalyzeStylePatterns:
    """Tests for analyze_style_patterns() function (AC: 3)."""

    def test_analyze_style_patterns_import_ordering(self) -> None:
        """Test analyze_style_patterns() import ordering."""
        from yolo_developer.agents.dev.pattern_utils import (
            StylePattern,
            analyze_style_patterns,
        )

        code = '''from __future__ import annotations

import os
import sys

import pytest
import structlog

from yolo_developer.config import load_config
'''

        patterns = [
            StylePattern(
                pattern_name="import_ordering",
                category="import_style",
                value="stdlib, third_party, local",
            ),
        ]

        deviations = analyze_style_patterns(code, patterns)

        # No deviations for correct import ordering
        assert len(deviations) == 0

    def test_analyze_style_patterns_docstring_format(self) -> None:
        """Test analyze_style_patterns() docstring format."""
        from yolo_developer.agents.dev.pattern_utils import (
            StylePattern,
            analyze_style_patterns,
        )

        code = '''
def process_data(data: dict) -> str:
    """Process input data.

    Args:
        data: Input dictionary to process.

    Returns:
        Processed string result.
    """
    return str(data)
'''

        patterns = [
            StylePattern(
                pattern_name="docstring_format",
                category="docstring_format",
                value="Google-style",
            ),
        ]

        deviations = analyze_style_patterns(code, patterns)

        # No deviations for Google-style docstrings
        assert len(deviations) == 0

    def test_analyze_style_patterns_type_annotations(self) -> None:
        """Test analyze_style_patterns() type annotation presence."""
        from yolo_developer.agents.dev.pattern_utils import (
            StylePattern,
            analyze_style_patterns,
        )

        code = '''
def process_data(data: dict[str, Any]) -> str:
    result: str = str(data)
    return result
'''

        patterns = [
            StylePattern(
                pattern_name="type_annotations",
                category="type_hint_style",
                value="full annotations required",
            ),
        ]

        deviations = analyze_style_patterns(code, patterns)

        # No deviations for fully annotated code
        assert len(deviations) == 0

    def test_analyze_style_patterns_multiple_deviations(self) -> None:
        """Test analyze_style_patterns() with multiple style deviations."""
        from yolo_developer.agents.dev.pattern_utils import (
            StylePattern,
            analyze_style_patterns,
        )

        # Code with multiple style issues
        code = '''from yolo_developer.config import load_config
import os

def process_data(data):
    result = str(data)
    return result
'''

        patterns = [
            StylePattern(
                pattern_name="import_ordering",
                category="import_style",
                value="stdlib, third_party, local",
            ),
            StylePattern(
                pattern_name="type_annotations",
                category="type_hint_style",
                value="full annotations required",
            ),
        ]

        deviations = analyze_style_patterns(code, patterns)

        # Should detect multiple deviations
        assert len(deviations) >= 1


# =============================================================================
# Task 15: Tests for Pattern Validation Aggregation
# =============================================================================


class TestValidatePatternAdherence:
    """Tests for validate_pattern_adherence() function (AC: 4, 5)."""

    def test_validate_pattern_adherence_compliant_code(self) -> None:
        """Test validate_pattern_adherence() with fully compliant code."""
        from yolo_developer.agents.dev.pattern_utils import validate_pattern_adherence

        code = '''from __future__ import annotations

import os

import structlog

from yolo_developer.config import load_config


def get_user(user_id: str) -> dict:
    """Get user by ID.

    Args:
        user_id: The user identifier.

    Returns:
        User data dictionary.
    """
    return {"id": user_id}


class UserService:
    """Service for user operations."""

    def process_request(self) -> None:
        """Process a user request."""
        try:
            result = self._do_work()
        except ValueError as e:
            structlog.get_logger().error("Error", error=str(e))
            raise
'''

        state: dict[str, Any] = {}

        result = validate_pattern_adherence(code, state)

        assert result.passed is True
        assert result.score >= 70

    def test_validate_pattern_adherence_multiple_deviations(self) -> None:
        """Test validate_pattern_adherence() with multiple deviations."""
        from yolo_developer.agents.dev.pattern_utils import validate_pattern_adherence

        # Code with multiple pattern deviations
        code = '''
def getUserData():
    pass

def processOrder():
    pass

class user_service:
    pass

try:
    x = 1
except:
    pass
'''

        state: dict[str, Any] = {}

        result = validate_pattern_adherence(code, state)

        # Should have deviations
        assert len(result.deviations) >= 1

    def test_validate_pattern_adherence_score_calculation(self) -> None:
        """Test validate_pattern_adherence() score calculation."""
        from yolo_developer.agents.dev.pattern_utils import validate_pattern_adherence

        code = '''
def getUserData():
    pass
'''

        state: dict[str, Any] = {}

        result = validate_pattern_adherence(code, state)

        # Score should be reduced for deviations
        assert result.score <= 100
        # Adherence percentage should be calculated
        assert result.adherence_percentage <= 100.0

    def test_validate_pattern_adherence_severity_aggregation(self) -> None:
        """Test validate_pattern_adherence() severity aggregation."""
        from yolo_developer.agents.dev.pattern_utils import validate_pattern_adherence

        # Code with high severity deviation (naming) and medium (bare except)
        code = '''
def getUserData():
    try:
        x = 1
    except:
        pass
'''

        state: dict[str, Any] = {}

        result = validate_pattern_adherence(code, state)

        # Should have deviations of different severities
        if result.deviations:
            severities = {d.severity for d in result.deviations}
            assert len(severities) >= 1

    def test_validate_pattern_adherence_threshold(self) -> None:
        """Test validate_pattern_adherence() with custom threshold."""
        from yolo_developer.agents.dev.pattern_utils import validate_pattern_adherence

        code = '''
def get_user():
    pass
'''

        state: dict[str, Any] = {}

        result = validate_pattern_adherence(code, state, threshold=90)

        # Should use provided threshold
        assert result.threshold == 90
