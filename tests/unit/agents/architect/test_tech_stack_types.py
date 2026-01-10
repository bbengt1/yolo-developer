"""Unit tests for tech stack type definitions (Story 7.6 - Task 1, 9).

Tests for TechStackCategory, ConstraintViolation, StackPattern, and TechStackValidation
types used in tech stack constraint validation.

AC Coverage:
    - AC1: TechStackValidation is a frozen dataclass importable from architect
    - AC7: Frozen (immutable), to_dict() method, overall_compliance field
"""

from __future__ import annotations

import pytest

# RED phase: These imports will fail until we implement the types
from yolo_developer.agents.architect import (
    ConstraintViolation,
    StackPattern,
    TechStackCategory,
    TechStackValidation,
)


class TestTechStackCategory:
    """Tests for TechStackCategory Literal type."""

    def test_valid_category_runtime(self) -> None:
        """Test runtime is a valid category."""
        category: TechStackCategory = "runtime"
        assert category == "runtime"

    def test_valid_category_framework(self) -> None:
        """Test framework is a valid category."""
        category: TechStackCategory = "framework"
        assert category == "framework"

    def test_valid_category_database(self) -> None:
        """Test database is a valid category."""
        category: TechStackCategory = "database"
        assert category == "database"

    def test_valid_category_testing(self) -> None:
        """Test testing is a valid category."""
        category: TechStackCategory = "testing"
        assert category == "testing"

    def test_valid_category_tooling(self) -> None:
        """Test tooling is a valid category."""
        category: TechStackCategory = "tooling"
        assert category == "tooling"


class TestConstraintViolation:
    """Tests for ConstraintViolation frozen dataclass."""

    def test_create_constraint_violation(self) -> None:
        """Test creating a constraint violation with all fields."""
        violation = ConstraintViolation(
            technology="SQLite",
            expected_version=None,
            actual_version="3.x",
            severity="critical",
            suggested_alternative="Use ChromaDB as configured for vector storage",
        )
        assert violation.technology == "SQLite"
        assert violation.expected_version is None
        assert violation.actual_version == "3.x"
        assert violation.severity == "critical"
        assert violation.suggested_alternative == "Use ChromaDB as configured for vector storage"

    def test_constraint_violation_with_version_mismatch(self) -> None:
        """Test violation with version mismatch."""
        violation = ConstraintViolation(
            technology="Python",
            expected_version="3.10",
            actual_version="3.8",
            severity="high",
            suggested_alternative="Upgrade to Python 3.10+",
        )
        assert violation.expected_version == "3.10"
        assert violation.actual_version == "3.8"
        assert violation.severity == "high"

    def test_constraint_violation_is_frozen(self) -> None:
        """Test that ConstraintViolation is immutable (frozen)."""
        violation = ConstraintViolation(
            technology="SQLite",
            expected_version=None,
            actual_version="3.x",
            severity="critical",
            suggested_alternative="Use ChromaDB",
        )
        with pytest.raises(AttributeError):
            violation.technology = "PostgreSQL"  # type: ignore[misc]

    def test_constraint_violation_to_dict(self) -> None:
        """Test to_dict serialization."""
        violation = ConstraintViolation(
            technology="SQLite",
            expected_version="5.0",
            actual_version="3.x",
            severity="critical",
            suggested_alternative="Use ChromaDB",
        )
        result = violation.to_dict()
        assert result == {
            "technology": "SQLite",
            "expected_version": "5.0",
            "actual_version": "3.x",
            "severity": "critical",
            "suggested_alternative": "Use ChromaDB",
        }

    def test_constraint_violation_to_dict_with_none_version(self) -> None:
        """Test to_dict with None expected_version."""
        violation = ConstraintViolation(
            technology="SQLite",
            expected_version=None,
            actual_version="3.x",
            severity="critical",
            suggested_alternative="Use ChromaDB",
        )
        result = violation.to_dict()
        assert result["expected_version"] is None


class TestStackPattern:
    """Tests for StackPattern frozen dataclass."""

    def test_create_stack_pattern(self) -> None:
        """Test creating a stack pattern with all fields."""
        pattern = StackPattern(
            pattern_name="pytest-fixtures",
            description="Use pytest fixtures for test setup/teardown",
            rationale="pytest is configured as test framework; fixtures provide clean test isolation",
            applicable_technologies=("pytest", "pytest-asyncio"),
        )
        assert pattern.pattern_name == "pytest-fixtures"
        assert pattern.description == "Use pytest fixtures for test setup/teardown"
        assert pattern.rationale == "pytest is configured as test framework; fixtures provide clean test isolation"
        assert pattern.applicable_technologies == ("pytest", "pytest-asyncio")

    def test_stack_pattern_is_frozen(self) -> None:
        """Test that StackPattern is immutable (frozen)."""
        pattern = StackPattern(
            pattern_name="uv-dependency-management",
            description="Use uv for fast dependency installation",
            rationale="uv is configured as package manager",
            applicable_technologies=("uv",),
        )
        with pytest.raises(AttributeError):
            pattern.pattern_name = "pip-install"  # type: ignore[misc]

    def test_stack_pattern_to_dict(self) -> None:
        """Test to_dict serialization."""
        pattern = StackPattern(
            pattern_name="pytest-fixtures",
            description="Use pytest fixtures",
            rationale="pytest is configured",
            applicable_technologies=("pytest", "pytest-asyncio"),
        )
        result = pattern.to_dict()
        assert result == {
            "pattern_name": "pytest-fixtures",
            "description": "Use pytest fixtures",
            "rationale": "pytest is configured",
            "applicable_technologies": ["pytest", "pytest-asyncio"],
        }

    def test_stack_pattern_to_dict_converts_tuple_to_list(self) -> None:
        """Test that applicable_technologies is converted to list in to_dict."""
        pattern = StackPattern(
            pattern_name="test",
            description="test",
            rationale="test",
            applicable_technologies=("a", "b", "c"),
        )
        result = pattern.to_dict()
        assert isinstance(result["applicable_technologies"], list)
        assert result["applicable_technologies"] == ["a", "b", "c"]


class TestTechStackValidation:
    """Tests for TechStackValidation frozen dataclass."""

    def test_create_compliant_validation(self) -> None:
        """Test creating a compliant validation result."""
        validation = TechStackValidation(
            overall_compliance=True,
            violations=(),
            suggested_patterns=(),
            summary="All design decisions comply with configured tech stack",
        )
        assert validation.overall_compliance is True
        assert validation.violations == ()
        assert validation.suggested_patterns == ()
        assert validation.summary == "All design decisions comply with configured tech stack"

    def test_create_non_compliant_validation(self) -> None:
        """Test creating a non-compliant validation result with violations."""
        violation = ConstraintViolation(
            technology="SQLite",
            expected_version=None,
            actual_version="3.x",
            severity="critical",
            suggested_alternative="Use ChromaDB",
        )
        pattern = StackPattern(
            pattern_name="pytest-fixtures",
            description="Use pytest fixtures",
            rationale="pytest is configured",
            applicable_technologies=("pytest",),
        )
        validation = TechStackValidation(
            overall_compliance=False,
            violations=(violation,),
            suggested_patterns=(pattern,),
            summary="1 constraint violation found",
        )
        assert validation.overall_compliance is False
        assert len(validation.violations) == 1
        assert len(validation.suggested_patterns) == 1

    def test_tech_stack_validation_is_frozen(self) -> None:
        """Test that TechStackValidation is immutable (frozen)."""
        validation = TechStackValidation(
            overall_compliance=True,
            violations=(),
            suggested_patterns=(),
            summary="Compliant",
        )
        with pytest.raises(AttributeError):
            validation.overall_compliance = False  # type: ignore[misc]

    def test_tech_stack_validation_to_dict(self) -> None:
        """Test to_dict serialization."""
        violation = ConstraintViolation(
            technology="SQLite",
            expected_version=None,
            actual_version="3.x",
            severity="critical",
            suggested_alternative="Use ChromaDB",
        )
        pattern = StackPattern(
            pattern_name="pytest-fixtures",
            description="Use pytest fixtures",
            rationale="pytest is configured",
            applicable_technologies=("pytest",),
        )
        validation = TechStackValidation(
            overall_compliance=False,
            violations=(violation,),
            suggested_patterns=(pattern,),
            summary="1 violation found",
        )
        result = validation.to_dict()
        assert result["overall_compliance"] is False
        assert result["summary"] == "1 violation found"
        assert len(result["violations"]) == 1
        assert len(result["suggested_patterns"]) == 1
        # Check nested serialization
        assert result["violations"][0]["technology"] == "SQLite"
        assert result["suggested_patterns"][0]["pattern_name"] == "pytest-fixtures"

    def test_tech_stack_validation_to_dict_empty_collections(self) -> None:
        """Test to_dict with empty collections."""
        validation = TechStackValidation(
            overall_compliance=True,
            violations=(),
            suggested_patterns=(),
            summary="All good",
        )
        result = validation.to_dict()
        assert result["violations"] == []
        assert result["suggested_patterns"] == []

    def test_tech_stack_validation_overall_compliance(self) -> None:
        """Test that overall_compliance indicates overall compliance (AC7)."""
        # When there are critical violations, overall_compliance should be False
        critical_violation = ConstraintViolation(
            technology="SQLite",
            expected_version=None,
            actual_version="3.x",
            severity="critical",
            suggested_alternative="Use ChromaDB",
        )
        validation = TechStackValidation(
            overall_compliance=False,
            violations=(critical_violation,),
            suggested_patterns=(),
            summary="Critical violation found",
        )
        assert validation.overall_compliance is False

    def test_multiple_violations(self) -> None:
        """Test validation with multiple violations."""
        violations = (
            ConstraintViolation(
                technology="SQLite",
                expected_version=None,
                actual_version="3.x",
                severity="critical",
                suggested_alternative="Use ChromaDB",
            ),
            ConstraintViolation(
                technology="Python",
                expected_version="3.10",
                actual_version="3.8",
                severity="high",
                suggested_alternative="Upgrade Python",
            ),
        )
        validation = TechStackValidation(
            overall_compliance=False,
            violations=violations,
            suggested_patterns=(),
            summary="2 violations found",
        )
        assert len(validation.violations) == 2
        result = validation.to_dict()
        assert len(result["violations"]) == 2

    def test_multiple_patterns(self) -> None:
        """Test validation with multiple suggested patterns."""
        patterns = (
            StackPattern(
                pattern_name="pytest-fixtures",
                description="Use pytest fixtures",
                rationale="pytest is configured",
                applicable_technologies=("pytest",),
            ),
            StackPattern(
                pattern_name="uv-dependency-management",
                description="Use uv for dependencies",
                rationale="uv is configured",
                applicable_technologies=("uv",),
            ),
        )
        validation = TechStackValidation(
            overall_compliance=True,
            violations=(),
            suggested_patterns=patterns,
            summary="All compliant, 2 patterns suggested",
        )
        assert len(validation.suggested_patterns) == 2
        result = validation.to_dict()
        assert len(result["suggested_patterns"]) == 2
