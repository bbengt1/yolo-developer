"""Unit tests for pattern matcher module (Story 7.8).

Tests cover:
- Pattern matching type definitions (Task 1)
- Pattern retrieval from memory (Task 2)
- Naming convention checking (Task 3)
- Architectural style checking (Task 4)
- Deviation detection (Task 5)
- Pass/fail decision logic (Task 6)
- LLM integration (Task 7)
- Main pattern matching function (Task 8)
"""

from __future__ import annotations

import pytest

from yolo_developer.agents.architect.types import (
    DesignDecision,
    PatternCheckSeverity,
    PatternDeviation,
    PatternMatchingResult,
    PatternViolation,
)
from yolo_developer.memory.patterns import CodePattern, PatternType

# =============================================================================
# Task 1: Type Definition Tests (AC: 1, 7)
# =============================================================================


class TestPatternViolation:
    """Tests for PatternViolation dataclass."""

    def test_creation_with_all_fields(self) -> None:
        """Test PatternViolation creation with all required fields."""
        violation = PatternViolation(
            pattern_type="naming_function",
            expected="snake_case",
            actual="camelCase",
            file_context="src/myModule.py",
            severity="high",
            justification=None,
        )

        assert violation.pattern_type == "naming_function"
        assert violation.expected == "snake_case"
        assert violation.actual == "camelCase"
        assert violation.file_context == "src/myModule.py"
        assert violation.severity == "high"
        assert violation.justification is None

    def test_creation_with_justification(self) -> None:
        """Test PatternViolation with justification for deviation."""
        violation = PatternViolation(
            pattern_type="design_pattern",
            expected="Factory pattern",
            actual="Builder pattern",
            file_context="src/services/auth.py",
            severity="medium",
            justification="Builder provides better configuration flexibility",
        )

        assert violation.justification == "Builder provides better configuration flexibility"

    def test_to_dict(self) -> None:
        """Test PatternViolation serialization to dictionary."""
        violation = PatternViolation(
            pattern_type="naming_class",
            expected="PascalCase",
            actual="camelCase",
            file_context="src/handlers.py",
            severity="high",
            justification=None,
        )

        result = violation.to_dict()

        assert result == {
            "pattern_type": "naming_class",
            "expected": "PascalCase",
            "actual": "camelCase",
            "file_context": "src/handlers.py",
            "severity": "high",
            "justification": None,
        }

    def test_immutability(self) -> None:
        """Test that PatternViolation is immutable (frozen)."""
        violation = PatternViolation(
            pattern_type="naming_function",
            expected="snake_case",
            actual="camelCase",
            file_context="src/test.py",
            severity="high",
            justification=None,
        )

        with pytest.raises(AttributeError):
            violation.pattern_type = "naming_class"  # type: ignore[misc]


class TestPatternDeviation:
    """Tests for PatternDeviation dataclass."""

    def test_creation_justified_deviation(self) -> None:
        """Test PatternDeviation with justified deviation."""
        deviation = PatternDeviation(
            pattern_type="design_pattern",
            standard_pattern="Factory pattern",
            proposed_pattern="Builder pattern",
            justification="Builder provides better configuration flexibility for complex objects",
            is_justified=True,
            severity="medium",
        )

        assert deviation.pattern_type == "design_pattern"
        assert deviation.standard_pattern == "Factory pattern"
        assert deviation.proposed_pattern == "Builder pattern"
        assert deviation.is_justified is True
        assert deviation.severity == "medium"

    def test_creation_unjustified_deviation(self) -> None:
        """Test PatternDeviation with unjustified deviation."""
        deviation = PatternDeviation(
            pattern_type="naming_function",
            standard_pattern="snake_case",
            proposed_pattern="camelCase",
            justification="",
            is_justified=False,
            severity="high",
        )

        assert deviation.is_justified is False
        assert deviation.justification == ""

    def test_to_dict(self) -> None:
        """Test PatternDeviation serialization to dictionary."""
        deviation = PatternDeviation(
            pattern_type="import_style",
            standard_pattern="absolute imports",
            proposed_pattern="relative imports",
            justification="Module is deeply nested, relative imports are clearer",
            is_justified=True,
            severity="low",
        )

        result = deviation.to_dict()

        assert result == {
            "pattern_type": "import_style",
            "standard_pattern": "absolute imports",
            "proposed_pattern": "relative imports",
            "justification": "Module is deeply nested, relative imports are clearer",
            "is_justified": True,
            "severity": "low",
        }

    def test_immutability(self) -> None:
        """Test that PatternDeviation is immutable (frozen)."""
        deviation = PatternDeviation(
            pattern_type="design_pattern",
            standard_pattern="Factory",
            proposed_pattern="Builder",
            justification="Needed for complex configuration",
            is_justified=True,
            severity="medium",
        )

        with pytest.raises(AttributeError):
            deviation.is_justified = False  # type: ignore[misc]


class TestPatternMatchingResult:
    """Tests for PatternMatchingResult dataclass."""

    def test_creation_passing_result(self) -> None:
        """Test PatternMatchingResult for a passing pattern check."""
        result = PatternMatchingResult(
            overall_pass=True,
            confidence=0.95,
            patterns_checked=("naming_function", "naming_class", "structure_directory"),
            violations=(),
            deviations=(),
            recommendations=(),
            summary="Design conforms to all established patterns",
        )

        assert result.overall_pass is True
        assert result.confidence == 0.95
        assert len(result.patterns_checked) == 3
        assert len(result.violations) == 0
        assert len(result.deviations) == 0

    def test_creation_failing_result(self) -> None:
        """Test PatternMatchingResult for a failing pattern check."""
        violation = PatternViolation(
            pattern_type="naming_function",
            expected="snake_case",
            actual="camelCase",
            file_context="src/service.py",
            severity="high",
            justification=None,
        )
        deviation = PatternDeviation(
            pattern_type="design_pattern",
            standard_pattern="Factory",
            proposed_pattern="Builder",
            justification="",
            is_justified=False,
            severity="critical",
        )

        result = PatternMatchingResult(
            overall_pass=False,
            confidence=0.45,
            patterns_checked=("naming_function", "design_pattern"),
            violations=(violation,),
            deviations=(deviation,),
            recommendations=("Rename getUserData to get_user_data",),
            summary="Design has 1 violation and 1 unjustified deviation",
        )

        assert result.overall_pass is False
        assert result.confidence == 0.45
        assert len(result.violations) == 1
        assert len(result.deviations) == 1
        assert len(result.recommendations) == 1

    def test_to_dict(self) -> None:
        """Test PatternMatchingResult serialization to dictionary."""
        result = PatternMatchingResult(
            overall_pass=True,
            confidence=0.85,
            patterns_checked=("naming_function",),
            violations=(),
            deviations=(),
            recommendations=("Consider adding type hints",),
            summary="Design mostly conforms",
        )

        result_dict = result.to_dict()

        assert result_dict["overall_pass"] is True
        assert result_dict["confidence"] == 0.85
        assert result_dict["patterns_checked"] == ["naming_function"]
        assert result_dict["violations"] == []
        assert result_dict["deviations"] == []
        assert result_dict["recommendations"] == ["Consider adding type hints"]
        assert result_dict["summary"] == "Design mostly conforms"

    def test_to_dict_with_nested_objects(self) -> None:
        """Test PatternMatchingResult serialization with nested violations/deviations."""
        violation = PatternViolation(
            pattern_type="naming_class",
            expected="PascalCase",
            actual="snake_case",
            file_context="src/models.py",
            severity="high",
            justification=None,
        )
        deviation = PatternDeviation(
            pattern_type="import_style",
            standard_pattern="absolute",
            proposed_pattern="relative",
            justification="Deep nesting",
            is_justified=True,
            severity="low",
        )

        result = PatternMatchingResult(
            overall_pass=True,
            confidence=0.75,
            patterns_checked=("naming_class", "import_style"),
            violations=(violation,),
            deviations=(deviation,),
            recommendations=(),
            summary="Minor issues found",
        )

        result_dict = result.to_dict()

        assert len(result_dict["violations"]) == 1
        assert result_dict["violations"][0]["pattern_type"] == "naming_class"
        assert len(result_dict["deviations"]) == 1
        assert result_dict["deviations"][0]["is_justified"] is True

    def test_immutability(self) -> None:
        """Test that PatternMatchingResult is immutable (frozen)."""
        result = PatternMatchingResult(
            overall_pass=True,
            confidence=0.9,
            patterns_checked=(),
            violations=(),
            deviations=(),
            recommendations=(),
            summary="All good",
        )

        with pytest.raises(AttributeError):
            result.overall_pass = False  # type: ignore[misc]


class TestPatternCheckSeverity:
    """Tests for PatternCheckSeverity type."""

    def test_valid_severity_values(self) -> None:
        """Test that all valid severity values work."""
        severities: list[PatternCheckSeverity] = ["critical", "high", "medium", "low"]

        for severity in severities:
            violation = PatternViolation(
                pattern_type="naming_function",
                expected="snake_case",
                actual="camelCase",
                file_context="test.py",
                severity=severity,
                justification=None,
            )
            assert violation.severity == severity


# =============================================================================
# Task 2: Pattern Retrieval Tests (AC: 1, 2, 3)
# =============================================================================


class TestGetLearnedPatterns:
    """Tests for _get_learned_patterns function."""

    @pytest.mark.asyncio
    async def test_retrieval_with_none_store(self) -> None:
        """Test pattern retrieval returns empty list when store is None."""
        from yolo_developer.agents.architect.pattern_matcher import _get_learned_patterns

        patterns = await _get_learned_patterns(None)

        assert patterns == []

    @pytest.mark.asyncio
    async def test_retrieval_with_mock_store(self) -> None:
        """Test pattern retrieval with mock ChromaPatternStore."""
        from unittest.mock import AsyncMock, MagicMock

        from yolo_developer.agents.architect.pattern_matcher import _get_learned_patterns

        # Create mock store
        mock_store = MagicMock()
        mock_store.get_patterns_by_type = AsyncMock(
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

        patterns = await _get_learned_patterns(mock_store)

        # Should have called get_patterns_by_type for each pattern type
        assert mock_store.get_patterns_by_type.call_count >= 1
        assert len(patterns) >= 1
        assert patterns[0].pattern_type == PatternType.NAMING_FUNCTION

    @pytest.mark.asyncio
    async def test_retrieval_of_all_pattern_types(self) -> None:
        """Test that retrieval queries all pattern types."""
        from unittest.mock import AsyncMock, MagicMock

        from yolo_developer.agents.architect.pattern_matcher import _get_learned_patterns

        # Create mock store that returns empty for all types
        mock_store = MagicMock()
        mock_store.get_patterns_by_type = AsyncMock(return_value=[])

        await _get_learned_patterns(mock_store)

        # Should have called get_patterns_by_type for each PatternType
        assert mock_store.get_patterns_by_type.call_count == 8  # All 8 pattern types

    @pytest.mark.asyncio
    async def test_retrieval_handles_store_errors(self) -> None:
        """Test that retrieval handles errors gracefully."""
        from unittest.mock import AsyncMock, MagicMock

        from yolo_developer.agents.architect.pattern_matcher import _get_learned_patterns

        # Create mock store that raises exception
        mock_store = MagicMock()
        mock_store.get_patterns_by_type = AsyncMock(side_effect=Exception("Store error"))

        # Should not raise, just return empty list
        patterns = await _get_learned_patterns(mock_store)

        assert patterns == []


# =============================================================================
# Task 12: Naming Convention Checking Tests (AC: 2)
# =============================================================================


class TestNamingConventionChecking:
    """Tests for _check_naming_conventions function."""

    def test_detect_snake_case(self) -> None:
        """Test snake_case detection."""
        from yolo_developer.agents.architect.pattern_matcher import _detect_naming_style

        assert _detect_naming_style("get_user_data") == "snake_case"
        assert _detect_naming_style("process_order") == "snake_case"

    def test_detect_camel_case(self) -> None:
        """Test camelCase detection."""
        from yolo_developer.agents.architect.pattern_matcher import _detect_naming_style

        assert _detect_naming_style("getUserData") == "camelCase"
        assert _detect_naming_style("processOrder") == "camelCase"

    def test_detect_pascal_case(self) -> None:
        """Test PascalCase detection."""
        from yolo_developer.agents.architect.pattern_matcher import _detect_naming_style

        assert _detect_naming_style("UserManager") == "PascalCase"
        assert _detect_naming_style("OrderProcessor") == "PascalCase"

    def test_extract_identifiers_from_decision(self) -> None:
        """Test identifier extraction from design decisions."""
        from yolo_developer.agents.architect.pattern_matcher import (
            _extract_identifiers_from_decision,
        )

        decision = DesignDecision(
            id="design-001",
            story_id="story-001",
            decision_type="pattern",
            description="Use getUserData() function in UserManager class",
            rationale="Follows existing pattern",
            alternatives_considered=(),
        )

        identifiers = _extract_identifiers_from_decision(decision)

        # Should find function and class identifiers
        assert len(identifiers) >= 1

    def test_check_naming_conventions_no_patterns(self) -> None:
        """Test naming check with no patterns returns empty violations."""
        from yolo_developer.agents.architect.pattern_matcher import (
            _check_naming_conventions,
        )

        decisions = [
            DesignDecision(
                id="design-001",
                story_id="story-001",
                decision_type="pattern",
                description="Use get_user_data() function",
                rationale="Standard approach",
                alternatives_considered=(),
            )
        ]

        violations, confidence = _check_naming_conventions(decisions, [])

        assert violations == []
        assert confidence == 1.0


# =============================================================================
# Task 13: Architectural Style Checking Tests (AC: 3)
# =============================================================================


class TestArchitecturalStyleChecking:
    """Tests for _check_architectural_style function."""

    def test_check_style_no_patterns(self) -> None:
        """Test style check with no patterns returns empty violations."""
        from yolo_developer.agents.architect.pattern_matcher import (
            _check_architectural_style,
        )

        decisions = [
            DesignDecision(
                id="design-001",
                story_id="story-001",
                decision_type="pattern",
                description="Use Factory pattern",
                rationale="Standard approach",
                alternatives_considered=(),
            )
        ]

        violations = _check_architectural_style(decisions, [])

        assert violations == []


# =============================================================================
# Task 14: Deviation Detection Tests (AC: 4)
# =============================================================================


class TestDeviationDetection:
    """Tests for _detect_pattern_deviations function."""

    def test_detect_deviations_empty_violations(self) -> None:
        """Test deviation detection with no violations."""
        from yolo_developer.agents.architect.pattern_matcher import (
            _detect_pattern_deviations,
        )

        decisions = [
            DesignDecision(
                id="design-001",
                story_id="story-001",
                decision_type="pattern",
                description="Use Repository pattern",
                rationale="Standard approach",
                alternatives_considered=(),
            )
        ]

        deviations = _detect_pattern_deviations([], decisions)

        assert deviations == []

    def test_detect_deviations_with_justification(self) -> None:
        """Test deviation detection finds justification in rationale."""
        from yolo_developer.agents.architect.pattern_matcher import (
            _detect_pattern_deviations,
        )

        violation = PatternViolation(
            pattern_type="design_pattern",
            expected="Factory pattern",
            actual="builder",
            file_context="src/service.py",
            severity="medium",
            justification=None,
        )

        decisions = [
            DesignDecision(
                id="design-001",
                story_id="story-001",
                decision_type="pattern",
                description="Use Builder pattern",
                rationale="Intentionally using builder for better configuration",
                alternatives_considered=(),
            )
        ]

        deviations = _detect_pattern_deviations([violation], decisions)

        assert len(deviations) == 1
        assert deviations[0].is_justified is True


# =============================================================================
# Task 15: Pass/Fail Decision Tests (AC: 4, 7)
# =============================================================================


class TestPassFailDecision:
    """Tests for _make_pattern_decision function."""

    def test_pass_with_no_violations(self) -> None:
        """Test pass decision with no violations."""
        from yolo_developer.agents.architect.pattern_matcher import (
            _make_pattern_decision,
        )

        overall_pass, reasons = _make_pattern_decision([], [], 0.9)

        assert overall_pass is True
        assert reasons == []

    def test_fail_with_low_confidence(self) -> None:
        """Test fail decision with low confidence."""
        from yolo_developer.agents.architect.pattern_matcher import (
            _make_pattern_decision,
        )

        overall_pass, reasons = _make_pattern_decision([], [], 0.5)

        assert overall_pass is False
        assert any("Confidence" in r for r in reasons)

    def test_fail_with_critical_unjustified_deviation(self) -> None:
        """Test fail decision with critical unjustified deviation."""
        from yolo_developer.agents.architect.pattern_matcher import (
            _make_pattern_decision,
        )

        deviation = PatternDeviation(
            pattern_type="design_pattern",
            standard_pattern="Factory",
            proposed_pattern="Builder",
            justification="",
            is_justified=False,
            severity="critical",
        )

        overall_pass, reasons = _make_pattern_decision([], [deviation], 0.9)

        assert overall_pass is False
        assert any("critical unjustified" in r for r in reasons)


# =============================================================================
# Task 16: Main Pattern Matching Function Tests (AC: 1, 5, 7)
# =============================================================================


class TestRunPatternMatching:
    """Tests for run_pattern_matching function."""

    @pytest.mark.asyncio
    async def test_run_pattern_matching_empty_decisions(self) -> None:
        """Test pattern matching with empty decisions list."""
        from yolo_developer.agents.architect.pattern_matcher import run_pattern_matching

        result = await run_pattern_matching([])

        assert result.overall_pass is True
        assert result.confidence == 1.0
        assert result.summary == "No design decisions to analyze"

    @pytest.mark.asyncio
    async def test_run_pattern_matching_no_pattern_store(self) -> None:
        """Test pattern matching without pattern store."""
        from yolo_developer.agents.architect.pattern_matcher import run_pattern_matching

        decisions = [
            DesignDecision(
                id="design-001",
                story_id="story-001",
                decision_type="pattern",
                description="Use Repository pattern",
                rationale="Standard approach",
                alternatives_considered=(),
            )
        ]

        result = await run_pattern_matching(decisions, pattern_store=None)

        assert result.overall_pass is True
        assert result.confidence == 1.0
        assert "No learned patterns" in result.summary

    @pytest.mark.asyncio
    async def test_run_pattern_matching_returns_result(self) -> None:
        """Test pattern matching returns PatternMatchingResult."""
        from yolo_developer.agents.architect.pattern_matcher import run_pattern_matching

        decisions = [
            DesignDecision(
                id="design-001",
                story_id="story-001",
                decision_type="pattern",
                description="Use get_user_data() function",
                rationale="Follows snake_case convention",
                alternatives_considered=(),
            )
        ]

        result = await run_pattern_matching(decisions)

        assert isinstance(result, PatternMatchingResult)
        assert hasattr(result, "overall_pass")
        assert hasattr(result, "confidence")
        assert hasattr(result, "violations")
        assert hasattr(result, "deviations")


# =============================================================================
# Task 17: Integration Tests (AC: 5)
# =============================================================================


class TestPatternMatchingIntegration:
    """Integration tests for pattern matching."""

    def test_pattern_matching_result_serialization(self) -> None:
        """Test PatternMatchingResult serializes correctly."""
        violation = PatternViolation(
            pattern_type="naming_function",
            expected="snake_case",
            actual="camelCase",
            file_context="src/test.py",
            severity="high",
            justification=None,
        )
        deviation = PatternDeviation(
            pattern_type="design_pattern",
            standard_pattern="Factory",
            proposed_pattern="Builder",
            justification="Builder is better for complex config",
            is_justified=True,
            severity="medium",
        )

        result = PatternMatchingResult(
            overall_pass=True,
            confidence=0.85,
            patterns_checked=("naming_function", "design_pattern"),
            violations=(violation,),
            deviations=(deviation,),
            recommendations=("Fix naming convention",),
            summary="Analysis complete",
        )

        result_dict = result.to_dict()

        assert result_dict["overall_pass"] is True
        assert result_dict["confidence"] == 0.85
        assert len(result_dict["violations"]) == 1
        assert len(result_dict["deviations"]) == 1
        assert result_dict["deviations"][0]["is_justified"] is True

    def test_architect_output_includes_pattern_matching(self) -> None:
        """Test ArchitectOutput includes pattern_matching_results field."""
        from yolo_developer.agents.architect.types import ArchitectOutput

        output = ArchitectOutput(
            design_decisions=(),
            adrs=(),
            processing_notes="Test",
            pattern_matching_results={"story-001": {"overall_pass": True}},
        )

        result_dict = output.to_dict()

        assert "pattern_matching_results" in result_dict
        assert result_dict["pattern_matching_results"]["story-001"]["overall_pass"] is True

    def test_run_pattern_matching_importable_from_architect(self) -> None:
        """Test run_pattern_matching is importable from architect module."""
        from yolo_developer.agents.architect import run_pattern_matching

        assert callable(run_pattern_matching)


# =============================================================================
# Code Review Fixes - Additional Tests
# =============================================================================


class TestLLMFallbackBehavior:
    """Tests for LLM fallback to rule-based analysis (AC: 6)."""

    @pytest.mark.asyncio
    async def test_fallback_to_rule_based_when_llm_fails(self) -> None:
        """Test that run_pattern_matching falls back to rule-based when LLM fails."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from yolo_developer.agents.architect.pattern_matcher import run_pattern_matching

        # Create mock store with patterns (so LLM path is attempted)
        mock_store = MagicMock()
        mock_store.get_patterns_by_type = AsyncMock(
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

        decisions = [
            DesignDecision(
                id="design-001",
                story_id="story-001",
                decision_type="pattern",
                description="Use get_user_data() function",
                rationale="Follows convention",
                alternatives_considered=(),
            )
        ]

        # Mock LLM to raise exception
        with patch(
            "yolo_developer.agents.architect.pattern_matcher._analyze_patterns_with_llm",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.side_effect = Exception("LLM API Error")

            # Should NOT raise - should fall back to rule-based
            result = await run_pattern_matching(decisions, pattern_store=mock_store)

            # Verify we got a result (from rule-based fallback)
            assert isinstance(result, PatternMatchingResult)
            assert result.overall_pass is True  # Rule-based should pass with good naming

    @pytest.mark.asyncio
    async def test_fallback_when_llm_returns_none(self) -> None:
        """Test fallback when LLM returns None (parse failure)."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from yolo_developer.agents.architect.pattern_matcher import run_pattern_matching

        mock_store = MagicMock()
        mock_store.get_patterns_by_type = AsyncMock(
            return_value=[
                CodePattern(
                    pattern_type=PatternType.NAMING_FUNCTION,
                    name="function_naming",
                    value="snake_case",
                    confidence=0.95,
                    examples=(),
                ),
            ]
        )

        decisions = [
            DesignDecision(
                id="design-001",
                story_id="story-001",
                decision_type="pattern",
                description="Use Repository pattern",
                rationale="Standard approach",
                alternatives_considered=(),
            )
        ]

        with patch(
            "yolo_developer.agents.architect.pattern_matcher._analyze_patterns_with_llm",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.return_value = None  # Simulates parse failure

            result = await run_pattern_matching(decisions, pattern_store=mock_store)

            assert isinstance(result, PatternMatchingResult)


class TestNamingConventionWithPatterns:
    """Tests for naming convention checking with actual patterns (AC: 2)."""

    def test_detect_naming_violation_with_patterns(self) -> None:
        """Test naming violation detection when patterns exist."""
        from yolo_developer.agents.architect.pattern_matcher import (
            _check_naming_conventions,
        )

        # Create patterns that expect snake_case
        patterns = [
            CodePattern(
                pattern_type=PatternType.NAMING_FUNCTION,
                name="function_naming",
                value="snake_case",
                confidence=0.95,
                examples=("get_user", "process_order"),
            ),
        ]

        # Decision with camelCase function name (violation)
        decisions = [
            DesignDecision(
                id="design-001",
                story_id="story-001",
                decision_type="pattern",
                description="Use getUserData() to fetch user information",
                rationale="Standard approach",
                alternatives_considered=(),
            )
        ]

        violations, confidence = _check_naming_conventions(decisions, patterns)

        # Should detect the camelCase violation
        assert len(violations) >= 1
        assert any(v.actual == "camelCase" for v in violations)
        assert confidence < 1.0  # Should have lower confidence due to violation

    def test_no_violations_when_naming_matches(self) -> None:
        """Test no violations when naming matches expected pattern."""
        from yolo_developer.agents.architect.pattern_matcher import (
            _check_naming_conventions,
        )

        patterns = [
            CodePattern(
                pattern_type=PatternType.NAMING_FUNCTION,
                name="function_naming",
                value="snake_case",
                confidence=0.95,
                examples=("get_user",),
            ),
        ]

        # Decision with proper snake_case
        decisions = [
            DesignDecision(
                id="design-001",
                story_id="story-001",
                decision_type="pattern",
                description="Use get_user_data() to fetch user",
                rationale="Standard approach",
                alternatives_considered=(),
            )
        ]

        _violations, confidence = _check_naming_conventions(decisions, patterns)

        # Should have high confidence, few/no violations
        assert confidence >= 0.8


class TestArchitecturalStyleWithPatterns:
    """Tests for architectural style checking with actual patterns (AC: 3)."""

    def test_detect_import_style_violation(self) -> None:
        """Test import style violation detection."""
        from yolo_developer.agents.architect.pattern_matcher import (
            _check_architectural_style,
        )

        patterns = [
            CodePattern(
                pattern_type=PatternType.IMPORT_STYLE,
                name="import_style",
                value="absolute",
                confidence=0.9,
                examples=("from yolo_developer.module import X",),
            ),
        ]

        # Decision mentioning relative imports when absolute is expected
        decisions = [
            DesignDecision(
                id="design-001",
                story_id="story-001",
                decision_type="pattern",
                description="Use relative import from ..utils",
                rationale="Simpler",
                alternatives_considered=(),
            )
        ]

        violations = _check_architectural_style(decisions, patterns)

        assert len(violations) >= 1
        assert any(v.pattern_type == "import_style" for v in violations)

    def test_detect_design_pattern_inconsistency(self) -> None:
        """Test design pattern inconsistency detection."""
        from yolo_developer.agents.architect.pattern_matcher import (
            _check_architectural_style,
        )

        patterns = [
            CodePattern(
                pattern_type=PatternType.DESIGN_PATTERN,
                name="primary_pattern",
                value="Repository",
                confidence=0.85,
                examples=("UserRepository", "OrderRepository"),
            ),
        ]

        # Decision using Factory when Repository is the standard
        decisions = [
            DesignDecision(
                id="design-001",
                story_id="story-001",
                decision_type="pattern",
                description="Use Factory pattern to create users",
                rationale="Flexible object creation",
                alternatives_considered=(),
            )
        ]

        violations = _check_architectural_style(decisions, patterns)

        assert len(violations) >= 1
        assert any(v.pattern_type == "design_pattern" for v in violations)
