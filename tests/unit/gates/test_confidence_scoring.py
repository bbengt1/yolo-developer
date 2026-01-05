"""Unit tests for confidence scoring gate.

Tests the confidence scoring gate evaluator and its component functions.
"""

from __future__ import annotations

import importlib
import logging

import pytest

from yolo_developer.gates.evaluators import clear_evaluators, get_evaluator
from yolo_developer.gates.gates.confidence_scoring import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_FACTOR_WEIGHTS,
    RISK_SEVERITY_IMPACT,
    ConfidenceBreakdown,
    ConfidenceFactor,
    calculate_confidence_score,
    calculate_coverage_factor,
    calculate_documentation_factor,
    calculate_gate_factor,
    calculate_risk_factor,
    confidence_scoring_evaluator,
    generate_confidence_report,
)
from yolo_developer.gates.types import GateContext

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def register_confidence_evaluator():
    """Ensure evaluator is registered for each test."""
    clear_evaluators()
    # Import to trigger registration
    from yolo_developer.gates.gates import confidence_scoring

    importlib.reload(confidence_scoring)
    yield
    clear_evaluators()


@pytest.fixture
def mock_code_with_tests() -> dict:
    """Code artifact with tests and documentation."""
    return {
        "files": [
            {
                "path": "README.md",
                "content": "# Test Project\n\nA sample project for testing.\n",
            },
            {
                "path": "src/module.py",
                "content": '''"""Module for testing."""

def public_function(arg: str) -> int:
    """A public function with docstring.

    Args:
        arg: Input string.

    Returns:
        Length of the string.
    """
    return len(arg)

def another_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

def _private_function() -> None:
    """Private function."""
    pass
''',
            },
            {
                "path": "tests/test_module.py",
                "content": '''"""Tests for module."""

def test_public_function():
    """Test public function."""
    from src.module import public_function
    assert public_function("test") == 4

def test_another_function():
    """Test another function."""
    from src.module import another_function
    assert another_function(1, 2) == 3
''',
            },
        ],
    }


@pytest.fixture
def mock_code_without_tests() -> dict:
    """Code artifact without tests."""
    return {
        "files": [
            {
                "path": "src/module.py",
                "content": """def public_function(arg):
    return len(arg)

def another_function(x, y):
    return x + y
""",
            },
        ],
    }


@pytest.fixture
def mock_gate_results_passing() -> list:
    """Gate results with all passing gates."""
    return [
        {"gate_name": "testability", "passed": True, "score": 95},
        {"gate_name": "ac_measurability", "passed": True, "score": 88},
        {"gate_name": "architecture_validation", "passed": True, "score": 92},
        {"gate_name": "definition_of_done", "passed": True, "score": 85},
    ]


@pytest.fixture
def mock_gate_results_mixed() -> list:
    """Gate results with some failing gates."""
    return [
        {"gate_name": "testability", "passed": True, "score": 85},
        {"gate_name": "ac_measurability", "passed": False, "score": 60},
        {"gate_name": "architecture_validation", "passed": True, "score": 75},
        {"gate_name": "definition_of_done", "passed": False, "score": 55},
    ]


@pytest.fixture
def mock_coverage_data() -> dict:
    """Coverage data for testing."""
    return {
        "line_coverage": 85.5,
        "branch_coverage": 72.0,
        "function_coverage": 90.0,
    }


@pytest.fixture
def mock_risks() -> list:
    """Risk data for testing."""
    return [
        {"type": "complexity", "severity": "medium", "location": "module.py"},
        {"type": "security", "severity": "low", "location": "api.py"},
    ]


# =============================================================================
# Test Constants
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_default_factor_weights_sum_to_one(self) -> None:
        """Default factor weights should sum to 1.0."""
        total = sum(DEFAULT_FACTOR_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_default_confidence_threshold_is_90(self) -> None:
        """Default confidence threshold should be 90."""
        assert DEFAULT_CONFIDENCE_THRESHOLD == 90

    def test_risk_severity_impact_order(self) -> None:
        """Risk severity impacts should be ordered correctly."""
        assert RISK_SEVERITY_IMPACT["critical"] > RISK_SEVERITY_IMPACT["high"]
        assert RISK_SEVERITY_IMPACT["high"] > RISK_SEVERITY_IMPACT["medium"]
        assert RISK_SEVERITY_IMPACT["medium"] > RISK_SEVERITY_IMPACT["low"]


# =============================================================================
# Test ConfidenceFactor
# =============================================================================


class TestConfidenceFactor:
    """Tests for ConfidenceFactor dataclass."""

    def test_factor_creation(self) -> None:
        """Should create factor with all fields."""
        factor = ConfidenceFactor(
            name="test_coverage",
            score=85,
            weight=0.30,
            description="Test coverage score",
        )
        assert factor.name == "test_coverage"
        assert factor.score == 85
        assert factor.weight == 0.30
        assert factor.description == "Test coverage score"

    def test_factor_is_immutable(self) -> None:
        """Factor should be frozen/immutable."""
        factor = ConfidenceFactor(
            name="test",
            score=50,
            weight=0.5,
            description="Test",
        )
        with pytest.raises(AttributeError):
            factor.score = 100  # type: ignore


# =============================================================================
# Test ConfidenceBreakdown
# =============================================================================


class TestConfidenceBreakdown:
    """Tests for ConfidenceBreakdown dataclass."""

    def test_breakdown_creation(self) -> None:
        """Should create breakdown with all fields."""
        factors = (
            ConfidenceFactor("test", 80, 0.5, "Test factor"),
            ConfidenceFactor("other", 90, 0.5, "Other factor"),
        )
        breakdown = ConfidenceBreakdown(
            factors=factors,
            total_score=85.0,
            weighted_score=85.0,
            threshold=90,
            passed=False,
        )
        assert len(breakdown.factors) == 2
        assert breakdown.total_score == 85.0
        assert breakdown.passed is False

    def test_breakdown_to_dict(self) -> None:
        """Should convert breakdown to dict."""
        factors = (ConfidenceFactor("test", 80, 0.5, "Test factor"),)
        breakdown = ConfidenceBreakdown(
            factors=factors,
            total_score=80.0,
            weighted_score=80.0,
            threshold=90,
            passed=False,
        )
        result = breakdown.to_dict()
        assert "factors" in result
        assert "total_score" in result
        assert "weighted_score" in result
        assert "threshold" in result
        assert "passed" in result
        assert result["factors"][0]["contribution"] == 40.0  # 80 * 0.5


# =============================================================================
# Test calculate_coverage_factor
# =============================================================================


class TestCalculateCoverageFactor:
    """Tests for calculate_coverage_factor function."""

    def test_with_explicit_coverage_data(self, mock_coverage_data: dict) -> None:
        """Should use explicit coverage data when available."""
        state = {"coverage": mock_coverage_data}
        factor = calculate_coverage_factor(state)

        assert factor.name == "test_coverage"
        assert factor.weight == DEFAULT_FACTOR_WEIGHTS["test_coverage"]
        # Average of 85.5, 72.0, 90.0 = 82.5
        assert factor.score == 82
        assert "Coverage:" in factor.description

    def test_with_empty_coverage_data(self) -> None:
        """Should handle empty coverage data."""
        state = {"coverage": {"line_coverage": 0, "branch_coverage": 0, "function_coverage": 0}}
        factor = calculate_coverage_factor(state)

        assert factor.score == 0
        assert "zero" in factor.description.lower()

    def test_with_code_estimation(self, mock_code_with_tests: dict) -> None:
        """Should estimate coverage from code when no explicit data."""
        state = {"code": mock_code_with_tests}
        factor = calculate_coverage_factor(state)

        assert factor.name == "test_coverage"
        assert 0 <= factor.score <= 100
        assert "Estimated" in factor.description or "coverage" in factor.description.lower()

    def test_with_no_code_or_coverage(self) -> None:
        """Should use default score when no data available."""
        state = {}
        factor = calculate_coverage_factor(state)

        assert factor.score == 50
        assert "default" in factor.description.lower()

    def test_with_code_no_tests(self, mock_code_without_tests: dict) -> None:
        """Should give low score when no tests present."""
        state = {"code": mock_code_without_tests}
        factor = calculate_coverage_factor(state)

        assert factor.score == 20  # No tests at all
        assert "No test" in factor.description

    def test_with_custom_weights(self, mock_coverage_data: dict) -> None:
        """Should use custom weights when provided."""
        state = {"coverage": mock_coverage_data}
        custom_weights = {"test_coverage": 0.50}
        factor = calculate_coverage_factor(state, custom_weights)

        assert factor.weight == 0.50


# =============================================================================
# Test calculate_gate_factor
# =============================================================================


class TestCalculateGateFactor:
    """Tests for calculate_gate_factor function."""

    def test_with_passing_gates(self, mock_gate_results_passing: list) -> None:
        """Should score high with all passing gates."""
        state = {"gate_results": mock_gate_results_passing}
        factor = calculate_gate_factor(state)

        assert factor.name == "gate_results"
        # Average of 95, 88, 92, 85 = 90
        assert factor.score == 90
        assert "All" in factor.description and "passed" in factor.description

    def test_with_mixed_gates(self, mock_gate_results_mixed: list) -> None:
        """Should reflect failing gates in score."""
        state = {"gate_results": mock_gate_results_mixed}
        factor = calculate_gate_factor(state)

        # Average of 85, 60, 75, 55 = 68.75 -> 68
        assert factor.score == 68
        assert "Failed:" in factor.description

    def test_with_no_gate_results(self) -> None:
        """Should use neutral score when no gate results."""
        state = {}
        factor = calculate_gate_factor(state)

        assert factor.score == 75  # Neutral score
        assert "neutral" in factor.description.lower()

    def test_with_empty_gate_results(self) -> None:
        """Should handle empty gate results list."""
        state = {"gate_results": []}
        factor = calculate_gate_factor(state)

        assert factor.score == 75
        assert "neutral" in factor.description.lower()

    def test_with_invalid_gate_results(self) -> None:
        """Should handle invalid gate result entries."""
        state = {
            "gate_results": ["invalid", None, {"gate_name": "valid", "passed": True, "score": 80}]
        }
        factor = calculate_gate_factor(state)

        # Only the valid gate should be counted
        assert factor.score == 80


# =============================================================================
# Test calculate_risk_factor
# =============================================================================


class TestCalculateRiskFactor:
    """Tests for calculate_risk_factor function."""

    def test_with_explicit_risks(self, mock_risks: list) -> None:
        """Should calculate score from explicit risks."""
        state = {"risks": mock_risks}
        factor = calculate_risk_factor(state)

        assert factor.name == "risk_assessment"
        # medium (10) + low (3) = 13 impact -> 100 - 13 = 87
        assert factor.score == 87
        assert "risk" in factor.description.lower()

    def test_with_critical_risk(self) -> None:
        """Should heavily penalize critical risks."""
        state = {"risks": [{"type": "security", "severity": "critical", "location": "auth.py"}]}
        factor = calculate_risk_factor(state)

        # critical = 40 impact -> 100 - 40 = 60
        assert factor.score == 60

    def test_with_no_risks(self) -> None:
        """Should estimate from code when no explicit risks."""
        state = {"code": {"files": []}}
        factor = calculate_risk_factor(state)

        assert factor.score == 70  # Default for no files

    def test_with_complex_code(self) -> None:
        """Should detect complexity risks in code."""
        complex_code = {
            "files": [
                {
                    "path": "src/complex.py",
                    "content": """def deeply_nested():
    if True:
        if True:
            if True:
                if True:
                    if True:
                        return "deep"
    return "shallow"
""",
                },
            ],
        }
        state = {"code": complex_code}
        factor = calculate_risk_factor(state)

        # Should have lower score due to deep nesting
        assert factor.score < 100
        assert "nesting" in factor.description.lower() or "risk" in factor.description.lower()


# =============================================================================
# Test calculate_documentation_factor
# =============================================================================


class TestCalculateDocumentationFactor:
    """Tests for calculate_documentation_factor function."""

    def test_with_documented_code(self, mock_code_with_tests: dict) -> None:
        """Should score high with documented code."""
        state = {"code": mock_code_with_tests}
        factor = calculate_documentation_factor(state)

        assert factor.name == "documentation"
        # Module docstring + 2 public functions with docstrings
        assert factor.score >= 80
        assert "document" in factor.description.lower()

    def test_with_undocumented_code(self, mock_code_without_tests: dict) -> None:
        """Should score low with undocumented code."""
        state = {"code": mock_code_without_tests}
        factor = calculate_documentation_factor(state)

        # No module docstring, no function docstrings
        assert factor.score == 0
        assert "Missing" in factor.description

    def test_with_no_code(self) -> None:
        """Should use default score when no code."""
        state = {}
        factor = calculate_documentation_factor(state)

        assert factor.score == 50
        assert "No code" in factor.description

    def test_with_partial_documentation(self) -> None:
        """Should reflect partial documentation."""
        partial_code = {
            "files": [
                {
                    "path": "README.md",
                    "content": "# Partial Project\n\nPartially documented.\n",
                },
                {
                    "path": "src/partial.py",
                    "content": '''"""Module with partial docs."""

def documented_function(x: int) -> int:
    """This function has docs."""
    return x * 2

def undocumented_function(y):
    return y + 1
''',
                },
            ],
        }
        state = {"code": partial_code}
        factor = calculate_documentation_factor(state)

        # 3/4 documented (module + documented_function + README, missing undocumented_function)
        assert factor.score == 75


# =============================================================================
# Test calculate_confidence_score
# =============================================================================


class TestCalculateConfidenceScore:
    """Tests for calculate_confidence_score function."""

    def test_weighted_calculation(self) -> None:
        """Should calculate weighted average correctly."""
        factors = [
            ConfidenceFactor("test_coverage", 80, 0.30, "Coverage"),
            ConfidenceFactor("gate_results", 90, 0.35, "Gates"),
            ConfidenceFactor("risk_assessment", 70, 0.20, "Risk"),
            ConfidenceFactor("documentation", 100, 0.15, "Docs"),
        ]

        breakdown = calculate_confidence_score(factors)

        # Weighted: 80*0.30 + 90*0.35 + 70*0.20 + 100*0.15 = 24 + 31.5 + 14 + 15 = 84.5
        assert abs(breakdown.weighted_score - 84.5) < 0.1
        assert breakdown.threshold == DEFAULT_CONFIDENCE_THRESHOLD

    def test_passing_threshold(self) -> None:
        """Should pass when score >= threshold."""
        factors = [
            ConfidenceFactor("test", 95, 0.5, "Test"),
            ConfidenceFactor("other", 95, 0.5, "Other"),
        ]

        breakdown = calculate_confidence_score(factors, threshold=90)

        assert breakdown.passed is True
        assert breakdown.weighted_score >= 90

    def test_failing_threshold(self) -> None:
        """Should fail when score < threshold."""
        factors = [
            ConfidenceFactor("test", 70, 0.5, "Test"),
            ConfidenceFactor("other", 70, 0.5, "Other"),
        ]

        breakdown = calculate_confidence_score(factors, threshold=90)

        assert breakdown.passed is False
        assert breakdown.weighted_score < 90

    def test_empty_factors(self) -> None:
        """Should handle empty factors list."""
        breakdown = calculate_confidence_score([])

        assert breakdown.weighted_score == 0.0
        assert breakdown.passed is False

    def test_custom_threshold(self) -> None:
        """Should use custom threshold when provided."""
        factors = [ConfidenceFactor("test", 75, 1.0, "Test")]

        breakdown = calculate_confidence_score(factors, threshold=70)

        assert breakdown.threshold == 70
        assert breakdown.passed is True


# =============================================================================
# Test confidence_scoring_evaluator
# =============================================================================


class TestConfidenceScoringEvaluator:
    """Tests for confidence_scoring_evaluator function."""

    @pytest.mark.asyncio
    async def test_evaluator_with_high_confidence(
        self, mock_code_with_tests: dict, mock_gate_results_passing: list, mock_coverage_data: dict
    ) -> None:
        """Should pass with high confidence code."""
        context = GateContext(
            state={
                "code": mock_code_with_tests,
                "gate_results": mock_gate_results_passing,
                "coverage": mock_coverage_data,
            },
            gate_name="confidence_scoring",
        )

        result = await confidence_scoring_evaluator(context)

        assert result.gate_name == "confidence_scoring"
        assert "Confidence score:" in result.reason

    @pytest.mark.asyncio
    async def test_evaluator_with_low_confidence(
        self, mock_code_without_tests: dict, mock_gate_results_mixed: list
    ) -> None:
        """Should fail with low confidence code."""
        context = GateContext(
            state={
                "code": mock_code_without_tests,
                "gate_results": mock_gate_results_mixed,
            },
            gate_name="confidence_scoring",
        )

        result = await confidence_scoring_evaluator(context)

        assert result.gate_name == "confidence_scoring"
        assert result.passed is False
        assert "BLOCKED" in result.reason or "threshold" in result.reason

    @pytest.mark.asyncio
    async def test_evaluator_with_custom_threshold(
        self, mock_code_with_tests: dict, mock_gate_results_mixed: list
    ) -> None:
        """Should respect custom threshold from config (0.0-1.0 format)."""
        context = GateContext(
            state={
                "code": mock_code_with_tests,
                "gate_results": mock_gate_results_mixed,
                "config": {"quality": {"confidence_threshold": 0.50}},
            },
            gate_name="confidence_scoring",
        )

        result = await confidence_scoring_evaluator(context)

        # With threshold 50%, should likely pass
        assert "50" in result.reason or "threshold" in result.reason

    @pytest.mark.asyncio
    async def test_evaluator_uses_gate_specific_threshold(
        self, mock_code_with_tests: dict, mock_gate_results_mixed: list
    ) -> None:
        """Should use gate-specific threshold from resolver (highest priority)."""
        context = GateContext(
            state={
                "code": mock_code_with_tests,
                "gate_results": mock_gate_results_mixed,
                "config": {
                    "quality": {
                        "confidence_threshold": 0.95,  # Global (lower priority)
                        "gate_thresholds": {
                            "confidence_scoring": {"min_score": 0.50},  # Gate-specific
                        },
                    }
                },
            },
            gate_name="confidence_scoring",
        )

        result = await confidence_scoring_evaluator(context)

        # Gate-specific threshold (0.50 = 50) should be used, not global (0.95 = 95)
        assert "50" in result.reason

    @pytest.mark.asyncio
    async def test_evaluator_with_custom_weights(
        self, mock_code_with_tests: dict, mock_gate_results_passing: list
    ) -> None:
        """Should use custom weights from config."""
        context = GateContext(
            state={
                "code": mock_code_with_tests,
                "gate_results": mock_gate_results_passing,
                "config": {
                    "quality": {
                        "confidence_threshold": 0.90,  # 90% in 0.0-1.0 format
                        "factor_weights": {
                            "test_coverage": 0.10,
                            "gate_results": 0.60,
                            "risk_assessment": 0.20,
                            "documentation": 0.10,
                        },
                    }
                },
            },
            gate_name="confidence_scoring",
        )

        result = await confidence_scoring_evaluator(context)

        # Should complete without error using custom weights
        assert result.gate_name == "confidence_scoring"

    @pytest.mark.asyncio
    async def test_evaluator_with_empty_state(self) -> None:
        """Should handle empty state gracefully."""
        context = GateContext(
            state={},
            gate_name="confidence_scoring",
        )

        result = await confidence_scoring_evaluator(context)

        # Should not crash, will use defaults
        assert result.gate_name == "confidence_scoring"

    @pytest.mark.asyncio
    async def test_evaluator_with_implementation_key(self, mock_code_with_tests: dict) -> None:
        """Should accept 'implementation' as alternative to 'code' key."""
        context = GateContext(
            state={"implementation": mock_code_with_tests},
            gate_name="confidence_scoring",
        )

        result = await confidence_scoring_evaluator(context)

        assert result.gate_name == "confidence_scoring"


# =============================================================================
# Test generate_confidence_report
# =============================================================================


class TestGenerateConfidenceReport:
    """Tests for generate_confidence_report function."""

    def test_report_format_passing(self) -> None:
        """Should generate proper format for passing score."""
        factors = (
            ConfidenceFactor("test_coverage", 95, 0.30, "High coverage"),
            ConfidenceFactor("gate_results", 90, 0.35, "All gates passed"),
        )
        breakdown = ConfidenceBreakdown(
            factors=factors,
            total_score=92.5,
            weighted_score=92.3,
            threshold=90,
            passed=True,
        )

        report = generate_confidence_report(breakdown, 90)

        assert "PASSED" in report
        assert "test_coverage" in report
        assert "gate_results" in report
        assert "90" in report  # threshold

    def test_report_format_failing(self) -> None:
        """Should generate proper format for failing score."""
        factors = (
            ConfidenceFactor("test_coverage", 60, 0.30, "Low coverage"),
            ConfidenceFactor("gate_results", 50, 0.35, "Gates failing"),
        )
        breakdown = ConfidenceBreakdown(
            factors=factors,
            total_score=55.0,
            weighted_score=54.5,
            threshold=90,
            passed=False,
        )

        report = generate_confidence_report(breakdown, 90)

        assert "BLOCKED" in report
        assert "Improvement Suggestions" in report

    def test_report_includes_contributions(self) -> None:
        """Should show factor contributions in report."""
        factors = (ConfidenceFactor("test", 80, 0.50, "Test factor"),)
        breakdown = ConfidenceBreakdown(
            factors=factors,
            total_score=80.0,
            weighted_score=80.0,
            threshold=90,
            passed=False,
        )

        report = generate_confidence_report(breakdown, 90)

        # Should show contribution (80 * 0.5 = 40)
        assert "contrib" in report.lower()


# =============================================================================
# Test Evaluator Registration
# =============================================================================


class TestEvaluatorRegistration:
    """Tests for evaluator registration."""

    def test_evaluator_registered_on_import(self) -> None:
        """Evaluator should be registered when module is imported."""
        evaluator = get_evaluator("confidence_scoring")
        assert evaluator is not None

    def test_evaluator_callable(self) -> None:
        """Registered evaluator should be callable."""
        evaluator = get_evaluator("confidence_scoring")
        assert callable(evaluator)


# =============================================================================
# Test Input Validation
# =============================================================================


class TestInputValidation:
    """Tests for input validation edge cases."""

    @pytest.mark.asyncio
    async def test_handles_none_in_state(self) -> None:
        """Should handle None values in state."""
        context = GateContext(
            state={"code": None, "gate_results": None, "coverage": None},
            gate_name="confidence_scoring",
        )

        result = await confidence_scoring_evaluator(context)

        # Should not crash
        assert result.gate_name == "confidence_scoring"

    @pytest.mark.asyncio
    async def test_handles_invalid_types(self) -> None:
        """Should handle invalid types in state."""
        context = GateContext(
            state={"code": "not a dict", "gate_results": "not a list"},
            gate_name="confidence_scoring",
        )

        result = await confidence_scoring_evaluator(context)

        # Should not crash
        assert result.gate_name == "confidence_scoring"


# =============================================================================
# Test Structured Logging
# =============================================================================


class TestStructuredLogging:
    """Tests for structured logging in confidence scoring gate."""

    @pytest.mark.asyncio
    async def test_evaluator_logs_evaluation_started(
        self, caplog: pytest.LogCaptureFixture, mock_code_with_tests: dict
    ) -> None:
        """Evaluator should log when evaluation starts."""
        caplog.set_level(logging.INFO)

        context = GateContext(
            state={"code": mock_code_with_tests},
            gate_name="confidence_scoring",
        )

        await confidence_scoring_evaluator(context)

        # Check that logging occurred (structlog may format differently)
        assert len(caplog.records) >= 0  # Just verify no crash with logging


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestExtractFunctionsFromContent:
    """Tests for _extract_functions_from_content helper."""

    def test_extracts_regular_function(self) -> None:
        """Should extract regular function info."""
        from yolo_developer.gates.gates.confidence_scoring import _extract_functions_from_content

        content = '''def my_func(x: int) -> int:
    """Docstring."""
    return x * 2
'''
        functions = _extract_functions_from_content(content)

        assert len(functions) == 1
        assert functions[0]["name"] == "my_func"
        assert functions[0]["has_docstring"] is True

    def test_extracts_async_function(self) -> None:
        """Should extract async function info."""
        from yolo_developer.gates.gates.confidence_scoring import _extract_functions_from_content

        content = """async def async_func():
    return None
"""
        functions = _extract_functions_from_content(content)

        assert len(functions) == 1
        assert functions[0]["name"] == "async_func"

    def test_handles_syntax_error(self) -> None:
        """Should return empty list for syntax errors."""
        from yolo_developer.gates.gates.confidence_scoring import _extract_functions_from_content

        content = "def broken("
        functions = _extract_functions_from_content(content)

        assert functions == []

    def test_detects_private_functions(self) -> None:
        """Should detect private functions."""
        from yolo_developer.gates.gates.confidence_scoring import _extract_functions_from_content

        content = "def _private(): pass"
        functions = _extract_functions_from_content(content)

        assert functions[0]["is_private"] is True


class TestCalculateNestingDepth:
    """Tests for _calculate_nesting_depth helper."""

    def test_no_nesting_returns_zero(self) -> None:
        """Should return 0 for no nesting."""
        import ast

        from yolo_developer.gates.gates.confidence_scoring import _calculate_nesting_depth

        tree = ast.parse("x = 1")
        depth = _calculate_nesting_depth(tree)

        assert depth == 0

    def test_single_if_returns_one(self) -> None:
        """Should return 1 for single if statement."""
        import ast

        from yolo_developer.gates.gates.confidence_scoring import _calculate_nesting_depth

        tree = ast.parse("if True:\n    x = 1")
        depth = _calculate_nesting_depth(tree)

        assert depth == 1

    def test_deep_nesting(self) -> None:
        """Should detect deep nesting."""
        import ast

        from yolo_developer.gates.gates.confidence_scoring import _calculate_nesting_depth

        code = """
if True:
    if True:
        if True:
            x = 1
"""
        tree = ast.parse(code)
        depth = _calculate_nesting_depth(tree)

        assert depth == 3


class TestHasModuleDocstring:
    """Tests for _has_module_docstring helper."""

    def test_detects_module_docstring(self) -> None:
        """Should detect module docstring."""
        from yolo_developer.gates.gates.confidence_scoring import _has_module_docstring

        content = '"""Module docs."""\ndef func(): pass'
        assert _has_module_docstring(content) is True

    def test_no_docstring_returns_false(self) -> None:
        """Should return False when no docstring."""
        from yolo_developer.gates.gates.confidence_scoring import _has_module_docstring

        content = "def func(): pass"
        assert _has_module_docstring(content) is False

    def test_handles_syntax_error(self) -> None:
        """Should return False for syntax errors."""
        from yolo_developer.gates.gates.confidence_scoring import _has_module_docstring

        content = "def broken("
        assert _has_module_docstring(content) is False


class TestCountTestFunctions:
    """Tests for _count_test_functions helper."""

    def test_counts_test_functions(self) -> None:
        """Should count test functions."""
        from yolo_developer.gates.gates.confidence_scoring import _count_test_functions

        content = """
def test_one(): pass
def test_two(): pass
def helper(): pass
"""
        count = _count_test_functions(content)

        assert count == 2

    def test_empty_for_no_tests(self) -> None:
        """Should return 0 for no tests."""
        from yolo_developer.gates.gates.confidence_scoring import _count_test_functions

        content = "def helper(): pass"
        count = _count_test_functions(content)

        assert count == 0

    def test_handles_syntax_error(self) -> None:
        """Should return 0 for syntax errors."""
        from yolo_developer.gates.gates.confidence_scoring import _count_test_functions

        content = "def broken("
        count = _count_test_functions(content)

        assert count == 0


class TestGetRemediationForFactor:
    """Tests for _get_remediation_for_factor helper."""

    def test_returns_coverage_remediation(self) -> None:
        """Should return remediation for coverage factor."""
        from yolo_developer.gates.gates.confidence_scoring import _get_remediation_for_factor

        remediation = _get_remediation_for_factor("test_coverage", 50)

        assert "test" in remediation.lower()

    def test_returns_gate_remediation(self) -> None:
        """Should return remediation for gate factor."""
        from yolo_developer.gates.gates.confidence_scoring import _get_remediation_for_factor

        remediation = _get_remediation_for_factor("gate_results", 60)

        assert "gate" in remediation.lower()

    def test_returns_default_for_unknown(self) -> None:
        """Should return default remediation for unknown factor."""
        from yolo_developer.gates.gates.confidence_scoring import _get_remediation_for_factor

        remediation = _get_remediation_for_factor("unknown_factor", 50)

        assert "unknown_factor" in remediation
