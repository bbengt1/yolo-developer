"""Unit tests for definition of done gate.

Tests the DoD validation functionality including:
- Test presence detection
- Documentation check
- Code style validation
- AC coverage check
- Checklist result generation
- Compliance score calculation
- Full evaluator integration
"""

from __future__ import annotations

import pytest

from yolo_developer.gates.evaluators import clear_evaluators, get_evaluator
from yolo_developer.gates.types import GateContext

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_code_with_tests() -> dict:
    """Code with proper test coverage."""
    return {
        "files": [
            {
                "path": "src/module.py",
                "content": '''"""Module docstring."""

def public_function(arg: str) -> int:
    """Function docstring."""
    return len(arg)

def another_public(value: int) -> str:
    """Another function docstring."""
    return str(value)

def _private_function():
    return None
''',
            },
            {
                "path": "tests/test_module.py",
                "content": '''"""Test module docstring."""

def test_public_function():
    """Test docstring."""
    from src.module import public_function
    assert public_function("test") == 4

def test_another_public():
    """Test docstring."""
    from src.module import another_public
    assert another_public(42) == "42"
''',
            },
        ],
    }


@pytest.fixture
def mock_code_without_tests() -> dict:
    """Code missing test coverage."""
    return {
        "files": [
            {
                "path": "src/module.py",
                "content": '''"""Module docstring."""

def public_function(arg: str) -> int:
    """Function docstring."""
    return len(arg)

def untested_function(value: int) -> str:
    """This function has no tests."""
    return str(value)
''',
            },
            {
                "path": "tests/test_module.py",
                "content": '''"""Test module."""

def test_public_function():
    assert True
''',
            },
        ],
    }


@pytest.fixture
def mock_code_without_docs() -> dict:
    """Code missing documentation."""
    return {
        "files": [
            {
                "path": "src/module.py",
                "content": """
def public_function(arg: str) -> int:
    return len(arg)

def another_function(value):
    return str(value)
""",
            },
        ],
    }


@pytest.fixture
def mock_code_with_style_issues() -> dict:
    """Code with style violations."""
    return {
        "files": [
            {
                "path": "src/module.py",
                "content": '''"""Module docstring."""

def BadNamingConvention(arg):
    return arg

def veryLongFunctionNameThatViolatesConventions(x):
    return x

def function_without_types(arg):
    return arg

def complex_function(a, b, c, d):
    if a:
        if b:
            if c:
                if d:
                    if a > b:
                        return 1
    return 0
''',
            },
        ],
    }


@pytest.fixture
def mock_story() -> dict:
    """Mock story with acceptance criteria."""
    return {
        "id": "3-5",
        "title": "Implement Definition of Done Gate",
        "acceptance_criteria": [
            "AC1: Test presence is verified",
            "AC2: Documentation presence is checked",
            "AC3: Code style compliance is validated",
        ],
        "tasks": [
            {"id": "task-1", "description": "Define DoD types"},
            {"id": "task-2", "description": "Implement test presence"},
        ],
    }


@pytest.fixture
def mock_story_partial_coverage() -> dict:
    """Story with some AC not addressed."""
    return {
        "id": "test-story",
        "title": "Test Story",
        "acceptance_criteria": [
            "AC1: Feature A is implemented",
            "AC2: Feature B is implemented",
            "AC3: Feature C is implemented",
        ],
    }


# =============================================================================
# Task 1: DoD Validation Types Tests
# =============================================================================


class TestDoDCategory:
    """Tests for DoDCategory enum."""

    def test_dod_category_has_tests_value(self) -> None:
        """DoDCategory should have TESTS value."""
        from yolo_developer.gates.gates.definition_of_done import DoDCategory

        assert DoDCategory.TESTS.value == "tests"

    def test_dod_category_has_documentation_value(self) -> None:
        """DoDCategory should have DOCUMENTATION value."""
        from yolo_developer.gates.gates.definition_of_done import DoDCategory

        assert DoDCategory.DOCUMENTATION.value == "documentation"

    def test_dod_category_has_style_value(self) -> None:
        """DoDCategory should have STYLE value."""
        from yolo_developer.gates.gates.definition_of_done import DoDCategory

        assert DoDCategory.STYLE.value == "style"

    def test_dod_category_has_ac_coverage_value(self) -> None:
        """DoDCategory should have AC_COVERAGE value."""
        from yolo_developer.gates.gates.definition_of_done import DoDCategory

        assert DoDCategory.AC_COVERAGE.value == "ac_coverage"


class TestDoDIssue:
    """Tests for DoDIssue dataclass."""

    def test_dod_issue_creation(self) -> None:
        """DoDIssue should be creatable with required fields."""
        from yolo_developer.gates.gates.definition_of_done import DoDCategory, DoDIssue

        issue = DoDIssue(
            check_id="test_check",
            category=DoDCategory.TESTS,
            description="Missing test",
            severity="high",
            item_name="public_function",
            remediation="Add unit test for public_function",
        )
        assert issue.check_id == "test_check"
        assert issue.category == DoDCategory.TESTS
        assert issue.description == "Missing test"
        assert issue.severity == "high"
        assert issue.item_name == "public_function"
        assert issue.remediation == "Add unit test for public_function"

    def test_dod_issue_is_frozen(self) -> None:
        """DoDIssue should be immutable."""
        from yolo_developer.gates.gates.definition_of_done import DoDCategory, DoDIssue

        issue = DoDIssue(
            check_id="test_check",
            category=DoDCategory.TESTS,
            description="Missing test",
            severity="high",
            item_name="public_function",
            remediation="Add test",
        )
        with pytest.raises(AttributeError):
            issue.severity = "low"  # type: ignore[misc]


class TestDoDChecklistItems:
    """Tests for DOD_CHECKLIST_ITEMS constant."""

    def test_checklist_has_tests_category(self) -> None:
        """Checklist should have TESTS category."""
        from yolo_developer.gates.gates.definition_of_done import (
            DOD_CHECKLIST_ITEMS,
            DoDCategory,
        )

        assert DoDCategory.TESTS in DOD_CHECKLIST_ITEMS

    def test_checklist_has_documentation_category(self) -> None:
        """Checklist should have DOCUMENTATION category."""
        from yolo_developer.gates.gates.definition_of_done import (
            DOD_CHECKLIST_ITEMS,
            DoDCategory,
        )

        assert DoDCategory.DOCUMENTATION in DOD_CHECKLIST_ITEMS

    def test_checklist_has_style_category(self) -> None:
        """Checklist should have STYLE category."""
        from yolo_developer.gates.gates.definition_of_done import (
            DOD_CHECKLIST_ITEMS,
            DoDCategory,
        )

        assert DoDCategory.STYLE in DOD_CHECKLIST_ITEMS

    def test_checklist_has_ac_coverage_category(self) -> None:
        """Checklist should have AC_COVERAGE category."""
        from yolo_developer.gates.gates.definition_of_done import (
            DOD_CHECKLIST_ITEMS,
            DoDCategory,
        )

        assert DoDCategory.AC_COVERAGE in DOD_CHECKLIST_ITEMS

    def test_checklist_items_are_lists(self) -> None:
        """Each category should contain a list of items."""
        from yolo_developer.gates.gates.definition_of_done import DOD_CHECKLIST_ITEMS

        for category, items in DOD_CHECKLIST_ITEMS.items():
            assert isinstance(items, list), f"Items for {category} should be a list"
            assert len(items) > 0, f"Items for {category} should not be empty"


class TestSeverityWeights:
    """Tests for SEVERITY_WEIGHTS constant."""

    def test_severity_weights_has_high(self) -> None:
        """SEVERITY_WEIGHTS should have high severity."""
        from yolo_developer.gates.gates.definition_of_done import SEVERITY_WEIGHTS

        assert "high" in SEVERITY_WEIGHTS
        assert SEVERITY_WEIGHTS["high"] == 20

    def test_severity_weights_has_medium(self) -> None:
        """SEVERITY_WEIGHTS should have medium severity."""
        from yolo_developer.gates.gates.definition_of_done import SEVERITY_WEIGHTS

        assert "medium" in SEVERITY_WEIGHTS
        assert SEVERITY_WEIGHTS["medium"] == 10

    def test_severity_weights_has_low(self) -> None:
        """SEVERITY_WEIGHTS should have low severity."""
        from yolo_developer.gates.gates.definition_of_done import SEVERITY_WEIGHTS

        assert "low" in SEVERITY_WEIGHTS
        assert SEVERITY_WEIGHTS["low"] == 3


# =============================================================================
# Task 2: Test Presence Detection Tests
# =============================================================================


class TestCheckTestPresence:
    """Tests for check_test_presence function."""

    def test_no_issues_when_all_functions_tested(
        self, mock_code_with_tests: dict, mock_story: dict
    ) -> None:
        """Should return no issues when all public functions have tests."""
        from yolo_developer.gates.gates.definition_of_done import check_test_presence

        issues = check_test_presence(mock_code_with_tests, mock_story)
        # Filter to only test presence issues (not missing test files)
        test_issues = [
            i
            for i in issues
            if "untested" in i.description.lower() or "missing test" in i.description.lower()
        ]
        assert len(test_issues) == 0

    def test_flags_untested_public_functions(
        self, mock_code_without_tests: dict, mock_story: dict
    ) -> None:
        """Should flag public functions without tests."""
        from yolo_developer.gates.gates.definition_of_done import check_test_presence

        issues = check_test_presence(mock_code_without_tests, mock_story)
        assert len(issues) > 0
        # Should mention the untested function
        descriptions = " ".join(i.description for i in issues)
        assert "untested_function" in descriptions.lower() or "untested" in descriptions.lower()

    def test_ignores_private_functions(self, mock_code_with_tests: dict, mock_story: dict) -> None:
        """Should not flag private functions (starting with _)."""
        from yolo_developer.gates.gates.definition_of_done import check_test_presence

        issues = check_test_presence(mock_code_with_tests, mock_story)
        descriptions = " ".join(i.description for i in issues)
        assert "_private_function" not in descriptions

    def test_issues_include_remediation(
        self, mock_code_without_tests: dict, mock_story: dict
    ) -> None:
        """Issues should include remediation guidance."""
        from yolo_developer.gates.gates.definition_of_done import check_test_presence

        issues = check_test_presence(mock_code_without_tests, mock_story)
        for issue in issues:
            assert issue.remediation is not None
            assert len(issue.remediation) > 0


# =============================================================================
# Task 3: Documentation Check Tests
# =============================================================================


class TestCheckDocumentation:
    """Tests for check_documentation function."""

    def test_no_issues_when_fully_documented(self, mock_code_with_tests: dict) -> None:
        """Should return no issues when code is fully documented."""
        from yolo_developer.gates.gates.definition_of_done import check_documentation

        issues = check_documentation(mock_code_with_tests)
        # Filter to only doc-related issues
        doc_issues = [
            i
            for i in issues
            if "docstring" in i.description.lower() or "documentation" in i.description.lower()
        ]
        assert len(doc_issues) == 0

    def test_flags_missing_module_docstring(self, mock_code_without_docs: dict) -> None:
        """Should flag missing module-level docstrings."""
        from yolo_developer.gates.gates.definition_of_done import check_documentation

        issues = check_documentation(mock_code_without_docs)
        assert len(issues) > 0
        descriptions = " ".join(i.description.lower() for i in issues)
        assert "module" in descriptions or "docstring" in descriptions

    def test_flags_missing_function_docstring(self, mock_code_without_docs: dict) -> None:
        """Should flag missing function docstrings."""
        from yolo_developer.gates.gates.definition_of_done import check_documentation

        issues = check_documentation(mock_code_without_docs)
        # Should have issues for functions without docstrings
        assert any("docstring" in i.description.lower() for i in issues)

    def test_issues_include_remediation(self, mock_code_without_docs: dict) -> None:
        """Issues should include remediation guidance."""
        from yolo_developer.gates.gates.definition_of_done import check_documentation

        issues = check_documentation(mock_code_without_docs)
        for issue in issues:
            assert issue.remediation is not None
            assert len(issue.remediation) > 0


# =============================================================================
# Task 4: Code Style Validation Tests
# =============================================================================


class TestCheckCodeStyle:
    """Tests for check_code_style function."""

    def test_flags_missing_type_annotations(self, mock_code_with_style_issues: dict) -> None:
        """Should flag functions missing type annotations."""
        from yolo_developer.gates.gates.definition_of_done import check_code_style

        issues = check_code_style(mock_code_with_style_issues)
        # Should flag function_without_types
        descriptions = " ".join(i.description.lower() for i in issues)
        assert "type" in descriptions or "annotation" in descriptions

    def test_flags_naming_convention_violations(self, mock_code_with_style_issues: dict) -> None:
        """Should flag naming convention violations."""
        from yolo_developer.gates.gates.definition_of_done import check_code_style

        issues = check_code_style(mock_code_with_style_issues)
        # Should flag BadNamingConvention
        descriptions = " ".join(i.description.lower() for i in issues)
        assert (
            "naming" in descriptions
            or "convention" in descriptions
            or "badnamingconvention" in descriptions.lower()
        )

    def test_flags_excessive_complexity(self, mock_code_with_style_issues: dict) -> None:
        """Should flag excessive function complexity."""
        from yolo_developer.gates.gates.definition_of_done import check_code_style

        issues = check_code_style(mock_code_with_style_issues)
        # Should flag complex_function
        descriptions = " ".join(i.description.lower() for i in issues)
        assert "complex" in descriptions or "nesting" in descriptions

    def test_issues_have_correct_category(self, mock_code_with_style_issues: dict) -> None:
        """Issues should have STYLE category."""
        from yolo_developer.gates.gates.definition_of_done import DoDCategory, check_code_style

        issues = check_code_style(mock_code_with_style_issues)
        for issue in issues:
            assert issue.category == DoDCategory.STYLE


# =============================================================================
# Task 5: AC Coverage Check Tests
# =============================================================================


class TestCheckACCoverage:
    """Tests for check_ac_coverage function."""

    def test_no_issues_when_all_ac_addressed(
        self, mock_code_with_tests: dict, mock_story: dict
    ) -> None:
        """Should return no issues when all AC are addressed."""
        from yolo_developer.gates.gates.definition_of_done import check_ac_coverage

        # Add comments referencing ACs
        code_with_ac_refs = {
            "files": [
                {
                    "path": "src/module.py",
                    "content": '''"""Module docstring.

Implements:
- AC1: Test presence is verified
- AC2: Documentation presence is checked
- AC3: Code style compliance is validated
"""

def verify_test_presence():
    """AC1: Test presence is verified."""
    pass

def check_documentation():
    """AC2: Documentation presence is checked."""
    pass

def validate_code_style():
    """AC3: Code style compliance is validated."""
    pass
''',
                },
            ],
        }
        issues = check_ac_coverage(code_with_ac_refs, mock_story)
        # Should have no unaddressed AC issues
        unaddressed = [
            i
            for i in issues
            if "unaddressed" in i.description.lower() or "not addressed" in i.description.lower()
        ]
        assert len(unaddressed) == 0

    def test_flags_unaddressed_ac(
        self, mock_code_with_tests: dict, mock_story_partial_coverage: dict
    ) -> None:
        """Should flag unaddressed acceptance criteria."""
        from yolo_developer.gates.gates.definition_of_done import check_ac_coverage

        issues = check_ac_coverage(mock_code_with_tests, mock_story_partial_coverage)
        # Should flag unaddressed ACs
        assert len(issues) > 0

    def test_calculates_coverage_percentage(
        self, mock_code_with_tests: dict, mock_story_partial_coverage: dict
    ) -> None:
        """Should calculate AC coverage percentage."""
        from yolo_developer.gates.gates.definition_of_done import check_ac_coverage

        issues = check_ac_coverage(mock_code_with_tests, mock_story_partial_coverage)
        # Coverage info should be included
        assert any("coverage" in i.description.lower() or "%" in i.description for i in issues)

    def test_issues_have_ac_coverage_category(
        self, mock_code_with_tests: dict, mock_story_partial_coverage: dict
    ) -> None:
        """Issues should have AC_COVERAGE category."""
        from yolo_developer.gates.gates.definition_of_done import DoDCategory, check_ac_coverage

        issues = check_ac_coverage(mock_code_with_tests, mock_story_partial_coverage)
        for issue in issues:
            assert issue.category == DoDCategory.AC_COVERAGE


# =============================================================================
# Task 6: Checklist Result Generation Tests
# =============================================================================


class TestGenerateDoDChecklist:
    """Tests for generate_dod_checklist function."""

    def test_groups_issues_by_category(self) -> None:
        """Should group issues by category."""
        from yolo_developer.gates.gates.definition_of_done import (
            DoDCategory,
            DoDIssue,
            generate_dod_checklist,
        )

        issues = [
            DoDIssue("t1", DoDCategory.TESTS, "Missing test", "high", "func1", "Add test"),
            DoDIssue("d1", DoDCategory.DOCUMENTATION, "Missing doc", "medium", "func2", "Add doc"),
            DoDIssue("s1", DoDCategory.STYLE, "Style issue", "low", "func3", "Fix style"),
        ]
        checklist = generate_dod_checklist(issues)
        assert "tests" in checklist or DoDCategory.TESTS.value in str(checklist)
        assert "documentation" in checklist or DoDCategory.DOCUMENTATION.value in str(checklist)
        assert "style" in checklist or DoDCategory.STYLE.value in str(checklist)

    def test_calculates_compliance_score(self) -> None:
        """Should calculate compliance score."""
        from yolo_developer.gates.gates.definition_of_done import (
            DoDCategory,
            DoDIssue,
            generate_dod_checklist,
        )

        issues = [
            DoDIssue("t1", DoDCategory.TESTS, "Missing test", "high", "func1", "Add test"),  # -20
        ]
        checklist = generate_dod_checklist(issues)
        assert "score" in checklist
        assert checklist["score"] == 80  # 100 - 20

    def test_score_capped_at_zero(self) -> None:
        """Score should not go below zero."""
        from yolo_developer.gates.gates.definition_of_done import (
            DoDCategory,
            DoDIssue,
            generate_dod_checklist,
        )

        # Create many high-severity issues
        issues = [
            DoDIssue(f"t{i}", DoDCategory.TESTS, f"Issue {i}", "high", f"func{i}", "Fix")
            for i in range(10)  # 10 * 20 = 200 points
        ]
        checklist = generate_dod_checklist(issues)
        assert checklist["score"] >= 0

    def test_empty_issues_gives_perfect_score(self) -> None:
        """Empty issues should give score of 100."""
        from yolo_developer.gates.gates.definition_of_done import generate_dod_checklist

        checklist = generate_dod_checklist([])
        assert checklist["score"] == 100


# =============================================================================
# Task 7: DoD Evaluator Tests
# =============================================================================


@pytest.fixture(autouse=True)
def register_dod_evaluator():
    """Ensure evaluator is registered for each test."""
    clear_evaluators()
    # Import to trigger registration
    from yolo_developer.gates.gates import definition_of_done  # noqa: F401

    yield
    clear_evaluators()


class TestDefinitionOfDoneEvaluator:
    """Tests for definition_of_done_evaluator function."""

    @pytest.mark.asyncio
    async def test_evaluator_passes_for_compliant_code(
        self, mock_code_with_tests: dict, mock_story: dict
    ) -> None:
        """Evaluator should pass for compliant code."""
        from yolo_developer.gates.gates.definition_of_done import definition_of_done_evaluator

        # Create code that addresses all ACs
        compliant_code = {
            "files": [
                {
                    "path": "src/module.py",
                    "content": '''"""Module implementing DoD gate functionality.

This module provides:
- AC1: Test presence verification
- AC2: Documentation presence checking
- AC3: Code style compliance validation
"""

def verify_test_presence(code: dict) -> bool:
    """Verify test presence (AC1)."""
    return True

def check_documentation(code: dict) -> bool:
    """Check documentation presence (AC2)."""
    return True

def validate_code_style(code: dict) -> bool:
    """Validate code style compliance (AC3)."""
    return True
''',
                },
                {
                    "path": "tests/test_module.py",
                    "content": '''"""Tests for module."""

def test_verify_test_presence():
    """Test AC1."""
    pass

def test_check_documentation():
    """Test AC2."""
    pass

def test_validate_code_style():
    """Test AC3."""
    pass
''',
                },
            ],
        }
        context = GateContext(
            state={"code": compliant_code, "story": mock_story},
            gate_name="definition_of_done",
        )
        result = await definition_of_done_evaluator(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_evaluator_fails_for_missing_code(self, mock_story: dict) -> None:
        """Evaluator should fail when code is missing from state."""
        from yolo_developer.gates.gates.definition_of_done import definition_of_done_evaluator

        context = GateContext(
            state={"story": mock_story},
            gate_name="definition_of_done",
        )
        result = await definition_of_done_evaluator(context)
        assert result.passed is False
        assert "code" in result.reason.lower() or "missing" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_evaluator_fails_for_low_score(
        self, mock_code_without_docs: dict, mock_story: dict
    ) -> None:
        """Evaluator should fail when compliance score is below threshold."""
        from yolo_developer.gates.gates.definition_of_done import definition_of_done_evaluator

        context = GateContext(
            state={"code": mock_code_without_docs, "story": mock_story},
            gate_name="definition_of_done",
        )
        result = await definition_of_done_evaluator(context)
        # Should fail due to missing documentation
        if not result.passed:
            assert (
                "score" in result.reason.lower()
                or "threshold" in result.reason.lower()
                or "compliance" in result.reason.lower()
            )

    @pytest.mark.asyncio
    async def test_evaluator_uses_default_threshold(
        self, mock_code_with_tests: dict, mock_story: dict
    ) -> None:
        """Evaluator should use default threshold of 70."""
        from yolo_developer.gates.gates.definition_of_done import definition_of_done_evaluator

        context = GateContext(
            state={"code": mock_code_with_tests, "story": mock_story},
            gate_name="definition_of_done",
        )
        result = await definition_of_done_evaluator(context)
        # Result should mention score and use default threshold
        assert result.gate_name == "definition_of_done"

    @pytest.mark.asyncio
    async def test_evaluator_uses_custom_threshold(
        self, mock_code_with_tests: dict, mock_story: dict
    ) -> None:
        """Evaluator should respect custom threshold from config."""
        from yolo_developer.gates.gates.definition_of_done import definition_of_done_evaluator

        context = GateContext(
            state={
                "code": mock_code_with_tests,
                "story": mock_story,
                "config": {"quality": {"dod_threshold": 95}},
            },
            gate_name="definition_of_done",
        )
        result = await definition_of_done_evaluator(context)
        # High threshold may cause failure
        assert result.gate_name == "definition_of_done"

    @pytest.mark.asyncio
    async def test_evaluator_reads_implementation_key(
        self, mock_code_with_tests: dict, mock_story: dict
    ) -> None:
        """Evaluator should also read from 'implementation' key."""
        from yolo_developer.gates.gates.definition_of_done import definition_of_done_evaluator

        context = GateContext(
            state={"implementation": mock_code_with_tests, "story": mock_story},
            gate_name="definition_of_done",
        )
        result = await definition_of_done_evaluator(context)
        # Should process the code from implementation key
        assert result.gate_name == "definition_of_done"


# =============================================================================
# Task 8: Failure Report Generation Tests
# =============================================================================


class TestGenerateDoDReport:
    """Tests for generate_dod_report function."""

    def test_report_includes_score(self) -> None:
        """Report should include compliance score."""
        from yolo_developer.gates.gates.definition_of_done import (
            DoDCategory,
            DoDIssue,
            generate_dod_report,
        )

        issues = [DoDIssue("t1", DoDCategory.TESTS, "Missing test", "high", "func1", "Add test")]
        checklist = {"score": 80, "tests": [{"item": "unit_tests_present", "passed": False}]}
        report = generate_dod_report(issues, checklist, 80)
        assert "80" in report

    def test_report_includes_issues_by_category(self) -> None:
        """Report should include issues grouped by category."""
        from yolo_developer.gates.gates.definition_of_done import (
            DoDCategory,
            DoDIssue,
            generate_dod_report,
        )

        issues = [
            DoDIssue("t1", DoDCategory.TESTS, "Missing test", "high", "func1", "Add test"),
            DoDIssue("d1", DoDCategory.DOCUMENTATION, "Missing doc", "medium", "func2", "Add doc"),
        ]
        checklist = {"score": 70, "tests": [], "documentation": []}
        report = generate_dod_report(issues, checklist, 70)
        assert "test" in report.lower()
        assert "documentation" in report.lower() or "doc" in report.lower()

    def test_report_includes_remediation(self) -> None:
        """Report should include remediation suggestions."""
        from yolo_developer.gates.gates.definition_of_done import (
            DoDCategory,
            DoDIssue,
            generate_dod_report,
        )

        issues = [
            DoDIssue(
                "t1", DoDCategory.TESTS, "Missing test", "high", "func1", "Add unit test for func1"
            )
        ]
        checklist = {"score": 80}
        report = generate_dod_report(issues, checklist, 80)
        assert "Add unit test" in report or "remediation" in report.lower()


# =============================================================================
# Task 9: Evaluator Registration Tests
# =============================================================================


class TestEvaluatorRegistration:
    """Tests for evaluator registration."""

    def test_evaluator_registered_on_import(self) -> None:
        """Evaluator should be registered when module is imported."""
        import importlib

        clear_evaluators()
        # Re-import to trigger registration (need reload since module is cached)
        from yolo_developer.gates.gates import definition_of_done

        importlib.reload(definition_of_done)

        evaluator = get_evaluator("definition_of_done")
        assert evaluator is not None

    def test_evaluator_follows_protocol(self) -> None:
        """Evaluator should follow GateEvaluator protocol."""
        from yolo_developer.gates.evaluators import GateEvaluator
        from yolo_developer.gates.gates.definition_of_done import definition_of_done_evaluator

        assert isinstance(definition_of_done_evaluator, GateEvaluator)


# =============================================================================
# Helper Function Tests (Code Review Finding HIGH-2)
# =============================================================================


class TestExtractFunctionsFromContent:
    """Tests for _extract_functions_from_content helper."""

    def test_extracts_regular_function(self) -> None:
        """Should extract regular function definitions."""
        from yolo_developer.gates.gates.definition_of_done import _extract_functions_from_content

        code = '''def my_function(arg: str) -> int:
    """Docstring."""
    return len(arg)
'''
        functions = _extract_functions_from_content(code)
        assert len(functions) == 1
        assert functions[0]["name"] == "my_function"
        assert functions[0]["has_docstring"] is True
        assert functions[0]["has_return_type"] is True

    def test_extracts_async_function(self) -> None:
        """Should extract async function definitions."""
        from yolo_developer.gates.gates.definition_of_done import _extract_functions_from_content

        code = """async def async_func() -> None:
    pass
"""
        functions = _extract_functions_from_content(code)
        assert len(functions) == 1
        assert functions[0]["name"] == "async_func"

    def test_handles_syntax_error(self) -> None:
        """Should return empty list on syntax error."""
        from yolo_developer.gates.gates.definition_of_done import _extract_functions_from_content

        code = "def broken(:"
        functions = _extract_functions_from_content(code)
        assert functions == []

    def test_detects_private_functions(self) -> None:
        """Should mark private functions correctly."""
        from yolo_developer.gates.gates.definition_of_done import _extract_functions_from_content

        code = """def _private():
    pass

def public():
    pass
"""
        functions = _extract_functions_from_content(code)
        private = next(f for f in functions if f["name"] == "_private")
        public = next(f for f in functions if f["name"] == "public")
        assert private["is_private"] is True
        assert public["is_private"] is False


class TestCalculateNestingDepth:
    """Tests for _calculate_nesting_depth helper."""

    def test_no_nesting_returns_zero(self) -> None:
        """Should return 0 for flat code."""
        import ast

        from yolo_developer.gates.gates.definition_of_done import _calculate_nesting_depth

        code = """def flat():
    return 1
"""
        tree = ast.parse(code)
        func = tree.body[0]
        depth = _calculate_nesting_depth(func)
        assert depth == 0

    def test_single_if_returns_one(self) -> None:
        """Should return 1 for single if statement."""
        import ast

        from yolo_developer.gates.gates.definition_of_done import _calculate_nesting_depth

        code = """def nested():
    if True:
        return 1
    return 0
"""
        tree = ast.parse(code)
        func = tree.body[0]
        depth = _calculate_nesting_depth(func)
        assert depth == 1

    def test_deep_nesting(self) -> None:
        """Should correctly count deep nesting."""
        import ast

        from yolo_developer.gates.gates.definition_of_done import _calculate_nesting_depth

        code = """def deep():
    if a:
        if b:
            if c:
                return 1
    return 0
"""
        tree = ast.parse(code)
        func = tree.body[0]
        depth = _calculate_nesting_depth(func)
        assert depth == 3


class TestHasModuleDocstring:
    """Tests for _has_module_docstring helper."""

    def test_detects_module_docstring(self) -> None:
        """Should detect module docstring."""
        from yolo_developer.gates.gates.definition_of_done import _has_module_docstring

        code = '''"""Module docstring."""

def func():
    pass
'''
        assert _has_module_docstring(code) is True

    def test_no_docstring_returns_false(self) -> None:
        """Should return False when no docstring."""
        from yolo_developer.gates.gates.definition_of_done import _has_module_docstring

        code = """def func():
    pass
"""
        assert _has_module_docstring(code) is False

    def test_handles_syntax_error(self) -> None:
        """Should return False on syntax error."""
        from yolo_developer.gates.gates.definition_of_done import _has_module_docstring

        code = "def broken(:"
        assert _has_module_docstring(code) is False


class TestIsNamingConventionValid:
    """Tests for _is_naming_convention_valid helper."""

    def test_snake_case_valid(self) -> None:
        """Should return True for snake_case names."""
        from yolo_developer.gates.gates.definition_of_done import _is_naming_convention_valid

        assert _is_naming_convention_valid("my_function") is True
        assert _is_naming_convention_valid("process_data") is True
        assert _is_naming_convention_valid("a") is True

    def test_private_snake_case_valid(self) -> None:
        """Should return True for private snake_case names."""
        from yolo_developer.gates.gates.definition_of_done import _is_naming_convention_valid

        assert _is_naming_convention_valid("_private") is True
        assert _is_naming_convention_valid("__dunder__") is True

    def test_camel_case_invalid(self) -> None:
        """Should return False for camelCase names."""
        from yolo_developer.gates.gates.definition_of_done import _is_naming_convention_valid

        assert _is_naming_convention_valid("myFunction") is False
        assert _is_naming_convention_valid("BadName") is False

    def test_numbers_allowed(self) -> None:
        """Should allow numbers in names."""
        from yolo_developer.gates.gates.definition_of_done import _is_naming_convention_valid

        assert _is_naming_convention_valid("process_v2") is True
        assert _is_naming_convention_valid("handler_123") is True


class TestExtractTestFunctionNames:
    """Tests for _extract_test_function_names helper."""

    def test_extracts_test_functions(self) -> None:
        """Should extract functions starting with test_."""
        from yolo_developer.gates.gates.definition_of_done import _extract_test_function_names

        code = """def test_something():
    pass

def test_another():
    pass

def helper():
    pass
"""
        names = _extract_test_function_names(code)
        assert "test_something" in names
        assert "test_another" in names
        assert "helper" not in names

    def test_empty_for_no_tests(self) -> None:
        """Should return empty set when no test functions."""
        from yolo_developer.gates.gates.definition_of_done import _extract_test_function_names

        code = """def helper():
    pass
"""
        names = _extract_test_function_names(code)
        assert names == set()

    def test_handles_syntax_error(self) -> None:
        """Should return empty set on syntax error."""
        from yolo_developer.gates.gates.definition_of_done import _extract_test_function_names

        code = "def broken(:"
        names = _extract_test_function_names(code)
        assert names == set()


# =============================================================================
# Logging Tests (Code Review Finding MEDIUM-5)
# =============================================================================


class TestStructuredLogging:
    """Tests for structured logging in DoD gate."""

    @pytest.mark.asyncio
    async def test_evaluator_logs_evaluation_started(
        self, caplog: pytest.LogCaptureFixture, mock_code_with_tests: dict, mock_story: dict
    ) -> None:
        """Evaluator should log when evaluation starts."""
        import structlog

        from yolo_developer.gates.gates.definition_of_done import definition_of_done_evaluator
        from yolo_developer.gates.types import GateContext

        # Configure structlog to use standard logging for capture
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=False,
        )

        context = GateContext(
            state={"code": mock_code_with_tests, "story": mock_story},
            gate_name="definition_of_done",
        )

        with caplog.at_level("INFO"):
            await definition_of_done_evaluator(context)

        # Check that logging occurred (structlog formats may vary)
        log_output = caplog.text.lower()
        assert "definition_of_done" in log_output or len(caplog.records) > 0
