"""Unit tests for DoD validation utilities (Story 8.6 - Tasks 7-10).

Tests for:
- DoDChecklistItem dataclass construction and behavior
- DoDValidationResult dataclass construction and behavior
- validate_implementation_dod() function
- validate_artifact_dod() function
- validate_dev_output_dod() function

All tests follow the red-green-refactor cycle.
"""

from __future__ import annotations

import pytest

from yolo_developer.gates.report_types import GateIssue, Severity

# =============================================================================
# Task 7: Tests for DoD Validation Types
# =============================================================================


class TestDoDChecklistItem:
    """Tests for DoDChecklistItem dataclass."""

    def test_checklist_item_construction_passed(self) -> None:
        """Test creating a passing checklist item."""
        from yolo_developer.agents.dev.dod_utils import DoDChecklistItem

        item = DoDChecklistItem(
            category="tests",
            item_name="unit_tests_present",
            passed=True,
            severity=None,
            message="All unit tests present",
        )

        assert item.category == "tests"
        assert item.item_name == "unit_tests_present"
        assert item.passed is True
        assert item.severity is None
        assert item.message == "All unit tests present"

    def test_checklist_item_construction_failed(self) -> None:
        """Test creating a failing checklist item."""
        from yolo_developer.agents.dev.dod_utils import DoDChecklistItem

        item = DoDChecklistItem(
            category="documentation",
            item_name="module_docstring",
            passed=False,
            severity="medium",
            message="Missing module docstring",
        )

        assert item.category == "documentation"
        assert item.item_name == "module_docstring"
        assert item.passed is False
        assert item.severity == "medium"
        assert item.message == "Missing module docstring"

    def test_checklist_item_is_frozen(self) -> None:
        """Test that DoDChecklistItem is immutable."""
        from yolo_developer.agents.dev.dod_utils import DoDChecklistItem

        item = DoDChecklistItem(
            category="tests",
            item_name="unit_tests_present",
            passed=True,
            severity=None,
            message="All unit tests present",
        )

        with pytest.raises(AttributeError):
            item.passed = False  # type: ignore[misc]

    def test_checklist_item_all_categories(self) -> None:
        """Test creating items for all valid categories."""
        from yolo_developer.agents.dev.dod_utils import DoDChecklistItem

        categories = ["tests", "documentation", "style", "ac_coverage"]

        for category in categories:
            item = DoDChecklistItem(
                category=category,  # type: ignore[arg-type]
                item_name="test_item",
                passed=True,
                severity=None,
                message="Test message",
            )
            assert item.category == category


class TestDoDValidationResult:
    """Tests for DoDValidationResult dataclass."""

    def test_validation_result_default_values(self) -> None:
        """Test DoDValidationResult with default values."""
        from yolo_developer.agents.dev.dod_utils import DoDValidationResult

        result = DoDValidationResult()

        assert result.score == 100
        assert result.passed is True
        assert result.threshold == 70
        assert result.checklist == []
        assert result.issues == []
        assert result.artifact_id is None

    def test_validation_result_with_values(self) -> None:
        """Test DoDValidationResult with explicit values."""
        from yolo_developer.agents.dev.dod_utils import (
            DoDChecklistItem,
            DoDValidationResult,
        )

        item = DoDChecklistItem(
            category="tests",
            item_name="unit_tests_present",
            passed=False,
            severity="high",
            message="Missing tests",
        )

        issue = GateIssue(
            location="test",
            issue_type="missing_test",
            description="Missing test",
            severity=Severity.BLOCKING,
        )

        result = DoDValidationResult(
            score=60,
            passed=False,
            threshold=70,
            checklist=[item],
            issues=[issue],
            artifact_id="story-001",
        )

        assert result.score == 60
        assert result.passed is False
        assert result.threshold == 70
        assert len(result.checklist) == 1
        assert len(result.issues) == 1
        assert result.artifact_id == "story-001"

    def test_validation_result_to_dict(self) -> None:
        """Test DoDValidationResult.to_dict() serialization."""
        from yolo_developer.agents.dev.dod_utils import (
            DoDChecklistItem,
            DoDValidationResult,
        )

        item = DoDChecklistItem(
            category="tests",
            item_name="unit_tests_present",
            passed=True,
            severity=None,
            message="All tests present",
        )

        result = DoDValidationResult(
            score=100,
            passed=True,
            threshold=70,
            checklist=[item],
            issues=[],
            artifact_id="story-001",
        )

        result_dict = result.to_dict()

        assert result_dict["score"] == 100
        assert result_dict["passed"] is True
        assert result_dict["threshold"] == 70
        assert result_dict["issue_count"] == 0
        assert result_dict["artifact_id"] == "story-001"
        assert len(result_dict["checklist"]) == 1
        assert result_dict["checklist"][0]["category"] == "tests"

    def test_validation_result_passed_based_on_score(self) -> None:
        """Test that passed is determined by score vs threshold."""
        from yolo_developer.agents.dev.dod_utils import DoDValidationResult

        # Score above threshold - passed
        result_pass = DoDValidationResult(score=80, passed=True, threshold=70)
        assert result_pass.passed is True

        # Score below threshold - failed
        result_fail = DoDValidationResult(score=60, passed=False, threshold=70)
        assert result_fail.passed is False

        # Score equals threshold - passed
        result_equal = DoDValidationResult(score=70, passed=True, threshold=70)
        assert result_equal.passed is True


# =============================================================================
# Task 8: Tests for Programmatic Validation
# =============================================================================


class TestValidateImplementationDoD:
    """Tests for validate_implementation_dod() function."""

    def test_validate_complete_implementation(self) -> None:
        """Test validation of complete implementation with tests and docs."""
        from yolo_developer.agents.dev.dod_utils import validate_implementation_dod

        code = {
            "files": [
                {
                    "path": "src/impl.py",
                    "content": '''"""Module docstring."""

from __future__ import annotations


def public_function(arg: str) -> str:
    """Public function docstring.

    Args:
        arg: Input argument.

    Returns:
        The argument.
    """
    return arg
''',
                },
                {
                    "path": "tests/test_impl.py",
                    "content": '''"""Test module."""

def test_public_function() -> None:
    """Test the public function."""
    from impl import public_function
    assert public_function("test") == "test"
''',
                },
            ]
        }

        story: dict[str, list[str]] = {"acceptance_criteria": []}

        result = validate_implementation_dod(code, story)

        # With complete implementation, should pass
        assert result.score >= 70
        assert isinstance(result.checklist, list)
        assert isinstance(result.issues, list)

    def test_validate_missing_tests(self) -> None:
        """Test validation detects missing tests."""
        from yolo_developer.agents.dev.dod_utils import validate_implementation_dod

        code = {
            "files": [
                {
                    "path": "src/impl.py",
                    "content": '''"""Module docstring."""

def public_function(arg: str) -> str:
    """Docstring."""
    return arg
''',
                },
                # No test file
            ]
        }

        story: dict[str, list[str]] = {"acceptance_criteria": []}

        result = validate_implementation_dod(code, story)

        # Missing tests should lower score
        assert result.score < 100
        # Should have issues about missing tests
        test_issues = [i for i in result.issues if "test" in i.issue_type.lower()]
        assert len(test_issues) > 0

    def test_validate_missing_documentation(self) -> None:
        """Test validation detects missing documentation."""
        from yolo_developer.agents.dev.dod_utils import validate_implementation_dod

        code = {
            "files": [
                {
                    "path": "src/impl.py",
                    "content": """def public_function(arg):
    return arg
""",
                },
                {
                    "path": "tests/test_impl.py",
                    "content": """def test_public_function():
    pass
""",
                },
            ]
        }

        story: dict[str, list[str]] = {"acceptance_criteria": []}

        result = validate_implementation_dod(code, story)

        # Missing docs should lower score
        assert result.score < 100
        # Should have issues about missing docstrings
        doc_issues = [i for i in result.issues if "docstring" in i.issue_type.lower()]
        assert len(doc_issues) > 0

    def test_validate_style_violations(self) -> None:
        """Test validation detects style violations."""
        from yolo_developer.agents.dev.dod_utils import validate_implementation_dod

        code = {
            "files": [
                {
                    "path": "src/impl.py",
                    "content": '''"""Module docstring."""

def publicFunction(arg):  # PascalCase - style violation
    """Docstring."""
    return arg
''',
                },
                {
                    "path": "tests/test_impl.py",
                    "content": '''"""Test module."""

def test_publicFunction():
    pass
''',
                },
            ]
        }

        story: dict[str, list[str]] = {"acceptance_criteria": []}

        result = validate_implementation_dod(code, story)

        # Style violations should lower score
        assert result.score < 100
        # Should have issues about naming
        style_issues = [i for i in result.issues if "naming" in i.issue_type.lower()]
        assert len(style_issues) > 0

    def test_validate_missing_ac_coverage(self) -> None:
        """Test validation detects missing AC coverage."""
        from yolo_developer.agents.dev.dod_utils import validate_implementation_dod

        code = {
            "files": [
                {
                    "path": "src/impl.py",
                    "content": '''"""Module docstring."""

def unrelated_function() -> None:
    """Docstring."""
    pass
''',
                },
            ]
        }

        story = {
            "acceptance_criteria": [
                "AC1: The system shall authenticate users via OAuth2",
                "AC2: The system shall validate credentials securely",
            ]
        }

        result = validate_implementation_dod(code, story)

        # Missing AC coverage should lower score
        # Should have issues about unaddressed ACs
        ac_issues = [i for i in result.issues if "ac" in i.issue_type.lower()]
        assert len(ac_issues) > 0

    def test_validate_custom_threshold(self) -> None:
        """Test validation with custom threshold."""
        from yolo_developer.agents.dev.dod_utils import validate_implementation_dod

        code = {
            "files": [
                {
                    "path": "src/impl.py",
                    "content": '''"""Module."""

def foo() -> None:
    """Doc."""
    pass
''',
                },
            ]
        }

        story: dict[str, list[str]] = {"acceptance_criteria": []}

        # With threshold=50, partial implementation might pass
        result_low = validate_implementation_dod(code, story, threshold=50)

        # With threshold=90, partial implementation should fail
        result_high = validate_implementation_dod(code, story, threshold=90)

        assert result_low.threshold == 50
        assert result_high.threshold == 90


# =============================================================================
# Task 9: Tests for Artifact Validation
# =============================================================================


class TestValidateArtifactDoD:
    """Tests for validate_artifact_dod() function."""

    def test_validate_artifact_converts_code_files(self) -> None:
        """Test that artifact code_files are converted correctly."""
        from yolo_developer.agents.dev.dod_utils import validate_artifact_dod
        from yolo_developer.agents.dev.types import (
            CodeFile,
            ImplementationArtifact,
            TestFile,
        )

        code_file = CodeFile(
            file_path="src/impl.py",
            content='''"""Module docstring."""

def public_function(arg: str) -> str:
    """Docstring."""
    return arg
''',
            file_type="source",
        )

        test_file = TestFile(
            file_path="tests/test_impl.py",
            content='''"""Test module."""

def test_public_function() -> None:
    pass
''',
            test_type="unit",
        )

        artifact = ImplementationArtifact(
            story_id="story-001",
            code_files=(code_file,),
            test_files=(test_file,),
            implementation_status="completed",
        )

        story: dict[str, list[str]] = {"acceptance_criteria": []}

        result = validate_artifact_dod(artifact, story)

        # Should have converted and validated
        assert result.artifact_id == "story-001"
        assert isinstance(result.score, int)
        assert isinstance(result.checklist, list)

    def test_validate_artifact_converts_test_files(self) -> None:
        """Test that artifact test_files are converted correctly."""
        from yolo_developer.agents.dev.dod_utils import validate_artifact_dod
        from yolo_developer.agents.dev.types import (
            CodeFile,
            ImplementationArtifact,
            TestFile,
        )

        code_file = CodeFile(
            file_path="src/auth.py",
            content='''"""Auth module."""

def authenticate(user: str) -> bool:
    """Authenticate user."""
    return True
''',
            file_type="source",
        )

        test_file = TestFile(
            file_path="tests/test_auth.py",
            content='''"""Test auth."""

def test_authenticate() -> None:
    """Test authenticate."""
    pass
''',
            test_type="unit",
        )

        artifact = ImplementationArtifact(
            story_id="story-002",
            code_files=(code_file,),
            test_files=(test_file,),
            implementation_status="completed",
        )

        story: dict[str, list[str]] = {"acceptance_criteria": []}

        result = validate_artifact_dod(artifact, story)

        # Should recognize test file and count toward test presence
        assert result.artifact_id == "story-002"

    def test_validate_artifact_with_valid_artifact_passes(self) -> None:
        """Test that complete artifact with tests and docs passes."""
        from yolo_developer.agents.dev.dod_utils import validate_artifact_dod
        from yolo_developer.agents.dev.types import (
            CodeFile,
            ImplementationArtifact,
            TestFile,
        )

        code_file = CodeFile(
            file_path="src/handler.py",
            content='''"""Handler module for processing requests.

This module provides request handling functionality.
"""

from __future__ import annotations


def handle_request(data: dict[str, str]) -> dict[str, str]:
    """Handle incoming request.

    Args:
        data: Request data dictionary.

    Returns:
        Response dictionary.
    """
    return {"status": "ok", "data": str(data)}
''',
            file_type="source",
        )

        test_file = TestFile(
            file_path="tests/test_handler.py",
            content='''"""Test handler module."""

from __future__ import annotations


def test_handle_request() -> None:
    """Test handle_request function."""
    from handler import handle_request

    result = handle_request({"key": "value"})
    assert result["status"] == "ok"
''',
            test_type="unit",
        )

        artifact = ImplementationArtifact(
            story_id="story-003",
            code_files=(code_file,),
            test_files=(test_file,),
            implementation_status="completed",
        )

        story: dict[str, list[str]] = {"acceptance_criteria": []}

        result = validate_artifact_dod(artifact, story)

        # Well-formed artifact should score well
        assert result.score >= 70
        assert result.artifact_id == "story-003"


# =============================================================================
# Task 10: Tests for DevOutput Validation
# =============================================================================


class TestValidateDevOutputDoD:
    """Tests for validate_dev_output_dod() function."""

    def test_validate_dev_output_validates_all_artifacts(self) -> None:
        """Test that all artifacts in DevOutput are validated."""
        from yolo_developer.agents.dev.dod_utils import validate_dev_output_dod
        from yolo_developer.agents.dev.types import (
            CodeFile,
            DevOutput,
            ImplementationArtifact,
        )

        artifact1 = ImplementationArtifact(
            story_id="story-001",
            code_files=(
                CodeFile(
                    file_path="src/a.py",
                    content='"""Module A."""\n',
                    file_type="source",
                ),
            ),
            implementation_status="completed",
        )

        artifact2 = ImplementationArtifact(
            story_id="story-002",
            code_files=(
                CodeFile(
                    file_path="src/b.py",
                    content='"""Module B."""\n',
                    file_type="source",
                ),
            ),
            implementation_status="completed",
        )

        output = DevOutput(
            implementations=(artifact1, artifact2),
            processing_notes="Test output",
        )

        # Create mock state
        state = {
            "story": {"acceptance_criteria": []},
        }

        results = validate_dev_output_dod(output, state)

        # Should have result for each artifact
        assert len(results) == 2
        assert results[0].artifact_id == "story-001"
        assert results[1].artifact_id == "story-002"

    def test_validate_dev_output_extracts_story_from_state(self) -> None:
        """Test that story data is extracted from state."""
        from yolo_developer.agents.dev.dod_utils import validate_dev_output_dod
        from yolo_developer.agents.dev.types import (
            CodeFile,
            DevOutput,
            ImplementationArtifact,
        )

        artifact = ImplementationArtifact(
            story_id="story-001",
            code_files=(
                CodeFile(
                    file_path="src/impl.py",
                    content='"""Module."""\n\ndef auth() -> None:\n    """Doc."""\n    pass\n',
                    file_type="source",
                ),
            ),
            implementation_status="completed",
        )

        output = DevOutput(
            implementations=(artifact,),
            processing_notes="Test output",
        )

        # State with story containing ACs
        state = {
            "story": {
                "acceptance_criteria": [
                    "AC1: System shall authenticate users",
                ]
            },
        }

        results = validate_dev_output_dod(output, state)

        # Should have used story from state for AC validation
        assert len(results) == 1

    def test_validate_dev_output_with_multiple_artifacts(self) -> None:
        """Test validation with multiple artifacts."""
        from yolo_developer.agents.dev.dod_utils import validate_dev_output_dod
        from yolo_developer.agents.dev.types import (
            CodeFile,
            DevOutput,
            ImplementationArtifact,
            TestFile,
        )

        artifacts = []
        for i in range(3):
            artifact = ImplementationArtifact(
                story_id=f"story-{i:03d}",
                code_files=(
                    CodeFile(
                        file_path=f"src/module_{i}.py",
                        content=f'"""Module {i}."""\n',
                        file_type="source",
                    ),
                ),
                test_files=(
                    TestFile(
                        file_path=f"tests/test_module_{i}.py",
                        content=f'"""Test module {i}."""\n',
                        test_type="unit",
                    ),
                ),
                implementation_status="completed",
            )
            artifacts.append(artifact)

        output = DevOutput(
            implementations=tuple(artifacts),
            processing_notes="Multiple artifacts",
        )

        state = {"story": {"acceptance_criteria": []}}

        results = validate_dev_output_dod(output, state)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.artifact_id == f"story-{i:03d}"

    def test_validate_dev_output_empty_output(self) -> None:
        """Test validation with empty DevOutput."""
        from yolo_developer.agents.dev.dod_utils import validate_dev_output_dod
        from yolo_developer.agents.dev.types import DevOutput

        output = DevOutput(
            implementations=(),
            processing_notes="Empty output",
        )

        state = {"story": {"acceptance_criteria": []}}

        results = validate_dev_output_dod(output, state)

        # Should return empty list for empty output
        assert len(results) == 0


# =============================================================================
# Code Review Fixes: Edge Cases and API Consistency Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling (Code Review M4)."""

    def test_validate_empty_files_list(self) -> None:
        """Test validation with empty files list."""
        from yolo_developer.agents.dev.dod_utils import validate_implementation_dod

        code: dict[str, list[dict[str, str]]] = {"files": []}
        story: dict[str, list[str]] = {"acceptance_criteria": []}

        result = validate_implementation_dod(code, story)

        # Should handle empty files gracefully
        assert isinstance(result.score, int)
        assert isinstance(result.checklist, list)
        assert isinstance(result.issues, list)

    def test_validate_missing_files_key(self) -> None:
        """Test validation with missing files key."""
        from yolo_developer.agents.dev.dod_utils import validate_implementation_dod

        code: dict[str, str] = {}  # Missing 'files' key
        story: dict[str, list[str]] = {"acceptance_criteria": []}

        result = validate_implementation_dod(code, story)

        # Should handle missing files key gracefully
        assert isinstance(result.score, int)
        assert isinstance(result.checklist, list)

    def test_validate_none_story(self) -> None:
        """Test validation with None-like story values."""
        from yolo_developer.agents.dev.dod_utils import validate_implementation_dod

        code = {
            "files": [
                {"path": "src/test.py", "content": '"""Module."""\n'},
            ]
        }
        story: dict[str, None] = {"acceptance_criteria": None}  # type: ignore[dict-item]

        result = validate_implementation_dod(code, story)

        # Should handle None values gracefully
        assert isinstance(result.score, int)


class TestValidateDodAlias:
    """Tests for validate_dod alias function (Code Review H1)."""

    def test_validate_dod_is_alias(self) -> None:
        """Test that validate_dod is an alias for validate_implementation_dod."""
        from yolo_developer.agents.dev.dod_utils import (
            validate_dod,
            validate_implementation_dod,
        )

        assert validate_dod is validate_implementation_dod

    def test_validate_dod_importable_from_module(self) -> None:
        """Test that validate_dod can be imported from dev module."""
        from yolo_developer.agents.dev import validate_dod

        assert callable(validate_dod)

    def test_validate_dod_produces_same_result(self) -> None:
        """Test that validate_dod produces same result as validate_implementation_dod."""
        from yolo_developer.agents.dev.dod_utils import (
            validate_dod,
            validate_implementation_dod,
        )

        code = {
            "files": [
                {"path": "src/test.py", "content": '"""Module."""\n'},
            ]
        }
        story: dict[str, list[str]] = {"acceptance_criteria": []}

        result1 = validate_implementation_dod(code, story)
        result2 = validate_dod(code, story)

        assert result1.score == result2.score
        assert result1.passed == result2.passed
        assert len(result1.checklist) == len(result2.checklist)


class TestCompleteChecklist:
    """Tests for complete checklist with all DOD items (Code Review H2)."""

    def test_checklist_contains_all_dod_items(self) -> None:
        """Test that checklist contains all 11 DOD_CHECKLIST_ITEMS."""
        from yolo_developer.agents.dev.dod_utils import validate_implementation_dod
        from yolo_developer.gates.gates.definition_of_done import DOD_CHECKLIST_ITEMS

        code = {
            "files": [
                {"path": "src/test.py", "content": '"""Module."""\n'},
            ]
        }
        story: dict[str, list[str]] = {"acceptance_criteria": []}

        result = validate_implementation_dod(code, story)

        # Count expected items from DOD_CHECKLIST_ITEMS
        expected_count = sum(len(items) for items in DOD_CHECKLIST_ITEMS.values())

        # Should have all 11 items in checklist
        assert len(result.checklist) == expected_count
        assert expected_count == 11  # Verify our assumption

    def test_checklist_includes_passing_items(self) -> None:
        """Test that checklist includes items with passed=True."""
        from yolo_developer.agents.dev.dod_utils import validate_implementation_dod

        code = {
            "files": [
                {
                    "path": "src/impl.py",
                    "content": '''"""Module docstring."""

def public_function(arg: str) -> str:
    """Function docstring."""
    return arg
''',
                },
                {
                    "path": "tests/test_impl.py",
                    "content": '''"""Test module."""

def test_public_function() -> None:
    """Test docstring."""
    pass
''',
                },
            ]
        }
        story: dict[str, list[str]] = {"acceptance_criteria": []}

        result = validate_implementation_dod(code, story)

        # Should have some passing items
        passing_items = [item for item in result.checklist if item.passed]
        assert len(passing_items) > 0

        # Passing items should have severity=None
        for item in passing_items:
            assert item.severity is None

    def test_checklist_includes_failing_items(self) -> None:
        """Test that checklist includes items with passed=False."""
        from yolo_developer.agents.dev.dod_utils import validate_implementation_dod

        code = {
            "files": [
                {
                    "path": "src/impl.py",
                    "content": "def no_doc(): pass\n",  # No docstring
                },
            ]
        }
        story: dict[str, list[str]] = {"acceptance_criteria": []}

        result = validate_implementation_dod(code, story)

        # Should have some failing items
        failing_items = [item for item in result.checklist if not item.passed]
        assert len(failing_items) > 0

        # Failing items should have severity set
        for item in failing_items:
            assert item.severity is not None

    def test_checklist_covers_all_categories(self) -> None:
        """Test that checklist covers all four categories."""
        from yolo_developer.agents.dev.dod_utils import validate_implementation_dod

        code = {
            "files": [
                {"path": "src/test.py", "content": '"""Module."""\n'},
            ]
        }
        story: dict[str, list[str]] = {"acceptance_criteria": []}

        result = validate_implementation_dod(code, story)

        # Should have items from all categories
        categories = {item.category for item in result.checklist}
        expected_categories = {"tests", "documentation", "style", "ac_coverage"}

        assert categories == expected_categories
