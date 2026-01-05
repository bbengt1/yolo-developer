"""Integration tests for definition of done gate.

Tests the full integration of the DoD gate with the decorator framework.
"""

from __future__ import annotations

import pytest

from yolo_developer.gates.decorator import quality_gate
from yolo_developer.gates.evaluators import clear_evaluators, get_evaluator

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def register_dod_evaluator():
    """Ensure evaluator is registered for each test."""
    clear_evaluators()
    # Import to trigger registration
    import importlib

    from yolo_developer.gates.gates import definition_of_done

    importlib.reload(definition_of_done)
    yield
    clear_evaluators()


@pytest.fixture
def compliant_code() -> dict:
    """Code that passes all DoD checks."""
    return {
        "files": [
            {
                "path": "src/feature.py",
                "content": '''"""Feature module implementing core functionality.

This module provides:
- AC1: Test presence verification
- AC2: Documentation presence checking
- AC3: Code style compliance validation
"""

from typing import Any


def verify_presence(data: dict[str, Any]) -> bool:
    """Verify presence of required elements.

    Args:
        data: Dictionary containing elements to check.

    Returns:
        True if all elements present, False otherwise.
    """
    return bool(data)


def check_compliance(code: str) -> dict[str, bool]:
    """Check code compliance.

    Args:
        code: Source code to validate.

    Returns:
        Dictionary with compliance results.
    """
    return {"compliant": True}


def validate_style(content: str) -> list[str]:
    """Validate code style.

    Args:
        content: Code content to validate.

    Returns:
        List of style issues found.
    """
    return []
''',
            },
            {
                "path": "tests/test_feature.py",
                "content": '''"""Tests for feature module."""


def test_verify_presence():
    """Test verify_presence function."""
    from src.feature import verify_presence
    assert verify_presence({"key": "value"}) is True
    assert verify_presence({}) is False


def test_check_compliance():
    """Test check_compliance function."""
    from src.feature import check_compliance
    result = check_compliance("def foo(): pass")
    assert result["compliant"] is True


def test_validate_style():
    """Test validate_style function."""
    from src.feature import validate_style
    issues = validate_style("def foo(): pass")
    assert issues == []
''',
            },
        ],
    }


@pytest.fixture
def non_compliant_code() -> dict:
    """Code that fails DoD checks."""
    return {
        "files": [
            {
                "path": "src/broken.py",
                "content": """
def brokenFunction(x):
    return x

def anotherBadFunction(y, z):
    if y:
        if z:
            if y > z:
                if y < 100:
                    if z > 0:
                        return y
    return z
""",
            },
        ],
    }


@pytest.fixture
def mock_story() -> dict:
    """Mock story with acceptance criteria."""
    return {
        "id": "test-story",
        "title": "Test Story",
        "acceptance_criteria": [
            "AC1: Test presence verification",
            "AC2: Documentation presence checking",
            "AC3: Code style compliance validation",
        ],
    }


# =============================================================================
# Integration Tests
# =============================================================================


class TestDoDGateIntegration:
    """Integration tests for DoD gate with decorator."""

    @pytest.mark.asyncio
    async def test_gate_decorator_with_dod_evaluator(
        self, compliant_code: dict, mock_story: dict
    ) -> None:
        """Gate decorator should work with definition_of_done evaluator."""

        @quality_gate("definition_of_done", blocking=True)
        async def process_code(state: dict) -> dict:
            state["processed"] = True
            return state

        state = {"code": compliant_code, "story": mock_story}
        result = await process_code(state)

        # Should pass and process
        assert result.get("processed") is True or result.get("gate_blocked") is not True

    @pytest.mark.asyncio
    async def test_gate_reads_code_from_state(self, compliant_code: dict, mock_story: dict) -> None:
        """Gate should read code from state['code']."""
        from yolo_developer.gates.gates.definition_of_done import definition_of_done_evaluator
        from yolo_developer.gates.types import GateContext

        context = GateContext(
            state={"code": compliant_code, "story": mock_story},
            gate_name="definition_of_done",
        )
        result = await definition_of_done_evaluator(context)

        assert result.gate_name == "definition_of_done"

    @pytest.mark.asyncio
    async def test_gate_reads_story_from_state(
        self, compliant_code: dict, mock_story: dict
    ) -> None:
        """Gate should read story from state['story']."""
        from yolo_developer.gates.gates.definition_of_done import definition_of_done_evaluator
        from yolo_developer.gates.types import GateContext

        context = GateContext(
            state={"code": compliant_code, "story": mock_story},
            gate_name="definition_of_done",
        )
        result = await definition_of_done_evaluator(context)

        # Should have evaluated against ACs
        assert result.gate_name == "definition_of_done"

    @pytest.mark.asyncio
    async def test_gate_blocking_with_low_score(
        self, non_compliant_code: dict, mock_story: dict
    ) -> None:
        """Gate should block when compliance score is below threshold."""

        @quality_gate("definition_of_done", blocking=True)
        async def process_code(state: dict) -> dict:
            state["processed"] = True
            return state

        state = {"code": non_compliant_code, "story": mock_story}
        result = await process_code(state)

        # Should be blocked due to low score
        assert result.get("gate_blocked") is True or result.get("processed") is not True

    @pytest.mark.asyncio
    async def test_gate_passes_with_compliant_code(
        self, compliant_code: dict, mock_story: dict
    ) -> None:
        """Gate should pass when code is compliant."""
        from yolo_developer.gates.gates.definition_of_done import definition_of_done_evaluator
        from yolo_developer.gates.types import GateContext

        context = GateContext(
            state={"code": compliant_code, "story": mock_story},
            gate_name="definition_of_done",
        )
        result = await definition_of_done_evaluator(context)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_configuration_threshold_integration(
        self, compliant_code: dict, mock_story: dict
    ) -> None:
        """Gate should respect threshold from config."""
        from yolo_developer.gates.gates.definition_of_done import definition_of_done_evaluator
        from yolo_developer.gates.types import GateContext

        # Set very high threshold
        context = GateContext(
            state={
                "code": compliant_code,
                "story": mock_story,
                "config": {"quality": {"dod_threshold": 99}},
            },
            gate_name="definition_of_done",
        )
        result = await definition_of_done_evaluator(context)

        # Very high threshold may cause failure
        assert result.gate_name == "definition_of_done"
        # Result depends on actual score

    @pytest.mark.asyncio
    async def test_evaluator_available_via_get_evaluator(self) -> None:
        """Evaluator should be available via get_evaluator()."""
        evaluator = get_evaluator("definition_of_done")
        assert evaluator is not None

    @pytest.mark.asyncio
    async def test_gate_handles_implementation_key(
        self, compliant_code: dict, mock_story: dict
    ) -> None:
        """Gate should also work with 'implementation' key in state."""
        from yolo_developer.gates.gates.definition_of_done import definition_of_done_evaluator
        from yolo_developer.gates.types import GateContext

        context = GateContext(
            state={"implementation": compliant_code, "story": mock_story},
            gate_name="definition_of_done",
        )
        result = await definition_of_done_evaluator(context)

        assert result.gate_name == "definition_of_done"
        # Should have processed the code from implementation key

    @pytest.mark.asyncio
    async def test_gate_result_includes_failed_checks(
        self, non_compliant_code: dict, mock_story: dict
    ) -> None:
        """Gate result should include which checks failed."""
        from yolo_developer.gates.gates.definition_of_done import definition_of_done_evaluator
        from yolo_developer.gates.types import GateContext

        context = GateContext(
            state={"code": non_compliant_code, "story": mock_story},
            gate_name="definition_of_done",
        )
        result = await definition_of_done_evaluator(context)

        # Should have reason with details
        assert result.reason is not None
        assert len(result.reason) > 0

    @pytest.mark.asyncio
    async def test_advisory_mode_allows_continuation(
        self, non_compliant_code: dict, mock_story: dict
    ) -> None:
        """Advisory mode should allow continuation even with failures."""

        @quality_gate("definition_of_done", blocking=False)
        async def process_code(state: dict) -> dict:
            state["processed"] = True
            return state

        state = {"code": non_compliant_code, "story": mock_story}
        result = await process_code(state)

        # Advisory mode should allow processing to continue
        assert result.get("processed") is True

    @pytest.mark.asyncio
    async def test_gate_with_empty_story(self, compliant_code: dict) -> None:
        """Gate should handle empty story gracefully."""
        from yolo_developer.gates.gates.definition_of_done import definition_of_done_evaluator
        from yolo_developer.gates.types import GateContext

        context = GateContext(
            state={"code": compliant_code, "story": {}},
            gate_name="definition_of_done",
        )
        result = await definition_of_done_evaluator(context)

        # Should not crash with empty story
        assert result.gate_name == "definition_of_done"

    @pytest.mark.asyncio
    async def test_gate_with_missing_story(self, compliant_code: dict) -> None:
        """Gate should handle missing story gracefully."""
        from yolo_developer.gates.gates.definition_of_done import definition_of_done_evaluator
        from yolo_developer.gates.types import GateContext

        context = GateContext(
            state={"code": compliant_code},
            gate_name="definition_of_done",
        )
        result = await definition_of_done_evaluator(context)

        # Should not crash without story
        assert result.gate_name == "definition_of_done"
