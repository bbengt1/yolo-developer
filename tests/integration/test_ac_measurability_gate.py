"""Integration tests for AC measurability gate.

Tests the integration of the AC measurability gate evaluator with the
quality gate decorator framework.
"""

from __future__ import annotations

from typing import Any

import pytest

from yolo_developer.gates.decorator import quality_gate
from yolo_developer.gates.evaluators import get_evaluator, register_evaluator
from yolo_developer.gates.gates.ac_measurability import ac_measurability_evaluator
from yolo_developer.gates.types import GateContext


@pytest.fixture(autouse=True)
def ensure_ac_measurability_evaluator_registered() -> None:
    """Ensure ac_measurability evaluator is registered for each test.

    Other tests may call clear_evaluators(), so we need to
    re-register the ac_measurability evaluator before tests that need it.
    """
    if get_evaluator("ac_measurability") is None:
        register_evaluator("ac_measurability", ac_measurability_evaluator)


class TestGateDecoratorIntegration:
    """Tests for gate decorator integration with AC measurability evaluator."""

    @pytest.mark.asyncio
    async def test_decorated_function_passes_with_measurable_acs(self) -> None:
        """Decorated function executes when ACs are measurable."""
        executed = False

        @quality_gate("ac_measurability")
        async def process_stories(state: dict[str, Any]) -> str:
            nonlocal executed
            executed = True
            return "processed"

        state: dict[str, Any] = {
            "stories": [
                {
                    "id": "story-001",
                    "acceptance_criteria": [
                        {
                            "content": "Given a user, When they login, Then they are redirected to dashboard"
                        },
                    ],
                }
            ]
        }

        result = await process_stories(state)
        assert executed is True
        assert result == "processed"

    @pytest.mark.asyncio
    async def test_decorated_function_blocked_without_gwt_structure(self) -> None:
        """Decorated function is blocked when ACs lack GWT structure."""
        executed = False

        @quality_gate("ac_measurability")
        async def process_stories(state: dict[str, Any]) -> dict[str, Any]:
            nonlocal executed
            executed = True
            state["processed"] = True
            return state

        state: dict[str, Any] = {
            "stories": [
                {
                    "id": "story-001",
                    "acceptance_criteria": [
                        {"content": "The system should be user-friendly"},  # No GWT
                    ],
                }
            ]
        }

        result = await process_stories(state)

        # Function should not have executed
        assert executed is False
        # Gate should have blocked execution
        assert result.get("gate_blocked") is True
        # Failure message should mention the story and the issue
        assert "story-001" in str(result.get("gate_failure", ""))
        assert (
            "Given" in str(result.get("gate_failure", ""))
            or "When" in str(result.get("gate_failure", ""))
            or "Then" in str(result.get("gate_failure", ""))
        )

    @pytest.mark.asyncio
    async def test_decorated_function_passes_with_warnings(self) -> None:
        """Decorated function executes with warnings when GWT valid but subjective terms present."""
        executed = False

        @quality_gate("ac_measurability")
        async def process_stories(state: dict[str, Any]) -> str:
            nonlocal executed
            executed = True
            return "processed"

        state: dict[str, Any] = {
            "stories": [
                {
                    "id": "story-001",
                    "acceptance_criteria": [
                        {
                            "content": "Given a user, When they view page, Then the layout is clean"
                        },  # Has GWT + subjective
                    ],
                }
            ]
        }

        result = await process_stories(state)
        # Should execute because GWT is valid (subjective terms are warnings only)
        assert executed is True
        assert result == "processed"


class TestStateIntegration:
    """Tests for state reading from stories key (AC6)."""

    @pytest.mark.asyncio
    async def test_reads_stories_from_state(self) -> None:
        """Gate reads stories from state['stories'] key."""
        state: dict[str, Any] = {
            "stories": [
                {
                    "id": "story-001",
                    "title": "User Login",
                    "acceptance_criteria": [
                        {
                            "content": "Given valid credentials, When login submitted, Then user is redirected to home"
                        },
                    ],
                }
            ],
            "other_key": "should be ignored",
        }
        context = GateContext(state=state, gate_name="ac_measurability")

        result = await ac_measurability_evaluator(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_identifies_unmeasurable_acs_by_story_and_index(self) -> None:
        """Gate identifies unmeasurable ACs by story ID and AC index."""
        state: dict[str, Any] = {
            "stories": [
                {
                    "id": "story-001",
                    "acceptance_criteria": [
                        {"content": "Given a, When b, Then c succeeds"},  # Valid
                        {"content": "The interface should be intuitive"},  # Invalid - index 1
                    ],
                },
                {
                    "id": "story-002",
                    "acceptance_criteria": [
                        {"content": "System should work well"},  # Invalid - index 0
                    ],
                },
            ]
        }
        context = GateContext(state=state, gate_name="ac_measurability")

        result = await ac_measurability_evaluator(context)
        assert result.passed is False
        # Should identify specific stories
        assert "story-001" in result.reason
        assert "story-002" in result.reason


class TestGateBlockingBehavior:
    """Tests for gate blocking behavior with unmeasurable ACs."""

    @pytest.mark.asyncio
    async def test_blocking_with_missing_gwt(self) -> None:
        """Gate blocks handoff when GWT structure is missing."""
        state: dict[str, Any] = {
            "stories": [
                {
                    "id": "story-001",
                    "acceptance_criteria": [
                        {"content": "The button should work properly"},
                    ],
                }
            ]
        }
        context = GateContext(state=state, gate_name="ac_measurability")

        result = await ac_measurability_evaluator(context)
        assert result.passed is False
        assert "blocking" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_partial_blocking_multiple_stories(self) -> None:
        """Gate fails if any story has blocking issues."""
        state: dict[str, Any] = {
            "stories": [
                {
                    "id": "story-001",
                    "acceptance_criteria": [
                        {"content": "Given x, When y, Then z succeeds"},  # Valid
                    ],
                },
                {
                    "id": "story-002",
                    "acceptance_criteria": [
                        {"content": "Make the app better"},  # Invalid
                    ],
                },
            ]
        }
        context = GateContext(state=state, gate_name="ac_measurability")

        result = await ac_measurability_evaluator(context)
        assert result.passed is False
        # story-001 should not appear as failing
        assert "story-002" in result.reason


class TestGatePassingBehavior:
    """Tests for gate passing behavior with measurable ACs."""

    @pytest.mark.asyncio
    async def test_passing_with_complete_gwt(self) -> None:
        """Gate passes when all ACs have complete GWT structure."""
        state: dict[str, Any] = {
            "stories": [
                {
                    "id": "story-001",
                    "acceptance_criteria": [
                        {
                            "content": "Given a logged-in user, When they click logout, Then they are redirected to login"
                        },
                        {
                            "content": "Given invalid password, When login submitted, Then error 'Invalid credentials' appears"
                        },
                    ],
                },
                {
                    "id": "story-002",
                    "acceptance_criteria": [
                        {
                            "content": "Given a form with data, When submitted, Then record is created"
                        },
                    ],
                },
            ]
        }
        context = GateContext(state=state, gate_name="ac_measurability")

        result = await ac_measurability_evaluator(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_passing_with_empty_stories(self) -> None:
        """Gate passes with empty stories list."""
        state: dict[str, Any] = {"stories": []}
        context = GateContext(state=state, gate_name="ac_measurability")

        result = await ac_measurability_evaluator(context)
        assert result.passed is True


class TestAdvisoryMode:
    """Tests for advisory mode with warnings only."""

    @pytest.mark.asyncio
    async def test_advisory_mode_with_subjective_terms(self) -> None:
        """Gate passes with warnings when GWT present but subjective terms found."""
        state: dict[str, Any] = {
            "stories": [
                {
                    "id": "story-001",
                    "acceptance_criteria": [
                        {
                            "content": "Given a user, When they view page, Then the design is elegant"
                        },
                    ],
                }
            ]
        }
        context = GateContext(
            state=state,
            gate_name="ac_measurability",
        )

        result = await ac_measurability_evaluator(context)
        # Should pass with warnings since GWT structure is present
        assert result.passed is True
        assert result.reason is not None
        assert "warning" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_warnings_include_specific_terms(self) -> None:
        """Warnings identify specific subjective terms found."""
        state: dict[str, Any] = {
            "stories": [
                {
                    "id": "story-001",
                    "acceptance_criteria": [
                        {
                            "content": "Given a user, When they view, Then the layout is intuitive and user-friendly"
                        },
                    ],
                }
            ]
        }
        context = GateContext(state=state, gate_name="ac_measurability")

        result = await ac_measurability_evaluator(context)
        # Should pass with warnings
        assert result.passed is True
        assert result.reason is not None
        # Should mention the subjective terms
        assert "intuitive" in result.reason.lower() or "user-friendly" in result.reason.lower()
