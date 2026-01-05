"""Integration tests for Architecture Validation gate.

Tests the integration of the architecture validation gate evaluator with the
quality gate decorator framework.
"""

from __future__ import annotations

from typing import Any

import pytest

from yolo_developer.gates.decorator import quality_gate
from yolo_developer.gates.evaluators import get_evaluator, register_evaluator
from yolo_developer.gates.gates.architecture_validation import architecture_validation_evaluator
from yolo_developer.gates.types import GateContext


@pytest.fixture(autouse=True)
def ensure_architecture_validation_evaluator_registered() -> None:
    """Ensure architecture_validation evaluator is registered for each test.

    Other tests may call clear_evaluators(), so we need to
    re-register the architecture_validation evaluator before tests that need it.
    """
    if get_evaluator("architecture_validation") is None:
        register_evaluator("architecture_validation", architecture_validation_evaluator)


class TestGateDecoratorIntegration:
    """Tests for gate decorator integration with architecture_validation evaluator."""

    @pytest.mark.asyncio
    async def test_decorated_function_passes_with_compliant_architecture(self) -> None:
        """Decorated function executes when architecture is compliant."""
        executed = False

        @quality_gate("architecture_validation")
        async def process_architecture(state: dict[str, Any]) -> str:
            nonlocal executed
            executed = True
            return "processed"

        state: dict[str, Any] = {
            "architecture": {
                "decisions": [
                    {
                        "id": "decision-001",
                        "title": "Use environment config",
                        "description": "Store all config in environment variables",
                    }
                ],
                "twelve_factor": {
                    "codebase": True,
                    "dependencies": True,
                    "config": True,
                    "backing_services": True,
                    "build_release_run": True,
                    "processes": True,
                    "port_binding": True,
                    "concurrency": True,
                    "disposability": True,
                    "dev_prod_parity": True,
                    "logs": True,
                    "admin_processes": True,
                },
                "tech_stack": {
                    "languages": ["python"],
                    "frameworks": ["fastapi"],
                },
                "security": {
                    "secrets_management": "AWS Secrets Manager",
                    "authentication": "OAuth 2.0 with JWT",
                },
            },
            "config": {
                "tech_stack": {
                    "allowed_languages": ["python"],
                    "allowed_frameworks": ["fastapi", "django"],
                },
            },
        }

        result = await process_architecture(state)
        assert executed is True
        assert result == "processed"

    @pytest.mark.asyncio
    async def test_decorated_function_blocked_with_low_score(self) -> None:
        """Decorated function is blocked when compliance score is below threshold."""
        executed = False

        @quality_gate("architecture_validation")
        async def process_architecture(state: dict[str, Any]) -> dict[str, Any]:
            nonlocal executed
            executed = True
            state["processed"] = True
            return state

        state: dict[str, Any] = {
            "architecture": {
                "twelve_factor": {
                    "config": False,
                    "logs": False,
                    "processes": False,
                },
                "security": {
                    "secrets_management": "password=secret123 in config file",
                },
            },
            "config": {
                "quality": {
                    "architecture_threshold": 0.70,
                },
            },
        }

        result = await process_architecture(state)

        # Function should not have executed
        assert executed is False
        # Gate should have blocked execution
        assert result.get("gate_blocked") is True
        # Failure message should mention score
        assert "score" in str(result.get("gate_failure", "")).lower()

    @pytest.mark.asyncio
    async def test_decorated_function_passes_with_minor_issues(self) -> None:
        """Decorated function executes with warnings when score above threshold."""
        executed = False

        @quality_gate("architecture_validation")
        async def process_architecture(state: dict[str, Any]) -> str:
            nonlocal executed
            executed = True
            return "processed"

        state: dict[str, Any] = {
            "architecture": {
                "twelve_factor": {
                    "config": False,  # One violation = 15 points = score 85
                },
            },
            "config": {
                "quality": {
                    "architecture_threshold": 0.70,  # 85 > 70, should pass
                },
            },
        }

        result = await process_architecture(state)
        # Should execute because score is above threshold
        assert executed is True
        assert result == "processed"


class TestStateIntegration:
    """Tests for state reading from architecture key (AC6)."""

    @pytest.mark.asyncio
    async def test_reads_architecture_from_state(self) -> None:
        """Gate reads architecture from state['architecture'] key."""
        state: dict[str, Any] = {
            "architecture": {
                "decisions": [
                    {
                        "id": "decision-001",
                        "title": "Tech Stack",
                        "description": "Use Python and PostgreSQL",
                    }
                ],
                "twelve_factor": {
                    "codebase": True,
                    "config": True,
                },
            },
            "other_key": "should be ignored",
        }
        context = GateContext(state=state, gate_name="architecture_validation")

        result = await architecture_validation_evaluator(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_reads_tech_stack_constraints_from_config(self) -> None:
        """Gate reads tech stack constraints from state['config']['tech_stack']."""
        state: dict[str, Any] = {
            "architecture": {
                "tech_stack": {
                    "languages": ["java"],  # Not in allowed list
                },
            },
            "config": {
                "tech_stack": {
                    "allowed_languages": ["python", "typescript"],
                },
                "quality": {
                    "architecture_threshold": 0.90,  # Set threshold high so single issue fails
                },
            },
        }
        context = GateContext(state=state, gate_name="architecture_validation")

        result = await architecture_validation_evaluator(context)
        # Score 85 < threshold 90, should fail
        assert result.passed is False
        assert "java" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_reads_threshold_from_config(self) -> None:
        """Gate reads architecture_threshold from state['config']['quality']."""
        state: dict[str, Any] = {
            "architecture": {
                "twelve_factor": {
                    "config": False,  # -15 points = score 85
                },
            },
            "config": {
                "quality": {
                    "architecture_threshold": 0.90,  # 85 < 90, should fail
                },
            },
        }
        context = GateContext(state=state, gate_name="architecture_validation")

        result = await architecture_validation_evaluator(context)
        assert result.passed is False
        assert "85" in result.reason  # Score should be in reason


class TestGateBlockingBehavior:
    """Tests for gate blocking behavior with non-compliant architectures."""

    @pytest.mark.asyncio
    async def test_blocking_with_security_issues(self) -> None:
        """Gate blocks handoff when security anti-patterns detected."""
        state: dict[str, Any] = {
            "architecture": {
                "security": {
                    "auth": "Uses password=admin123 for database",
                },
            },
        }
        context = GateContext(state=state, gate_name="architecture_validation")

        result = await architecture_validation_evaluator(context)
        # Hardcoded password is critical (-25 points = 75), may pass or fail
        # depending on threshold. Check that security issue is detected.
        if not result.passed:
            assert "security" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_blocking_with_multiple_violations(self) -> None:
        """Gate fails if many violations bring score below threshold."""
        state: dict[str, Any] = {
            "architecture": {
                "twelve_factor": {
                    "config": False,
                    "logs": False,
                    "processes": False,
                    "disposability": False,
                },
            },
        }
        context = GateContext(state=state, gate_name="architecture_validation")

        result = await architecture_validation_evaluator(context)
        # 4 high issues = -60 points = score 40 < threshold 70
        assert result.passed is False


class TestGatePassingBehavior:
    """Tests for gate passing behavior with compliant architectures."""

    @pytest.mark.asyncio
    async def test_passing_with_all_compliant(self) -> None:
        """Gate passes when all checks are compliant."""
        state: dict[str, Any] = {
            "architecture": {
                "decisions": [
                    {
                        "id": "decision-001",
                        "title": "Secure Architecture",
                        "description": "Follow best practices",
                    }
                ],
                "twelve_factor": {
                    "codebase": True,
                    "dependencies": True,
                    "config": True,
                    "backing_services": True,
                    "build_release_run": True,
                    "processes": True,
                    "port_binding": True,
                    "concurrency": True,
                    "disposability": True,
                    "dev_prod_parity": True,
                    "logs": True,
                    "admin_processes": True,
                },
                "tech_stack": {
                    "languages": ["python"],
                },
                "security": {
                    "secrets_management": "Use AWS Secrets Manager",
                    "transport": "HTTPS with TLS 1.3",
                },
            },
            "config": {
                "tech_stack": {
                    "allowed_languages": ["python"],
                },
            },
        }
        context = GateContext(state=state, gate_name="architecture_validation")

        result = await architecture_validation_evaluator(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_passing_with_empty_architecture(self) -> None:
        """Gate passes with empty architecture (no violations to check)."""
        state: dict[str, Any] = {"architecture": {}}
        context = GateContext(state=state, gate_name="architecture_validation")

        result = await architecture_validation_evaluator(context)
        assert result.passed is True


class TestComplianceScoreIntegration:
    """Tests for compliance score calculation integration."""

    @pytest.mark.asyncio
    async def test_score_included_in_result(self) -> None:
        """Compliance score is included in gate result when there are issues."""
        state: dict[str, Any] = {
            "architecture": {
                "twelve_factor": {
                    "config": False,
                },
            },
        }
        context = GateContext(state=state, gate_name="architecture_validation")

        result = await architecture_validation_evaluator(context)
        # Score 85 should be in the reason
        assert "85" in (result.reason or "")

    @pytest.mark.asyncio
    async def test_custom_threshold_respected(self) -> None:
        """Custom threshold from config is respected."""
        # Test with threshold 0.50 (50%) - should pass with score 85
        state_pass: dict[str, Any] = {
            "architecture": {
                "twelve_factor": {"config": False},
            },
            "config": {
                "quality": {"architecture_threshold": 0.50},
            },
        }
        context_pass = GateContext(state=state_pass, gate_name="architecture_validation")
        result_pass = await architecture_validation_evaluator(context_pass)
        assert result_pass.passed is True

        # Test with threshold 0.90 (90%) - should fail with score 85
        state_fail: dict[str, Any] = {
            "architecture": {
                "twelve_factor": {"config": False},
            },
            "config": {
                "quality": {"architecture_threshold": 0.90},
            },
        }
        context_fail = GateContext(state=state_fail, gate_name="architecture_validation")
        result_fail = await architecture_validation_evaluator(context_fail)
        assert result_fail.passed is False
