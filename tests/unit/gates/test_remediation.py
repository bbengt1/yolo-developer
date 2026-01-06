"""Unit tests for remediation suggestions (Story 3.8 - Task 3).

Tests the remediation registry and lookup functionality.
"""

from __future__ import annotations

import pytest

from yolo_developer.gates.remediation import (
    DEFAULT_REMEDIATION,
    GATE_SPECIFIC_REMEDIATION,
    get_remediation_suggestion,
)


class TestDefaultRemediation:
    """Tests for DEFAULT_REMEDIATION registry."""

    def test_contains_testability_suggestions(self) -> None:
        """Registry should contain testability gate issue types."""
        assert "vague_term" in DEFAULT_REMEDIATION
        assert "no_success_criteria" in DEFAULT_REMEDIATION

    def test_contains_ac_measurability_suggestions(self) -> None:
        """Registry should contain AC measurability gate issue types."""
        assert "unmeasurable_ac" in DEFAULT_REMEDIATION
        assert "missing_assertion" in DEFAULT_REMEDIATION

    def test_contains_architecture_suggestions(self) -> None:
        """Registry should contain architecture validation gate issue types."""
        assert "adr_violation" in DEFAULT_REMEDIATION
        assert "pattern_mismatch" in DEFAULT_REMEDIATION
        assert "missing_component" in DEFAULT_REMEDIATION

    def test_contains_dod_suggestions(self) -> None:
        """Registry should contain definition of done gate issue types."""
        assert "tests_missing" in DEFAULT_REMEDIATION
        assert "coverage_gap" in DEFAULT_REMEDIATION
        assert "documentation_missing" in DEFAULT_REMEDIATION

    def test_contains_confidence_suggestions(self) -> None:
        """Registry should contain confidence scoring gate issue types."""
        assert "low_gate_score" in DEFAULT_REMEDIATION
        assert "low_coverage" in DEFAULT_REMEDIATION
        assert "high_risk" in DEFAULT_REMEDIATION
        assert "low_documentation" in DEFAULT_REMEDIATION

    def test_suggestions_are_strings(self) -> None:
        """All suggestions should be non-empty strings."""
        for issue_type, suggestion in DEFAULT_REMEDIATION.items():
            assert isinstance(suggestion, str), f"{issue_type} suggestion not a string"
            assert len(suggestion) > 0, f"{issue_type} suggestion is empty"

    def test_vague_term_includes_example(self) -> None:
        """vague_term suggestion should include a concrete example."""
        suggestion = DEFAULT_REMEDIATION["vague_term"]
        assert "example" in suggestion.lower() or "instead of" in suggestion.lower()


class TestGateSpecificRemediation:
    """Tests for GATE_SPECIFIC_REMEDIATION registry."""

    def test_is_dict_of_dicts(self) -> None:
        """Registry should be a dict mapping gate names to suggestion dicts."""
        assert isinstance(GATE_SPECIFIC_REMEDIATION, dict)
        for gate_name, overrides in GATE_SPECIFIC_REMEDIATION.items():
            assert isinstance(gate_name, str)
            assert isinstance(overrides, dict)


class TestGetRemediationSuggestion:
    """Tests for get_remediation_suggestion function."""

    def test_returns_default_suggestion(self) -> None:
        """Should return default suggestion for known issue type."""
        result = get_remediation_suggestion("vague_term", "testability")

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_none_for_unknown_issue_type(self) -> None:
        """Should return None for unknown issue type."""
        result = get_remediation_suggestion("unknown_issue_type", "testability")

        assert result is None

    def test_returns_gate_specific_override_if_exists(self) -> None:
        """Should return gate-specific override when available."""
        # Add a temporary override for testing
        original = GATE_SPECIFIC_REMEDIATION.get("test_gate", {})
        GATE_SPECIFIC_REMEDIATION["test_gate"] = {"test_issue": "Gate-specific suggestion"}

        try:
            result = get_remediation_suggestion("test_issue", "test_gate")
            assert result == "Gate-specific suggestion"
        finally:
            # Restore original state
            if original:
                GATE_SPECIFIC_REMEDIATION["test_gate"] = original
            else:
                del GATE_SPECIFIC_REMEDIATION["test_gate"]

    def test_falls_back_to_default_when_no_gate_override(self) -> None:
        """Should fall back to default when no gate-specific override exists."""
        # Ensure no override exists for this combination
        result = get_remediation_suggestion("vague_term", "nonexistent_gate")

        assert result == DEFAULT_REMEDIATION["vague_term"]

    def test_different_gates_same_issue_type(self) -> None:
        """Same issue type should return same default for different gates."""
        result1 = get_remediation_suggestion("coverage_gap", "testability")
        result2 = get_remediation_suggestion("coverage_gap", "definition_of_done")

        # Both should get the default suggestion
        assert result1 == result2
        assert result1 == DEFAULT_REMEDIATION["coverage_gap"]

    @pytest.mark.parametrize(
        "issue_type",
        [
            "vague_term",
            "no_success_criteria",
            "unmeasurable_ac",
            "adr_violation",
            "tests_missing",
            "low_gate_score",
        ],
    )
    def test_all_common_issue_types_have_suggestions(self, issue_type: str) -> None:
        """All common issue types should have suggestions."""
        result = get_remediation_suggestion(issue_type, "any_gate")

        assert result is not None
        assert len(result) > 10  # Should be a meaningful suggestion
