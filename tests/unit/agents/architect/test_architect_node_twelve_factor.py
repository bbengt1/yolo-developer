"""Tests for architect_node integration with 12-Factor analysis (Story 7.2, Task 8, Task 12).

Tests verify the architect_node integrates with twelve-factor analysis.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


class TestArchitectNodeTwelveFactorIntegration:
    """Test architect_node includes twelve-factor analysis."""

    @pytest.mark.asyncio
    async def test_architect_node_calls_twelve_factor_analyzer(self) -> None:
        """Test that architect_node calls twelve-factor analyzer for stories."""
        from yolo_developer.agents.architect import architect_node
        from yolo_developer.agents.architect.types import TwelveFactorAnalysis

        state = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
            "pm_output": {
                "stories": [
                    {
                        "id": "story-001",
                        "title": "Configure database",
                        "description": "Setup PostgreSQL connection",
                    }
                ]
            },
        }

        mock_result = TwelveFactorAnalysis(
            factor_results={},
            applicable_factors=(),
            overall_compliance=1.0,
            recommendations=(),
        )

        with patch(
            "yolo_developer.agents.architect.node.analyze_twelve_factor",
            new_callable=AsyncMock,
        ) as mock_analyze:
            mock_analyze.return_value = mock_result

            await architect_node(state)

            # Twelve-factor analyzer should be called
            assert mock_analyze.called

    @pytest.mark.asyncio
    async def test_architect_output_includes_twelve_factor_analysis(self) -> None:
        """Test that architect_output includes twelve_factor_analysis."""
        from yolo_developer.agents.architect import architect_node

        state = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
            "pm_output": {
                "stories": [
                    {
                        "id": "story-001",
                        "title": "Test story",
                        "description": "Description",
                    }
                ]
            },
        }

        result = await architect_node(state)

        # Check architect_output contains twelve_factor_analyses
        assert "architect_output" in result
        output = result["architect_output"]
        assert "twelve_factor_analyses" in output

    @pytest.mark.asyncio
    async def test_design_decision_rationale_includes_twelve_factor(self) -> None:
        """Test that DesignDecision rationale includes 12-Factor compliance."""
        from yolo_developer.agents.architect import architect_node

        state = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
            "pm_output": {
                "stories": [
                    {
                        "id": "story-001",
                        "title": "Database config",
                        "description": "Connect to postgresql://localhost/db",
                    }
                ]
            },
        }

        result = await architect_node(state)

        # Check that design decisions mention 12-factor
        output = result["architect_output"]
        if output.get("design_decisions"):
            decision = output["design_decisions"][0]
            # Rationale should mention twelve-factor or compliance
            rationale = decision.get("rationale", "")
            assert "12-factor" in rationale.lower() or "twelve" in rationale.lower() or "compliance" in rationale.lower()


class TestDesignDecisionEnhancement:
    """Test _generate_design_decisions enhancement with 12-Factor."""

    @pytest.mark.asyncio
    async def test_generate_design_decisions_includes_twelve_factor(self) -> None:
        """Test _generate_design_decisions includes twelve-factor analysis."""
        from yolo_developer.agents.architect.node import _generate_design_decisions

        stories = [
            {
                "id": "story-001",
                "title": "Add caching",
                "description": "Use Redis for session caching",
            }
        ]

        decisions, _twelve_factor_analyses = await _generate_design_decisions(stories)

        assert len(decisions) > 0
        # Decision should have twelve_factor_compliance in rationale
        for decision in decisions:
            assert decision.rationale is not None
            # Rationale should mention 12-factor compliance
            assert "12-factor" in decision.rationale.lower() or "compliance" in decision.rationale.lower()


class TestArchitectOutputTwelveFactorField:
    """Test ArchitectOutput twelve_factor_analyses field."""

    def test_architect_output_has_twelve_factor_analyses_field(self) -> None:
        """Test that ArchitectOutput includes twelve_factor_analyses."""
        from yolo_developer.agents.architect.types import ArchitectOutput

        output = ArchitectOutput(
            design_decisions=(),
            adrs=(),
            processing_notes="Test",
            twelve_factor_analyses={},
        )

        assert hasattr(output, "twelve_factor_analyses")
        assert output.twelve_factor_analyses == {}

    def test_architect_output_to_dict_includes_twelve_factor(self) -> None:
        """Test to_dict includes twelve_factor_analyses."""
        from yolo_developer.agents.architect.types import ArchitectOutput

        output = ArchitectOutput(
            design_decisions=(),
            adrs=(),
            processing_notes="Test",
            twelve_factor_analyses={"story-001": {"overall_compliance": 0.8}},
        )

        d = output.to_dict()

        assert "twelve_factor_analyses" in d
        assert d["twelve_factor_analyses"] == {"story-001": {"overall_compliance": 0.8}}


class TestStateReturnsTwelveFactorResults:
    """Test architect_node returns twelve-factor results in state."""

    @pytest.mark.asyncio
    async def test_state_update_includes_twelve_factor(self) -> None:
        """Test that state update dict includes twelve-factor analysis."""
        from yolo_developer.agents.architect import architect_node

        state = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
            "pm_output": {
                "stories": [
                    {
                        "id": "story-001",
                        "title": "Test",
                        "description": "Test story",
                    }
                ]
            },
        }

        result = await architect_node(state)

        # architect_output should have twelve_factor_analyses
        assert "architect_output" in result
        assert "twelve_factor_analyses" in result["architect_output"]
