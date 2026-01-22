"""Unit tests for tech stack validator (Story 7.6 - Tasks 2-5, 7, 10, 11).

Tests for config extraction, technology validation, version compatibility,
pattern suggestion, and main validation function.

AC Coverage:
    - AC1: validate_tech_stack_constraints() is importable from architect
    - AC2: Version compatibility verification
    - AC3: Stack-specific pattern suggestions
    - AC4: Constraint violations flagged with severity
    - AC5: Integration with YoloConfig
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yolo_developer.config import LLM_CHEAP_MODEL_DEFAULT
from yolo_developer.agents.architect import DesignDecision

# RED phase: These imports will fail until we implement the module
from yolo_developer.agents.architect.tech_stack_validator import (
    _check_version_compatibility,
    _extract_tech_stack_from_config,
    _suggest_stack_patterns,
    _validate_technology_choices,
    validate_tech_stack_constraints,
)
from yolo_developer.agents.architect.types import (
    TechStackValidation,
)


class TestExtractTechStackFromConfig:
    """Tests for _extract_tech_stack_from_config (Task 2)."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock YoloConfig."""
        config = MagicMock()
        config.llm.cheap_model = LLM_CHEAP_MODEL_DEFAULT
        config.llm.premium_model = "claude-sonnet-4-20250514"
        config.memory.vector_store_type = "chromadb"
        config.memory.graph_store_type = "json"
        return config

    def test_extract_llm_provider_from_config(self, mock_config: MagicMock) -> None:
        """Test extracting LLM provider information."""
        with patch(
            "yolo_developer.agents.architect.tech_stack_validator.load_config",
            return_value=mock_config,
        ):
            result = _extract_tech_stack_from_config()
            assert "llm_models" in result
            assert LLM_CHEAP_MODEL_DEFAULT in result["llm_models"]

    def test_extract_memory_store_types(self, mock_config: MagicMock) -> None:
        """Test extracting memory store type information."""
        with patch(
            "yolo_developer.agents.architect.tech_stack_validator.load_config",
            return_value=mock_config,
        ):
            result = _extract_tech_stack_from_config()
            assert "vector_store" in result
            assert result["vector_store"] == "chromadb"

    def test_extract_returns_dict(self, mock_config: MagicMock) -> None:
        """Test that extraction returns a dictionary."""
        with patch(
            "yolo_developer.agents.architect.tech_stack_validator.load_config",
            return_value=mock_config,
        ):
            result = _extract_tech_stack_from_config()
            assert isinstance(result, dict)


class TestValidateTechnologyChoices:
    """Tests for _validate_technology_choices (Task 3)."""

    @pytest.fixture
    def sample_tech_stack(self) -> dict[str, str | list[str]]:
        """Sample tech stack configuration."""
        return {
            "runtime": "Python 3.10+",
            "database": "chromadb",
            "testing": "pytest",
            "tooling": ["uv", "ruff", "mypy"],
        }

    @pytest.fixture
    def compliant_decision(self) -> DesignDecision:
        """A design decision that complies with the tech stack."""
        return DesignDecision(
            id="design-001",
            story_id="story-001",
            decision_type="technology",
            description="Use ChromaDB for vector storage",
            rationale="Configured vector store",
        )

    @pytest.fixture
    def non_compliant_decision(self) -> DesignDecision:
        """A design decision that violates the tech stack."""
        return DesignDecision(
            id="design-002",
            story_id="story-001",
            decision_type="technology",
            description="Use SQLite for data storage",
            rationale="Simple embedded database",
        )

    def test_compliant_decision_no_violations(
        self,
        sample_tech_stack: dict[str, str | list[str]],
        compliant_decision: DesignDecision,
    ) -> None:
        """Test that compliant decisions produce no violations."""
        violations = _validate_technology_choices([compliant_decision], sample_tech_stack)
        assert len(violations) == 0

    def test_non_compliant_decision_flagged(
        self,
        sample_tech_stack: dict[str, str | list[str]],
        non_compliant_decision: DesignDecision,
    ) -> None:
        """Test that non-compliant decisions produce violations."""
        violations = _validate_technology_choices([non_compliant_decision], sample_tech_stack)
        assert len(violations) > 0
        # Check case-insensitively since technology names are title-cased
        assert any(v.technology.lower() == "sqlite" for v in violations)

    def test_violation_includes_severity(
        self,
        sample_tech_stack: dict[str, str | list[str]],
        non_compliant_decision: DesignDecision,
    ) -> None:
        """Test that violations include severity level."""
        violations = _validate_technology_choices([non_compliant_decision], sample_tech_stack)
        assert len(violations) > 0
        assert violations[0].severity in ("critical", "high", "medium", "low")

    def test_violation_includes_suggested_alternative(
        self,
        sample_tech_stack: dict[str, str | list[str]],
        non_compliant_decision: DesignDecision,
    ) -> None:
        """Test that violations include suggested alternative."""
        violations = _validate_technology_choices([non_compliant_decision], sample_tech_stack)
        assert len(violations) > 0
        assert violations[0].suggested_alternative


class TestCheckVersionCompatibility:
    """Tests for _check_version_compatibility (Task 4)."""

    def test_compatible_versions_no_violation(self) -> None:
        """Test compatible versions produce no violation."""
        result = _check_version_compatibility("Python", "3.10", "3.10")
        assert result is None

    def test_major_version_mismatch_critical(self) -> None:
        """Test major version mismatch is critical."""
        result = _check_version_compatibility("Python", "3.10", "2.7")
        assert result is not None
        assert result.severity == "critical"

    def test_minor_version_mismatch_medium(self) -> None:
        """Test minor version mismatch is medium severity."""
        result = _check_version_compatibility("Python", "3.10", "3.8")
        assert result is not None
        assert result.severity in ("medium", "high")

    def test_version_compatibility_includes_expected_and_actual(self) -> None:
        """Test violation includes expected and actual versions."""
        result = _check_version_compatibility("Python", "3.10", "3.8")
        assert result is not None
        assert result.expected_version == "3.10"
        assert result.actual_version == "3.8"

    def test_version_prefix_handling(self) -> None:
        """Test that version prefixes like 'v' are handled correctly."""
        # v3.10 should match 3.10
        result = _check_version_compatibility("Python", "v3.10", "3.10")
        assert result is None

    def test_version_prefix_both_have_v(self) -> None:
        """Test both versions with v prefix."""
        result = _check_version_compatibility("Python", "v3.10", "v3.10")
        assert result is None

    def test_version_plus_suffix_handling(self) -> None:
        """Test that trailing + is handled correctly."""
        result = _check_version_compatibility("Python", "3.10+", "3.10")
        assert result is None


class TestSuggestStackPatterns:
    """Tests for _suggest_stack_patterns (Task 5)."""

    @pytest.fixture
    def python_tech_stack(self) -> dict[str, str | list[str]]:
        """Python tech stack configuration."""
        return {
            "runtime": "Python 3.10+",
            "testing": "pytest",
            "tooling": ["uv", "ruff", "mypy"],
        }

    def test_suggests_pytest_patterns(self, python_tech_stack: dict[str, str | list[str]]) -> None:
        """Test that pytest patterns are suggested for Python stack."""
        patterns = _suggest_stack_patterns(python_tech_stack, [])
        pattern_names = [p.pattern_name for p in patterns]
        assert any("pytest" in name.lower() for name in pattern_names)

    def test_suggests_uv_patterns(self, python_tech_stack: dict[str, str | list[str]]) -> None:
        """Test that uv patterns are suggested for Python stack."""
        patterns = _suggest_stack_patterns(python_tech_stack, [])
        pattern_names = [p.pattern_name for p in patterns]
        assert any("uv" in name.lower() for name in pattern_names)

    def test_patterns_include_rationale(
        self, python_tech_stack: dict[str, str | list[str]]
    ) -> None:
        """Test that patterns include rationale tied to tech stack."""
        patterns = _suggest_stack_patterns(python_tech_stack, [])
        assert len(patterns) > 0
        for pattern in patterns:
            assert pattern.rationale
            assert len(pattern.rationale) > 0

    def test_patterns_include_applicable_technologies(
        self, python_tech_stack: dict[str, str | list[str]]
    ) -> None:
        """Test that patterns include applicable technologies."""
        patterns = _suggest_stack_patterns(python_tech_stack, [])
        assert len(patterns) > 0
        for pattern in patterns:
            assert pattern.applicable_technologies
            assert len(pattern.applicable_technologies) > 0

    def test_empty_stack_no_patterns(self) -> None:
        """Test that empty stack produces no patterns."""
        patterns = _suggest_stack_patterns({}, [])
        assert len(patterns) == 0


class TestValidateTechStackConstraints:
    """Tests for validate_tech_stack_constraints (Task 7)."""

    @pytest.fixture
    def sample_decisions(self) -> list[DesignDecision]:
        """Sample design decisions for validation."""
        return [
            DesignDecision(
                id="design-001",
                story_id="story-001",
                decision_type="technology",
                description="Use ChromaDB for vector storage",
                rationale="Configured vector store",
            ),
        ]

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock YoloConfig with proper string values."""
        config = MagicMock()
        config.project_name = "test-project"
        config.llm.cheap_model = LLM_CHEAP_MODEL_DEFAULT
        config.llm.premium_model = "claude-sonnet-4"
        config.llm.best_model = "claude-opus-4"
        config.memory.vector_store_type = "chromadb"
        config.memory.graph_store_type = "json"
        return config

    @pytest.mark.asyncio
    async def test_returns_tech_stack_validation(
        self, sample_decisions: list[DesignDecision], mock_config: MagicMock
    ) -> None:
        """Test that validation returns TechStackValidation."""
        with patch(
            "yolo_developer.agents.architect.tech_stack_validator.load_config",
            return_value=mock_config,
        ):
            result = await validate_tech_stack_constraints(sample_decisions, use_llm=False)
            assert isinstance(result, TechStackValidation)

    @pytest.mark.asyncio
    async def test_validation_includes_compliance_status(
        self, sample_decisions: list[DesignDecision], mock_config: MagicMock
    ) -> None:
        """Test that validation includes overall_compliance field."""
        with patch(
            "yolo_developer.agents.architect.tech_stack_validator.load_config",
            return_value=mock_config,
        ):
            result = await validate_tech_stack_constraints(sample_decisions, use_llm=False)
            assert isinstance(result.overall_compliance, bool)

    @pytest.mark.asyncio
    async def test_validation_includes_summary(
        self, sample_decisions: list[DesignDecision], mock_config: MagicMock
    ) -> None:
        """Test that validation includes summary."""
        with patch(
            "yolo_developer.agents.architect.tech_stack_validator.load_config",
            return_value=mock_config,
        ):
            result = await validate_tech_stack_constraints(sample_decisions, use_llm=False)
            assert isinstance(result.summary, str)
            assert len(result.summary) > 0

    @pytest.mark.asyncio
    async def test_critical_violation_sets_non_compliant(self, mock_config: MagicMock) -> None:
        """Test that critical violation sets overall_compliance to False."""
        decisions = [
            DesignDecision(
                id="design-001",
                story_id="story-001",
                decision_type="technology",
                description="Use SQLite for storage",
                rationale="Simple database",
            ),
        ]
        with patch(
            "yolo_developer.agents.architect.tech_stack_validator.load_config",
            return_value=mock_config,
        ):
            result = await validate_tech_stack_constraints(decisions, use_llm=False)
            # If SQLite is flagged as critical, overall_compliance should be False
            if any(v.severity == "critical" for v in result.violations):
                assert result.overall_compliance is False

    @pytest.mark.asyncio
    async def test_to_dict_serialization(
        self, sample_decisions: list[DesignDecision], mock_config: MagicMock
    ) -> None:
        """Test that result can be serialized."""
        with patch(
            "yolo_developer.agents.architect.tech_stack_validator.load_config",
            return_value=mock_config,
        ):
            result = await validate_tech_stack_constraints(sample_decisions, use_llm=False)
            serialized = result.to_dict()
            assert isinstance(serialized, dict)
            assert "overall_compliance" in serialized
            assert "violations" in serialized
            assert "suggested_patterns" in serialized
            assert "summary" in serialized


class TestLLMIntegration:
    """Tests for LLM-powered tech stack analysis (Task 12)."""

    @pytest.fixture
    def sample_tech_stack(self) -> dict[str, str | list[str]]:
        """Sample tech stack configuration."""
        return {
            "runtime": "Python 3.10+",
            "testing": "pytest",
            "vector_store": "chromadb",
            "tooling": ["uv", "ruff", "mypy"],
            "frameworks": ["langgraph", "pydantic"],
            "libraries": ["litellm", "structlog"],
        }

    @pytest.fixture
    def sample_decisions(self) -> list[DesignDecision]:
        """Sample design decisions for validation."""
        return [
            DesignDecision(
                id="design-001",
                story_id="story-001",
                decision_type="technology",
                description="Use ChromaDB for vector storage",
                rationale="Configured vector store",
            ),
        ]

    @pytest.mark.asyncio
    async def test_llm_analysis_with_mocked_llm(
        self, sample_decisions: list[DesignDecision]
    ) -> None:
        """Test LLM analysis with mocked LLM response."""
        from yolo_developer.agents.architect.tech_stack_validator import (
            _analyze_tech_stack_with_llm,
        )

        mock_response = """```json
{
  "overall_compliance": true,
  "violations": [],
  "suggested_patterns": [
    {
      "pattern_name": "pytest-fixtures",
      "description": "Use pytest fixtures",
      "rationale": "pytest configured",
      "applicable_technologies": ["pytest"]
    }
  ],
  "summary": "All compliant"
}
```"""

        with patch(
            "yolo_developer.agents.architect.tech_stack_validator._call_tech_stack_llm",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await _analyze_tech_stack_with_llm({"testing": "pytest"}, sample_decisions)

            assert result is not None
            assert result.overall_compliance is True
            assert len(result.violations) == 0
            assert len(result.suggested_patterns) == 1

    @pytest.mark.asyncio
    async def test_llm_analysis_parses_violations(
        self, sample_decisions: list[DesignDecision]
    ) -> None:
        """Test that LLM response violations are properly parsed."""
        from yolo_developer.agents.architect.tech_stack_validator import (
            _analyze_tech_stack_with_llm,
        )

        mock_response = """```json
{
  "overall_compliance": false,
  "violations": [
    {
      "technology": "SQLite",
      "expected_version": null,
      "actual_version": "3.x",
      "severity": "critical",
      "suggested_alternative": "Use ChromaDB"
    }
  ],
  "suggested_patterns": [],
  "summary": "1 violation found"
}
```"""

        with patch(
            "yolo_developer.agents.architect.tech_stack_validator._call_tech_stack_llm",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await _analyze_tech_stack_with_llm(
                {"vector_store": "chromadb"}, sample_decisions
            )

            assert result is not None
            assert result.overall_compliance is False
            assert len(result.violations) == 1
            assert result.violations[0].technology == "SQLite"
            assert result.violations[0].severity == "critical"

    @pytest.mark.asyncio
    async def test_llm_analysis_returns_none_on_json_error(
        self, sample_decisions: list[DesignDecision]
    ) -> None:
        """Test that LLM analysis returns None on JSON parse error."""
        from yolo_developer.agents.architect.tech_stack_validator import (
            _analyze_tech_stack_with_llm,
        )

        with patch(
            "yolo_developer.agents.architect.tech_stack_validator._call_tech_stack_llm",
            new_callable=AsyncMock,
            return_value="not valid json",
        ):
            result = await _analyze_tech_stack_with_llm({"testing": "pytest"}, sample_decisions)

            assert result is None

    @pytest.mark.asyncio
    async def test_llm_analysis_returns_none_on_exception(
        self, sample_decisions: list[DesignDecision]
    ) -> None:
        """Test that LLM analysis returns None on exception."""
        from yolo_developer.agents.architect.tech_stack_validator import (
            _analyze_tech_stack_with_llm,
        )

        with patch(
            "yolo_developer.agents.architect.tech_stack_validator._call_tech_stack_llm",
            new_callable=AsyncMock,
            side_effect=Exception("LLM error"),
        ):
            result = await _analyze_tech_stack_with_llm({"testing": "pytest"}, sample_decisions)

            assert result is None

    @pytest.mark.asyncio
    async def test_fallback_to_rule_based_on_llm_failure(
        self, sample_decisions: list[DesignDecision]
    ) -> None:
        """Test fallback to rule-based validation when LLM fails."""
        mock_config = MagicMock()
        mock_config.project_name = "test-project"
        mock_config.llm.cheap_model = LLM_CHEAP_MODEL_DEFAULT
        mock_config.llm.premium_model = "claude-sonnet-4"
        mock_config.llm.best_model = "claude-opus-4"
        mock_config.memory.vector_store_type = "chromadb"
        mock_config.memory.graph_store_type = "json"

        with (
            patch(
                "yolo_developer.agents.architect.tech_stack_validator.load_config",
                return_value=mock_config,
            ),
            patch(
                "yolo_developer.agents.architect.tech_stack_validator._analyze_tech_stack_with_llm",
                new_callable=AsyncMock,
                return_value=None,  # LLM fails
            ),
        ):
            result = await validate_tech_stack_constraints(sample_decisions, use_llm=True)

            # Should still return a valid result using rule-based validation
            assert isinstance(result, TechStackValidation)
            assert isinstance(result.overall_compliance, bool)

    def test_call_tech_stack_llm_has_retry_decorator(self) -> None:
        """Test that _call_tech_stack_llm has retry decorator."""
        from yolo_developer.agents.architect.tech_stack_validator import (
            _call_tech_stack_llm,
        )

        # Check for retry decorator attributes
        assert hasattr(_call_tech_stack_llm, "retry")

    @pytest.mark.asyncio
    async def test_validate_with_llm_disabled(self, sample_decisions: list[DesignDecision]) -> None:
        """Test validation with LLM explicitly disabled."""
        mock_config = MagicMock()
        mock_config.project_name = "test-project"
        mock_config.llm.cheap_model = LLM_CHEAP_MODEL_DEFAULT
        mock_config.llm.premium_model = "claude-sonnet-4"
        mock_config.llm.best_model = "claude-opus-4"
        mock_config.memory.vector_store_type = "chromadb"
        mock_config.memory.graph_store_type = "json"

        with (
            patch(
                "yolo_developer.agents.architect.tech_stack_validator.load_config",
                return_value=mock_config,
            ),
            patch(
                "yolo_developer.agents.architect.tech_stack_validator._analyze_tech_stack_with_llm",
                new_callable=AsyncMock,
            ) as mock_llm,
        ):
            await validate_tech_stack_constraints(sample_decisions, use_llm=False)

            # LLM should not be called when disabled
            mock_llm.assert_not_called()


class TestArchitectNodeIntegration:
    """Integration tests for architect_node with tech stack validation (Task 13)."""

    @pytest.fixture
    def sample_state(self) -> dict[str, object]:
        """Sample orchestration state with stories."""
        return {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
            "pm_output": {
                "stories": [
                    {
                        "id": "story-001",
                        "title": "Test Story",
                        "description": "Test description",
                    }
                ]
            },
        }

    @pytest.mark.asyncio
    async def test_architect_node_includes_tech_stack_validation(
        self, sample_state: dict[str, object]
    ) -> None:
        """Test architect_node includes tech_stack_validations in output."""
        from yolo_developer.agents.architect import architect_node

        mock_validation = TechStackValidation(
            overall_compliance=True,
            violations=(),
            suggested_patterns=(),
            summary="All compliant",
        )

        with patch(
            "yolo_developer.agents.architect.node.validate_tech_stack_constraints",
            new_callable=AsyncMock,
            return_value=mock_validation,
        ):
            result = await architect_node(sample_state)

            assert "architect_output" in result
            output = result["architect_output"]
            assert "tech_stack_validations" in output

    @pytest.mark.asyncio
    async def test_architect_output_serializes_tech_stack_validation(
        self, sample_state: dict[str, object]
    ) -> None:
        """Test that tech_stack_validations are properly serialized."""
        from yolo_developer.agents.architect import architect_node

        mock_validation = TechStackValidation(
            overall_compliance=True,
            violations=(),
            suggested_patterns=(),
            summary="All compliant",
        )

        with patch(
            "yolo_developer.agents.architect.node.validate_tech_stack_constraints",
            new_callable=AsyncMock,
            return_value=mock_validation,
        ):
            result = await architect_node(sample_state)

            output = result["architect_output"]
            validations = output["tech_stack_validations"]

            # Should have validation for story-001
            assert "story-001" in validations
            story_validation = validations["story-001"]
            assert story_validation["overall_compliance"] is True
            assert story_validation["violations"] == []
            assert story_validation["summary"] == "All compliant"

    @pytest.mark.asyncio
    async def test_architect_output_to_dict_includes_tech_stack(self) -> None:
        """Test ArchitectOutput.to_dict includes tech_stack_validations."""
        from yolo_developer.agents.architect.types import ArchitectOutput

        output = ArchitectOutput(
            design_decisions=(),
            adrs=(),
            processing_notes="Test",
            tech_stack_validations={"story-001": {"overall_compliance": True}},
        )

        d = output.to_dict()

        assert "tech_stack_validations" in d
        assert d["tech_stack_validations"] == {"story-001": {"overall_compliance": True}}

    @pytest.mark.asyncio
    async def test_tech_stack_validation_runs_after_risk_identification(
        self, sample_state: dict[str, object]
    ) -> None:
        """Test tech stack validation runs after risk identification (AC5)."""
        from yolo_developer.agents.architect import architect_node

        call_order: list[str] = []

        async def mock_risk_identifier(*args: object, **kwargs: object) -> object:
            call_order.append("risk")
            from yolo_developer.agents.architect.types import TechnicalRiskReport

            return TechnicalRiskReport(
                risks=(),
                overall_risk_level="low",
                summary="No risks",
            )

        async def mock_tech_stack_validator(*args: object, **kwargs: object) -> TechStackValidation:
            call_order.append("tech_stack")
            return TechStackValidation(
                overall_compliance=True,
                violations=(),
                suggested_patterns=(),
                summary="All compliant",
            )

        with (
            patch(
                "yolo_developer.agents.architect.node.identify_technical_risks",
                new_callable=AsyncMock,
                side_effect=mock_risk_identifier,
            ),
            patch(
                "yolo_developer.agents.architect.node.validate_tech_stack_constraints",
                new_callable=AsyncMock,
                side_effect=mock_tech_stack_validator,
            ),
        ):
            await architect_node(sample_state)

            # Tech stack validation should come after risk identification
            assert "risk" in call_order
            assert "tech_stack" in call_order
            assert call_order.index("risk") < call_order.index("tech_stack")

    @pytest.mark.asyncio
    async def test_validate_tech_stack_constraints_importable_from_architect(self) -> None:
        """Test that validate_tech_stack_constraints is importable from architect (AC1)."""
        from yolo_developer.agents.architect import validate_tech_stack_constraints

        assert callable(validate_tech_stack_constraints)
