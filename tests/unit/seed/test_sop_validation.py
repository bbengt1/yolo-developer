"""Unit tests for SOP constraint validation (Story 4.5)."""

from __future__ import annotations

import pytest

from yolo_developer.seed.sop import (
    CATEGORY_SEVERITY_MAP,
    ConflictSeverity,
    InMemorySOPStore,
    SOPCategory,
    SOPConflict,
    SOPConstraint,
    SOPValidationResult,
    generate_constraint_id,
)


class TestSOPCategory:
    """Test SOPCategory enum."""

    def test_all_categories_defined(self) -> None:
        """Test all expected categories exist."""
        assert SOPCategory.ARCHITECTURE.value == "architecture"
        assert SOPCategory.SECURITY.value == "security"
        assert SOPCategory.PERFORMANCE.value == "performance"
        assert SOPCategory.NAMING.value == "naming"
        assert SOPCategory.TESTING.value == "testing"
        assert SOPCategory.DEPENDENCY.value == "dependency"

    def test_category_count(self) -> None:
        """Test we have exactly 6 categories."""
        assert len(SOPCategory) == 6


class TestConflictSeverity:
    """Test ConflictSeverity enum."""

    def test_severity_values(self) -> None:
        """Test severity enum values."""
        assert ConflictSeverity.HARD.value == "hard"
        assert ConflictSeverity.SOFT.value == "soft"

    def test_severity_count(self) -> None:
        """Test we have exactly 2 severities."""
        assert len(ConflictSeverity) == 2


class TestCategorySeverityMap:
    """Test category to severity mapping."""

    def test_hard_categories(self) -> None:
        """Test categories that default to HARD severity."""
        assert CATEGORY_SEVERITY_MAP[SOPCategory.ARCHITECTURE] == ConflictSeverity.HARD
        assert CATEGORY_SEVERITY_MAP[SOPCategory.SECURITY] == ConflictSeverity.HARD
        assert CATEGORY_SEVERITY_MAP[SOPCategory.DEPENDENCY] == ConflictSeverity.HARD

    def test_soft_categories(self) -> None:
        """Test categories that default to SOFT severity."""
        assert CATEGORY_SEVERITY_MAP[SOPCategory.PERFORMANCE] == ConflictSeverity.SOFT
        assert CATEGORY_SEVERITY_MAP[SOPCategory.NAMING] == ConflictSeverity.SOFT
        assert CATEGORY_SEVERITY_MAP[SOPCategory.TESTING] == ConflictSeverity.SOFT


class TestSOPConstraint:
    """Test SOPConstraint dataclass."""

    def test_create_constraint(self) -> None:
        """Test creating a constraint."""
        constraint = SOPConstraint(
            id="arch-001",
            rule_text="Use REST API conventions",
            category=SOPCategory.ARCHITECTURE,
            source="architecture.md",
            severity=ConflictSeverity.HARD,
        )

        assert constraint.id == "arch-001"
        assert constraint.rule_text == "Use REST API conventions"
        assert constraint.category == SOPCategory.ARCHITECTURE
        assert constraint.source == "architecture.md"
        assert constraint.severity == ConflictSeverity.HARD
        assert constraint.created_at is not None

    def test_constraint_is_frozen(self) -> None:
        """Test constraint is immutable."""
        constraint = SOPConstraint(
            id="arch-001",
            rule_text="Use REST API",
            category=SOPCategory.ARCHITECTURE,
            source="test",
            severity=ConflictSeverity.HARD,
        )

        with pytest.raises(AttributeError):
            constraint.id = "new-id"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        constraint = SOPConstraint(
            id="sec-001",
            rule_text="Require authentication",
            category=SOPCategory.SECURITY,
            source="security.md",
            severity=ConflictSeverity.HARD,
            created_at="2026-01-08T00:00:00+00:00",
        )

        result = constraint.to_dict()

        assert result["id"] == "sec-001"
        assert result["rule_text"] == "Require authentication"
        assert result["category"] == "security"
        assert result["source"] == "security.md"
        assert result["severity"] == "hard"
        assert result["created_at"] == "2026-01-08T00:00:00+00:00"

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "id": "perf-001",
            "rule_text": "Response time < 100ms",
            "category": "performance",
            "source": "perf-guide.md",
            "severity": "soft",
            "created_at": "2026-01-08T00:00:00+00:00",
        }

        constraint = SOPConstraint.from_dict(data)

        assert constraint.id == "perf-001"
        assert constraint.rule_text == "Response time < 100ms"
        assert constraint.category == SOPCategory.PERFORMANCE
        assert constraint.severity == ConflictSeverity.SOFT

    def test_from_dict_without_created_at(self) -> None:
        """Test from_dict generates created_at if missing."""
        data = {
            "id": "test-001",
            "rule_text": "Test rule",
            "category": "testing",
            "source": "test.md",
            "severity": "soft",
        }

        constraint = SOPConstraint.from_dict(data)

        assert constraint.created_at is not None


class TestSOPConflict:
    """Test SOPConflict dataclass."""

    @pytest.fixture
    def sample_constraint(self) -> SOPConstraint:
        """Create a sample constraint for testing."""
        return SOPConstraint(
            id="arch-001",
            rule_text="Use REST API",
            category=SOPCategory.ARCHITECTURE,
            source="test.md",
            severity=ConflictSeverity.HARD,
        )

    def test_create_conflict(self, sample_constraint: SOPConstraint) -> None:
        """Test creating a conflict."""
        conflict = SOPConflict(
            constraint=sample_constraint,
            seed_text="Use GraphQL for API",
            severity=ConflictSeverity.HARD,
            description="Conflicts with REST requirement",
            resolution_options=("Use REST instead", "Request exception"),
        )

        assert conflict.constraint == sample_constraint
        assert conflict.seed_text == "Use GraphQL for API"
        assert conflict.severity == ConflictSeverity.HARD
        assert conflict.description == "Conflicts with REST requirement"
        assert len(conflict.resolution_options) == 2

    def test_conflict_is_frozen(self, sample_constraint: SOPConstraint) -> None:
        """Test conflict is immutable."""
        conflict = SOPConflict(
            constraint=sample_constraint,
            seed_text="test",
            severity=ConflictSeverity.SOFT,
            description="test",
        )

        with pytest.raises(AttributeError):
            conflict.seed_text = "new text"  # type: ignore[misc]

    def test_to_dict(self, sample_constraint: SOPConstraint) -> None:
        """Test conversion to dictionary."""
        conflict = SOPConflict(
            constraint=sample_constraint,
            seed_text="Use GraphQL",
            severity=ConflictSeverity.HARD,
            description="Conflicts with REST",
            resolution_options=("option1", "option2"),
        )

        result = conflict.to_dict()

        assert result["seed_text"] == "Use GraphQL"
        assert result["severity"] == "hard"
        assert result["description"] == "Conflicts with REST"
        assert result["resolution_options"] == ["option1", "option2"]
        assert "constraint" in result
        assert result["constraint"]["id"] == "arch-001"

    def test_default_empty_resolution_options(self, sample_constraint: SOPConstraint) -> None:
        """Test default empty resolution options."""
        conflict = SOPConflict(
            constraint=sample_constraint,
            seed_text="test",
            severity=ConflictSeverity.SOFT,
            description="test",
        )

        assert conflict.resolution_options == ()


class TestSOPValidationResult:
    """Test SOPValidationResult dataclass."""

    @pytest.fixture
    def sample_constraint(self) -> SOPConstraint:
        """Create a sample constraint for testing."""
        return SOPConstraint(
            id="arch-001",
            rule_text="Use REST API",
            category=SOPCategory.ARCHITECTURE,
            source="test.md",
            severity=ConflictSeverity.HARD,
        )

    def test_empty_result(self) -> None:
        """Test empty result defaults."""
        result = SOPValidationResult()

        assert result.conflicts == []
        assert result.passed is True
        assert result.override_applied is False
        assert result.has_conflicts is False
        assert result.hard_conflict_count == 0
        assert result.soft_conflict_count == 0

    def test_result_with_conflicts(self, sample_constraint: SOPConstraint) -> None:
        """Test result with conflicts."""
        hard_conflict = SOPConflict(
            constraint=sample_constraint,
            seed_text="test1",
            severity=ConflictSeverity.HARD,
            description="hard conflict",
        )
        soft_conflict = SOPConflict(
            constraint=sample_constraint,
            seed_text="test2",
            severity=ConflictSeverity.SOFT,
            description="soft conflict",
        )

        result = SOPValidationResult(
            conflicts=[hard_conflict, soft_conflict],
            passed=False,
        )

        assert result.has_conflicts is True
        assert result.hard_conflict_count == 1
        assert result.soft_conflict_count == 1
        assert len(result.hard_conflicts) == 1
        assert len(result.soft_conflicts) == 1

    def test_to_dict(self, sample_constraint: SOPConstraint) -> None:
        """Test conversion to dictionary."""
        conflict = SOPConflict(
            constraint=sample_constraint,
            seed_text="test",
            severity=ConflictSeverity.HARD,
            description="test",
        )

        result = SOPValidationResult(
            conflicts=[conflict],
            passed=False,
            override_applied=True,
        )

        result_dict = result.to_dict()

        assert len(result_dict["conflicts"]) == 1
        assert result_dict["passed"] is False
        assert result_dict["override_applied"] is True
        assert result_dict["hard_conflict_count"] == 1
        assert result_dict["soft_conflict_count"] == 0


class TestInMemorySOPStore:
    """Test InMemorySOPStore implementation."""

    @pytest.fixture
    def store(self) -> InMemorySOPStore:
        """Create a fresh store for each test."""
        return InMemorySOPStore()

    @pytest.fixture
    def sample_constraint(self) -> SOPConstraint:
        """Create a sample constraint."""
        return SOPConstraint(
            id="arch-001",
            rule_text="Use REST API",
            category=SOPCategory.ARCHITECTURE,
            source="test.md",
            severity=ConflictSeverity.HARD,
        )

    @pytest.mark.asyncio
    async def test_add_and_get_constraint(
        self, store: InMemorySOPStore, sample_constraint: SOPConstraint
    ) -> None:
        """Test adding and retrieving constraints."""
        await store.add_constraint(sample_constraint)

        constraints = await store.get_constraints()

        assert len(constraints) == 1
        assert constraints[0].id == "arch-001"

    @pytest.mark.asyncio
    async def test_get_constraints_by_category(self, store: InMemorySOPStore) -> None:
        """Test filtering constraints by category."""
        arch_constraint = SOPConstraint(
            id="arch-001",
            rule_text="Arch rule",
            category=SOPCategory.ARCHITECTURE,
            source="test.md",
            severity=ConflictSeverity.HARD,
        )
        sec_constraint = SOPConstraint(
            id="sec-001",
            rule_text="Sec rule",
            category=SOPCategory.SECURITY,
            source="test.md",
            severity=ConflictSeverity.HARD,
        )

        await store.add_constraint(arch_constraint)
        await store.add_constraint(sec_constraint)

        arch_only = await store.get_constraints(category=SOPCategory.ARCHITECTURE)
        sec_only = await store.get_constraints(category=SOPCategory.SECURITY)
        all_constraints = await store.get_constraints()

        assert len(arch_only) == 1
        assert len(sec_only) == 1
        assert len(all_constraints) == 2

    @pytest.mark.asyncio
    async def test_search_similar(
        self, store: InMemorySOPStore, sample_constraint: SOPConstraint
    ) -> None:
        """Test searching for similar constraints."""
        await store.add_constraint(sample_constraint)

        results = await store.search_similar("REST")

        assert len(results) == 1
        assert results[0].id == "arch-001"

    @pytest.mark.asyncio
    async def test_search_similar_no_match(
        self, store: InMemorySOPStore, sample_constraint: SOPConstraint
    ) -> None:
        """Test search with no matches."""
        await store.add_constraint(sample_constraint)

        results = await store.search_similar("GraphQL")

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_similar_limit(self, store: InMemorySOPStore) -> None:
        """Test search respects limit."""
        for i in range(5):
            await store.add_constraint(
                SOPConstraint(
                    id=f"test-{i}",
                    rule_text=f"Test rule {i}",
                    category=SOPCategory.TESTING,
                    source="test.md",
                    severity=ConflictSeverity.SOFT,
                )
            )

        results = await store.search_similar("Test", limit=2)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_remove_constraint(
        self, store: InMemorySOPStore, sample_constraint: SOPConstraint
    ) -> None:
        """Test removing a constraint."""
        await store.add_constraint(sample_constraint)

        removed = await store.remove_constraint("arch-001")

        assert removed is True
        constraints = await store.get_constraints()
        assert len(constraints) == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent_constraint(self, store: InMemorySOPStore) -> None:
        """Test removing a nonexistent constraint."""
        removed = await store.remove_constraint("nonexistent")

        assert removed is False

    @pytest.mark.asyncio
    async def test_clear(self, store: InMemorySOPStore, sample_constraint: SOPConstraint) -> None:
        """Test clearing all constraints."""
        await store.add_constraint(sample_constraint)

        await store.clear()

        constraints = await store.get_constraints()
        assert len(constraints) == 0


class TestGenerateConstraintId:
    """Test constraint ID generation."""

    def test_generates_unique_ids(self) -> None:
        """Test IDs are unique."""
        ids = [generate_constraint_id(SOPCategory.ARCHITECTURE) for _ in range(10)]
        assert len(set(ids)) == 10

    def test_id_has_category_prefix(self) -> None:
        """Test ID starts with category prefix."""
        arch_id = generate_constraint_id(SOPCategory.ARCHITECTURE)
        sec_id = generate_constraint_id(SOPCategory.SECURITY)
        perf_id = generate_constraint_id(SOPCategory.PERFORMANCE)

        assert arch_id.startswith("arch-")
        assert sec_id.startswith("secu-")
        assert perf_id.startswith("perf-")


class TestValidateAgainstSOP:
    """Test validate_against_sop function."""

    from unittest.mock import AsyncMock, patch

    @pytest.fixture
    def store(self) -> InMemorySOPStore:
        """Create a fresh store for each test."""
        return InMemorySOPStore()

    @pytest.fixture
    def sample_constraint(self) -> SOPConstraint:
        """Create a sample constraint."""
        return SOPConstraint(
            id="arch-001",
            rule_text="All API endpoints must use REST conventions",
            category=SOPCategory.ARCHITECTURE,
            source="architecture.md",
            severity=ConflictSeverity.HARD,
        )

    @pytest.mark.asyncio
    async def test_empty_store_passes(self, store: InMemorySOPStore) -> None:
        """Test validation passes with empty store."""
        from yolo_developer.seed.sop import validate_against_sop

        result = await validate_against_sop("Build a GraphQL API", store)

        assert result.passed is True
        assert len(result.conflicts) == 0

    @pytest.mark.asyncio
    async def test_no_conflicts_detected(
        self, store: InMemorySOPStore, sample_constraint: SOPConstraint
    ) -> None:
        """Test validation with no conflicts."""
        from unittest.mock import AsyncMock, patch

        from yolo_developer.seed.sop import validate_against_sop

        await store.add_constraint(sample_constraint)

        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock(message=AsyncMock(content='{"conflicts": []}'))]

        with patch("yolo_developer.seed.sop.litellm.acompletion", return_value=mock_response):
            result = await validate_against_sop("Build a REST API", store)

        assert result.passed is True
        assert len(result.conflicts) == 0

    @pytest.mark.asyncio
    async def test_hard_conflict_detected(
        self, store: InMemorySOPStore, sample_constraint: SOPConstraint
    ) -> None:
        """Test validation detects HARD conflict."""
        from unittest.mock import AsyncMock, patch

        from yolo_developer.seed.sop import validate_against_sop

        await store.add_constraint(sample_constraint)

        mock_response = AsyncMock()
        mock_response.choices = [
            AsyncMock(
                message=AsyncMock(
                    content="""{
                        "conflicts": [{
                            "constraint_id": "arch-001",
                            "seed_text": "Use GraphQL for the API",
                            "severity": "HARD",
                            "description": "GraphQL conflicts with REST requirement",
                            "resolution_options": ["Use REST instead", "Request exception"]
                        }]
                    }"""
                )
            )
        ]

        with patch("yolo_developer.seed.sop.litellm.acompletion", return_value=mock_response):
            result = await validate_against_sop("Use GraphQL for the API", store)

        assert result.passed is False
        assert len(result.conflicts) == 1
        assert result.conflicts[0].severity == ConflictSeverity.HARD
        assert result.hard_conflict_count == 1

    @pytest.mark.asyncio
    async def test_soft_conflict_passes(self, store: InMemorySOPStore) -> None:
        """Test validation passes with only SOFT conflicts."""
        from unittest.mock import AsyncMock, patch

        from yolo_developer.seed.sop import validate_against_sop

        soft_constraint = SOPConstraint(
            id="name-001",
            rule_text="Use snake_case for function names",
            category=SOPCategory.NAMING,
            source="style.md",
            severity=ConflictSeverity.SOFT,
        )
        await store.add_constraint(soft_constraint)

        mock_response = AsyncMock()
        mock_response.choices = [
            AsyncMock(
                message=AsyncMock(
                    content="""{
                        "conflicts": [{
                            "constraint_id": "name-001",
                            "seed_text": "Use camelCase naming",
                            "severity": "SOFT",
                            "description": "camelCase conflicts with snake_case convention",
                            "resolution_options": ["Use snake_case", "Document exception"]
                        }]
                    }"""
                )
            )
        ]

        with patch("yolo_developer.seed.sop.litellm.acompletion", return_value=mock_response):
            result = await validate_against_sop("Use camelCase naming", store)

        assert result.passed is True  # SOFT conflicts don't block
        assert len(result.conflicts) == 1
        assert result.soft_conflict_count == 1

    @pytest.mark.asyncio
    async def test_mixed_conflicts(self, store: InMemorySOPStore) -> None:
        """Test validation with both HARD and SOFT conflicts."""
        from unittest.mock import AsyncMock, patch

        from yolo_developer.seed.sop import validate_against_sop

        hard_constraint = SOPConstraint(
            id="arch-001",
            rule_text="Use REST API",
            category=SOPCategory.ARCHITECTURE,
            source="arch.md",
            severity=ConflictSeverity.HARD,
        )
        soft_constraint = SOPConstraint(
            id="name-001",
            rule_text="Use snake_case",
            category=SOPCategory.NAMING,
            source="style.md",
            severity=ConflictSeverity.SOFT,
        )
        await store.add_constraint(hard_constraint)
        await store.add_constraint(soft_constraint)

        mock_response = AsyncMock()
        mock_response.choices = [
            AsyncMock(
                message=AsyncMock(
                    content="""{
                        "conflicts": [
                            {
                                "constraint_id": "arch-001",
                                "seed_text": "Use GraphQL",
                                "severity": "HARD",
                                "description": "Conflicts with REST",
                                "resolution_options": []
                            },
                            {
                                "constraint_id": "name-001",
                                "seed_text": "Use camelCase",
                                "severity": "SOFT",
                                "description": "Conflicts with snake_case",
                                "resolution_options": []
                            }
                        ]
                    }"""
                )
            )
        ]

        with patch("yolo_developer.seed.sop.litellm.acompletion", return_value=mock_response):
            result = await validate_against_sop("Use GraphQL with camelCase", store)

        assert result.passed is False  # HARD conflict blocks
        assert result.hard_conflict_count == 1
        assert result.soft_conflict_count == 1

    @pytest.mark.asyncio
    async def test_unknown_constraint_id_ignored(
        self, store: InMemorySOPStore, sample_constraint: SOPConstraint
    ) -> None:
        """Test unknown constraint IDs are gracefully handled."""
        from unittest.mock import AsyncMock, patch

        from yolo_developer.seed.sop import validate_against_sop

        await store.add_constraint(sample_constraint)

        mock_response = AsyncMock()
        mock_response.choices = [
            AsyncMock(
                message=AsyncMock(
                    content="""{
                        "conflicts": [{
                            "constraint_id": "unknown-999",
                            "seed_text": "test",
                            "severity": "HARD",
                            "description": "test",
                            "resolution_options": []
                        }]
                    }"""
                )
            )
        ]

        with patch("yolo_developer.seed.sop.litellm.acompletion", return_value=mock_response):
            result = await validate_against_sop("test content", store)

        # Unknown constraint ID should be ignored
        assert result.passed is True
        assert len(result.conflicts) == 0

    @pytest.mark.asyncio
    async def test_json_in_markdown_code_block(
        self, store: InMemorySOPStore, sample_constraint: SOPConstraint
    ) -> None:
        """Test handling JSON wrapped in markdown code blocks."""
        from unittest.mock import AsyncMock, patch

        from yolo_developer.seed.sop import validate_against_sop

        await store.add_constraint(sample_constraint)

        mock_response = AsyncMock()
        mock_response.choices = [
            AsyncMock(
                message=AsyncMock(
                    content="""```json
                    {"conflicts": []}
                    ```"""
                )
            )
        ]

        with patch("yolo_developer.seed.sop.litellm.acompletion", return_value=mock_response):
            result = await validate_against_sop("test", store)

        assert result.passed is True


class TestParseJsonResponse:
    """Test _parse_json_response helper."""

    def test_parse_valid_json(self) -> None:
        """Test parsing valid JSON."""
        from yolo_developer.seed.sop import _parse_json_response

        result = _parse_json_response('{"conflicts": []}')
        assert result == {"conflicts": []}

    def test_parse_json_with_markdown_wrapper(self) -> None:
        """Test parsing JSON wrapped in markdown."""
        from yolo_developer.seed.sop import _parse_json_response

        result = _parse_json_response('```json\n{"conflicts": []}\n```')
        assert result == {"conflicts": []}

    def test_parse_empty_content(self) -> None:
        """Test parsing empty content."""
        from yolo_developer.seed.sop import _parse_json_response

        result = _parse_json_response(None)
        assert result == {"conflicts": []}

        result = _parse_json_response("")
        assert result == {"conflicts": []}

    def test_parse_invalid_json(self) -> None:
        """Test handling invalid JSON."""
        from yolo_developer.seed.sop import _parse_json_response

        result = _parse_json_response("not valid json")
        assert result == {"conflicts": []}


class TestSeedParseResultSOPIntegration:
    """Test SeedParseResult SOP-related properties (Story 4.5)."""

    def test_has_sop_conflicts_none(self) -> None:
        """Test has_sop_conflicts when no validation done."""
        from yolo_developer.seed.types import SeedParseResult, SeedSource

        result = SeedParseResult(
            goals=(),
            features=(),
            constraints=(),
            raw_content="test",
            source=SeedSource.TEXT,
            sop_validation=None,
        )
        assert result.has_sop_conflicts is False

    def test_has_sop_conflicts_empty(self) -> None:
        """Test has_sop_conflicts with empty validation result."""
        from yolo_developer.seed.sop import SOPValidationResult
        from yolo_developer.seed.types import SeedParseResult, SeedSource

        result = SeedParseResult(
            goals=(),
            features=(),
            constraints=(),
            raw_content="test",
            source=SeedSource.TEXT,
            sop_validation=SOPValidationResult(),
        )
        assert result.has_sop_conflicts is False

    def test_has_sop_conflicts_with_conflicts(self) -> None:
        """Test has_sop_conflicts when conflicts present."""
        from yolo_developer.seed.sop import (
            ConflictSeverity,
            SOPCategory,
            SOPConflict,
            SOPConstraint,
            SOPValidationResult,
        )
        from yolo_developer.seed.types import SeedParseResult, SeedSource

        constraint = SOPConstraint(
            id="test",
            rule_text="test rule",
            category=SOPCategory.ARCHITECTURE,
            source="test",
            severity=ConflictSeverity.HARD,
        )
        conflict = SOPConflict(
            constraint=constraint,
            seed_text="test",
            severity=ConflictSeverity.HARD,
            description="test conflict",
        )
        result = SeedParseResult(
            goals=(),
            features=(),
            constraints=(),
            raw_content="test",
            source=SeedSource.TEXT,
            sop_validation=SOPValidationResult(conflicts=[conflict], passed=False),
        )
        assert result.has_sop_conflicts is True

    def test_sop_passed_none(self) -> None:
        """Test sop_passed when no validation done."""
        from yolo_developer.seed.types import SeedParseResult, SeedSource

        result = SeedParseResult(
            goals=(),
            features=(),
            constraints=(),
            raw_content="test",
            source=SeedSource.TEXT,
            sop_validation=None,
        )
        assert result.sop_passed is True

    def test_sop_passed_with_pass(self) -> None:
        """Test sop_passed when validation passed."""
        from yolo_developer.seed.sop import SOPValidationResult
        from yolo_developer.seed.types import SeedParseResult, SeedSource

        result = SeedParseResult(
            goals=(),
            features=(),
            constraints=(),
            raw_content="test",
            source=SeedSource.TEXT,
            sop_validation=SOPValidationResult(passed=True),
        )
        assert result.sop_passed is True

    def test_sop_passed_with_failure(self) -> None:
        """Test sop_passed when validation failed."""
        from yolo_developer.seed.sop import SOPValidationResult
        from yolo_developer.seed.types import SeedParseResult, SeedSource

        result = SeedParseResult(
            goals=(),
            features=(),
            constraints=(),
            raw_content="test",
            source=SeedSource.TEXT,
            sop_validation=SOPValidationResult(passed=False),
        )
        assert result.sop_passed is False

    def test_to_dict_includes_sop_validation(self) -> None:
        """Test to_dict includes sop_validation field."""
        from yolo_developer.seed.sop import SOPValidationResult
        from yolo_developer.seed.types import SeedParseResult, SeedSource

        result = SeedParseResult(
            goals=(),
            features=(),
            constraints=(),
            raw_content="test",
            source=SeedSource.TEXT,
            sop_validation=SOPValidationResult(passed=True),
        )
        d = result.to_dict()
        assert "sop_validation" in d
        assert d["sop_validation"]["passed"] is True

    def test_to_dict_with_none_sop_validation(self) -> None:
        """Test to_dict with None sop_validation."""
        from yolo_developer.seed.types import SeedParseResult, SeedSource

        result = SeedParseResult(
            goals=(),
            features=(),
            constraints=(),
            raw_content="test",
            source=SeedSource.TEXT,
            sop_validation=None,
        )
        d = result.to_dict()
        assert "sop_validation" in d
        assert d["sop_validation"] is None


class TestParseSeedSOPValidation:
    """Test parse_seed with SOP validation (Story 4.5)."""

    @pytest.mark.asyncio
    async def test_parse_seed_requires_store_for_validation(self) -> None:
        """Test parse_seed raises error when validate_sop=True but no store."""
        from yolo_developer.seed.api import parse_seed

        with pytest.raises(ValueError, match="sop_store is required"):
            await parse_seed("test", validate_sop=True, sop_store=None)

    @pytest.mark.asyncio
    async def test_parse_seed_with_validation_empty_store(self) -> None:
        """Test parse_seed with validation and empty store."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from yolo_developer.seed.api import parse_seed
        from yolo_developer.seed.sop import InMemorySOPStore, SOPValidationResult
        from yolo_developer.seed.types import SeedParseResult, SeedSource

        # Mock the LLM parser result
        mock_parse_result = SeedParseResult(
            goals=(),
            features=(),
            constraints=(),
            raw_content="test content",
            source=SeedSource.TEXT,
        )

        with (
            patch("yolo_developer.seed.api.LLMSeedParser") as mock_parser_class,
            patch(
                "yolo_developer.seed.api._validate_against_sop",
                new_callable=AsyncMock,
            ) as mock_validate,
        ):
            mock_parser = MagicMock()
            mock_parser.parse = AsyncMock(return_value=mock_parse_result)
            mock_parser_class.return_value = mock_parser

            mock_validate.return_value = SOPValidationResult(passed=True)

            store = InMemorySOPStore()
            result = await parse_seed(
                "test content",
                validate_sop=True,
                sop_store=store,
            )

            # Verify validation was called
            mock_validate.assert_called_once()
            assert result.sop_validation is not None
            assert result.sop_validation.passed is True
