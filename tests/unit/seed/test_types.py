"""Unit tests for seed types module (Story 4.1 - Task 1).

Tests the data models for seed document parsing:
- SeedSource enum
- ComponentType enum
- ConstraintCategory enum
- SeedComponent dataclass
- SeedGoal dataclass
- SeedFeature dataclass
- SeedConstraint dataclass
- SeedParseResult dataclass
"""

from __future__ import annotations

import pytest

from yolo_developer.seed.types import (
    ComponentType,
    ConstraintCategory,
    SeedComponent,
    SeedConstraint,
    SeedFeature,
    SeedGoal,
    SeedParseResult,
    SeedSource,
)

# =============================================================================
# SeedSource Enum Tests
# =============================================================================


class TestSeedSource:
    """Tests for SeedSource enum."""

    def test_seed_source_values(self) -> None:
        """Test that SeedSource has expected values."""
        assert SeedSource.FILE.value == "file"
        assert SeedSource.TEXT.value == "text"
        assert SeedSource.URL.value == "url"

    def test_seed_source_from_string(self) -> None:
        """Test creating SeedSource from string values."""
        assert SeedSource("file") == SeedSource.FILE
        assert SeedSource("text") == SeedSource.TEXT
        assert SeedSource("url") == SeedSource.URL

    def test_seed_source_invalid(self) -> None:
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError):
            SeedSource("invalid")


# =============================================================================
# ComponentType Enum Tests
# =============================================================================


class TestComponentType:
    """Tests for ComponentType enum."""

    def test_component_type_values(self) -> None:
        """Test that ComponentType has expected values."""
        assert ComponentType.GOAL.value == "goal"
        assert ComponentType.FEATURE.value == "feature"
        assert ComponentType.CONSTRAINT.value == "constraint"
        assert ComponentType.CONTEXT.value == "context"
        assert ComponentType.UNKNOWN.value == "unknown"

    def test_component_type_from_string(self) -> None:
        """Test creating ComponentType from string values."""
        assert ComponentType("goal") == ComponentType.GOAL
        assert ComponentType("feature") == ComponentType.FEATURE
        assert ComponentType("constraint") == ComponentType.CONSTRAINT


# =============================================================================
# ConstraintCategory Enum Tests
# =============================================================================


class TestConstraintCategory:
    """Tests for ConstraintCategory enum."""

    def test_constraint_category_values(self) -> None:
        """Test that ConstraintCategory has expected values."""
        assert ConstraintCategory.TECHNICAL.value == "technical"
        assert ConstraintCategory.BUSINESS.value == "business"
        assert ConstraintCategory.TIMELINE.value == "timeline"
        assert ConstraintCategory.RESOURCE.value == "resource"
        assert ConstraintCategory.COMPLIANCE.value == "compliance"


# =============================================================================
# SeedComponent Tests
# =============================================================================


class TestSeedComponent:
    """Tests for SeedComponent dataclass."""

    def test_seed_component_creation(self) -> None:
        """Test creating a SeedComponent with required fields."""
        component = SeedComponent(
            component_type=ComponentType.GOAL,
            content="Build a web application",
            confidence=0.95,
        )
        assert component.component_type == ComponentType.GOAL
        assert component.content == "Build a web application"
        assert component.confidence == 0.95
        assert component.source_line is None
        assert component.metadata == ()

    def test_seed_component_with_optional_fields(self) -> None:
        """Test creating a SeedComponent with all fields."""
        component = SeedComponent(
            component_type=ComponentType.FEATURE,
            content="User authentication",
            confidence=0.85,
            source_line=42,
            metadata=(("priority", "high"),),
        )
        assert component.source_line == 42
        assert component.metadata == (("priority", "high"),)

    def test_seed_component_is_frozen(self) -> None:
        """Test that SeedComponent is immutable."""
        component = SeedComponent(
            component_type=ComponentType.GOAL,
            content="Test content",
            confidence=0.9,
        )
        with pytest.raises(AttributeError):
            component.content = "Modified"  # type: ignore[misc]

    def test_seed_component_to_dict(self) -> None:
        """Test SeedComponent serialization to dict."""
        component = SeedComponent(
            component_type=ComponentType.CONSTRAINT,
            content="Must use Python 3.10+",
            confidence=1.0,
            source_line=15,
            metadata=(("category", "technical"),),
        )
        result = component.to_dict()
        assert result["component_type"] == "constraint"
        assert result["content"] == "Must use Python 3.10+"
        assert result["confidence"] == 1.0
        assert result["source_line"] == 15
        assert result["metadata"] == {"category": "technical"}

    def test_seed_component_to_dict_empty_metadata(self) -> None:
        """Test SeedComponent serialization with empty metadata."""
        component = SeedComponent(
            component_type=ComponentType.GOAL,
            content="Test",
            confidence=0.5,
        )
        result = component.to_dict()
        assert result["metadata"] == {}


# =============================================================================
# SeedGoal Tests
# =============================================================================


class TestSeedGoal:
    """Tests for SeedGoal dataclass."""

    def test_seed_goal_creation(self) -> None:
        """Test creating a SeedGoal with required fields."""
        goal = SeedGoal(
            title="Build E-commerce Platform",
            description="Create a modern e-commerce platform for selling products online",
            priority=1,
        )
        assert goal.title == "Build E-commerce Platform"
        assert goal.description == "Create a modern e-commerce platform for selling products online"
        assert goal.priority == 1
        assert goal.rationale is None

    def test_seed_goal_with_rationale(self) -> None:
        """Test creating a SeedGoal with rationale."""
        goal = SeedGoal(
            title="Improve Performance",
            description="Optimize application response times",
            priority=2,
            rationale="User feedback indicates slow loading times are a pain point",
        )
        assert goal.rationale == "User feedback indicates slow loading times are a pain point"

    def test_seed_goal_is_frozen(self) -> None:
        """Test that SeedGoal is immutable."""
        goal = SeedGoal(
            title="Test",
            description="Test goal",
            priority=1,
        )
        with pytest.raises(AttributeError):
            goal.title = "Modified"  # type: ignore[misc]

    def test_seed_goal_to_dict(self) -> None:
        """Test SeedGoal serialization to dict."""
        goal = SeedGoal(
            title="Build API",
            description="Create REST API for mobile app",
            priority=1,
            rationale="Mobile app team needs backend support",
        )
        result = goal.to_dict()
        assert result == {
            "title": "Build API",
            "description": "Create REST API for mobile app",
            "priority": 1,
            "rationale": "Mobile app team needs backend support",
        }

    def test_seed_goal_to_dict_none_rationale(self) -> None:
        """Test SeedGoal serialization with None rationale."""
        goal = SeedGoal(
            title="Test",
            description="Test",
            priority=3,
        )
        result = goal.to_dict()
        assert result["rationale"] is None


# =============================================================================
# SeedFeature Tests
# =============================================================================


class TestSeedFeature:
    """Tests for SeedFeature dataclass."""

    def test_seed_feature_creation(self) -> None:
        """Test creating a SeedFeature with required fields."""
        feature = SeedFeature(
            name="User Registration",
            description="Allow users to create accounts with email verification",
        )
        assert feature.name == "User Registration"
        assert feature.description == "Allow users to create accounts with email verification"
        assert feature.user_value is None
        assert feature.related_goals == ()

    def test_seed_feature_with_optional_fields(self) -> None:
        """Test creating a SeedFeature with all fields."""
        feature = SeedFeature(
            name="Shopping Cart",
            description="Add products to cart and manage quantities",
            user_value="Users can collect items before checkout",
            related_goals=("Build E-commerce Platform",),
        )
        assert feature.user_value == "Users can collect items before checkout"
        assert feature.related_goals == ("Build E-commerce Platform",)

    def test_seed_feature_is_frozen(self) -> None:
        """Test that SeedFeature is immutable."""
        feature = SeedFeature(
            name="Test",
            description="Test feature",
        )
        with pytest.raises(AttributeError):
            feature.name = "Modified"  # type: ignore[misc]

    def test_seed_feature_to_dict(self) -> None:
        """Test SeedFeature serialization to dict."""
        feature = SeedFeature(
            name="Search",
            description="Full-text search for products",
            user_value="Quickly find products",
            related_goals=("Goal 1", "Goal 2"),
        )
        result = feature.to_dict()
        assert result == {
            "name": "Search",
            "description": "Full-text search for products",
            "user_value": "Quickly find products",
            "related_goals": ["Goal 1", "Goal 2"],
        }

    def test_seed_feature_to_dict_empty_goals(self) -> None:
        """Test SeedFeature serialization with empty related_goals."""
        feature = SeedFeature(
            name="Test",
            description="Test",
        )
        result = feature.to_dict()
        assert result["related_goals"] == []


# =============================================================================
# SeedConstraint Tests
# =============================================================================


class TestSeedConstraint:
    """Tests for SeedConstraint dataclass."""

    def test_seed_constraint_creation(self) -> None:
        """Test creating a SeedConstraint with required fields."""
        constraint = SeedConstraint(
            category=ConstraintCategory.TECHNICAL,
            description="Must use Python 3.10 or higher",
        )
        assert constraint.category == ConstraintCategory.TECHNICAL
        assert constraint.description == "Must use Python 3.10 or higher"
        assert constraint.impact is None
        assert constraint.related_items == ()

    def test_seed_constraint_with_optional_fields(self) -> None:
        """Test creating a SeedConstraint with all fields."""
        constraint = SeedConstraint(
            category=ConstraintCategory.TIMELINE,
            description="Must launch by Q2 2026",
            impact="Limits scope of MVP features",
            related_items=("Feature A", "Feature B"),
        )
        assert constraint.impact == "Limits scope of MVP features"
        assert constraint.related_items == ("Feature A", "Feature B")

    def test_seed_constraint_all_categories(self) -> None:
        """Test creating constraints with all category types."""
        categories = [
            ConstraintCategory.TECHNICAL,
            ConstraintCategory.BUSINESS,
            ConstraintCategory.TIMELINE,
            ConstraintCategory.RESOURCE,
            ConstraintCategory.COMPLIANCE,
        ]
        for category in categories:
            constraint = SeedConstraint(
                category=category,
                description=f"Test {category.value} constraint",
            )
            assert constraint.category == category

    def test_seed_constraint_is_frozen(self) -> None:
        """Test that SeedConstraint is immutable."""
        constraint = SeedConstraint(
            category=ConstraintCategory.BUSINESS,
            description="Test",
        )
        with pytest.raises(AttributeError):
            constraint.description = "Modified"  # type: ignore[misc]

    def test_seed_constraint_to_dict(self) -> None:
        """Test SeedConstraint serialization to dict."""
        constraint = SeedConstraint(
            category=ConstraintCategory.RESOURCE,
            description="Team of 3 developers max",
            impact="Cannot parallelize all work",
            related_items=("Feature 1",),
        )
        result = constraint.to_dict()
        assert result == {
            "category": "resource",
            "description": "Team of 3 developers max",
            "impact": "Cannot parallelize all work",
            "related_items": ["Feature 1"],
        }


# =============================================================================
# SeedParseResult Tests
# =============================================================================


class TestSeedParseResult:
    """Tests for SeedParseResult dataclass."""

    def test_seed_parse_result_creation(self) -> None:
        """Test creating a SeedParseResult with required fields."""
        result = SeedParseResult(
            goals=(),
            features=(),
            constraints=(),
            raw_content="Build a web app",
            source=SeedSource.TEXT,
        )
        assert result.goals == ()
        assert result.features == ()
        assert result.constraints == ()
        assert result.raw_content == "Build a web app"
        assert result.source == SeedSource.TEXT
        assert result.metadata == ()

    def test_seed_parse_result_with_data(self) -> None:
        """Test creating a SeedParseResult with actual data."""
        goal = SeedGoal(title="Goal 1", description="Desc", priority=1)
        feature = SeedFeature(name="Feature 1", description="Desc")
        constraint = SeedConstraint(
            category=ConstraintCategory.TECHNICAL,
            description="Python 3.10+",
        )

        result = SeedParseResult(
            goals=(goal,),
            features=(feature,),
            constraints=(constraint,),
            raw_content="Test content",
            source=SeedSource.FILE,
            metadata=(("filename", "test.md"),),
        )
        assert len(result.goals) == 1
        assert len(result.features) == 1
        assert len(result.constraints) == 1
        assert result.metadata == (("filename", "test.md"),)

    def test_seed_parse_result_is_frozen(self) -> None:
        """Test that SeedParseResult is immutable."""
        result = SeedParseResult(
            goals=(),
            features=(),
            constraints=(),
            raw_content="Test",
            source=SeedSource.TEXT,
        )
        with pytest.raises(AttributeError):
            result.raw_content = "Modified"  # type: ignore[misc]

    def test_seed_parse_result_to_dict(self) -> None:
        """Test SeedParseResult serialization to dict."""
        goal = SeedGoal(title="Goal", description="Desc", priority=1)
        feature = SeedFeature(name="Feature", description="Desc")
        constraint = SeedConstraint(
            category=ConstraintCategory.TECHNICAL,
            description="Constraint",
        )

        result = SeedParseResult(
            goals=(goal,),
            features=(feature,),
            constraints=(constraint,),
            raw_content="Content",
            source=SeedSource.TEXT,
            metadata=(("key", "value"),),
        )
        serialized = result.to_dict()

        assert serialized["raw_content"] == "Content"
        assert serialized["source"] == "text"
        assert len(serialized["goals"]) == 1
        assert len(serialized["features"]) == 1
        assert len(serialized["constraints"]) == 1
        assert serialized["metadata"] == {"key": "value"}

    def test_seed_parse_result_to_dict_empty(self) -> None:
        """Test SeedParseResult serialization with empty collections."""
        result = SeedParseResult(
            goals=(),
            features=(),
            constraints=(),
            raw_content="",
            source=SeedSource.TEXT,
        )
        serialized = result.to_dict()

        assert serialized["goals"] == []
        assert serialized["features"] == []
        assert serialized["constraints"] == []
        assert serialized["metadata"] == {}

    def test_seed_parse_result_goal_count(self) -> None:
        """Test goal_count property."""
        result = SeedParseResult(
            goals=(
                SeedGoal(title="G1", description="D1", priority=1),
                SeedGoal(title="G2", description="D2", priority=2),
            ),
            features=(),
            constraints=(),
            raw_content="Test",
            source=SeedSource.TEXT,
        )
        assert result.goal_count == 2

    def test_seed_parse_result_feature_count(self) -> None:
        """Test feature_count property."""
        result = SeedParseResult(
            goals=(),
            features=(
                SeedFeature(name="F1", description="D1"),
                SeedFeature(name="F2", description="D2"),
                SeedFeature(name="F3", description="D3"),
            ),
            constraints=(),
            raw_content="Test",
            source=SeedSource.TEXT,
        )
        assert result.feature_count == 3

    def test_seed_parse_result_constraint_count(self) -> None:
        """Test constraint_count property."""
        result = SeedParseResult(
            goals=(),
            features=(),
            constraints=(SeedConstraint(category=ConstraintCategory.TECHNICAL, description="C1"),),
            raw_content="Test",
            source=SeedSource.TEXT,
        )
        assert result.constraint_count == 1
