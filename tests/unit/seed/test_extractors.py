"""Unit tests for seed extraction helpers (Story 4.1 - Tasks 4-6).

Tests the extraction helpers for parsing LLM output:
- _extract_goals
- _extract_features
- _extract_constraints
"""

from __future__ import annotations

from yolo_developer.seed.parser import (
    _extract_constraints,
    _extract_features,
    _extract_goals,
)
from yolo_developer.seed.types import (
    ConstraintCategory,
    SeedGoal,
)

# =============================================================================
# _extract_goals Tests
# =============================================================================


class TestExtractGoals:
    """Tests for _extract_goals helper function."""

    def test_extract_goals_basic(self) -> None:
        """Test extracting a single goal from LLM output."""
        llm_output = {
            "goals": [
                {
                    "title": "Build E-commerce Platform",
                    "description": "Create an online store",
                    "priority": 1,
                    "rationale": "Market demand",
                }
            ],
            "features": [],
            "constraints": [],
        }
        goals = _extract_goals(llm_output)
        assert len(goals) == 1
        assert goals[0].title == "Build E-commerce Platform"
        assert goals[0].description == "Create an online store"
        assert goals[0].priority == 1
        assert goals[0].rationale == "Market demand"

    def test_extract_goals_multiple(self) -> None:
        """Test extracting multiple goals."""
        llm_output = {
            "goals": [
                {"title": "Goal 1", "description": "Desc 1", "priority": 1},
                {"title": "Goal 2", "description": "Desc 2", "priority": 2},
                {"title": "Goal 3", "description": "Desc 3", "priority": 3},
            ],
            "features": [],
            "constraints": [],
        }
        goals = _extract_goals(llm_output)
        assert len(goals) == 3
        assert goals[0].title == "Goal 1"
        assert goals[1].title == "Goal 2"
        assert goals[2].title == "Goal 3"

    def test_extract_goals_empty(self) -> None:
        """Test extracting from empty goals list."""
        llm_output = {"goals": [], "features": [], "constraints": []}
        goals = _extract_goals(llm_output)
        assert len(goals) == 0

    def test_extract_goals_missing_key(self) -> None:
        """Test extracting when goals key is missing."""
        llm_output = {"features": [], "constraints": []}
        goals = _extract_goals(llm_output)
        assert len(goals) == 0

    def test_extract_goals_invalid_priority_clamped(self) -> None:
        """Test that invalid priority values are defaulted to 3."""
        llm_output = {
            "goals": [
                {"title": "Goal 1", "description": "Desc", "priority": 0},
                {"title": "Goal 2", "description": "Desc", "priority": 10},
                {"title": "Goal 3", "description": "Desc", "priority": "high"},
            ],
            "features": [],
            "constraints": [],
        }
        goals = _extract_goals(llm_output)
        assert len(goals) == 3
        # All invalid priorities should default to 3
        assert goals[0].priority == 3
        assert goals[1].priority == 3
        assert goals[2].priority == 3

    def test_extract_goals_missing_fields_use_defaults(self) -> None:
        """Test that missing fields use default values."""
        llm_output = {
            "goals": [{"title": "Minimal Goal"}],
            "features": [],
            "constraints": [],
        }
        goals = _extract_goals(llm_output)
        assert len(goals) == 1
        assert goals[0].title == "Minimal Goal"
        assert goals[0].description == ""
        assert goals[0].priority == 3  # Default priority
        assert goals[0].rationale is None

    def test_extract_goals_null_rationale(self) -> None:
        """Test extracting goal with null rationale."""
        llm_output = {
            "goals": [
                {
                    "title": "Goal",
                    "description": "Desc",
                    "priority": 1,
                    "rationale": None,
                }
            ],
            "features": [],
            "constraints": [],
        }
        goals = _extract_goals(llm_output)
        assert len(goals) == 1
        assert goals[0].rationale is None

    def test_extract_goals_returns_seedgoal_type(self) -> None:
        """Test that extracted goals are SeedGoal instances."""
        llm_output = {
            "goals": [{"title": "Goal", "description": "Desc", "priority": 1}],
            "features": [],
            "constraints": [],
        }
        goals = _extract_goals(llm_output)
        assert isinstance(goals[0], SeedGoal)


# =============================================================================
# _extract_features Tests
# =============================================================================


class TestExtractFeatures:
    """Tests for _extract_features helper function."""

    def test_extract_features_basic(self) -> None:
        """Test extracting a single feature."""
        llm_output = {
            "goals": [],
            "features": [
                {
                    "name": "User Authentication",
                    "description": "Allow users to log in",
                    "user_value": "Secure access",
                    "related_goals": [],
                }
            ],
            "constraints": [],
        }
        features = _extract_features(llm_output, [])
        assert len(features) == 1
        assert features[0].name == "User Authentication"
        assert features[0].description == "Allow users to log in"
        assert features[0].user_value == "Secure access"

    def test_extract_features_multiple(self) -> None:
        """Test extracting multiple features."""
        llm_output = {
            "goals": [],
            "features": [
                {"name": "Feature 1", "description": "Desc 1"},
                {"name": "Feature 2", "description": "Desc 2"},
            ],
            "constraints": [],
        }
        features = _extract_features(llm_output, [])
        assert len(features) == 2

    def test_extract_features_empty(self) -> None:
        """Test extracting from empty features list."""
        llm_output = {"goals": [], "features": [], "constraints": []}
        features = _extract_features(llm_output, [])
        assert len(features) == 0

    def test_extract_features_missing_key(self) -> None:
        """Test extracting when features key is missing."""
        llm_output = {"goals": [], "constraints": []}
        features = _extract_features(llm_output, [])
        assert len(features) == 0

    def test_extract_features_with_valid_related_goals(self) -> None:
        """Test that related_goals are filtered to valid goal titles."""
        goals = [
            SeedGoal(title="Goal A", description="Desc", priority=1),
            SeedGoal(title="Goal B", description="Desc", priority=2),
        ]
        llm_output = {
            "goals": [],
            "features": [
                {
                    "name": "Feature",
                    "description": "Desc",
                    "related_goals": ["Goal A", "Goal C", "Goal B"],
                }
            ],
            "constraints": [],
        }
        features = _extract_features(llm_output, goals)
        assert len(features) == 1
        # Only valid goals should be included
        assert features[0].related_goals == ("Goal A", "Goal B")

    def test_extract_features_with_no_valid_related_goals(self) -> None:
        """Test that invalid related_goals are filtered out."""
        goals = [SeedGoal(title="Real Goal", description="Desc", priority=1)]
        llm_output = {
            "goals": [],
            "features": [
                {
                    "name": "Feature",
                    "description": "Desc",
                    "related_goals": ["Fake Goal 1", "Fake Goal 2"],
                }
            ],
            "constraints": [],
        }
        features = _extract_features(llm_output, goals)
        assert len(features) == 1
        assert features[0].related_goals == ()

    def test_extract_features_null_related_goals(self) -> None:
        """Test extracting feature with null related_goals."""
        llm_output = {
            "goals": [],
            "features": [
                {
                    "name": "Feature",
                    "description": "Desc",
                    "related_goals": None,
                }
            ],
            "constraints": [],
        }
        features = _extract_features(llm_output, [])
        assert len(features) == 1
        assert features[0].related_goals == ()

    def test_extract_features_missing_fields_use_defaults(self) -> None:
        """Test that missing fields use default values."""
        llm_output = {
            "goals": [],
            "features": [{"name": "Minimal Feature"}],
            "constraints": [],
        }
        features = _extract_features(llm_output, [])
        assert len(features) == 1
        assert features[0].name == "Minimal Feature"
        assert features[0].description == ""
        assert features[0].user_value is None
        assert features[0].related_goals == ()


# =============================================================================
# _extract_constraints Tests
# =============================================================================


class TestExtractConstraints:
    """Tests for _extract_constraints helper function."""

    def test_extract_constraints_basic(self) -> None:
        """Test extracting a single constraint."""
        llm_output = {
            "goals": [],
            "features": [],
            "constraints": [
                {
                    "category": "technical",
                    "description": "Must use Python 3.10+",
                    "impact": "Limits hosting options",
                    "related_items": ["Backend API"],
                }
            ],
        }
        constraints = _extract_constraints(llm_output)
        assert len(constraints) == 1
        assert constraints[0].category == ConstraintCategory.TECHNICAL
        assert constraints[0].description == "Must use Python 3.10+"
        assert constraints[0].impact == "Limits hosting options"
        assert constraints[0].related_items == ("Backend API",)

    def test_extract_constraints_all_categories(self) -> None:
        """Test extracting constraints with all category types."""
        llm_output = {
            "goals": [],
            "features": [],
            "constraints": [
                {"category": "technical", "description": "Tech constraint"},
                {"category": "business", "description": "Business constraint"},
                {"category": "timeline", "description": "Timeline constraint"},
                {"category": "resource", "description": "Resource constraint"},
                {"category": "compliance", "description": "Compliance constraint"},
            ],
        }
        constraints = _extract_constraints(llm_output)
        assert len(constraints) == 5
        assert constraints[0].category == ConstraintCategory.TECHNICAL
        assert constraints[1].category == ConstraintCategory.BUSINESS
        assert constraints[2].category == ConstraintCategory.TIMELINE
        assert constraints[3].category == ConstraintCategory.RESOURCE
        assert constraints[4].category == ConstraintCategory.COMPLIANCE

    def test_extract_constraints_empty(self) -> None:
        """Test extracting from empty constraints list."""
        llm_output = {"goals": [], "features": [], "constraints": []}
        constraints = _extract_constraints(llm_output)
        assert len(constraints) == 0

    def test_extract_constraints_missing_key(self) -> None:
        """Test extracting when constraints key is missing."""
        llm_output = {"goals": [], "features": []}
        constraints = _extract_constraints(llm_output)
        assert len(constraints) == 0

    def test_extract_constraints_invalid_category_defaults_to_technical(self) -> None:
        """Test that invalid category defaults to TECHNICAL."""
        llm_output = {
            "goals": [],
            "features": [],
            "constraints": [
                {"category": "invalid_category", "description": "Constraint"},
                {"category": "unknown", "description": "Another constraint"},
            ],
        }
        constraints = _extract_constraints(llm_output)
        assert len(constraints) == 2
        assert constraints[0].category == ConstraintCategory.TECHNICAL
        assert constraints[1].category == ConstraintCategory.TECHNICAL

    def test_extract_constraints_null_related_items(self) -> None:
        """Test extracting constraint with null related_items."""
        llm_output = {
            "goals": [],
            "features": [],
            "constraints": [
                {
                    "category": "business",
                    "description": "Constraint",
                    "related_items": None,
                }
            ],
        }
        constraints = _extract_constraints(llm_output)
        assert len(constraints) == 1
        assert constraints[0].related_items == ()

    def test_extract_constraints_missing_fields_use_defaults(self) -> None:
        """Test that missing fields use default values."""
        llm_output = {
            "goals": [],
            "features": [],
            "constraints": [{"description": "Minimal constraint"}],
        }
        constraints = _extract_constraints(llm_output)
        assert len(constraints) == 1
        assert constraints[0].category == ConstraintCategory.TECHNICAL  # Default
        assert constraints[0].description == "Minimal constraint"
        assert constraints[0].impact is None
        assert constraints[0].related_items == ()

    def test_extract_constraints_case_insensitive_category(self) -> None:
        """Test that category matching is case-insensitive."""
        llm_output = {
            "goals": [],
            "features": [],
            "constraints": [
                {"category": "TECHNICAL", "description": "C1"},
                {"category": "Business", "description": "C2"},
                {"category": "TIMELINE", "description": "C3"},
            ],
        }
        constraints = _extract_constraints(llm_output)
        assert len(constraints) == 3
        assert constraints[0].category == ConstraintCategory.TECHNICAL
        assert constraints[1].category == ConstraintCategory.BUSINESS
        assert constraints[2].category == ConstraintCategory.TIMELINE


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestExtractionEdgeCases:
    """Tests for edge cases in extraction helpers."""

    def test_extract_with_malformed_goal_entry(self) -> None:
        """Test that malformed goal entries are skipped."""
        llm_output = {
            "goals": [
                {"title": "Valid Goal", "description": "Desc", "priority": 1},
                "not a dict",  # Invalid entry
                {"title": "Another Valid", "description": "Desc", "priority": 2},
            ],
            "features": [],
            "constraints": [],
        }
        # Should skip invalid entries and extract valid ones
        goals = _extract_goals(llm_output)
        assert len(goals) == 2

    def test_extract_with_malformed_feature_entry(self) -> None:
        """Test that malformed feature entries are skipped."""
        llm_output = {
            "goals": [],
            "features": [
                {"name": "Valid Feature", "description": "Desc"},
                123,  # Invalid entry
                {"name": "Another Valid", "description": "Desc"},
            ],
            "constraints": [],
        }
        features = _extract_features(llm_output, [])
        assert len(features) == 2

    def test_extract_with_malformed_constraint_entry(self) -> None:
        """Test that malformed constraint entries are skipped."""
        llm_output = {
            "goals": [],
            "features": [],
            "constraints": [
                {"category": "technical", "description": "Valid"},
                None,  # Invalid entry
                {"category": "business", "description": "Another Valid"},
            ],
        }
        constraints = _extract_constraints(llm_output)
        assert len(constraints) == 2

    def test_extract_goals_with_unicode(self) -> None:
        """Test extracting goals with unicode content."""
        llm_output = {
            "goals": [
                {
                    "title": "Build App with Emoji Support",
                    "description": "Handle unicode properly",
                    "priority": 1,
                }
            ],
            "features": [],
            "constraints": [],
        }
        goals = _extract_goals(llm_output)
        assert len(goals) == 1
        assert "" in goals[0].title

    def test_extract_features_with_long_description(self) -> None:
        """Test extracting features with very long descriptions."""
        long_description = "A" * 10000
        llm_output = {
            "goals": [],
            "features": [{"name": "Feature", "description": long_description}],
            "constraints": [],
        }
        features = _extract_features(llm_output, [])
        assert len(features) == 1
        assert len(features[0].description) == 10000
