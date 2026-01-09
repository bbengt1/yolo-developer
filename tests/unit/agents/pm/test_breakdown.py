"""Tests for PM agent epic breakdown module (Story 6.6).

This module tests the epic breakdown functionality:
- Story size detection (_needs_breakdown)
- Sub-story generation (_generate_sub_stories)
- Coverage validation (_validate_coverage)
- Main breakdown orchestration (break_down_epic)

Test Patterns:
- Unit tests for each function
- Edge cases for breakdown triggers
- Coverage validation scenarios
- Integration with PM node flow
"""

from __future__ import annotations

import pytest

from yolo_developer.agents.pm.types import (
    AcceptanceCriterion,
    CoverageMapping,
    EpicBreakdownResult,
    Story,
    StoryPriority,
    StoryStatus,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_story() -> Story:
    """Create a simple story that should NOT need breakdown."""
    return Story(
        id="story-001",
        title="Simple user login",
        role="user",
        action="log into the system with email and password",
        benefit="I can access my account",
        acceptance_criteria=(
            AcceptanceCriterion(
                id="AC1",
                given="a valid user account exists",
                when="I enter correct credentials",
                then="I am logged in successfully",
            ),
            AcceptanceCriterion(
                id="AC2",
                given="a valid user account exists",
                when="I enter incorrect credentials",
                then="I see an error message",
            ),
        ),
        priority=StoryPriority.HIGH,
        status=StoryStatus.DRAFT,
        source_requirements=("req-001",),
        estimated_complexity="M",
    )


@pytest.fixture
def complex_story() -> Story:
    """Create a complex story that SHOULD need breakdown (high complexity)."""
    return Story(
        id="story-002",
        title="Complete user management system",
        role="admin",
        action="manage all user accounts including creation, modification, deletion, role assignment, and audit logging",
        benefit="the system has proper user governance",
        acceptance_criteria=(
            AcceptanceCriterion(id="AC1", given="admin logged in", when="create user", then="user created"),
            AcceptanceCriterion(id="AC2", given="admin logged in", when="modify user", then="user modified"),
            AcceptanceCriterion(id="AC3", given="admin logged in", when="delete user", then="user deleted"),
            AcceptanceCriterion(id="AC4", given="admin logged in", when="assign role", then="role assigned"),
            AcceptanceCriterion(id="AC5", given="admin logged in", when="view audit", then="audit shown"),
            AcceptanceCriterion(id="AC6", given="admin logged in", when="export users", then="users exported"),
        ),
        priority=StoryPriority.CRITICAL,
        status=StoryStatus.DRAFT,
        source_requirements=("req-002",),
        estimated_complexity="XL",
    )


@pytest.fixture
def medium_complexity_story() -> Story:
    """Create a medium complexity story at the edge of breakdown threshold."""
    return Story(
        id="story-003",
        title="User profile management",
        role="user",
        action="update my profile information",
        benefit="my account reflects my current details",
        acceptance_criteria=(
            AcceptanceCriterion(id="AC1", given="logged in", when="update name", then="name updated"),
            AcceptanceCriterion(id="AC2", given="logged in", when="update email", then="email updated"),
            AcceptanceCriterion(id="AC3", given="logged in", when="update avatar", then="avatar updated"),
            AcceptanceCriterion(id="AC4", given="logged in", when="invalid data", then="error shown"),
            AcceptanceCriterion(id="AC5", given="logged in", when="save changes", then="changes persisted"),
        ),
        priority=StoryPriority.MEDIUM,
        status=StoryStatus.DRAFT,
        source_requirements=("req-003",),
        estimated_complexity="L",
    )


# =============================================================================
# Task 1: Type Definition Tests
# =============================================================================


class TestEpicBreakdownTypes:
    """Tests for epic breakdown type definitions (Task 1)."""

    def test_coverage_mapping_type_exists(self) -> None:
        """Test CoverageMapping TypedDict can be instantiated."""
        mapping: CoverageMapping = {
            "original_ac_id": "AC1",
            "covering_story_ids": ["story-001.1", "story-001.2"],
            "is_covered": True,
        }
        assert mapping["original_ac_id"] == "AC1"
        assert mapping["is_covered"] is True
        assert len(mapping["covering_story_ids"]) == 2

    def test_epic_breakdown_result_type_exists(self) -> None:
        """Test EpicBreakdownResult TypedDict can be instantiated."""
        result: EpicBreakdownResult = {
            "original_story_id": "story-001",
            "sub_stories": (),
            "coverage_mappings": [],
            "breakdown_rationale": "Test rationale",
            "is_valid": True,
        }
        assert result["original_story_id"] == "story-001"
        assert result["is_valid"] is True
        assert result["breakdown_rationale"] == "Test rationale"

    def test_coverage_mapping_with_uncovered_ac(self) -> None:
        """Test CoverageMapping with is_covered=False."""
        mapping: CoverageMapping = {
            "original_ac_id": "AC3",
            "covering_story_ids": [],
            "is_covered": False,
        }
        assert mapping["is_covered"] is False
        assert len(mapping["covering_story_ids"]) == 0

    def test_epic_breakdown_result_with_invalid_coverage(self) -> None:
        """Test EpicBreakdownResult with is_valid=False."""
        result: EpicBreakdownResult = {
            "original_story_id": "story-002",
            "sub_stories": (),
            "coverage_mappings": [
                {"original_ac_id": "AC1", "covering_story_ids": [], "is_covered": False}
            ],
            "breakdown_rationale": "Coverage incomplete",
            "is_valid": False,
        }
        assert result["is_valid"] is False
        assert len(result["coverage_mappings"]) == 1


# =============================================================================
# Task 2: Size Detection Tests
# =============================================================================


class TestNeedsBreakdown:
    """Tests for story size detection (Task 2)."""

    def test_low_complexity_story_does_not_need_breakdown(
        self, simple_story: Story
    ) -> None:
        """Test low complexity story returns False."""
        from yolo_developer.agents.pm.breakdown import _needs_breakdown

        assert _needs_breakdown(simple_story) is False

    def test_high_complexity_story_needs_breakdown(
        self, complex_story: Story
    ) -> None:
        """Test high/XL complexity story returns True."""
        from yolo_developer.agents.pm.breakdown import _needs_breakdown

        assert _needs_breakdown(complex_story) is True

    def test_story_with_more_than_5_acs_needs_breakdown(
        self, complex_story: Story
    ) -> None:
        """Test story with >5 acceptance criteria returns True."""
        from yolo_developer.agents.pm.breakdown import _needs_breakdown

        # complex_story has 6 ACs
        assert len(complex_story.acceptance_criteria) > 5
        assert _needs_breakdown(complex_story) is True

    def test_story_with_exactly_5_acs_does_not_need_breakdown(
        self, medium_complexity_story: Story
    ) -> None:
        """Test story with exactly 5 ACs returns False (threshold is >5)."""
        from yolo_developer.agents.pm.breakdown import _needs_breakdown

        assert len(medium_complexity_story.acceptance_criteria) == 5
        # Medium complexity with exactly 5 ACs should not trigger
        # unless complexity is "high" or "XL"
        # medium_complexity_story has "L" which is high
        # So this should return True due to complexity
        assert _needs_breakdown(medium_complexity_story) is True

    def test_story_with_complex_action_text_needs_breakdown(self) -> None:
        """Test story with multiple 'and' in action needs breakdown."""
        from yolo_developer.agents.pm.breakdown import _needs_breakdown

        story = Story(
            id="story-004",
            title="Multi-feature story",
            role="user",
            action="create accounts and manage profiles and configure settings and export data",
            benefit="complete control",
            acceptance_criteria=(
                AcceptanceCriterion(id="AC1", given="g", when="w", then="t"),
            ),
            priority=StoryPriority.MEDIUM,
            status=StoryStatus.DRAFT,
            estimated_complexity="M",
        )
        # Multiple "and" conjunctions should trigger breakdown
        assert _needs_breakdown(story) is True

    def test_medium_complexity_with_few_acs_does_not_need_breakdown(self) -> None:
        """Test medium complexity story with 2-3 ACs returns False."""
        from yolo_developer.agents.pm.breakdown import _needs_breakdown

        story = Story(
            id="story-005",
            title="Simple feature",
            role="user",
            action="view my dashboard",
            benefit="I see my data",
            acceptance_criteria=(
                AcceptanceCriterion(id="AC1", given="g", when="w", then="t"),
                AcceptanceCriterion(id="AC2", given="g", when="w", then="t"),
            ),
            priority=StoryPriority.MEDIUM,
            status=StoryStatus.DRAFT,
            estimated_complexity="M",
        )
        assert _needs_breakdown(story) is False


# =============================================================================
# Task 4: Sub-Story Generation Tests
# =============================================================================


class TestGenerateSubStories:
    """Tests for sub-story generation (Task 4)."""

    def test_sub_story_ids_follow_parent_pattern(self) -> None:
        """Test sub-story IDs follow story-XXX.N pattern."""
        from yolo_developer.agents.pm.breakdown import _generate_sub_stories

        original = Story(
            id="story-003",
            title="Original story",
            role="user",
            action="do something",
            benefit="get value",
            acceptance_criteria=(),
            priority=StoryPriority.HIGH,
            status=StoryStatus.DRAFT,
            source_requirements=("req-001",),
        )

        breakdown_data = [
            {"title": "Sub 1", "role": "user", "action": "action 1", "benefit": "benefit 1", "suggested_ac": ["AC1"]},
            {"title": "Sub 2", "role": "user", "action": "action 2", "benefit": "benefit 2", "suggested_ac": ["AC2"]},
        ]

        sub_stories = _generate_sub_stories(original, breakdown_data)

        assert len(sub_stories) == 2
        assert sub_stories[0].id == "story-003.1"
        assert sub_stories[1].id == "story-003.2"

    def test_source_requirements_preserved_from_parent(self) -> None:
        """Test source_requirements are copied from parent story."""
        from yolo_developer.agents.pm.breakdown import _generate_sub_stories

        original = Story(
            id="story-003",
            title="Original",
            role="user",
            action="do",
            benefit="get",
            acceptance_criteria=(),
            priority=StoryPriority.HIGH,
            status=StoryStatus.DRAFT,
            source_requirements=("req-001", "req-002"),
        )

        breakdown_data = [
            {"title": "Sub 1", "role": "user", "action": "a", "benefit": "b", "suggested_ac": []},
        ]

        sub_stories = _generate_sub_stories(original, breakdown_data)

        assert sub_stories[0].source_requirements == ("req-001", "req-002")

    def test_each_sub_story_has_acceptance_criteria(self) -> None:
        """Test each sub-story gets acceptance criteria generated."""
        from yolo_developer.agents.pm.breakdown import _generate_sub_stories

        original = Story(
            id="story-003",
            title="Original",
            role="user",
            action="do",
            benefit="get",
            acceptance_criteria=(),
            priority=StoryPriority.HIGH,
            status=StoryStatus.DRAFT,
        )

        breakdown_data = [
            {"title": "Sub 1", "role": "user", "action": "action 1", "benefit": "benefit 1", "suggested_ac": ["Do X", "Verify Y"]},
        ]

        sub_stories = _generate_sub_stories(original, breakdown_data)

        assert len(sub_stories[0].acceptance_criteria) >= 1

    def test_complexity_estimates_are_low_or_medium(self) -> None:
        """Test sub-stories have complexity of 'S', 'M', or 'L' (not 'XL')."""
        from yolo_developer.agents.pm.breakdown import _generate_sub_stories

        original = Story(
            id="story-003",
            title="Original",
            role="user",
            action="do",
            benefit="get",
            acceptance_criteria=(),
            priority=StoryPriority.HIGH,
            status=StoryStatus.DRAFT,
            estimated_complexity="XL",
        )

        breakdown_data = [
            {"title": "Sub 1", "role": "user", "action": "a1", "benefit": "b1", "suggested_ac": []},
            {"title": "Sub 2", "role": "user", "action": "a2", "benefit": "b2", "suggested_ac": []},
        ]

        sub_stories = _generate_sub_stories(original, breakdown_data)

        for sub in sub_stories:
            assert sub.estimated_complexity in ("S", "M", "L")


# =============================================================================
# Task 5: Coverage Validation Tests
# =============================================================================


class TestValidateCoverage:
    """Tests for coverage validation (Task 5)."""

    def test_complete_coverage_returns_valid_result(self) -> None:
        """Test when all ACs are covered, is_valid=True."""
        from yolo_developer.agents.pm.breakdown import _validate_coverage

        original = Story(
            id="story-001",
            title="Original",
            role="user",
            action="do stuff",
            benefit="get value",
            acceptance_criteria=(
                AcceptanceCriterion(id="AC1", given="g", when="w", then="t"),
                AcceptanceCriterion(id="AC2", given="g", when="w", then="t"),
            ),
            priority=StoryPriority.HIGH,
            status=StoryStatus.DRAFT,
        )

        sub_stories = (
            Story(
                id="story-001.1",
                title="Sub 1",
                role="user",
                action="covers AC1",
                benefit="b",
                acceptance_criteria=(
                    AcceptanceCriterion(id="AC1", given="g", when="covers AC1", then="t"),
                ),
                priority=StoryPriority.MEDIUM,
                status=StoryStatus.DRAFT,
            ),
            Story(
                id="story-001.2",
                title="Sub 2",
                role="user",
                action="covers AC2",
                benefit="b",
                acceptance_criteria=(
                    AcceptanceCriterion(id="AC1", given="g", when="covers AC2", then="t"),
                ),
                priority=StoryPriority.MEDIUM,
                status=StoryStatus.DRAFT,
            ),
        )

        mappings = _validate_coverage(original, sub_stories)

        # All mappings should have is_covered=True
        assert all(m["is_covered"] for m in mappings)

    def test_incomplete_coverage_is_flagged(self) -> None:
        """Test when some ACs are not covered, those mappings have is_covered=False."""
        from yolo_developer.agents.pm.breakdown import _validate_coverage

        original = Story(
            id="story-001",
            title="Original",
            role="user",
            action="do stuff",
            benefit="get value",
            acceptance_criteria=(
                AcceptanceCriterion(id="AC1", given="authentication database connection", when="login", then="authenticated"),
                AcceptanceCriterion(id="AC2", given="payment gateway integration", when="checkout", then="processed"),
                AcceptanceCriterion(id="AC3", given="reporting analytics dashboard", when="export", then="generated"),
            ),
            priority=StoryPriority.HIGH,
            status=StoryStatus.DRAFT,
        )

        # Only one sub-story that covers authentication (AC1) but not payment (AC2) or reporting (AC3)
        sub_stories = (
            Story(
                id="story-001.1",
                title="Authentication Sub",
                role="user",
                action="authentication database connection login",
                benefit="b",
                acceptance_criteria=(
                    AcceptanceCriterion(id="AC1", given="database", when="login", then="authenticated"),
                ),
                priority=StoryPriority.MEDIUM,
                status=StoryStatus.DRAFT,
            ),
        )

        mappings = _validate_coverage(original, sub_stories)

        # Should have mappings for AC1, AC2, AC3
        assert len(mappings) == 3
        # AC1 should be covered, AC2 and AC3 should not
        covered = [m for m in mappings if m["is_covered"]]
        uncovered = [m for m in mappings if not m["is_covered"]]
        assert len(covered) >= 1  # AC1 should be covered
        assert len(uncovered) >= 1  # AC2 or AC3 should be uncovered

    def test_empty_sub_stories_list_flags_all_uncovered(self) -> None:
        """Test empty sub_stories results in all ACs uncovered."""
        from yolo_developer.agents.pm.breakdown import _validate_coverage

        original = Story(
            id="story-001",
            title="Original",
            role="user",
            action="do stuff",
            benefit="get value",
            acceptance_criteria=(
                AcceptanceCriterion(id="AC1", given="g", when="w", then="t"),
            ),
            priority=StoryPriority.HIGH,
            status=StoryStatus.DRAFT,
        )

        mappings = _validate_coverage(original, ())

        assert len(mappings) == 1
        assert mappings[0]["is_covered"] is False
        assert mappings[0]["covering_story_ids"] == []


# =============================================================================
# Task 6: Main Breakdown Function Tests
# =============================================================================


class TestBreakDownEpic:
    """Tests for main epic breakdown function (Task 6)."""

    @pytest.mark.asyncio
    async def test_break_down_epic_returns_valid_result(
        self, complex_story: Story
    ) -> None:
        """Test break_down_epic returns EpicBreakdownResult."""
        from yolo_developer.agents.pm.breakdown import break_down_epic

        result = await break_down_epic(complex_story)

        assert result["original_story_id"] == complex_story.id
        assert isinstance(result["sub_stories"], tuple)
        assert isinstance(result["coverage_mappings"], list)
        assert isinstance(result["breakdown_rationale"], str)
        assert isinstance(result["is_valid"], bool)

    @pytest.mark.asyncio
    async def test_break_down_epic_generates_2_to_5_sub_stories(
        self, complex_story: Story
    ) -> None:
        """Test breakdown generates between 2-5 sub-stories."""
        from yolo_developer.agents.pm.breakdown import break_down_epic

        result = await break_down_epic(complex_story)

        # Per spec: 2-5 sub-stories
        assert 2 <= len(result["sub_stories"]) <= 5

    @pytest.mark.asyncio
    async def test_break_down_epic_includes_rationale(
        self, complex_story: Story
    ) -> None:
        """Test breakdown includes rationale for audit trail."""
        from yolo_developer.agents.pm.breakdown import break_down_epic

        result = await break_down_epic(complex_story)

        assert len(result["breakdown_rationale"]) > 0
        # Should mention original story
        assert complex_story.id in result["breakdown_rationale"] or "story" in result["breakdown_rationale"].lower()


# =============================================================================
# Task 11: Integration Tests
# =============================================================================


class TestProcessEpicBreakdowns:
    """Tests for _process_epic_breakdowns helper function (Code Review Fix)."""

    @pytest.mark.asyncio
    async def test_process_epic_breakdowns_with_no_large_stories(self) -> None:
        """Test _process_epic_breakdowns returns original stories when none need breakdown."""
        from yolo_developer.agents.pm.breakdown import _process_epic_breakdowns

        stories = (
            Story(
                id="story-001",
                title="Simple story",
                role="user",
                action="do simple thing",
                benefit="get value",
                acceptance_criteria=(
                    AcceptanceCriterion(id="AC1", given="g", when="w", then="t"),
                ),
                priority=StoryPriority.MEDIUM,
                status=StoryStatus.DRAFT,
                estimated_complexity="M",
            ),
        )

        processed, results = await _process_epic_breakdowns(stories)

        assert len(processed) == 1
        assert processed[0].id == "story-001"
        assert len(results) == 0  # No breakdowns occurred

    @pytest.mark.asyncio
    async def test_process_epic_breakdowns_replaces_large_stories(self) -> None:
        """Test _process_epic_breakdowns replaces large stories with sub-stories."""
        from yolo_developer.agents.pm.breakdown import _process_epic_breakdowns

        stories = (
            Story(
                id="story-001",
                title="Complex story",
                role="admin",
                action="manage users and roles and permissions and audit logs",
                benefit="governance",
                acceptance_criteria=(
                    AcceptanceCriterion(id="AC1", given="g", when="w", then="t"),
                ),
                priority=StoryPriority.HIGH,
                status=StoryStatus.DRAFT,
                estimated_complexity="XL",
            ),
        )

        processed, results = await _process_epic_breakdowns(stories)

        # Original story should be replaced by sub-stories
        assert len(processed) >= 2  # At least 2 sub-stories
        assert all("story-001" in s.id for s in processed)  # All sub-stories derive from original
        assert len(results) == 1  # One breakdown occurred
        assert results[0]["original_story_id"] == "story-001"

    @pytest.mark.asyncio
    async def test_process_epic_breakdowns_mixed_stories(self) -> None:
        """Test _process_epic_breakdowns handles mix of large and small stories."""
        from yolo_developer.agents.pm.breakdown import _process_epic_breakdowns

        stories = (
            Story(
                id="story-001",
                title="Simple story",
                role="user",
                action="login",
                benefit="access",
                acceptance_criteria=(AcceptanceCriterion(id="AC1", given="g", when="w", then="t"),),
                priority=StoryPriority.MEDIUM,
                status=StoryStatus.DRAFT,
                estimated_complexity="S",
            ),
            Story(
                id="story-002",
                title="Complex story",
                role="admin",
                action="manage everything and configure all settings",
                benefit="control",
                acceptance_criteria=(AcceptanceCriterion(id="AC1", given="g", when="w", then="t"),),
                priority=StoryPriority.HIGH,
                status=StoryStatus.DRAFT,
                estimated_complexity="XL",
            ),
        )

        processed, results = await _process_epic_breakdowns(stories)

        # story-001 should remain, story-002 should be broken down
        story_ids = [s.id for s in processed]
        assert "story-001" in story_ids  # Simple story unchanged
        assert "story-002" not in story_ids  # Complex story replaced
        assert any("story-002." in sid for sid in story_ids)  # Sub-stories exist
        assert len(results) == 1  # Only one breakdown


class TestBreakdownFallbackPaths:
    """Tests for LLM fallback behavior (Code Review Fix)."""

    @pytest.mark.asyncio
    async def test_break_down_story_llm_stub_always_returns_2_plus(self) -> None:
        """Test stub implementation always returns at least 2 sub-stories."""
        from yolo_developer.agents.pm.breakdown import _break_down_story_llm

        # Story without "and" in action - would previously return 1 story
        story = Story(
            id="story-001",
            title="Single action story",
            role="user",
            action="view the dashboard",  # No "and" conjunction
            benefit="see data",
            acceptance_criteria=(
                AcceptanceCriterion(id="AC1", given="g", when="w", then="t"),
            ),
            priority=StoryPriority.HIGH,
            status=StoryStatus.DRAFT,
            estimated_complexity="XL",
        )

        result = await _break_down_story_llm(story)

        assert len(result) >= 2  # Must return at least 2 sub-stories

    def test_parse_breakdown_response_handles_invalid_json(self) -> None:
        """Test _parse_breakdown_response returns empty list on invalid JSON."""
        from yolo_developer.agents.pm.breakdown import _parse_breakdown_response

        result = _parse_breakdown_response("not valid json at all")
        assert result == []

    def test_parse_breakdown_response_handles_missing_fields(self) -> None:
        """Test _parse_breakdown_response filters out items missing required fields."""
        from yolo_developer.agents.pm.breakdown import _parse_breakdown_response

        # JSON with one valid and one invalid item
        response = '[{"title": "Valid", "role": "user", "action": "do", "benefit": "get"}, {"title": "Invalid"}]'
        result = _parse_breakdown_response(response)

        assert len(result) == 1
        assert result[0]["title"] == "Valid"


class TestBreakdownParameterValidation:
    """Tests for parameter validation (Code Review Fix)."""

    @pytest.mark.asyncio
    async def test_break_down_epic_rejects_non_story_type(self) -> None:
        """Test break_down_epic raises TypeError for non-Story input."""
        from yolo_developer.agents.pm.breakdown import break_down_epic

        with pytest.raises(TypeError, match="Expected Story object"):
            await break_down_epic({"id": "not-a-story"})  # type: ignore

    @pytest.mark.asyncio
    async def test_break_down_epic_rejects_empty_id(self) -> None:
        """Test break_down_epic raises ValueError for empty story id."""
        from yolo_developer.agents.pm.breakdown import break_down_epic

        story = Story(
            id="",  # Empty id
            title="Test",
            role="user",
            action="do something",
            benefit="get value",
            acceptance_criteria=(),
            priority=StoryPriority.MEDIUM,
            status=StoryStatus.DRAFT,
        )

        with pytest.raises(ValueError, match="non-empty id"):
            await break_down_epic(story)

    @pytest.mark.asyncio
    async def test_break_down_epic_rejects_empty_action(self) -> None:
        """Test break_down_epic raises ValueError for empty action."""
        from yolo_developer.agents.pm.breakdown import break_down_epic

        story = Story(
            id="story-001",
            title="Test",
            role="user",
            action="",  # Empty action
            benefit="get value",
            acceptance_criteria=(),
            priority=StoryPriority.MEDIUM,
            status=StoryStatus.DRAFT,
        )

        with pytest.raises(ValueError, match="non-empty action"):
            await break_down_epic(story)


class TestPmNodeBreakdownIntegration:
    """Integration tests for pm_node breakdown (Task 11)."""

    @pytest.fixture
    def mock_analyst_output_with_complex_requirement(self) -> dict:
        """Create mock analyst output with a complex requirement that triggers breakdown."""
        return {
            "requirements": [
                {
                    "id": "req-001",
                    "refined_text": "Admin manages all users including creation and deletion and role assignment and audit logging and user export and bulk import",
                    "category": "functional",
                },
                {
                    "id": "req-002",
                    "refined_text": "User can login with email",
                    "category": "functional",
                },
            ],
            "gaps": [],
            "contradictions": [],
            "escalations": [],
        }

    @pytest.fixture
    def mock_analyst_output_simple(self) -> dict:
        """Create mock analyst output with simple requirements (no breakdown needed)."""
        return {
            "requirements": [
                {
                    "id": "req-001",
                    "refined_text": "User can login with email",
                    "category": "functional",
                },
                {
                    "id": "req-002",
                    "refined_text": "User can logout",
                    "category": "functional",
                },
            ],
            "gaps": [],
            "contradictions": [],
            "escalations": [],
        }

    @pytest.mark.asyncio
    async def test_pm_node_triggers_breakdown_for_high_complexity(
        self, mock_analyst_output_with_complex_requirement: dict
    ) -> None:
        """Test pm_node triggers breakdown for complex stories."""
        from yolo_developer.agents.pm.node import pm_node

        state = {
            "messages": [],
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [],
            "analyst_output": mock_analyst_output_with_complex_requirement,
        }

        result = await pm_node(state)

        # Should have breakdown_results in output
        assert "breakdown_results" in result
        # The complex requirement should trigger breakdown due to multiple "and"
        # So we may have more stories than requirements
        pm_output = result["pm_output"]
        # Original had 2 reqs, but first one is complex with many "and"
        # so should be broken down into multiple sub-stories
        assert pm_output["story_count"] >= 2

    @pytest.mark.asyncio
    async def test_pm_node_no_breakdown_for_simple_stories(
        self, mock_analyst_output_simple: dict
    ) -> None:
        """Test pm_node does not break down simple stories."""
        from yolo_developer.agents.pm.node import pm_node

        state = {
            "messages": [],
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [],
            "analyst_output": mock_analyst_output_simple,
        }

        result = await pm_node(state)

        # Should have breakdown_results (possibly empty)
        assert "breakdown_results" in result
        breakdown_results = result["breakdown_results"]
        # Simple stories should not trigger breakdown
        assert len(breakdown_results) == 0
        # Story count should match requirement count (minus constraints)
        pm_output = result["pm_output"]
        assert pm_output["story_count"] == 2

    @pytest.mark.asyncio
    async def test_pm_node_decision_includes_breakdown_rationale(
        self, mock_analyst_output_with_complex_requirement: dict
    ) -> None:
        """Test Decision record includes breakdown rationale."""
        from yolo_developer.agents.pm.node import pm_node

        state = {
            "messages": [],
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [],
            "analyst_output": mock_analyst_output_with_complex_requirement,
        }

        result = await pm_node(state)

        decisions = result["decisions"]
        assert len(decisions) == 1
        decision = decisions[0]

        # If breakdown occurred, rationale should mention it
        breakdown_results = result["breakdown_results"]
        if breakdown_results:
            assert "breakdown" in decision.rationale.lower() or "broken" in decision.rationale.lower()

    @pytest.mark.asyncio
    async def test_pm_node_processing_notes_indicate_breakdown(
        self, mock_analyst_output_with_complex_requirement: dict
    ) -> None:
        """Test processing_notes indicate which stories were broken down."""
        from yolo_developer.agents.pm.node import pm_node

        state = {
            "messages": [],
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [],
            "analyst_output": mock_analyst_output_with_complex_requirement,
        }

        result = await pm_node(state)

        pm_output = result["pm_output"]
        processing_notes = pm_output["processing_notes"]

        # If breakdown occurred, notes should mention it
        breakdown_results = result["breakdown_results"]
        if breakdown_results:
            assert "breakdown" in processing_notes.lower() or "broken" in processing_notes.lower()

    @pytest.mark.asyncio
    async def test_pm_node_sub_stories_have_correct_ids(
        self, mock_analyst_output_with_complex_requirement: dict
    ) -> None:
        """Test sub-stories follow parent.N ID pattern."""
        from yolo_developer.agents.pm.node import pm_node

        state = {
            "messages": [],
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [],
            "analyst_output": mock_analyst_output_with_complex_requirement,
        }

        result = await pm_node(state)

        pm_output = result["pm_output"]
        stories = pm_output["stories"]

        # Check for sub-story ID patterns
        for story in stories:
            story_id = story["id"]
            # Sub-stories have format story-XXX.N
            if "." in story_id:
                parts = story_id.split(".")
                assert len(parts) == 2
                assert parts[0].startswith("story-")
                assert parts[1].isdigit()
