"""Tests for PM agent dependency analysis module (Story 6.5).

This module tests story dependency analysis including:
- Dependency extraction from text (keyword matching)
- Dependency graph building
- Cycle detection (DFS with coloring)
- Critical path analysis (longest path in DAG)
- Story updates with dependencies
- Full analysis orchestration

Test Organization:
    TestDependencyExtraction: Tests for _extract_dependencies_from_text function
    TestDependencyGraphBuilding: Tests for _build_dependency_graph function
    TestCycleDetection: Tests for _detect_cycles function
    TestCriticalPath: Tests for _find_critical_path function
    TestStoryUpdates: Tests for _update_stories_with_dependencies function
    TestAnalyzeDependencies: Tests for analyze_dependencies function
"""

from __future__ import annotations

import pytest

from yolo_developer.agents.pm.dependencies import (
    _build_dependency_graph,
    _detect_cycles,
    _extract_dependencies_from_text,
    _find_critical_path,
    _update_stories_with_dependencies,
    analyze_dependencies,
)
from yolo_developer.agents.pm.types import (
    AcceptanceCriterion,
    DependencyAnalysisResult,
    DependencyGraph,
    Story,
    StoryPriority,
    StoryStatus,
)


def _make_ac(
    ac_id: str = "AC1",
    given: str = "precondition",
    when: str = "action",
    then: str = "result",
) -> AcceptanceCriterion:
    """Create an acceptance criterion for testing."""
    return AcceptanceCriterion(
        id=ac_id,
        given=given,
        when=when,
        then=then,
    )


def _make_story(
    story_id: str = "story-001",
    title: str = "Test Story",
    action: str = "do something",
    benefit: str = "get value",
    dependencies: tuple[str, ...] = (),
    acs: tuple[AcceptanceCriterion, ...] | None = None,
) -> Story:
    """Create a story with configurable attributes for testing."""
    if acs is None:
        acs = (_make_ac(),)

    return Story(
        id=story_id,
        title=title,
        role="user",
        action=action,
        benefit=benefit,
        acceptance_criteria=acs,
        priority=StoryPriority.HIGH,
        status=StoryStatus.DRAFT,
        source_requirements=("req-001",),
        dependencies=dependencies,
        estimated_complexity="M",
    )


class TestDependencyExtraction:
    """Tests for _extract_dependencies_from_text function."""

    def test_no_dependencies_for_independent_story(self) -> None:
        """Independent story should have no dependencies extracted."""
        story = _make_story(
            story_id="story-001",
            title="User Login",
            action="log into the system",
        )
        other_stories = (
            _make_story(story_id="story-002", title="Dashboard View"),
            _make_story(story_id="story-003", title="Settings Page"),
        )
        all_stories = (story, *other_stories)

        deps = _extract_dependencies_from_text(story, all_stories)

        assert deps == []

    def test_after_keyword_finds_dependency(self) -> None:
        """'after X is complete' pattern should find dependency."""
        story = _make_story(
            story_id="story-002",
            title="Dashboard View",
            action="view my dashboard after authentication is complete",
        )
        auth_story = _make_story(
            story_id="story-001",
            title="User Authentication",
            action="authenticate users",
        )
        all_stories = (story, auth_story)

        deps = _extract_dependencies_from_text(story, all_stories)

        assert len(deps) == 1
        assert deps[0][0] == "story-001"
        assert "completion dependency" in deps[0][1]

    def test_requires_keyword_finds_dependency(self) -> None:
        """'requires X' pattern should find dependency."""
        story = _make_story(
            story_id="story-002",
            title="Protected Resource",
            action="access resource which requires authentication",
        )
        auth_story = _make_story(
            story_id="story-001",
            title="Authentication",
            action="implement authentication",
        )
        all_stories = (story, auth_story)

        deps = _extract_dependencies_from_text(story, all_stories)

        assert len(deps) == 1
        assert deps[0][0] == "story-001"
        assert "requirement dependency" in deps[0][1]

    def test_depends_on_keyword_finds_dependency(self) -> None:
        """'depends on X' pattern should find dependency."""
        story = _make_story(
            story_id="story-002",
            title="Feature B",
            action="implement feature B which depends on authentication",
        )
        auth_story = _make_story(
            story_id="story-001",
            title="Authentication Feature",
            action="implement authentication",
        )
        all_stories = (story, auth_story)

        deps = _extract_dependencies_from_text(story, all_stories)

        assert len(deps) == 1
        assert deps[0][0] == "story-001"

    def test_multiple_dependencies_found(self) -> None:
        """Story with multiple dependency keywords should find all."""
        story = _make_story(
            story_id="story-003",
            title="Report Generation",
            action="generate reports after data collection is complete and requires authentication",
        )
        auth_story = _make_story(
            story_id="story-001",
            title="Authentication",
            action="implement authentication",
        )
        data_story = _make_story(
            story_id="story-002",
            title="Data Collection",
            action="collect data from sources",
        )
        all_stories = (story, auth_story, data_story)

        deps = _extract_dependencies_from_text(story, all_stories)

        dep_ids = [d[0] for d in deps]
        # Should find BOTH dependencies based on different keyword patterns
        # "after data collection is complete" -> story-002 (completion dependency)
        # "requires authentication" -> story-001 (requirement dependency)
        assert len(dep_ids) == 2, f"Expected 2 dependencies, found {len(dep_ids)}: {dep_ids}"
        assert "story-001" in dep_ids, "Should find authentication dependency"
        assert "story-002" in dep_ids, "Should find data collection dependency"

    def test_dependency_in_acceptance_criteria(self) -> None:
        """Dependency keywords in AC should be detected."""
        ac = _make_ac(
            given="the user requires authentication first",
            when="user views dashboard",
            then="dashboard loads after authentication is complete",
        )
        story = _make_story(
            story_id="story-002",
            title="Dashboard",
            action="view dashboard",
            acs=(ac,),
        )
        auth_story = _make_story(
            story_id="story-001",
            title="Authentication",
            action="authenticate users",
        )
        all_stories = (story, auth_story)

        deps = _extract_dependencies_from_text(story, all_stories)

        assert len(deps) >= 1
        assert deps[0][0] == "story-001"

    def test_no_self_dependency(self) -> None:
        """Story should not depend on itself."""
        story = _make_story(
            story_id="story-001",
            title="Self Reference",
            action="this story requires self reference after self reference",
        )
        all_stories = (story,)

        deps = _extract_dependencies_from_text(story, all_stories)

        assert deps == []


class TestDependencyGraphBuilding:
    """Tests for _build_dependency_graph function."""

    def test_empty_stories_produces_empty_graph(self) -> None:
        """Empty stories should produce empty graph."""
        graph = _build_dependency_graph((), {})

        assert graph["nodes"] == []
        assert graph["edges"] == []
        assert graph["adjacency_list"] == {}
        assert graph["reverse_adjacency_list"] == {}

    def test_single_story_graph(self) -> None:
        """Single story should produce graph with one node, no edges."""
        story = _make_story(story_id="story-001")

        graph = _build_dependency_graph((story,), {})

        assert graph["nodes"] == ["story-001"]
        assert graph["edges"] == []
        assert graph["adjacency_list"]["story-001"] == []
        assert graph["reverse_adjacency_list"]["story-001"] == []

    def test_linear_chain_graph(self) -> None:
        """Linear chain A->B->C should produce correct edges."""
        stories = (
            _make_story(story_id="story-001"),
            _make_story(story_id="story-002"),
            _make_story(story_id="story-003"),
        )
        dependencies = {
            "story-002": [("story-001", "reason1")],
            "story-003": [("story-002", "reason2")],
        }

        graph = _build_dependency_graph(stories, dependencies)

        assert len(graph["nodes"]) == 3
        assert len(graph["edges"]) == 2

        # Check adjacency: story-002 depends on story-001
        assert "story-001" in graph["adjacency_list"]["story-002"]
        # Check adjacency: story-003 depends on story-002
        assert "story-002" in graph["adjacency_list"]["story-003"]

        # Check reverse adjacency
        assert "story-002" in graph["reverse_adjacency_list"]["story-001"]
        assert "story-003" in graph["reverse_adjacency_list"]["story-002"]

    def test_fan_out_graph(self) -> None:
        """Fan-out (A->B, A->C) should produce correct edges."""
        stories = (
            _make_story(story_id="story-001"),
            _make_story(story_id="story-002"),
            _make_story(story_id="story-003"),
        )
        dependencies = {
            "story-002": [("story-001", "reason1")],
            "story-003": [("story-001", "reason2")],
        }

        graph = _build_dependency_graph(stories, dependencies)

        # Both B and C depend on A
        assert "story-001" in graph["adjacency_list"]["story-002"]
        assert "story-001" in graph["adjacency_list"]["story-003"]

        # A is depended on by both B and C
        assert "story-002" in graph["reverse_adjacency_list"]["story-001"]
        assert "story-003" in graph["reverse_adjacency_list"]["story-001"]

    def test_fan_in_graph(self) -> None:
        """Fan-in (B->A, C->A) should produce correct edges."""
        stories = (
            _make_story(story_id="story-001"),
            _make_story(story_id="story-002"),
            _make_story(story_id="story-003"),
        )
        # A depends on both B and C
        dependencies = {
            "story-001": [("story-002", "reason1"), ("story-003", "reason2")],
        }

        graph = _build_dependency_graph(stories, dependencies)

        # A depends on B and C
        assert "story-002" in graph["adjacency_list"]["story-001"]
        assert "story-003" in graph["adjacency_list"]["story-001"]

        # B and C are depended on by A
        assert "story-001" in graph["reverse_adjacency_list"]["story-002"]
        assert "story-001" in graph["reverse_adjacency_list"]["story-003"]

    def test_invalid_dependency_filtered(self) -> None:
        """Dependencies to non-existent stories should be filtered."""
        story = _make_story(story_id="story-001")
        dependencies = {
            "story-001": [("nonexistent", "invalid")],
        }

        graph = _build_dependency_graph((story,), dependencies)

        assert graph["adjacency_list"]["story-001"] == []


class TestCycleDetection:
    """Tests for _detect_cycles function."""

    def test_no_cycles_in_linear_chain(self) -> None:
        """Linear chain A->B->C should have no cycles."""
        graph: DependencyGraph = {
            "nodes": ["A", "B", "C"],
            "edges": [
                {"from_story_id": "B", "to_story_id": "A", "reason": "dep"},
                {"from_story_id": "C", "to_story_id": "B", "reason": "dep"},
            ],
            "adjacency_list": {"A": [], "B": ["A"], "C": ["B"]},
            "reverse_adjacency_list": {"A": ["B"], "B": ["C"], "C": []},
        }

        cycles = _detect_cycles(graph)

        assert cycles == []

    def test_simple_cycle_detected(self) -> None:
        """Simple cycle A->B->A should be detected."""
        graph: DependencyGraph = {
            "nodes": ["A", "B"],
            "edges": [
                {"from_story_id": "A", "to_story_id": "B", "reason": "dep"},
                {"from_story_id": "B", "to_story_id": "A", "reason": "dep"},
            ],
            "adjacency_list": {"A": ["B"], "B": ["A"]},
            "reverse_adjacency_list": {"A": ["B"], "B": ["A"]},
        }

        cycles = _detect_cycles(graph)

        assert len(cycles) >= 1
        # The cycle should contain both A and B
        cycle_nodes = set()
        for cycle in cycles:
            cycle_nodes.update(cycle)
        assert "A" in cycle_nodes or "B" in cycle_nodes

    def test_longer_cycle_detected(self) -> None:
        """Longer cycle A->B->C->A should be detected."""
        graph: DependencyGraph = {
            "nodes": ["A", "B", "C"],
            "edges": [
                {"from_story_id": "A", "to_story_id": "B", "reason": "dep"},
                {"from_story_id": "B", "to_story_id": "C", "reason": "dep"},
                {"from_story_id": "C", "to_story_id": "A", "reason": "dep"},
            ],
            "adjacency_list": {"A": ["B"], "B": ["C"], "C": ["A"]},
            "reverse_adjacency_list": {"A": ["C"], "B": ["A"], "C": ["B"]},
        }

        cycles = _detect_cycles(graph)

        assert len(cycles) >= 1
        # Should find cycle involving A, B, C
        cycle_nodes = set()
        for cycle in cycles:
            cycle_nodes.update(cycle)
        assert len(cycle_nodes & {"A", "B", "C"}) >= 2

    def test_self_loop_detected(self) -> None:
        """Self-loop A->A should be detected."""
        graph: DependencyGraph = {
            "nodes": ["A"],
            "edges": [{"from_story_id": "A", "to_story_id": "A", "reason": "self"}],
            "adjacency_list": {"A": ["A"]},
            "reverse_adjacency_list": {"A": ["A"]},
        }

        cycles = _detect_cycles(graph)

        assert len(cycles) >= 1
        assert "A" in cycles[0]

    def test_multiple_independent_cycles(self) -> None:
        """Multiple independent cycles should all be detected."""
        graph: DependencyGraph = {
            "nodes": ["A", "B", "C", "D"],
            "edges": [
                {"from_story_id": "A", "to_story_id": "B", "reason": "dep"},
                {"from_story_id": "B", "to_story_id": "A", "reason": "dep"},
                {"from_story_id": "C", "to_story_id": "D", "reason": "dep"},
                {"from_story_id": "D", "to_story_id": "C", "reason": "dep"},
            ],
            "adjacency_list": {"A": ["B"], "B": ["A"], "C": ["D"], "D": ["C"]},
            "reverse_adjacency_list": {"A": ["B"], "B": ["A"], "C": ["D"], "D": ["C"]},
        }

        cycles = _detect_cycles(graph)

        # Should find at least one cycle involving A-B and one involving C-D
        all_cycle_nodes = set()
        for cycle in cycles:
            all_cycle_nodes.update(cycle)
        assert len(all_cycle_nodes & {"A", "B"}) >= 1 or len(all_cycle_nodes & {"C", "D"}) >= 1

    def test_empty_graph_no_cycles(self) -> None:
        """Empty graph should have no cycles."""
        graph: DependencyGraph = {
            "nodes": [],
            "edges": [],
            "adjacency_list": {},
            "reverse_adjacency_list": {},
        }

        cycles = _detect_cycles(graph)

        assert cycles == []


class TestCriticalPath:
    """Tests for _find_critical_path function."""

    def test_empty_graph_empty_path(self) -> None:
        """Empty graph should have empty critical paths."""
        graph: DependencyGraph = {
            "nodes": [],
            "edges": [],
            "adjacency_list": {},
            "reverse_adjacency_list": {},
        }

        paths, length = _find_critical_path(graph, [])

        assert paths == []
        assert length == 0

    def test_single_node_path_length_1(self) -> None:
        """Single node should have path length 1."""
        graph: DependencyGraph = {
            "nodes": ["A"],
            "edges": [],
            "adjacency_list": {"A": []},
            "reverse_adjacency_list": {"A": []},
        }

        paths, length = _find_critical_path(graph, [])

        assert len(paths) == 1
        assert paths[0] == ["A"]
        assert length == 1

    def test_linear_chain_critical_path(self) -> None:
        """Linear chain A->B->C should have path length 3."""
        graph: DependencyGraph = {
            "nodes": ["A", "B", "C"],
            "edges": [
                {"from_story_id": "B", "to_story_id": "A", "reason": "dep"},
                {"from_story_id": "C", "to_story_id": "B", "reason": "dep"},
            ],
            "adjacency_list": {"A": [], "B": ["A"], "C": ["B"]},
            "reverse_adjacency_list": {"A": ["B"], "B": ["C"], "C": []},
        }

        paths, length = _find_critical_path(graph, [])

        assert length == 3
        assert len(paths) == 1
        path = paths[0]
        assert len(path) == 3
        # Path should be A -> B -> C (in dependency order)
        assert "A" in path
        assert "B" in path
        assert "C" in path

    def test_diamond_pattern_finds_all_paths(self) -> None:
        """Diamond pattern should find ALL equal-length longest paths (AC4)."""
        # Diamond: A <- B, A <- C, B <- D, C <- D
        # D depends on both B and C, which both depend on A
        # Should return BOTH paths: A -> B -> D and A -> C -> D (length 3)
        graph: DependencyGraph = {
            "nodes": ["A", "B", "C", "D"],
            "edges": [
                {"from_story_id": "B", "to_story_id": "A", "reason": "dep"},
                {"from_story_id": "C", "to_story_id": "A", "reason": "dep"},
                {"from_story_id": "D", "to_story_id": "B", "reason": "dep"},
                {"from_story_id": "D", "to_story_id": "C", "reason": "dep"},
            ],
            "adjacency_list": {"A": [], "B": ["A"], "C": ["A"], "D": ["B", "C"]},
            "reverse_adjacency_list": {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []},
        }

        paths, length = _find_critical_path(graph, [])

        assert length == 3
        # Should find BOTH equal-length paths per AC4
        assert len(paths) == 2
        # Each path should contain A and D
        for path in paths:
            assert "A" in path
            assert "D" in path
            assert len(path) == 3

    def test_graph_with_cycles_returns_empty(self) -> None:
        """Graph with cycles should return empty paths."""
        graph: DependencyGraph = {
            "nodes": ["A", "B"],
            "edges": [
                {"from_story_id": "A", "to_story_id": "B", "reason": "dep"},
                {"from_story_id": "B", "to_story_id": "A", "reason": "dep"},
            ],
            "adjacency_list": {"A": ["B"], "B": ["A"]},
            "reverse_adjacency_list": {"A": ["B"], "B": ["A"]},
        }
        cycles = [["A", "B", "A"]]

        paths, length = _find_critical_path(graph, cycles)

        assert paths == []
        assert length == 0

    def test_disconnected_components(self) -> None:
        """Disconnected components should find paths for each."""
        graph: DependencyGraph = {
            "nodes": ["A", "B", "C"],
            "edges": [],
            "adjacency_list": {"A": [], "B": [], "C": []},
            "reverse_adjacency_list": {"A": [], "B": [], "C": []},
        }

        paths, length = _find_critical_path(graph, [])

        # Should find multiple single-node paths
        assert length == 1
        assert len(paths) >= 1


class TestStoryUpdates:
    """Tests for _update_stories_with_dependencies function."""

    def test_updates_story_dependencies(self) -> None:
        """Stories should have dependencies populated from graph."""
        stories = (
            _make_story(story_id="story-001"),
            _make_story(story_id="story-002"),
        )
        result: DependencyAnalysisResult = {
            "graph": {
                "nodes": ["story-001", "story-002"],
                "edges": [
                    {"from_story_id": "story-002", "to_story_id": "story-001", "reason": "dep"}
                ],
                "adjacency_list": {"story-001": [], "story-002": ["story-001"]},
                "reverse_adjacency_list": {"story-001": ["story-002"], "story-002": []},
            },
            "cycles": [],
            "critical_paths": [["story-001", "story-002"]],
            "critical_path_length": 2,
            "has_cycles": False,
            "analysis_notes": "test",
        }

        updated = _update_stories_with_dependencies(stories, result)

        assert updated[0].dependencies == ()  # story-001 has no deps
        assert updated[1].dependencies == ("story-001",)  # story-002 depends on story-001

    def test_preserves_other_story_fields(self) -> None:
        """Update should preserve all other story fields."""
        original = _make_story(
            story_id="story-001",
            title="Original Title",
            action="original action",
        )
        result: DependencyAnalysisResult = {
            "graph": {
                "nodes": ["story-001"],
                "edges": [],
                "adjacency_list": {"story-001": []},
                "reverse_adjacency_list": {"story-001": []},
            },
            "cycles": [],
            "critical_paths": [],
            "critical_path_length": 0,
            "has_cycles": False,
            "analysis_notes": "test",
        }

        updated = _update_stories_with_dependencies((original,), result)

        assert updated[0].id == original.id
        assert updated[0].title == original.title
        assert updated[0].action == original.action
        assert updated[0].role == original.role
        assert updated[0].benefit == original.benefit
        assert updated[0].priority == original.priority
        assert updated[0].status == original.status

    def test_empty_stories_returns_empty(self) -> None:
        """Empty stories should return empty tuple."""
        result: DependencyAnalysisResult = {
            "graph": {
                "nodes": [],
                "edges": [],
                "adjacency_list": {},
                "reverse_adjacency_list": {},
            },
            "cycles": [],
            "critical_paths": [],
            "critical_path_length": 0,
            "has_cycles": False,
            "analysis_notes": "test",
        }

        updated = _update_stories_with_dependencies((), result)

        assert updated == ()


class TestAnalyzeDependencies:
    """Tests for analyze_dependencies function (integration)."""

    @pytest.mark.asyncio
    async def test_empty_stories_analysis(self) -> None:
        """Empty stories should return empty analysis result."""
        result = await analyze_dependencies(())

        assert result["graph"]["nodes"] == []
        assert result["graph"]["edges"] == []
        assert result["cycles"] == []
        assert result["critical_paths"] == []
        assert result["critical_path_length"] == 0
        assert result["has_cycles"] is False

    @pytest.mark.asyncio
    async def test_single_story_analysis(self) -> None:
        """Single story should produce valid analysis."""
        story = _make_story(story_id="story-001")

        result = await analyze_dependencies((story,))

        assert result["graph"]["nodes"] == ["story-001"]
        assert result["cycles"] == []
        assert result["has_cycles"] is False

    @pytest.mark.asyncio
    async def test_analysis_with_dependency_keywords(self) -> None:
        """Analysis should detect dependencies from keywords."""
        # Story 2 depends on story 1's authentication
        story1 = _make_story(
            story_id="story-001",
            title="User Authentication",
            action="implement user authentication",
        )
        story2 = _make_story(
            story_id="story-002",
            title="Dashboard",
            action="show dashboard after authentication is complete",
        )

        result = await analyze_dependencies((story1, story2))

        assert len(result["graph"]["nodes"]) == 2
        # Should find dependency (story-002 depends on story-001)
        # The keyword matching should find 'authentication'
        assert result["has_cycles"] is False

    @pytest.mark.asyncio
    async def test_analysis_result_structure(self) -> None:
        """Analysis result should have all required fields."""
        story = _make_story(story_id="story-001")

        result = await analyze_dependencies((story,))

        # Check all required fields exist
        assert "graph" in result
        assert "cycles" in result
        assert "critical_paths" in result
        assert "critical_path_length" in result
        assert "has_cycles" in result
        assert "analysis_notes" in result

        # Check graph structure
        assert "nodes" in result["graph"]
        assert "edges" in result["graph"]
        assert "adjacency_list" in result["graph"]
        assert "reverse_adjacency_list" in result["graph"]

    @pytest.mark.asyncio
    async def test_analysis_notes_contain_summary(self) -> None:
        """Analysis notes should contain meaningful summary."""
        stories = (
            _make_story(story_id="story-001"),
            _make_story(story_id="story-002"),
        )

        result = await analyze_dependencies(stories)

        assert "2 stories" in result["analysis_notes"]

    @pytest.mark.asyncio
    async def test_multiple_stories_analysis(self) -> None:
        """Multiple stories should be analyzed correctly."""
        stories = tuple(_make_story(story_id=f"story-{i:03d}") for i in range(1, 6))

        result = await analyze_dependencies(stories)

        assert len(result["graph"]["nodes"]) == 5
        assert result["has_cycles"] is False
