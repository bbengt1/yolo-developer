"""Dependency analysis for PM agent stories (Story 6.5).

This module provides dependency analysis capabilities for user stories:

- _extract_dependencies_from_text: Extract dependencies using keyword matching
- _infer_dependencies_llm: Infer dependencies using LLM analysis
- _build_dependency_graph: Build a graph structure from dependencies
- _detect_cycles: Detect circular dependencies in the graph
- _find_critical_path: Find the longest dependency chain
- analyze_dependencies: Main orchestration function
- update_stories_with_dependencies: Update stories with their dependencies

Key Concepts:
- **Text-Based Extraction**: Looks for keywords like "after", "requires", "depends"
- **LLM Inference**: Uses LLM for smart dependency detection (optional)
- **Graph Algorithms**: DFS for cycle detection, topological sort for critical path
- **Immutable Updates**: Stories are immutable dataclasses, creates new instances

Example:
    >>> from yolo_developer.agents.pm.dependencies import analyze_dependencies
    >>> stories = (story1, story2, story3)
    >>> result = await analyze_dependencies(stories)
    >>> result["has_cycles"]
    False
    >>> result["critical_path_length"]
    2

Architecture Note:
    Per ADR-001, all result types are TypedDict for internal state.
    Per ADR-005, this module doesn't mutate state directly.
    Per ARCH-QUALITY-5, LLM calls are async, graph algorithms are sync.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from yolo_developer.agents.pm.types import (
    DependencyAnalysisResult,
    DependencyEdge,
    DependencyGraph,
    Story,
)

logger = structlog.get_logger(__name__)


# LLM prompts for dependency inference (module constants per Story 6.4 patterns)
DEPENDENCY_SYSTEM_PROMPT = """You are a software architect AI that analyzes user stories to identify dependencies.

Your task is to identify which stories must be completed BEFORE a given story can be implemented.

DEPENDENCY TYPES TO CONSIDER:
1. Technical Dependencies: Infrastructure, database schemas, auth systems must exist first
2. Functional Dependencies: Features that build on other features (login before dashboard)
3. Data Dependencies: Stories that create data other stories need to read
4. API Dependencies: Stories that provide APIs other stories consume

OUTPUT FORMAT:
Return a JSON array of dependencies:
[
    {
        "depends_on_story_id": "story-001",
        "reason": "Requires user authentication from story-001"
    }
]

RULES:
- Only identify REAL dependencies that would block implementation
- Do NOT identify preferences or nice-to-haves
- Return empty array [] if no dependencies found
- Be specific about WHY the dependency exists"""

DEPENDENCY_USER_PROMPT_TEMPLATE = """Analyze this story for dependencies on other stories:

TARGET STORY:
ID: {story_id}
Title: {story_title}
As a {role}, I want {action}, so that {benefit}

ALL AVAILABLE STORIES:
{all_stories_summary}

Return JSON array of dependencies this story has on other stories."""


# Keyword patterns for dependency extraction
# These patterns indicate implicit dependencies between stories
DEPENDENCY_KEYWORDS: dict[str, str] = {
    r"\bafter\s+(\w+(?:\s+\w+){0,3})\s+(?:is\s+)?(?:complete|done|finished|implemented)": "completion dependency",
    r"\brequires?\s+(\w+(?:\s+\w+){0,3})": "requirement dependency",
    r"\bdepends?\s+on\s+(\w+(?:\s+\w+){0,3})": "explicit dependency",
    r"\buses?\s+(\w+(?:\s+\w+){0,3})\s+from": "usage dependency",
    r"\bneeds?\s+(\w+(?:\s+\w+){0,3})\s+(?:to\s+be|first)": "prerequisite dependency",
    r"\bbuild(?:s|ing)?\s+on\s+(\w+(?:\s+\w+){0,3})": "build dependency",
}


def _extract_dependencies_from_text(
    story: Story,
    all_stories: tuple[Story, ...],
) -> list[tuple[str, str]]:
    """Extract dependencies from story text using keyword matching.

    Analyzes the story's action, benefit, and acceptance criteria for keywords
    that indicate dependencies on other stories (e.g., "after X is complete",
    "requires Y", "depends on Z").

    Args:
        story: The story to analyze for dependencies.
        all_stories: All available stories to match against.

    Returns:
        List of (dependency_story_id, reason) tuples.

    Example:
        >>> story = Story(
        ...     id="story-002",
        ...     action="view dashboard after authentication is complete",
        ...     ...
        ... )
        >>> deps = _extract_dependencies_from_text(story, all_stories)
        >>> deps
        [("story-001", "completion dependency: authentication")]
    """
    dependencies: list[tuple[str, str]] = []
    seen_deps: set[str] = set()

    # Build searchable text from story
    story_text_parts = [
        story.title,
        story.action,
        story.benefit,
    ]
    # Add acceptance criteria text
    for ac in story.acceptance_criteria:
        story_text_parts.extend([ac.given, ac.when, ac.then])
        story_text_parts.extend(ac.and_clauses)

    story_text = " ".join(story_text_parts).lower()

    # Build index of other stories for matching
    story_index: dict[str, str] = {}  # keyword -> story_id
    for other in all_stories:
        if other.id == story.id:
            continue
        # Index by title words
        title_words = other.title.lower().split()
        for word in title_words:
            if len(word) > 3:  # Skip short words
                story_index[word] = other.id
        # Index by key action words
        action_words = other.action.lower().split()
        for word in action_words:
            if len(word) > 3:
                story_index[word] = other.id

    # Search for dependency patterns
    for pattern, reason_type in DEPENDENCY_KEYWORDS.items():
        matches = re.finditer(pattern, story_text, re.IGNORECASE)
        for match in matches:
            keyword = match.group(1).lower().strip()

            # Try to match keyword to a story
            for word in keyword.split():
                if word in story_index:
                    dep_story_id = story_index[word]
                    if dep_story_id not in seen_deps:
                        seen_deps.add(dep_story_id)
                        dependencies.append((dep_story_id, f"{reason_type}: {keyword}"))
                        logger.debug(
                            "dependency_extracted_from_text",
                            story_id=story.id,
                            dependency_id=dep_story_id,
                            reason=f"{reason_type}: {keyword}",
                        )
                    break

    return dependencies


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def _call_llm_with_retry(user_prompt: str, system_prompt: str) -> str:
    """Call LLM with retry logic per ADR-007.

    Wraps the LLM call with Tenacity retry for transient failures.

    Args:
        user_prompt: User message to send.
        system_prompt: System message for context.

    Returns:
        LLM response string.

    Raises:
        Exception: After 3 retry attempts fail.
    """
    from yolo_developer.agents.pm.llm import _call_llm

    return await _call_llm(user_prompt, system_prompt)


async def _infer_dependencies_llm(
    story: Story,
    all_stories: tuple[Story, ...],
) -> list[tuple[str, str]]:
    """Infer dependencies using LLM analysis.

    Uses LLM to analyze story content and identify logical dependencies
    considering technical, functional, data, and API dependencies.
    Includes retry logic per ADR-007 for transient failures.

    Args:
        story: The story to analyze for dependencies.
        all_stories: All available stories to match against.

    Returns:
        List of (dependency_story_id, reason) tuples.

    Example:
        >>> deps = await _infer_dependencies_llm(story, all_stories)
        >>> deps
        [("story-001", "Requires user authentication")]
    """
    from yolo_developer.agents.pm.llm import _USE_LLM

    if not _USE_LLM:
        # Stub implementation for testing - returns empty list
        logger.debug("llm_dependency_inference_disabled", story_id=story.id)
        return []

    # Build stories summary for context
    stories_summary_parts = []
    for other in all_stories:
        if other.id != story.id:
            stories_summary_parts.append(
                f"- {other.id}: {other.title} (As a {other.role}, I want {other.action})"
            )
    all_stories_summary = "\n".join(stories_summary_parts)

    # Format user prompt from template
    user_prompt = DEPENDENCY_USER_PROMPT_TEMPLATE.format(
        story_id=story.id,
        story_title=story.title,
        role=story.role,
        action=story.action,
        benefit=story.benefit,
        all_stories_summary=all_stories_summary,
    )

    try:
        response = await _call_llm_with_retry(user_prompt, DEPENDENCY_SYSTEM_PROMPT)

        # Extract JSON array from response
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if isinstance(data, list):
                dependencies = []
                valid_story_ids = {s.id for s in all_stories}
                for item in data:
                    if isinstance(item, dict):
                        dep_id = item.get("depends_on_story_id", "")
                        reason = item.get("reason", "LLM inferred dependency")
                        if dep_id in valid_story_ids and dep_id != story.id:
                            dependencies.append((dep_id, reason))
                            logger.debug(
                                "dependency_inferred_by_llm",
                                story_id=story.id,
                                dependency_id=dep_id,
                                reason=reason,
                            )
                return dependencies

        logger.warning("llm_dependency_response_invalid", story_id=story.id)
        return []

    except Exception as e:
        logger.warning(
            "llm_dependency_inference_failed",
            story_id=story.id,
            error=str(e),
        )
        return []


def _build_dependency_graph(
    stories: tuple[Story, ...],
    dependencies: dict[str, list[tuple[str, str]]],
) -> DependencyGraph:
    """Build a dependency graph from stories and their dependencies.

    Creates a graph structure with nodes (story IDs), edges (dependencies),
    and adjacency lists for efficient traversal in both directions.

    Args:
        stories: All stories to include as nodes.
        dependencies: Dict mapping story_id to list of (dep_id, reason) tuples.

    Returns:
        DependencyGraph TypedDict with nodes, edges, and adjacency lists.

    Example:
        >>> graph = _build_dependency_graph(stories, deps)
        >>> graph["nodes"]
        ["story-001", "story-002", "story-003"]
        >>> graph["adjacency_list"]["story-002"]
        ["story-001"]
    """
    nodes = [story.id for story in stories]
    edges: list[DependencyEdge] = []
    adjacency_list: dict[str, list[str]] = {node: [] for node in nodes}
    reverse_adjacency_list: dict[str, list[str]] = {node: [] for node in nodes}

    valid_story_ids = set(nodes)

    for story_id, deps in dependencies.items():
        if story_id not in valid_story_ids:
            continue

        for dep_id, reason in deps:
            if dep_id not in valid_story_ids:
                continue
            if dep_id == story_id:
                # Self-loop - still record it
                pass

            edge: DependencyEdge = {
                "from_story_id": story_id,
                "to_story_id": dep_id,
                "reason": reason,
            }
            edges.append(edge)

            # Adjacency: story_id depends on dep_id
            if dep_id not in adjacency_list[story_id]:
                adjacency_list[story_id].append(dep_id)

            # Reverse adjacency: dep_id is depended on by story_id
            if story_id not in reverse_adjacency_list[dep_id]:
                reverse_adjacency_list[dep_id].append(story_id)

    logger.info(
        "dependency_graph_built",
        node_count=len(nodes),
        edge_count=len(edges),
    )

    return {
        "nodes": nodes,
        "edges": edges,
        "adjacency_list": adjacency_list,
        "reverse_adjacency_list": reverse_adjacency_list,
    }


def _detect_cycles(graph: DependencyGraph) -> list[list[str]]:
    """Detect all cycles in the dependency graph using DFS.

    Uses three-color DFS algorithm:
    - WHITE (0): Unvisited node
    - GRAY (1): Node currently being processed (in DFS stack)
    - BLACK (2): Node fully processed

    When we encounter a GRAY node during DFS, we've found a cycle.

    Args:
        graph: The dependency graph to analyze.

    Returns:
        List of cycles, each cycle is a list of story IDs forming the cycle.

    Example:
        >>> cycles = _detect_cycles(graph)
        >>> cycles
        [["story-001", "story-002", "story-001"]]  # A->B->A
    """
    # DFS color constants: 0=unvisited, 1=in progress, 2=done
    _white, _gray, _black = 0, 1, 2
    color: dict[str, int] = dict.fromkeys(graph["nodes"], _white)
    cycles: list[list[str]] = []
    path: list[str] = []

    def dfs(node: str) -> None:
        color[node] = _gray
        path.append(node)

        for neighbor in graph["adjacency_list"].get(node, []):
            if color[neighbor] == _gray:
                # Found cycle - extract it from path
                if neighbor in path:
                    cycle_start = path.index(neighbor)
                    cycle = [*path[cycle_start:], neighbor]
                    # Only add unique cycles
                    if cycle not in cycles:
                        cycles.append(cycle)
                        logger.warning(
                            "dependency_cycle_detected",
                            cycle=cycle,
                        )
            elif color[neighbor] == _white:
                dfs(neighbor)

        path.pop()
        color[node] = _black

    for node in graph["nodes"]:
        if color[node] == _white:
            dfs(node)

    return cycles


def _find_critical_path(
    graph: DependencyGraph,
    cycles: list[list[str]],
) -> tuple[list[list[str]], int]:
    """Find ALL critical paths (longest dependency chains) in the DAG.

    Uses topological sort followed by dynamic programming to find
    all longest paths. Per AC4, returns ALL equal-length paths.
    Only runs if there are no cycles.

    Args:
        graph: The dependency graph (must be acyclic for valid results).
        cycles: List of detected cycles (returns empty if cycles exist).

    Returns:
        Tuple of (list_of_critical_paths, path_length).
        Returns ([], 0) if cycles exist or graph is empty.

    Example:
        >>> paths, length = _find_critical_path(graph, [])
        >>> paths
        [["story-001", "story-002", "story-003"]]
        >>> length
        3
    """
    if cycles:
        logger.info("critical_path_skipped_cycles_exist", cycle_count=len(cycles))
        return [], 0

    if not graph["nodes"]:
        return [], 0

    # Calculate in-degree for topological sort
    # in_degree[v] = number of edges pointing TO v (number of stories v depends on)
    in_degree: dict[str, int] = dict.fromkeys(graph["nodes"], 0)
    for node in graph["nodes"]:
        in_degree[node] = len(graph["adjacency_list"].get(node, []))

    # Find nodes with no dependencies (starting points)
    queue = [node for node in graph["nodes"] if in_degree[node] == 0]

    # If no root nodes exist but we have nodes, the graph might be fully connected
    if not queue and graph["nodes"]:
        queue = [graph["nodes"][0]]

    # Distance and ALL parents for path reconstruction (to capture all paths)
    dist: dict[str, int] = dict.fromkeys(graph["nodes"], 1)
    parents: dict[str, list[str]] = {node: [] for node in graph["nodes"]}

    # BFS-like traversal following reverse adjacency (who depends on me)
    visited: set[str] = set()
    process_queue = list(queue)

    while process_queue:
        node = process_queue.pop(0)
        if node in visited:
            continue
        visited.add(node)

        # For each story that depends on this node
        for dependent in graph["reverse_adjacency_list"].get(node, []):
            new_dist = dist[node] + 1
            if new_dist > dist[dependent]:
                # Found a longer path - replace parents
                dist[dependent] = new_dist
                parents[dependent] = [node]
            elif new_dist == dist[dependent] and node not in parents[dependent]:
                # Found an equal-length path - add parent
                parents[dependent].append(node)

            if dependent not in visited:
                process_queue.append(dependent)

    # If nothing was visited, just process all nodes
    if not visited and graph["nodes"]:
        for node in graph["nodes"]:
            dist[node] = 1

    # Find the endpoint(s) with maximum distance
    if not dist:
        return [], 0

    max_dist = max(dist.values())
    end_nodes = [node for node, d in dist.items() if d == max_dist]

    # Reconstruct ALL paths from ALL max-distance endpoints
    all_paths: list[list[str]] = []

    def reconstruct_paths(node: str, current_path: list[str]) -> None:
        """Recursively reconstruct all paths ending at node."""
        current_path = [node, *current_path]
        if not parents[node]:
            # Reached a root node
            all_paths.append(current_path)
        else:
            for parent_node in parents[node]:
                reconstruct_paths(parent_node, current_path)

    for end_node in end_nodes:
        reconstruct_paths(end_node, [])

    logger.info(
        "critical_paths_found",
        path_count=len(all_paths),
        length=max_dist,
        paths=all_paths,
    )

    return all_paths, max_dist


async def analyze_dependencies(
    stories: tuple[Story, ...],
) -> DependencyAnalysisResult:
    """Analyze dependencies between stories.

    Main orchestration function that:
    1. Extracts dependencies from text for all stories
    2. Infers dependencies via LLM (if enabled)
    3. Merges and deduplicates dependencies
    4. Builds dependency graph
    5. Detects cycles
    6. Finds critical path (if no cycles)

    Args:
        stories: Tuple of stories to analyze.

    Returns:
        DependencyAnalysisResult with graph, cycles, and critical path.

    Example:
        >>> result = await analyze_dependencies(stories)
        >>> result["has_cycles"]
        False
        >>> result["critical_path_length"]
        3
    """
    if not stories:
        logger.info("dependency_analysis_empty_stories")
        empty_graph: DependencyGraph = {
            "nodes": [],
            "edges": [],
            "adjacency_list": {},
            "reverse_adjacency_list": {},
        }
        return {
            "graph": empty_graph,
            "cycles": [],
            "critical_paths": [],
            "critical_path_length": 0,
            "has_cycles": False,
            "analysis_notes": "No stories to analyze",
        }

    logger.info("dependency_analysis_started", story_count=len(stories))

    # Step 1 & 2: Extract and infer dependencies for all stories
    all_dependencies: dict[str, list[tuple[str, str]]] = {}
    total_deps = 0

    for story in stories:
        story_deps: list[tuple[str, str]] = []

        # Text-based extraction
        text_deps = _extract_dependencies_from_text(story, stories)
        story_deps.extend(text_deps)

        # LLM-based inference
        llm_deps = await _infer_dependencies_llm(story, stories)
        story_deps.extend(llm_deps)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_deps: list[tuple[str, str]] = []
        for dep_id, reason in story_deps:
            if dep_id not in seen:
                seen.add(dep_id)
                unique_deps.append((dep_id, reason))

        all_dependencies[story.id] = unique_deps
        total_deps += len(unique_deps)

    logger.info(
        "dependencies_collected",
        story_count=len(stories),
        total_dependencies=total_deps,
    )

    # Step 3: Build dependency graph
    graph = _build_dependency_graph(stories, all_dependencies)

    # Step 4: Detect cycles
    cycles = _detect_cycles(graph)
    has_cycles = len(cycles) > 0

    # Step 5: Find critical paths (only if no cycles) - returns ALL equal-length paths per AC4
    critical_paths, critical_path_length = _find_critical_path(graph, cycles)

    # Build analysis notes
    notes_parts = [
        f"Analyzed {len(stories)} stories",
        f"found {total_deps} dependencies",
    ]
    if has_cycles:
        notes_parts.append(f"detected {len(cycles)} cycle(s) (ERROR)")
    else:
        path_info = f"critical path length: {critical_path_length}"
        if len(critical_paths) > 1:
            path_info += f" ({len(critical_paths)} equal-length paths)"
        notes_parts.append(path_info)

    analysis_notes = ", ".join(notes_parts)

    logger.info(
        "dependency_analysis_complete",
        story_count=len(stories),
        dependency_count=total_deps,
        cycle_count=len(cycles),
        critical_path_length=critical_path_length,
        critical_path_count=len(critical_paths),
    )

    return {
        "graph": graph,
        "cycles": cycles,
        "critical_paths": critical_paths,
        "critical_path_length": critical_path_length,
        "has_cycles": has_cycles,
        "analysis_notes": analysis_notes,
    }


def _update_stories_with_dependencies(
    stories: tuple[Story, ...],
    result: DependencyAnalysisResult,
) -> tuple[Story, ...]:
    """Update stories with their dependencies field populated.

    Creates new Story instances with the dependencies field set based
    on the analysis results. Stories are immutable dataclasses.

    Args:
        stories: Original stories without dependencies populated.
        result: Dependency analysis result containing the graph.

    Returns:
        New tuple of stories with dependencies field populated.

    Example:
        >>> updated = update_stories_with_dependencies(stories, result)
        >>> updated[1].dependencies
        ("story-001",)
    """
    updated_stories: list[Story] = []

    for story in stories:
        # Get dependencies from the graph's adjacency list
        deps = result["graph"]["adjacency_list"].get(story.id, [])
        deps_tuple = tuple(deps)

        # Create new story with updated dependencies
        updated_story = replace(story, dependencies=deps_tuple)
        updated_stories.append(updated_story)

        if deps:
            logger.debug(
                "story_dependencies_updated",
                story_id=story.id,
                dependencies=deps,
            )

    return tuple(updated_stories)
