# Story 6.5: Dependency Identification

Status: Ready for Sign-Off

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want story dependencies explicitly identified,
So that work is sequenced correctly.

## Acceptance Criteria

1. **AC1: Stories That Block Other Stories Are Identified**
   - **Given** a set of stories with potential dependencies
   - **When** dependency analysis runs
   - **Then** stories that block other stories are identified
   - **And** each blocking relationship has a clear reason (e.g., "story-002 depends on story-001 for user authentication")
   - **And** the blocking relationship is bidirectional (blocker knows what it blocks, blocked knows what blocks it)

2. **AC2: A Dependency Graph Is Created**
   - **Given** stories have been analyzed for dependencies
   - **When** the dependency graph is produced
   - **Then** the graph contains all stories as nodes
   - **And** edges represent dependency relationships (directed from dependent to dependency)
   - **And** the graph is represented as a data structure suitable for traversal
   - **And** the graph supports querying "what does X depend on" and "what depends on X"

3. **AC3: Circular Dependencies Are Flagged As Errors**
   - **Given** a set of stories with potential circular dependencies
   - **When** cycle detection runs
   - **Then** circular dependencies are detected (e.g., A->B->C->A)
   - **And** all cycles are reported with the full chain of story IDs
   - **And** cycles are flagged as errors (not just warnings)
   - **And** the cycle detection handles multiple independent cycles

4. **AC4: The Critical Path Is Identified**
   - **Given** a dependency graph with no cycles (or cycles resolved)
   - **When** critical path analysis runs
   - **Then** the longest dependency chain is identified as the critical path
   - **And** stories on the critical path are marked
   - **And** the critical path length is reported
   - **And** if multiple paths have the same length, all are reported

5. **AC5: Dependency Analysis Integrates with PM Node**
   - **Given** stories have been transformed from requirements
   - **When** pm_node completes processing
   - **Then** dependency analysis runs on all generated stories
   - **And** stories are updated with their dependencies field populated
   - **And** dependency analysis results are included in PMOutput or a new dedicated output
   - **And** Decision record includes dependency summary for audit trail

## Tasks / Subtasks

- [x] Task 1: Create Dependency Analysis Types (AC: 2, 4)
  - [x] Create `DependencyGraph` TypedDict in `src/yolo_developer/agents/pm/types.py`
  - [x] Define fields: nodes (list of story IDs), edges (list of (from_id, to_id, reason) tuples)
  - [x] Create `DependencyAnalysisResult` TypedDict for full analysis output
  - [x] Define fields: graph, cycles, critical_path, critical_path_length, dependency_reasons
  - [x] Add type exports to `__init__.py`

- [x] Task 2: Implement Dependency Extraction from Requirements (AC: 1)
  - [x] Create `dependencies.py` module in `src/yolo_developer/agents/pm/`
  - [x] Implement `_extract_dependencies_from_text(story: Story, all_stories: tuple[Story, ...]) -> list[tuple[str, str]]`
  - [x] Use keyword matching for implicit dependencies:
    - "after X is complete" -> depends on story with X
    - "requires X" -> depends on story containing X
    - "uses X from Y" -> depends on story containing Y
  - [x] Check for explicit dependency hints in acceptance criteria
  - [x] Return list of (dependency_story_id, reason) tuples

- [x] Task 3: Implement LLM-Based Dependency Inference (AC: 1, 5)
  - [x] Create `_infer_dependencies_llm(story: Story, all_stories: tuple[Story, ...]) -> list[tuple[str, str]]`
  - [x] Use LLM to analyze story content and identify logical dependencies
  - [x] LLM prompt should consider:
    - Technical dependencies (DB before API, auth before protected features)
    - Functional dependencies (user creation before user login)
    - Data dependencies (schema before data access)
  - [x] Return list of (dependency_story_id, reason) tuples
  - [x] Support _USE_LLM flag for testing (stub returns empty list)

- [x] Task 4: Build Dependency Graph (AC: 2)
  - [x] Create `_build_dependency_graph(stories: tuple[Story, ...], dependencies: dict[str, list[tuple[str, str]]]) -> DependencyGraph`
  - [x] Add all story IDs as nodes
  - [x] Add edges from dependent to dependency (direction: dependent -> dependency)
  - [x] Include dependency reason in each edge
  - [x] Return DependencyGraph TypedDict

- [x] Task 5: Implement Cycle Detection (AC: 3)
  - [x] Create `_detect_cycles(graph: DependencyGraph) -> list[list[str]]`
  - [x] Use Tarjan's algorithm or DFS with coloring for cycle detection
  - [x] Return all cycles as lists of story ID chains
  - [x] Handle multiple independent cycles
  - [x] Return empty list if no cycles found

- [x] Task 6: Implement Critical Path Analysis (AC: 4)
  - [x] Create `_find_critical_path(graph: DependencyGraph, cycles: list[list[str]]) -> tuple[list[str], int]`
  - [x] Only run if no cycles (return empty if cycles exist)
  - [x] Use topological sort + longest path algorithm
  - [x] Return (critical_path_story_ids, path_length) tuple
  - [x] Handle multiple equal-length paths (return all)

- [x] Task 7: Create Main Dependency Analysis Function (AC: all)
  - [x] Create `analyze_dependencies(stories: tuple[Story, ...]) -> DependencyAnalysisResult`
  - [x] Orchestrate:
    1. Extract dependencies from text for all stories
    2. Infer dependencies via LLM (if enabled)
    3. Merge and deduplicate dependencies
    4. Build dependency graph
    5. Detect cycles
    6. Find critical path (if no cycles)
    7. Build result with all data
  - [x] Add structured logging for analysis steps
  - [x] Return DependencyAnalysisResult

- [x] Task 8: Create Story Update Function (AC: 5)
  - [x] Create `_update_stories_with_dependencies(stories: tuple[Story, ...], result: DependencyAnalysisResult) -> tuple[Story, ...]`
  - [x] For each story, create new Story with dependencies field populated
  - [x] Preserve all other story fields (immutable dataclass)
  - [x] Return new tuple of updated stories

- [x] Task 9: Integrate into pm_node (AC: 5)
  - [x] Import `analyze_dependencies` in `node.py`
  - [x] After story transformation, call `analyze_dependencies(stories)`
  - [x] Update stories with dependencies using `_update_stories_with_dependencies`
  - [x] Pass updated stories to `prioritize_stories()` (so prioritization has dependency info)
  - [x] Add dependency summary to processing_notes
  - [x] Include dependency data in Decision rationale
  - [x] Add `dependency_analysis_result` to return dict
  - [x] Log dependency summary

- [x] Task 10: Write Unit Tests for Dependency Extraction (AC: 1)
  - [x] Test keyword matching finds "after X" patterns
  - [x] Test keyword matching finds "requires X" patterns
  - [x] Test explicit dependency detection from AC text
  - [x] Test no dependencies found for independent stories
  - [x] Test multiple dependencies found for complex stories

- [x] Task 11: Write Unit Tests for Graph Building (AC: 2)
  - [x] Test empty stories produces empty graph
  - [x] Test single story produces graph with one node, no edges
  - [x] Test linear chain A->B->C produces correct edges
  - [x] Test fan-out (A->B, A->C) produces correct edges
  - [x] Test fan-in (B->A, C->A) produces correct edges

- [x] Task 12: Write Unit Tests for Cycle Detection (AC: 3)
  - [x] Test no cycles in linear chain
  - [x] Test simple cycle A->B->A detected
  - [x] Test longer cycle A->B->C->A detected
  - [x] Test multiple independent cycles detected
  - [x] Test self-loop A->A detected

- [x] Task 13: Write Unit Tests for Critical Path (AC: 4)
  - [x] Test empty graph has empty critical path
  - [x] Test single node has path length 1
  - [x] Test linear chain has path length = chain length
  - [x] Test diamond pattern finds longest path
  - [x] Test multiple equal paths all returned
  - [x] Test graph with cycles returns empty (cycles must be resolved first)

- [x] Task 14: Write Integration Tests (AC: 5)
  - [x] Test pm_node includes dependency analysis in output
  - [x] Test stories have dependencies field populated after pm_node
  - [x] Test prioritization receives stories with dependencies
  - [x] Test Decision record includes dependency summary

## Dev Notes

### Architecture Compliance

- **ADR-001 (State Management):** Use TypedDict for `DependencyGraph` and `DependencyAnalysisResult` (internal state)
- **ADR-003 (LLM Provider Abstraction):** Use LiteLLM for dependency inference via cheap_model
- **ADR-005 (LangGraph Communication):** Dependency output added to node return dict, not via direct mutation
- **ADR-007 (Error Handling):** Use Tenacity retry for LLM calls with exponential backoff
- **ARCH-QUALITY-5:** Async for LLM calls, sync for pure computation (graph algorithms)
- **ARCH-QUALITY-6:** Use structlog for all dependency analysis logging
- **ARCH-QUALITY-7:** Full type annotations on all functions

### Technical Requirements

- Use `from __future__ import annotations` at top of all files
- Use snake_case for all function names and variables
- LLM-powered dependency inference is async (uses _call_llm)
- Graph algorithms (cycle detection, critical path) are synchronous (pure computation)
- Follow existing patterns from `prioritization.py` and `testability.py`
- Story objects are immutable - create new Story objects with updated dependencies

### LLM Prompts for Dependency Inference

**System Prompt:**
```python
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
```

**User Prompt Template:**
```python
DEPENDENCY_USER_PROMPT_TEMPLATE = """Analyze this story for dependencies on other stories:

TARGET STORY:
ID: {story_id}
Title: {story_title}
As a {role}, I want {action}, so that {benefit}

ALL AVAILABLE STORIES:
{all_stories_summary}

Return JSON array of dependencies this story has on other stories."""
```

### Type Definitions

```python
class DependencyEdge(TypedDict):
    """A single edge in the dependency graph."""
    from_story_id: str    # The dependent story
    to_story_id: str      # The dependency (what it depends on)
    reason: str           # Why this dependency exists

class DependencyGraph(TypedDict):
    """Graph representation of story dependencies."""
    nodes: list[str]                    # All story IDs
    edges: list[DependencyEdge]         # Dependency relationships
    adjacency_list: dict[str, list[str]] # story_id -> list of dependency IDs

class DependencyAnalysisResult(TypedDict):
    """Complete dependency analysis result."""
    graph: DependencyGraph
    cycles: list[list[str]]             # Detected cycles as story ID chains
    critical_path: list[str]            # Story IDs on critical path
    critical_path_length: int           # Length of critical path
    has_cycles: bool                    # True if any cycles detected
    analysis_notes: str                 # Summary of analysis
```

### Graph Algorithms

**Cycle Detection (Tarjan's SCC or DFS with coloring):**
```python
def _detect_cycles(graph: DependencyGraph) -> list[list[str]]:
    """Detect all cycles in the dependency graph using DFS.

    Uses three-color DFS:
    - WHITE (0): Unvisited
    - GRAY (1): Currently being processed (in current DFS path)
    - BLACK (2): Fully processed

    When we encounter a GRAY node, we've found a cycle.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph["nodes"]}
    cycles = []
    path = []

    def dfs(node: str) -> None:
        color[node] = GRAY
        path.append(node)

        for neighbor in graph["adjacency_list"].get(node, []):
            if color[neighbor] == GRAY:
                # Found cycle - extract it from path
                cycle_start = path.index(neighbor)
                cycles.append(path[cycle_start:] + [neighbor])
            elif color[neighbor] == WHITE:
                dfs(neighbor)

        path.pop()
        color[node] = BLACK

    for node in graph["nodes"]:
        if color[node] == WHITE:
            dfs(node)

    return cycles
```

**Critical Path (Longest Path in DAG):**
```python
def _find_critical_path(graph: DependencyGraph) -> tuple[list[str], int]:
    """Find the critical path (longest dependency chain) in the DAG.

    Uses dynamic programming with topological sort.
    dist[v] = max(dist[u] + 1) for all edges u -> v
    """
    if not graph["nodes"]:
        return [], 0

    # Topological sort using Kahn's algorithm
    in_degree = {node: 0 for node in graph["nodes"]}
    for edge in graph["edges"]:
        in_degree[edge["from_story_id"]] += 1

    queue = [node for node in graph["nodes"] if in_degree[node] == 0]
    topo_order = []

    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for neighbor in graph["adjacency_list"].get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Find longest path using DP
    dist = {node: 1 for node in graph["nodes"]}  # Distance from any start
    parent = {node: None for node in graph["nodes"]}

    for node in topo_order:
        for neighbor in graph["adjacency_list"].get(node, []):
            if dist[node] + 1 > dist[neighbor]:
                dist[neighbor] = dist[node] + 1
                parent[neighbor] = node

    # Find the endpoint with maximum distance
    max_dist = max(dist.values())
    end_node = [node for node, d in dist.items() if d == max_dist][0]

    # Reconstruct path
    path = []
    current = end_node
    while current is not None:
        path.append(current)
        current = parent[current]

    return list(reversed(path)), max_dist
```

### File Structure (ARCH-STRUCT)

```
src/yolo_developer/agents/pm/
├── __init__.py          # Add DependencyGraph, DependencyAnalysisResult, analyze_dependencies exports
├── types.py             # MODIFY: Add DependencyEdge, DependencyGraph, DependencyAnalysisResult TypedDicts
├── dependencies.py      # NEW: Dependency analysis implementation
├── prioritization.py    # Existing - no changes needed (will receive stories with dependencies)
├── testability.py       # Existing - no changes needed
├── llm.py               # MODIFY: Add DEPENDENCY_SYSTEM_PROMPT, DEPENDENCY_USER_PROMPT_TEMPLATE, _infer_dependencies_llm
└── node.py              # MODIFY: Import and call analyze_dependencies, update stories

tests/unit/agents/pm/
├── test_dependencies.py    # NEW: Tests for dependency analysis module
└── test_node.py            # MODIFY: Add dependency integration tests
```

### Previous Story Intelligence (Story 6.4)

Key learnings to apply:
1. **TypedDict for results:** Use TypedDict (not dataclass) for analysis results since they're internal state
2. **Module constants:** Define scoring constants and prompts at module level
3. **Pure computation for graphs:** Graph algorithms don't need async - pure computation
4. **LLM for inference:** Use LLM with _USE_LLM flag for the "smart" dependency inference
5. **Comprehensive testing:** Follow the 54-test pattern from prioritization.py

Code review patterns from 6.4 to follow:
- Integrate results into processing notes and Decision rationale
- Create new module for the main functionality (dependencies.py)
- Keep LLM-related code in llm.py module
- Update node.py to orchestrate the flow

### Git Intelligence

Recent commits show pattern:
```
c31fd62 feat: Implement story prioritization with code review fixes (Story 6.4)
da2808b feat: Implement AC testability validation with code review fixes (Story 6.3)
d6342f1 feat: Implement LLM-powered story transformation with code review fixes (Story 6.2)
```

All stories follow "feat: Implement X with code review fixes (Story Y.Z)" format.

### Integration Points

**Input (from story transformation):**
- `tuple[Story, ...]` from `_transform_requirements_to_stories()`
- Each Story has: id, title, role, action, benefit, acceptance_criteria, etc.
- Initial dependencies field is empty `()`

**Output (to pm_node return):**
- `DependencyAnalysisResult` with graph, cycles, critical path
- Updated `tuple[Story, ...]` with dependencies field populated
- Results added to `processing_notes` string
- Summary added to `Decision.rationale`

**Integration Order in pm_node:**
1. Transform requirements to stories → `stories = await _transform_requirements_to_stories(...)`
2. Analyze dependencies → `dep_result = await analyze_dependencies(stories)`
3. Update stories with dependencies → `stories = _update_stories_with_dependencies(stories, dep_result)`
4. Prioritize (with dependency info) → `priority_result = prioritize_stories(stories)`
5. Return all results

### Relationship to Other Stories

- **Story 6.4 (Prioritization):** Consumes stories WITH dependencies for better scoring
- **Story 6.6 (Epic Breakdown):** May use dependency analysis for sub-story relationships
- **Story 6.7 (Escalation to Analyst):** May escalate circular dependencies for resolution
- **Story 10.3 (SM Sprint Planning):** Will use critical path for sprint capacity planning
- **Story 10.6 (Circular Logic Detection):** Different context but similar cycle detection algorithm

### Project Structure Notes

- PM module at `src/yolo_developer/agents/pm/`
- New `dependencies.py` alongside existing modules
- Tests at `tests/unit/agents/pm/`
- No circular imports - PM only imports from config, orchestrator (if needed)

### Testing Strategy

**Unit Tests (mix of sync and async):**
- Graph algorithms are sync (pure computation) - direct testing
- LLM inference is async - use pytest-asyncio with mocked LLM
- Edge case coverage for empty graphs, single nodes, complex graphs

**Integration Tests:**
- Test pm_node flow includes dependency analysis
- Verify stories have dependencies field populated
- Test prioritization receives updated stories
- Test with empty stories list (graceful handling)

### References

- [Source: _bmad-output/planning-artifacts/epics.md - Story 6.5: Dependency Identification]
- [Source: _bmad-output/planning-artifacts/epics.md - Epic 6: PM Agent overview]
- [Source: _bmad-output/planning-artifacts/architecture.md - ADR-001: State Management Pattern]
- [Source: _bmad-output/planning-artifacts/architecture.md - ADR-003: LLM Provider Abstraction]
- [Source: _bmad-output/planning-artifacts/prd.md - FR45: Identify story dependencies and sequencing constraints]
- [Source: src/yolo_developer/agents/pm/types.py - PM type definitions, Story, DependencyInfo (from 6.4)]
- [Source: src/yolo_developer/agents/pm/node.py - PM node implementation pattern]
- [Source: src/yolo_developer/agents/pm/prioritization.py - Related module pattern (uses DependencyInfo)]
- [Source: src/yolo_developer/agents/pm/llm.py - LLM integration pattern]
- [Source: _bmad-output/implementation-artifacts/6-4-story-prioritization.md - Previous story patterns]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- All 267 PM module tests pass
- mypy strict mode passes on all 7 PM module files
- ruff linting passes on all PM source files

### Completion Notes List

- Implemented dependency analysis module with text-based extraction and LLM inference
- Created DependencyGraph and DependencyAnalysisResult TypedDicts per ADR-001
- Implemented DFS-based cycle detection algorithm with three-color marking
- Implemented critical path analysis using BFS traversal and longest path algorithm
- Integrated dependency analysis into pm_node after story transformation
- Stories now have dependencies field populated before prioritization runs
- Dependency summary included in processing_notes, Decision rationale, and AIMessage
- Added 34 new tests for dependency analysis module
- Added 11 new integration tests for pm_node dependency integration

### Code Review Fixes Applied

**HIGH Issues (3):**
1. Added Tenacity retry decorator for LLM calls per ADR-007 (`_call_llm_with_retry`)
2. Changed `critical_path` to `critical_paths` (list of lists) to return ALL equal-length paths per AC4
3. Renamed `update_stories_with_dependencies` to `_update_stories_with_dependencies` (private function)

**MEDIUM Issues (4):**
4. sprint-status.yaml added to File List (see below)
5. Moved LLM prompts to module-level constants (`DEPENDENCY_SYSTEM_PROMPT`, `DEPENDENCY_USER_PROMPT_TEMPLATE`)
6. Moved `import json` to top of file (was inside function)
7. Strengthened weak test `test_multiple_dependencies_found` to verify BOTH dependencies found

### File List

**New Files:**
- src/yolo_developer/agents/pm/dependencies.py - Dependency analysis implementation
- tests/unit/agents/pm/test_dependencies.py - Dependency analysis tests

**Modified Files:**
- src/yolo_developer/agents/pm/types.py - Added DependencyEdge, DependencyGraph, DependencyAnalysisResult TypedDicts
- src/yolo_developer/agents/pm/__init__.py - Added exports for new types and functions
- src/yolo_developer/agents/pm/node.py - Integrated dependency analysis into pm_node flow
- tests/unit/agents/pm/test_node.py - Added dependency integration tests
- _bmad-output/implementation-artifacts/sprint-status.yaml - Updated story status

