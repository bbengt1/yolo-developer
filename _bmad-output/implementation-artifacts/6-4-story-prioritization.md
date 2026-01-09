# Story 6.4: Story Prioritization

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want stories prioritized by value and dependencies,
So that the most important work is done first.

## Acceptance Criteria

1. **AC1: Stories Are Ranked by User Value**
   - **Given** a set of stories with various priority levels and characteristics
   - **When** prioritization scoring runs
   - **Then** each story receives a numeric priority score (0-100)
   - **And** the score incorporates user value factors (business impact, user reach, complexity-to-value ratio)
   - **And** stories can be sorted by score for execution ordering

2. **AC2: Technical Dependencies Are Considered**
   - **Given** stories with dependencies on other stories
   - **When** prioritization runs
   - **Then** blocked stories (dependencies not done) receive reduced effective priority
   - **And** stories that unblock many others receive priority boost
   - **And** dependency cycles are detected and flagged (not blocking, just warning)

3. **AC3: Quick Wins Are Identified**
   - **Given** prioritized stories with complexity and value data
   - **When** quick win identification runs
   - **Then** stories with high value and low complexity are flagged as "quick_win"
   - **And** quick wins are highlighted in the output for visibility
   - **And** quick win threshold is based on value/complexity ratio

4. **AC4: Priority Scores Are Assigned**
   - **Given** prioritization analysis is complete
   - **When** results are returned
   - **Then** each story has a `PriorityScore` with:
     - `raw_score`: int (0-100) base priority score
     - `dependency_adjustment`: int (-20 to +20) adjustment based on dependencies
     - `final_score`: int (0-100) = raw_score + dependency_adjustment (clamped)
     - `is_quick_win`: bool
     - `scoring_rationale`: str (explanation of score factors)
   - **And** the prioritization output includes all scores and a recommended execution order

5. **AC5: Prioritization Integrates with PM Node**
   - **Given** stories have been transformed from requirements
   - **When** pm_node completes processing
   - **Then** prioritization runs automatically on all generated stories
   - **And** prioritization results are included in PMOutput or a new dedicated output
   - **And** Decision record includes prioritization summary for audit trail

## Tasks / Subtasks

- [x] Task 1: Create Prioritization Types (AC: 4)
  - [x] Create `PriorityScore` TypedDict in `src/yolo_developer/agents/pm/types.py`
  - [x] Define score fields: raw_score, dependency_adjustment, final_score, is_quick_win, scoring_rationale
  - [x] Create `PrioritizationResult` TypedDict for full prioritization output
  - [x] Add type exports to `__init__.py`

- [x] Task 2: Implement Value Scoring (AC: 1)
  - [x] Create `prioritization.py` module in `src/yolo_developer/agents/pm/`
  - [x] Implement `_calculate_value_score(story: Story) -> int` function
  - [x] Score factors:
    - Base score from StoryPriority enum (CRITICAL=100, HIGH=75, MEDIUM=50, LOW=25)
    - Complexity adjustment: S=+10, M=0, L=-10, XL=-20 (value vs effort)
    - AC count bonus: Well-specified stories (3-5 ACs) get +5
    - Source requirement count: Multiple sources = higher coverage = +5 per extra req
  - [x] Return score 0-100

- [x] Task 3: Implement Dependency Analysis (AC: 2)
  - [x] Create `_analyze_dependencies(stories: tuple[Story, ...]) -> dict[str, DependencyInfo]`
  - [x] Build dependency graph from story.dependencies fields
  - [x] Calculate "blocking_count" for each story (how many stories depend on it)
  - [x] Calculate "blocked_by_count" for each story (how many unfinished dependencies)
  - [x] Detect cycles using depth-first search with path tracking
  - [x] Return dict mapping story_id to dependency info

- [x] Task 4: Calculate Dependency Adjustment (AC: 2, 4)
  - [x] Create `_calculate_dependency_adjustment(dep_info: DependencyInfo) -> tuple[int, list[str]]`
  - [x] Apply adjustments:
    - If story blocks 3+ others: +20 (critical path)
    - If story blocks 1-2 others: +10 (important)
    - If story is blocked by 1-2 unfinished: -10
    - If story is blocked by 3+ unfinished: -20 (heavily blocked)
    - If story is in cycle: -5 (needs resolution)
  - [x] Return adjustment in range -20 to +20

- [x] Task 5: Identify Quick Wins (AC: 3)
  - [x] Create `_is_quick_win(story: Story, raw_score: int, dep_info: DependencyInfo) -> bool`
  - [x] Quick win criteria:
    - Complexity is "S" or "M"
    - Raw value score >= 60 (HIGH or CRITICAL priority base)
    - AC count <= 4 (manageable scope)
    - No blocking dependencies (dependencies tuple empty or all done)
  - [x] Return True if story qualifies as quick win

- [x] Task 6: Create Main Prioritization Function (AC: all)
  - [x] Create `prioritize_stories(stories: tuple[Story, ...]) -> PrioritizationResult`
  - [x] Orchestrate:
    1. Calculate value scores for all stories
    2. Analyze dependencies
    3. Calculate dependency adjustments
    4. Identify quick wins
    5. Build PriorityScore for each story
    6. Sort stories by final_score descending
    7. Create recommended_execution_order list
  - [x] Return PrioritizationResult with all data
  - [x] Add structured logging for prioritization steps

- [x] Task 7: Integrate into pm_node (AC: 5)
  - [x] Import `prioritize_stories` in `node.py`
  - [x] After story transformation (line ~335), call `prioritize_stories(stories)`
  - [x] Add prioritization summary to processing_notes
  - [x] Include prioritization data in Decision rationale
  - [x] Add `prioritization_result` to return dict (or extend pm_output)
  - [x] Log prioritization summary

- [x] Task 8: Write Unit Tests for Value Scoring (AC: 1)
  - [x] Test CRITICAL priority yields highest base score
  - [x] Test complexity adjustments work correctly
  - [x] Test AC count bonus applied properly
  - [x] Test score bounds (0-100 range)

- [x] Task 9: Write Unit Tests for Dependency Analysis (AC: 2)
  - [x] Test stories with no dependencies
  - [x] Test stories with linear dependencies (A -> B -> C)
  - [x] Test stories that unblock multiple stories
  - [x] Test cycle detection (A -> B -> C -> A)
  - [x] Test blocked_by counting

- [x] Task 10: Write Unit Tests for Quick Win Detection (AC: 3)
  - [x] Test S complexity + HIGH priority = quick win
  - [x] Test XL complexity = not quick win (regardless of priority)
  - [x] Test blocked story = not quick win
  - [x] Test LOW priority = not quick win (value too low)

- [x] Task 11: Write Unit Tests for Full Prioritization (AC: 4)
  - [x] Test PriorityScore fields populated correctly
  - [x] Test final_score is clamped to 0-100
  - [x] Test recommended_execution_order is sorted correctly
  - [x] Test scoring_rationale includes relevant factors

- [x] Task 12: Write Integration Tests (AC: 5)
  - [x] Test pm_node includes prioritization in output
  - [x] Test prioritization results in Decision record
  - [x] Test stories remain unchanged (prioritization is analysis only)

## Dev Notes

### Architecture Compliance

- **ADR-001 (State Management):** Use TypedDict for `PriorityScore` and `PrioritizationResult` (internal state)
- **ADR-005 (LangGraph Communication):** Prioritization output added to node return dict, not via direct mutation
- **ARCH-QUALITY-5:** No async needed - prioritization is pure computation (no I/O)
- **ARCH-QUALITY-6:** Use structlog for all prioritization logging
- **ARCH-QUALITY-7:** Full type annotations on all functions

### Technical Requirements

- Use `from __future__ import annotations` at top of all files
- Use snake_case for all function names and variables
- All prioritization functions are synchronous (no async needed - pure computation)
- Follow existing patterns from `testability.py` (computation-only module)
- Story objects are immutable - prioritization creates NEW data, doesn't modify stories

### Prioritization Algorithm Design

**Value Score Calculation (0-100):**
```python
# Base scores from StoryPriority
PRIORITY_BASE_SCORES = {
    StoryPriority.CRITICAL: 100,
    StoryPriority.HIGH: 75,
    StoryPriority.MEDIUM: 50,
    StoryPriority.LOW: 25,
}

# Complexity adjustments (value vs effort consideration)
COMPLEXITY_ADJUSTMENTS = {
    "S": +10,   # Low effort, high relative value
    "M": 0,     # Neutral
    "L": -10,   # Higher effort reduces priority
    "XL": -20,  # Much higher effort, may need breakdown
}

def _calculate_value_score(story: Story) -> int:
    score = PRIORITY_BASE_SCORES[story.priority]
    score += COMPLEXITY_ADJUSTMENTS.get(story.estimated_complexity, 0)

    # Well-specified stories get bonus
    ac_count = len(story.acceptance_criteria)
    if 3 <= ac_count <= 5:
        score += 5

    # Multi-source stories (covers more requirements)
    if len(story.source_requirements) > 1:
        score += 5 * (len(story.source_requirements) - 1)

    return max(0, min(100, score))  # Clamp
```

**Dependency Adjustment (-20 to +20):**
```python
def _calculate_dependency_adjustment(
    story_id: str,
    stories: tuple[Story, ...],
    dep_info: dict[str, DependencyInfo],
) -> int:
    info = dep_info.get(story_id)
    if not info:
        return 0

    adjustment = 0

    # Boost for blocking many stories (critical path)
    if info.blocking_count >= 3:
        adjustment += 20
    elif info.blocking_count >= 1:
        adjustment += 10

    # Penalty for being blocked
    if info.blocked_by_count >= 3:
        adjustment -= 20
    elif info.blocked_by_count >= 1:
        adjustment -= 10

    # Penalty for cycle involvement
    if info.in_cycle:
        adjustment -= 5

    return max(-20, min(20, adjustment))
```

**Quick Win Criteria:**
```python
def _is_quick_win(story: Story, raw_score: int, dep_info: DependencyInfo) -> bool:
    # Must have reasonable value
    if raw_score < 60:
        return False

    # Must be low effort
    if story.estimated_complexity not in ("S", "M"):
        return False

    # Must have manageable scope
    if len(story.acceptance_criteria) > 4:
        return False

    # Must not be blocked
    if dep_info.blocked_by_count > 0:
        return False

    return True
```

### Type Definitions

```python
class DependencyInfo(TypedDict):
    """Dependency analysis for a single story."""
    blocking_count: int       # How many stories depend on this one
    blocked_by_count: int     # How many unfinished dependencies this story has
    blocking_story_ids: list[str]   # IDs of stories that depend on this
    blocked_by_story_ids: list[str] # IDs of stories this depends on
    in_cycle: bool            # True if story is part of a dependency cycle

class PriorityScore(TypedDict):
    """Priority score breakdown for a single story."""
    story_id: str
    raw_score: int            # 0-100, value-based score
    dependency_adjustment: int # -20 to +20
    final_score: int          # 0-100, clamped sum
    is_quick_win: bool
    scoring_rationale: str    # Human-readable explanation

class PrioritizationResult(TypedDict):
    """Complete prioritization analysis result."""
    scores: list[PriorityScore]                    # All story scores
    recommended_execution_order: list[str]         # Story IDs in priority order
    quick_wins: list[str]                          # Story IDs flagged as quick wins
    dependency_cycles: list[list[str]]             # Detected cycles (list of story ID chains)
    analysis_notes: str                            # Summary of prioritization analysis
```

### File Structure (ARCH-STRUCT)

```
src/yolo_developer/agents/pm/
├── __init__.py          # Add PriorityScore, PrioritizationResult, prioritize_stories exports
├── types.py             # MODIFY: Add PriorityScore, PrioritizationResult, DependencyInfo
├── prioritization.py    # NEW: Prioritization algorithm implementation
├── testability.py       # Existing - no changes needed
├── llm.py               # Existing - no changes needed
└── node.py              # MODIFY: Import and call prioritize_stories

tests/unit/agents/pm/
├── test_prioritization.py  # NEW: Tests for prioritization module
└── test_node.py            # MODIFY: Add prioritization integration tests
```

### Previous Story Intelligence (Story 6.3)

Key learnings to apply:
1. **TypedDict for results:** Use TypedDict (not dataclass) for validation results since they're internal state
2. **Module constants:** Define scoring constants at module level (like `ERROR_PATTERNS`)
3. **Pure computation:** Prioritization is pure computation - no async needed
4. **Comprehensive testing:** Follow the 39-test pattern from testability.py

Code review fixes from 6.3 to avoid:
- Don't create unused types (TestabilitySeverity was removed as dead code)
- Integrate results into processing notes and Decision rationale
- Use single source of truth for constants (don't duplicate)

### Git Intelligence

Recent commits show pattern:
```
da2808b feat: Implement AC testability validation with code review fixes (Story 6.3)
d6342f1 feat: Implement LLM-powered story transformation with code review fixes (Story 6.2)
4f00abd feat: Implement PM agent node with code review fixes (Story 6.1)
```

All stories follow "feat: Implement X with code review fixes (Story Y.Z)" format.

### Integration Points

**Input (from story transformation):**
- `tuple[Story, ...]` from `_transform_requirements_to_stories()`
- Each Story has: priority enum, estimated_complexity, dependencies tuple, acceptance_criteria

**Output (to pm_node return):**
- `PrioritizationResult` with scores and execution order
- Results added to `processing_notes` string
- Summary added to `Decision.rationale`

**Note on Dependencies:**
- Story 6.5 (Dependency Identification) will populate `story.dependencies`
- For Story 6.4, dependencies may be empty - prioritization should handle this gracefully
- Current implementation in `_transform_single_requirement` sets `dependencies=()` with comment referencing Story 6.5

### Testing Strategy

**Unit Tests (synchronous):**
- No mocking needed (pure computation)
- Direct function testing with fixture data
- Edge case coverage for scoring bounds and cycles

**Integration Tests:**
- Test pm_node flow includes prioritization
- Verify prioritization results in output
- Test with empty stories list (graceful handling)

### Relationship to Other Stories

- **Story 6.3 (AC Testability):** Precedent for computation-only modules with TypedDict results
- **Story 6.5 (Dependency Identification):** Will populate story.dependencies field
- **Story 6.6 (Epic Breakdown):** May use prioritization for sub-story ordering
- **Story 10.3 (SM Sprint Planning):** Will consume prioritization for sprint capacity planning

### Project Structure Notes

- PM module at `src/yolo_developer/agents/pm/`
- New `prioritization.py` alongside existing `testability.py`, `types.py`, `node.py`, `llm.py`
- Tests at `tests/unit/agents/pm/`
- No circular imports - PM only imports from config, orchestrator (if needed)

### References

- [Source: _bmad-output/planning-artifacts/epics.md - Story 6.4: Story Prioritization]
- [Source: _bmad-output/planning-artifacts/epics.md - Epic 6: PM Agent overview]
- [Source: _bmad-output/planning-artifacts/architecture.md - ADR-001: State Management Pattern]
- [Source: _bmad-output/planning-artifacts/prd.md - FR44: Prioritize stories based on value and dependencies]
- [Source: src/yolo_developer/agents/pm/types.py - PM type definitions, Story, StoryPriority]
- [Source: src/yolo_developer/agents/pm/node.py - PM node implementation pattern]
- [Source: src/yolo_developer/agents/pm/testability.py - Computation-only module pattern]
- [Source: _bmad-output/implementation-artifacts/6-3-ac-testability-validation.md - Previous story patterns]
- [Source: src/yolo_developer/gates/gates/confidence_scoring.py - Weighted scoring pattern reference]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

### Completion Notes List

- All 12 tasks completed successfully
- 54 new tests in test_prioritization.py for prioritization module
- 9 new integration tests added to test_node.py for pm_node prioritization
- All 223 PM agent tests pass (including 63 new tests for Story 6.4)
- mypy strict mode passes for all modified files
- ruff check and format pass for all modified files

### File List

**New Files:**
- `src/yolo_developer/agents/pm/prioritization.py` - Story prioritization algorithm implementation

**Modified Files:**
- `src/yolo_developer/agents/pm/types.py` - Added DependencyInfo, PriorityScore, PrioritizationResult TypedDicts
- `src/yolo_developer/agents/pm/node.py` - Integrated prioritize_stories() call and output
- `src/yolo_developer/agents/pm/__init__.py` - Added exports for new types and prioritize_stories function

**New Test Files:**
- `tests/unit/agents/pm/test_prioritization.py` - 54 tests for prioritization module

**Modified Test Files:**
- `tests/unit/agents/pm/test_node.py` - Added 9 integration tests for prioritization

**Tracking Files:**
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Updated status from backlog → in-progress → review → done
