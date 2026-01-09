# Story 6.6: Epic Breakdown

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want large features broken into appropriately-sized stories,
So that each story is completable in a single dev session.

## Acceptance Criteria

1. **AC1: Each Resulting Story Is Independently Valuable**
   - **Given** a large feature requirement (epic or complex requirement)
   - **When** the PM breaks it down into stories
   - **Then** each resulting story delivers user-visible value on its own
   - **And** stories can be deployed/released independently if needed
   - **And** no story is just "setup" without user value

2. **AC2: Stories Are Small Enough for Single-Session Completion**
   - **Given** stories generated from epic breakdown
   - **When** story size is evaluated
   - **Then** each story is estimated to take less than 4 hours to implement
   - **And** stories with more than 5 acceptance criteria are flagged for further breakdown
   - **And** estimated complexity is "low" or "medium" (not "high")

3. **AC3: The Original Requirement Is Fully Covered**
   - **Given** a large requirement broken into stories
   - **When** coverage analysis runs
   - **Then** all aspects of the original requirement are covered by at least one story
   - **And** no functionality gaps exist between stories
   - **And** the combined stories are traceable back to the source requirement

4. **AC4: Story Numbering Is Consistent**
   - **Given** stories generated from epic breakdown
   - **When** story IDs are assigned
   - **Then** sub-stories follow a consistent numbering scheme (e.g., "story-003a", "story-003b" or "story-003.1", "story-003.2")
   - **And** parent-child relationships are trackable via ID patterns
   - **And** the original epic/requirement ID is preserved as source reference

5. **AC5: Epic Breakdown Integrates with PM Node**
   - **Given** requirements with high complexity estimates
   - **When** pm_node processes them
   - **Then** large requirements trigger the epic breakdown flow
   - **And** breakdown results replace the original oversized story
   - **And** Decision record includes breakdown rationale for audit trail
   - **And** processing_notes indicate which requirements were broken down

## Tasks / Subtasks

- [x] Task 1: Define Epic Breakdown Types (AC: 3, 4)
  - [x] Create `EpicBreakdownResult` TypedDict in `types.py`
  - [x] Define fields: original_story_id, sub_stories, coverage_notes, breakdown_rationale
  - [x] Create `CoverageMapping` TypedDict for requirement-to-story traceability
  - [x] Add type exports to `__init__.py`

- [x] Task 2: Implement Story Size Detection (AC: 2)
  - [x] Create `breakdown.py` module in `src/yolo_developer/agents/pm/`
  - [x] Implement `_needs_breakdown(story: Story) -> bool`
  - [x] Check estimated_complexity == "high" (XL, L)
  - [x] Check len(acceptance_criteria) > 5
  - [x] Check action text for multiple "and" conjunctions
  - [x] Return True if any breakdown trigger is met

- [x] Task 3: Implement LLM-Based Epic Breakdown (AC: 1, 2, 3)
  - [x] Create `_break_down_story_llm(story: Story) -> list[dict]`
  - [x] Design prompt for story decomposition (BREAKDOWN_SYSTEM_PROMPT, BREAKDOWN_USER_PROMPT_TEMPLATE)
  - [x] LLM considers: independent value, logical boundaries, single-session target, no setup-only stories
  - [x] Return list of sub-story dicts with role, action, benefit, suggested_ac
  - [x] Support _USE_LLM flag for testing (stub splits by "and" or returns 2 generic sub-stories)

- [x] Task 4: Implement Sub-Story Generation (AC: 1, 4)
  - [x] Create `_generate_sub_stories(original: Story, breakdown_data: list[dict]) -> tuple[Story, ...]`
  - [x] Generate consistent sub-story IDs (story-003.1, story-003.2)
  - [x] Preserve source_requirements from original story
  - [x] Generate acceptance criteria for each sub-story
  - [x] Set appropriate complexity estimates (S, M, or L - never XL)

- [x] Task 5: Implement Coverage Validation (AC: 3)
  - [x] Create `_validate_coverage(original: Story, sub_stories: tuple[Story, ...]) -> list[CoverageMapping]`
  - [x] Use keyword matching to check AC coverage
  - [x] Return mapping with is_covered status for each original AC
  - [x] Flag incomplete coverage

- [x] Task 6: Create Main Epic Breakdown Function (AC: all)
  - [x] Create `break_down_epic(story: Story) -> EpicBreakdownResult`
  - [x] Orchestrate: LLM breakdown -> sub-story generation -> coverage validation
  - [x] Add structured logging for breakdown steps
  - [x] Return EpicBreakdownResult with rationale

- [x] Task 7: Integrate into pm_node (AC: 5)
  - [x] Import `_process_epic_breakdowns` in `node.py`
  - [x] Call breakdown after story transformation, BEFORE dependency analysis
  - [x] Replace original stories with sub-stories in output
  - [x] Add breakdown summary to processing_notes
  - [x] Include breakdown decisions in Decision rationale
  - [x] Add breakdown_results to return dict
  - [x] Log breakdown activity

- [x] Task 8: Write Unit Tests for Size Detection (AC: 2)
  - [x] Test low complexity story does not need breakdown
  - [x] Test high complexity story needs breakdown
  - [x] Test story with >5 ACs needs breakdown
  - [x] Test story with complex action text needs breakdown
  - [x] Test edge cases (exactly 5 ACs, medium complexity)

- [x] Task 9: Write Unit Tests for Sub-Story Generation (AC: 1, 4)
  - [x] Test sub-story IDs follow parent.N pattern
  - [x] Test source_requirements preserved from parent
  - [x] Test each sub-story has acceptance criteria
  - [x] Test complexity estimates are "low" or "medium"

- [x] Task 10: Write Unit Tests for Coverage Validation (AC: 3)
  - [x] Test complete coverage returns valid result
  - [x] Test incomplete coverage is flagged
  - [x] Test empty sub-stories list is flagged as error

- [x] Task 11: Write Integration Tests (AC: 5)
  - [x] Test pm_node triggers breakdown for high complexity stories
  - [x] Test pm_node no breakdown for simple stories
  - [x] Test Decision record includes breakdown rationale
  - [x] Test processing_notes indicate breakdown occurred
  - [x] Test sub-story IDs follow correct pattern

## Dev Notes

### Architecture Compliance

- **ADR-001 (State Management):** Use TypedDict for `EpicBreakdownResult` and `CoverageMapping` (internal state)
- **ADR-003 (LLM Provider Abstraction):** Use LiteLLM for breakdown via cheap_model
- **ADR-005 (LangGraph Communication):** Breakdown results modify node return, not via direct mutation
- **ADR-007 (Error Handling):** Use Tenacity retry for LLM calls with exponential backoff
- **ARCH-QUALITY-5:** Async for LLM calls, sync for pure computation (coverage validation)
- **ARCH-QUALITY-6:** Use structlog for all breakdown analysis logging
- **ARCH-QUALITY-7:** Full type annotations on all functions

### Technical Requirements

- Use `from __future__ import annotations` at top of all files
- Use snake_case for all function names and variables
- LLM-powered breakdown is async (uses _call_llm)
- Coverage validation is synchronous (pure computation)
- Follow existing patterns from `dependencies.py` and `prioritization.py`
- Story objects are immutable - create new Story objects for sub-stories

### Integration Order in pm_node

Implemented flow with breakdown:
1. Transform requirements to stories -> `stories = await _transform_requirements_to_stories(...)`
2. **Break down large stories** -> `stories, breakdown_results = await _process_epic_breakdowns(stories)`
3. Analyze dependencies -> `dep_result = await analyze_dependencies(stories)`
4. Update stories with dependencies -> `stories = _update_stories_with_dependencies(stories, dep_result)`
5. Prioritize (with dependency info) -> `priority_result = prioritize_stories(stories)`

Breakdown occurs BEFORE dependency analysis so sub-stories get proper dependency mapping.

### Breakdown Triggers

A story needs breakdown if ANY of these conditions are true:
- `estimated_complexity in ("XL", "L")` (HIGH_COMPLEXITY_TRIGGERS)
- `len(acceptance_criteria) > 5` (MAX_AC_THRESHOLD)
- Story action text contains >= 2 "and" conjunctions (MIN_AND_CONJUNCTION_TRIGGER)

### Sub-Story ID Pattern

For story "story-003" broken into 3 sub-stories:
- story-003.1 (first sub-story)
- story-003.2 (second sub-story)
- story-003.3 (third sub-story)

This preserves sortability and parent-child relationship tracking.

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

### Completion Notes List

1. Created `CoverageMapping` and `EpicBreakdownResult` TypedDicts in `types.py`
2. Created `breakdown.py` module with complete epic breakdown implementation:
   - `_needs_breakdown()` - Story size detection
   - `_break_down_story_llm()` - LLM-powered breakdown with stub fallback
   - `_generate_sub_stories()` - Sub-story generation with parent.N IDs
   - `_validate_coverage()` - Coverage validation using keyword matching
   - `break_down_epic()` - Main orchestration function
   - `_process_epic_breakdowns()` - Helper for pm_node integration
3. Integrated breakdown into `pm_node`:
   - Calls `_process_epic_breakdowns()` after story transformation
   - Breakdown happens BEFORE dependency analysis for proper dependency mapping
   - Updates `processing_notes` and `Decision.rationale` with breakdown summary
   - Adds `breakdown_results` to return dict
4. All 34 breakdown tests pass (including 9 new tests from code review)
5. All 301 PM tests pass
6. mypy and ruff checks pass

### Code Review Fixes Applied

**HIGH Issues Fixed:**
- HIGH #1: Added `break_down_epic` export to `__init__.py`
- HIGH #2: Added 3 tests for `_process_epic_breakdowns` helper function
- HIGH #3: Fixed stub to always return 2+ sub-stories (Core + Validation pattern)

**MEDIUM Issues Fixed:**
- MEDIUM #1: Added rationale comments for 30% coverage threshold
- MEDIUM #2: Added 3 tests for LLM fallback/parsing paths
- MEDIUM #3: Changed hardcoded "authenticated" to "ready to proceed" in AC generation
- MEDIUM #4: Added parameter validation with TypeError/ValueError for `break_down_epic`

### File List

**New Files:**
- `src/yolo_developer/agents/pm/breakdown.py` - Epic breakdown module (660+ lines)
- `tests/unit/agents/pm/test_breakdown.py` - Breakdown tests (25 tests)

**Modified Files:**
- `src/yolo_developer/agents/pm/types.py` - Added CoverageMapping, EpicBreakdownResult TypedDicts
- `src/yolo_developer/agents/pm/__init__.py` - Added exports for new types
- `src/yolo_developer/agents/pm/node.py` - Integrated breakdown into pm_node flow
