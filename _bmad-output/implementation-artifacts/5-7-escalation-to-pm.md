# Story 5.7: Escalation to PM

Status: done

## Story

As a developer,
I want unresolvable issues escalated to PM,
So that product decisions are made at the right level.

## Acceptance Criteria

1. **AC1: Issue Packaging with Context**
   - **Given** the Analyst cannot resolve a requirement issue
   - **When** escalation is triggered
   - **Then** the issue is packaged with complete context
   - **And** context includes: original requirement text, analysis attempts, why it cannot be resolved
   - **And** relevant contradictions or gaps are included
   - **And** the decision request is clear and actionable

2. **AC2: PM Agent Receives Escalation**
   - **Given** an issue is escalated
   - **When** the PM agent receives it
   - **Then** the escalation is delivered via state update with clear decision request
   - **And** the handoff includes all context needed for PM to make a decision
   - **And** the PM agent can understand the issue without re-analyzing

3. **AC3: Escalation Logging**
   - **Given** an escalation occurs
   - **When** the escalation is processed
   - **Then** the escalation is logged with structured logging
   - **And** log includes: reason for escalation, requirement IDs involved, timestamp
   - **And** the escalation appears in the Decision audit trail

4. **AC4: Workflow Continues Appropriately**
   - **Given** an escalation has been created
   - **When** the analyst node completes processing
   - **Then** workflow state is updated with escalation_needed flag
   - **And** the orchestrator can route to PM based on this flag
   - **And** non-escalated requirements continue processing normally

## Tasks / Subtasks

- [x] Task 1: Define Escalation Types and Enums (AC: 1, 3)
  - [x] Create `EscalationReason` enum with values: UNRESOLVABLE_AMBIGUITY, CONFLICTING_REQUIREMENTS, MISSING_DOMAIN_KNOWLEDGE, STAKEHOLDER_DECISION_NEEDED, SCOPE_CLARIFICATION
  - [x] Create `EscalationPriority` enum: URGENT, HIGH, NORMAL

- [x] Task 2: Create Escalation Dataclass (AC: 1, 2, 3)
  - [x] Create frozen `Escalation` dataclass with fields:
    - `id`: str (unique escalation identifier)
    - `reason`: EscalationReason
    - `priority`: EscalationPriority
    - `summary`: str (brief description)
    - `context`: str (full context for PM)
    - `original_requirements`: tuple[str, ...] (requirement IDs involved)
    - `analysis_attempts`: tuple[str, ...] (what was tried)
    - `decision_requested`: str (specific question for PM)
    - `related_gaps`: tuple[str, ...] (gap IDs if applicable)
    - `related_contradictions`: tuple[str, ...] (contradiction IDs if applicable)
    - `timestamp`: datetime
  - [x] Add `to_dict()` method for serialization

- [x] Task 3: Extend AnalystOutput with Escalations (AC: 1, 4)
  - [x] Add `escalations` field to AnalystOutput dataclass (tuple of Escalation)
  - [x] Add `escalation_needed` property returning bool
  - [x] Maintain backward compatibility with default empty tuple

- [x] Task 4: Implement Escalation Detection Logic (AC: 1)
  - [x] Create `_should_escalate_requirement()` function that checks:
    - Critical contradictions that cannot be resolved
    - Gaps with missing domain knowledge
    - Requirements that failed implementability with unresolvable issues
    - Multiple ambiguities in same requirement area
  - [x] Create `_identify_escalation_reason()` to determine reason type

- [x] Task 5: Implement Escalation Packaging Function (AC: 1, 2)
  - [x] Create `_package_escalation()` function that:
    - Gathers all context from analysis
    - Formulates clear decision request for PM
    - Includes relevant gaps and contradictions
    - Generates unique escalation ID
  - [x] Create `_format_decision_request()` helper for clear questions

- [x] Task 6: Implement `_analyze_escalations()` Main Function (AC: 1, 4)
  - [x] Analyze requirements, gaps, and contradictions for escalation needs
  - [x] Return list of Escalation objects sorted by priority
  - [x] Log escalation analysis with structlog

- [x] Task 7: Integrate into Analyst Node (AC: 2, 3, 4)
  - [x] Call `_analyze_escalations()` after contradiction analysis
  - [x] Add escalations to AnalystOutput
  - [x] Include escalation info in Decision record
  - [x] Include escalation info in output message

- [x] Task 8: Update State Return for Orchestrator Routing (AC: 4)
  - [x] If escalation_needed, include `escalation_needed: True` in return dict
  - [x] Orchestrator can use this flag for conditional routing to PM

- [x] Task 9: Export New Types from `__init__.py` (AC: all)
  - [x] Add EscalationReason, EscalationPriority, Escalation to exports

- [x] Task 10: Write Unit Tests for Types (AC: all)
  - [x] Test EscalationReason enum values and membership
  - [x] Test EscalationPriority enum values and membership
  - [x] Test Escalation dataclass creation and to_dict()
  - [x] Test AnalystOutput with escalations field
  - [x] Test escalation_needed property

- [x] Task 11: Write Unit Tests for Node Functions (AC: all)
  - [x] Test `_should_escalate_requirement()` with various scenarios
  - [x] Test `_identify_escalation_reason()` categorization
  - [x] Test `_package_escalation()` context assembly
  - [x] Test `_format_decision_request()` question formatting
  - [x] Test `_analyze_escalations()` full pipeline
  - [x] Test integration in analyst_node

## Dev Notes

### Architecture Compliance

- Follow ADR-001: Use frozen dataclasses for Escalation type (immutable internal state)
- Follow ADR-005: State updates return dict, never mutate state
- Follow ADR-006: Integrate with existing quality_gate decorator pattern
- Use structlog for all logging (ARCH-QUALITY-6)
- Full type annotations on all functions (ARCH-QUALITY-7)

### Technical Requirements

- Use `from __future__ import annotations` at top of all files
- All I/O operations must be async (but escalation detection is pure computation)
- Use snake_case for all state dictionary keys
- Return dict updates from analyst_node, never mutate state
- Follow existing analyst module patterns from Stories 5.1-5.6

### File Structure (ARCH-STRUCT)

Files to modify:
- `src/yolo_developer/agents/analyst/types.py` - Add Escalation types
- `src/yolo_developer/agents/analyst/node.py` - Add escalation detection and packaging
- `src/yolo_developer/agents/analyst/__init__.py` - Export new types
- `tests/unit/agents/analyst/test_types.py` - Tests for escalation types
- `tests/unit/agents/analyst/test_node.py` - Tests for escalation functions

### Existing Code Patterns to Follow

**From Story 5.6 (Contradiction Flagging):**
- Pattern: Add enum + dataclass in types.py, functions in node.py
- Pattern: Use `to_dict()` method on dataclasses for serialization
- Pattern: Sort results by severity/priority
- Pattern: Integrate into `_enhance_with_gap_analysis()` or similar pipeline function
- Pattern: Use `@dataclass(frozen=True)` for immutable types

**From analyst_node function:**
```python
# Current return pattern (line 3125-3128)
return {
    "messages": [message],
    "decisions": [decision],
}
# Extend to include escalation_needed flag when applicable
```

**Decision record pattern (lines 3082-3094):**
```python
decision = Decision(
    agent="analyst",
    summary=f"...",
    rationale=f"...",
    related_artifacts=tuple(...)
)
```

### Integration Points

- **Handoff Context**: When escalation_needed is True, orchestrator will use `create_handoff_context()` to package context for PM agent
- **State Update**: Add `escalation_needed: bool` to return dict (not in YoloState TypedDict, but as routing signal)
- **Message Content**: Include escalation summary in AIMessage for transparency

### Key Implementation Details

**Escalation Triggers (based on FR41):**
1. Critical contradictions that block implementation entirely
2. Requirements missing essential domain knowledge
3. Scope ambiguity affecting multiple requirements
4. Multiple failed implementability validations in same area
5. Conflicting stakeholder requirements needing prioritization

**Decision Request Format:**
Clear, actionable questions for PM:
- "Should we prioritize [X] over [Y] given the conflict?"
- "What is the expected behavior when [ambiguous scenario]?"
- "Is [feature X] in scope for this iteration?"

### Testing Strategy

- Test with requirements that should NOT trigger escalation (happy path)
- Test with critical contradictions (should trigger)
- Test with unresolvable ambiguities (should trigger)
- Test with missing domain knowledge (should trigger)
- Test edge case: multiple escalations from same analysis
- Test backward compatibility: existing tests must pass

### Previous Story Learnings (from Story 5.6)

1. Use dict.get() with defaults to avoid KeyError/ValueError on invalid values
2. Ensure new types are properly exported from __init__.py
3. Add both positive and edge case tests
4. Integration tests can be covered via unit tests that test full pipeline
5. Maintain backward compatibility with default values for new fields

### Project Structure Notes

- Module location: `src/yolo_developer/agents/analyst/`
- Test location: `tests/unit/agents/analyst/`
- Follows existing analyst module organization from Stories 5.1-5.6
- No new files needed - extend existing types.py and node.py

### References

- [Source: _bmad-output/planning-artifacts/epics.md - Story 5.7]
- [Source: _bmad-output/planning-artifacts/architecture.md - ADR-001, ADR-005, ADR-006]
- [Source: _bmad-output/planning-artifacts/prd.md - FR41: Analyst Agent can escalate to PM]
- [Source: src/yolo_developer/agents/analyst/types.py - Existing type patterns]
- [Source: src/yolo_developer/agents/analyst/node.py - Current analyst_node implementation]
- [Source: src/yolo_developer/orchestrator/context.py - Decision, HandoffContext patterns]
- [Source: _bmad-output/implementation-artifacts/5-6-contradiction-flagging.md - Previous story patterns]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A - All tests passing

### Completion Notes List

- All 11 tasks completed successfully
- 328 total tests passing (148 type tests + 180 node tests) after code review
- 26 new type tests for escalation types
- 27 new node function tests for escalation logic (3 added during code review)
- mypy type checking passes
- ruff linter passes
- Backward compatibility maintained (all existing tests pass)

### Code Review Findings (Fixed)

1. **Issue 1 (Fixed):** Removed unused threshold constants `_CRITICAL_CONTRADICTION_THRESHOLD` and `_MISSING_DOMAIN_THRESHOLD` from node.py
2. **Issue 2 (Fixed):** Added boundary test `test_req_with_two_gaps_does_not_trigger_escalation` to verify 2 gaps below threshold doesn't trigger
3. **Issue 4 (Fixed):** Added missing keywords ("stakeholder", "regulation") to `_identify_escalation_reason` for consistency with `_should_escalate_for_gap`
4. **Issue 5 (Fixed):** Added test `test_all_none_returns_stakeholder_decision` for edge case when all args are None
5. **Issue 7 (Fixed):** Added test `test_escalation_ids_are_unique` to verify escalation ID uniqueness
6. **Bonus:** Removed unused `Escalation` import from test_node.py

### File List

**Modified:**
- `src/yolo_developer/agents/analyst/types.py` - Added EscalationReason, EscalationPriority, Escalation types; extended AnalystOutput
- `src/yolo_developer/agents/analyst/node.py` - Added escalation detection, packaging, and analysis functions; integrated into analyst_node
- `src/yolo_developer/agents/analyst/__init__.py` - Exported new escalation types
- `tests/unit/agents/analyst/test_types.py` - Added 26 escalation type tests
- `tests/unit/agents/analyst/test_node.py` - Added 24 escalation node function tests
