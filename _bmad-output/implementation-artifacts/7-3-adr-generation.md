# Story 7.3: ADR Generation

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want Architecture Decision Records auto-generated,
so that design rationale is documented for future reference.

## Acceptance Criteria

1. **Given** a significant architectural decision
   **When** an ADR is generated
   **Then** it follows standard ADR format (Title, Status, Context, Decision, Consequences)
   **And** the ADR includes all required sections with meaningful content

2. **Given** a design decision from the Architect agent
   **When** an ADR is created
   **Then** alternatives considered are documented with pros/cons
   **And** the rationale for the chosen approach is clear

3. **Given** an ADR being generated
   **When** it is linked to stories
   **Then** the decision is linked to all relevant story IDs
   **And** the relationship is bidirectional (ADR references stories, stories can reference ADRs)

4. **Given** generated ADRs
   **When** they are stored
   **Then** ADRs are included in the ArchitectOutput
   **And** they can be serialized via to_dict() method
   **And** they are accessible in the architect_output state

5. **Given** a design decision with 12-Factor analysis
   **When** an ADR is generated
   **Then** the ADR context includes 12-Factor compliance information
   **And** recommendations from the analysis are incorporated

6. **Given** LLM-powered ADR generation
   **When** generating ADR content
   **Then** it uses litellm with configurable model
   **And** it includes tenacity retry with exponential backoff
   **And** it handles LLM failures gracefully

7. **Given** the ADR dataclass
   **When** all ADRs are generated
   **Then** they are frozen (immutable) per ADR-001
   **And** they have unique IDs in format ADR-{number:03d}
   **And** they include created_at ISO timestamp

## Tasks / Subtasks

- [x] Task 1: Enhance ADR Content Generation (AC: 1, 5)
  - [x] Create `_generate_adr_context()` function that uses 12-Factor analysis
  - [x] Create `_generate_adr_decision()` function with proper formatting
  - [x] Create `_generate_adr_consequences()` function with pros/cons analysis
  - [x] Include 12-Factor compliance percentage in context

- [x] Task 2: Implement Alternatives Documentation (AC: 2)
  - [x] Add `alternatives_rationale` field or expand `consequences` to include alternatives
  - [x] Create `_document_alternatives()` function to format alternatives with pros/cons
  - [x] Extract alternatives from DesignDecision.alternatives_considered

- [x] Task 3: Implement Story Linking (AC: 3)
  - [x] Ensure ADR.story_ids contains all related story IDs
  - [x] Group related decisions by story when generating ADRs
  - [x] Add cross-reference metadata for bidirectional linking

- [x] Task 4: Integrate LLM-Powered Generation (AC: 6)
  - [x] Create `_generate_adr_with_llm()` async function
  - [x] Design prompt template for ADR content generation
  - [x] Add tenacity @retry decorator with exponential backoff
  - [x] Use configurable model via YOLO_LLM__ROUTINE_MODEL env var
  - [x] Handle LLM failures with graceful fallback to pattern-based generation

- [x] Task 5: Update _generate_adrs Function (AC: 1, 4, 7)
  - [x] Make `_generate_adrs` async to support LLM calls
  - [x] Accept twelve_factor_analyses parameter for context enrichment
  - [x] Update architect_node to pass analyses to _generate_adrs
  - [x] Ensure proper ID generation (ADR-{counter:03d})
  - [x] Verify ADR immutability (frozen dataclass)

- [x] Task 6: Write Unit Tests for ADR Content (AC: 1, 2, 5)
  - [x] Test ADR follows standard format (Title, Status, Context, Decision, Consequences)
  - [x] Test alternatives are documented
  - [x] Test 12-Factor context is included

- [x] Task 7: Write Unit Tests for Story Linking (AC: 3)
  - [x] Test ADR includes story_ids
  - [x] Test multiple stories can be linked to one ADR
  - [x] Test ADR can be found by story ID

- [x] Task 8: Write Unit Tests for LLM Integration (AC: 6)
  - [x] Test LLM generation with mocked LLM
  - [x] Test retry behavior on transient failures
  - [x] Test fallback to pattern-based on LLM failure

- [x] Task 9: Write Integration Tests (AC: 4, 7)
  - [x] Test architect_node returns ADRs in output
  - [x] Test ADRs are serializable via to_dict()
  - [x] Test ADR IDs are unique and properly formatted
  - [x] Test created_at timestamp is set

## Dev Notes

### Architecture Compliance

- **ADR-001 (State Management):** ADR dataclass is frozen (immutable) for internal state
- **ADR-003 (LLM Abstraction):** Use litellm for LLM calls with configurable model
- **ADR-007 (Error Handling):** Use tenacity @retry for LLM calls with exponential backoff
- **ARCH-QUALITY-5:** All I/O operations (LLM calls) must be async/await
- **ARCH-QUALITY-7:** Full type annotations on all functions

### Existing Implementation (from Story 7.1)

The current `_generate_adrs()` function in `node.py` is a stub:
```python
def _generate_adrs(decisions: list[DesignDecision]) -> list[ADR]:
    """Generate Architecture Decision Records from design decisions.

    Creates ADRs for significant decisions (technology and pattern types).
    This is a stub implementation - full ADR generation will be implemented
    in Story 7.3.
    """
    # Only generates ADRs for "technology" and "pattern" types
    # Uses basic context and decision text
    # consequences="Stub implementation - full consequences analysis in Story 7.3"
```

### ADR Dataclass (already exists in types.py)

```python
@dataclass(frozen=True)
class ADR:
    id: str                              # Format: ADR-{number:03d}
    title: str                           # Descriptive title
    status: ADRStatus                    # proposed, accepted, deprecated, superseded
    context: str                         # Why this decision was needed
    decision: str                        # What was decided
    consequences: str                    # Positive and negative effects
    story_ids: tuple[str, ...] = ()      # Stories this ADR relates to
    created_at: str = <ISO timestamp>    # Auto-generated

    def to_dict(self) -> dict[str, Any]: ...
```

### Standard ADR Format Reference

From architecture.md ADR examples:
```
**Decision:** <Title of decision>
**Context:** <Why this decision was needed>
**Choice:** <What was chosen>
**Rationale:** <Why this was chosen>
**Consequences:** <Positive and negative effects>
**Sources:** <References>
```

### ADR Content Patterns from 12-Factor Analysis

Use TwelveFactorAnalysis to enrich ADR content:
- If compliance < 100%: Include recommendations in consequences
- If violations found: Document trade-offs in context
- Link factor results to decision rationale

### LLM Prompt Template (suggested)

```python
ADR_GENERATION_PROMPT = """Generate an Architecture Decision Record for the following design decision.

Design Decision:
- Type: {decision_type}
- Description: {description}
- Rationale: {rationale}
- Alternatives Considered: {alternatives}
- Story ID: {story_id}

12-Factor Analysis:
- Compliance: {compliance_percentage}%
- Applicable Factors: {applicable_factors}
- Recommendations: {recommendations}

Generate ADR content in JSON format:
{
  "title": "Brief, descriptive title",
  "context": "Why this decision was needed (2-3 sentences)",
  "decision": "What was decided and the chosen approach",
  "consequences": "Positive effects, negative effects, trade-offs"
}
"""
```

### Testing Approach

Follow RED-GREEN-REFACTOR TDD cycle:
1. RED: Write failing tests first
2. GREEN: Implement minimal code to pass
3. REFACTOR: Clean up while tests pass

### Story Dependencies

- **Depends on:** Story 7.1 (architect_node, ADR dataclass), Story 7.2 (twelve-factor analysis)
- **Enables:** Story 7.4 (Quality Attribute Evaluation)
- **FR Covered:** FR50: Architect Agent can produce Architecture Decision Records (ADRs)

### Previous Story Context (7.2)

From Story 7.2 implementation:
- `analyze_twelve_factor()` returns `TwelveFactorAnalysis` with overall_compliance and recommendations
- `architect_node` now passes `twelve_factor_analyses` dict to ArchitectOutput
- LLM integration pattern established with `_call_llm()` using tenacity retry

### Git Intelligence (Recent Commits)

Recent commit pattern: `feat: Implement X with code review fixes (Story X.X)`

Files from Story 7.2:
- `src/yolo_developer/agents/architect/twelve_factor.py` - LLM integration pattern
- `src/yolo_developer/agents/architect/node.py` - `_generate_design_decisions` is async
- `tests/unit/agents/architect/test_adr_generation.py` - Existing ADR tests to extend

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story-7.3] - Story definition
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-001 through ADR-009] - ADR format examples
- [Source: src/yolo_developer/agents/architect/types.py:143-198] - ADR dataclass definition
- [Source: src/yolo_developer/agents/architect/node.py:239-296] - Current _generate_adrs stub
- [Source: src/yolo_developer/agents/architect/twelve_factor.py] - LLM integration pattern
- [FR50: Architect Agent can produce Architecture Decision Records (ADRs)]

## Dev Agent Record

### Agent Model Used

claude-opus-4-5-20251101

### Debug Log References

- All 190 architect tests passing
- ruff check passes with no errors
- mypy type checking passes

### Completion Notes List

1. Created new `adr_generator.py` module with comprehensive ADR content generation:
   - `_generate_adr_context()` - Context section with 12-Factor analysis integration
   - `_generate_adr_decision()` - Decision section with rationale
   - `_generate_adr_consequences()` - Consequences with pros/cons analysis
   - `_document_alternatives()` - Alternatives with brief analysis
   - `_generate_adr_title()` - Descriptive ADR titles
   - `_generate_adr_with_llm()` - LLM-powered generation with fallback
   - `_call_adr_llm()` - Low-level LLM call with @retry decorator
   - `generate_adr()` - Main function for single ADR generation
   - `generate_adrs()` - Batch function for multiple ADRs

2. Integrated with architect_node:
   - Replaced stub `_generate_adrs` with new async `generate_adrs` from adr_generator
   - Updated imports and documentation to reflect Story 7.3

3. Exported new functions from `__init__.py`:
   - `generate_adr` and `generate_adrs` now publicly available

4. Test coverage:
   - 16 unit tests for ADR content generation (test_adr_content.py)
   - 8 unit tests for LLM integration (test_adr_llm.py)
   - 10 unit tests for story linking (test_adr_story_linking.py)
   - 11 updated tests for ADR generation (test_adr_generation.py)
   - 13 integration tests (test_adr_integration.py)

### File List

**New Files:**
- `src/yolo_developer/agents/architect/adr_generator.py` - ADR content generation module

**Modified Files:**
- `src/yolo_developer/agents/architect/node.py` - Integrated new generate_adrs, removed stub
- `src/yolo_developer/agents/architect/__init__.py` - Added exports for generate_adr, generate_adrs
- `tests/unit/agents/architect/test_adr_generation.py` - Updated to use new async API
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Updated story status

**New Test Files:**
- `tests/unit/agents/architect/test_adr_content.py` - Content generation tests
- `tests/unit/agents/architect/test_adr_llm.py` - LLM integration tests
- `tests/unit/agents/architect/test_adr_story_linking.py` - Story linking tests
- `tests/integration/agents/architect/test_adr_integration.py` - Integration tests

## Senior Developer Review (AI)

**Review Date:** 2026-01-10
**Reviewer:** Claude Opus 4.5 (Adversarial Code Review)

### Issues Found & Fixed

| Severity | Issue | Resolution |
|----------|-------|------------|
| MEDIUM | Unused `decision_type` parameter in `_analyze_alternative()` | Removed parameter, improved docstring |
| MEDIUM | File List missing `sprint-status.yaml` | Added to File List |
| MEDIUM | Generic fallback message in `_analyze_alternative()` | Improved message, added more technology patterns |
| MEDIUM | Title generation edge cases not tested | Added 3 edge case tests |

### Verification

- All 61 ADR-related tests passing (including 3 new edge case tests)
- ruff check passes with no errors
- mypy strict type checking passes
- All Acceptance Criteria verified as implemented

### Outcome

**APPROVED** - All HIGH and MEDIUM issues fixed. Story ready for merge.
