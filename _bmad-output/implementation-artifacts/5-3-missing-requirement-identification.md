# Story 5.3: Missing Requirement Identification

Status: done

## Story

As a developer,
I want gaps in requirements identified,
So that important functionality isn't overlooked.

## Acceptance Criteria

1. **AC1: Missing Edge Cases Identified**
   - **Given** a set of requirements
   - **When** the Analyst analyzes them
   - **Then** missing edge cases are identified
   - **And** edge cases include error handling, boundary conditions, and unusual inputs
   - **And** each identified edge case is traceable to the source requirement

2. **AC2: Implied But Unstated Requirements Surfaced**
   - **Given** crystallized requirements from the Analyst
   - **When** gap analysis runs
   - **Then** implied but unstated requirements are surfaced
   - **And** each implied requirement includes rationale for why it's needed
   - **And** implied requirements reference the source requirement(s) that suggest them

3. **AC3: Common Patterns Suggest Likely Missing Features**
   - **Given** requirements for a specific domain (e.g., user authentication)
   - **When** pattern matching is applied
   - **Then** common patterns suggest likely missing features
   - **And** suggestions are based on industry-standard patterns
   - **And** each suggestion explains why it's typically needed

4. **AC4: Gaps Flagged with Severity**
   - **Given** identified gaps and missing requirements
   - **When** the gap analysis completes
   - **Then** gaps are flagged with severity (critical, high, medium, low)
   - **And** severity is based on impact to implementation success
   - **And** gaps are sorted by severity for prioritization

## Tasks / Subtasks

- [x] Task 1: Define Gap and Missing Requirement Types (AC: all)
  - [x] Create `IdentifiedGap` dataclass in `agents/analyst/types.py`
  - [x] Add fields: id, description, gap_type, severity, source_requirements, rationale
  - [x] Define `GapType` enum: "edge_case", "implied_requirement", "pattern_suggestion"
  - [x] Define `Severity` enum: "critical", "high", "medium", "low"
  - [x] Add `to_dict()` method for serialization
  - [x] Maintain backward compatibility with existing AnalystOutput

- [x] Task 2: Implement Edge Case Detection (AC: 1)
  - [x] Create `_identify_edge_cases()` function in node.py
  - [x] Define common edge case patterns (empty inputs, null values, boundary conditions)
  - [x] Map requirement types to expected edge cases
  - [x] Generate edge case descriptions with traceability to source requirements
  - [x] Add unit tests for edge case detection

- [x] Task 3: Implement Implied Requirement Detection (AC: 2)
  - [x] Create `_identify_implied_requirements()` function
  - [x] Define rules for common implications (e.g., "user login" implies "logout")
  - [x] Generate rationale for each implied requirement
  - [x] Link implied requirements to source requirements
  - [x] Add unit tests for implied requirement detection

- [x] Task 4: Implement Pattern-Based Suggestion Engine (AC: 3)
  - [x] Create `_suggest_from_patterns()` function
  - [x] Define domain pattern knowledge base (auth, CRUD, API, etc.)
  - [x] Match requirements against known domain patterns
  - [x] Generate suggestions with explanations
  - [x] Add unit tests for pattern matching

- [x] Task 5: Implement Severity Assessment (AC: 4)
  - [x] Create `_assess_gap_severity()` function
  - [x] Define severity rules based on gap type and impact
  - [x] Critical: Security, data integrity, core functionality
  - [x] High: Major feature gaps, integration issues
  - [x] Medium: User experience gaps, edge cases
  - [x] Low: Nice-to-have features, optimization
  - [x] Add unit tests for severity assessment

- [x] Task 6: Update AnalystOutput with Gap Information (AC: all)
  - [x] Extend AnalystOutput to include structured gaps (not just strings)
  - [x] Maintain backward compatibility with existing string-based gaps
  - [x] Update `to_dict()` to serialize IdentifiedGap objects
  - [x] Update `_parse_llm_response()` to handle enhanced gap format

- [x] Task 7: Update LLM Prompts for Gap Analysis (AC: all)
  - [x] Add gap analysis instructions to ANALYST_SYSTEM_PROMPT
  - [x] Include edge case detection guidelines in prompt
  - [x] Add domain pattern examples for suggestion generation
  - [x] Define severity assessment criteria in prompt
  - [x] Update JSON schema for enhanced output format

- [x] Task 8: Integrate Gap Analysis into Node Function (AC: all)
  - [x] Update `analyst_node()` to call gap analysis functions
  - [x] Combine crystallized requirements with gap analysis
  - [x] Log gap analysis results with structlog
  - [x] Add decision record for gap analysis findings
  - [x] Handle both LLM and placeholder modes

- [x] Task 9: Write Unit Tests (AC: all)
  - [x] Test IdentifiedGap dataclass creation and serialization
  - [x] Test edge case detection for various requirement types
  - [x] Test implied requirement detection rules
  - [x] Test pattern-based suggestions
  - [x] Test severity assessment accuracy
  - [x] Test backward compatibility with existing tests

- [x] Task 10: Write Integration Tests (AC: all)
  - [x] Test full gap analysis flow with realistic requirements
  - [x] Test gap severity ordering
  - [x] Test traceability from gaps to source requirements
  - [x] Test integration with crystallization output
  - [x] Verify audit trail includes gap analysis

## Dev Notes

### Architecture Compliance

- **ADR-001 (TypedDict State):** Continue using frozen dataclasses for types
- **ADR-003 (LiteLLM):** Use LiteLLM SDK for LLM calls with model tiering
- **ADR-005 (LangGraph):** Maintain node pattern returning state updates
- **FR37:** Analyst Agent can identify missing requirements from seed documents
- [Source: architecture.md#ADR-001] - TypedDict for internal state
- [Source: architecture.md#ADR-003] - LiteLLM SDK integration
- [Source: epics.md#Story-5.3] - Missing Requirement Identification requirements

### Technical Requirements

- **Immutable Types:** Use frozen dataclasses with enums
- **Pure Functions:** Gap detection functions should be side-effect free
- **Async Required:** LLM-based detection continues to use async/await
- **Structured Logging:** Log gap analysis details for audit trail
- **Type Annotations:** Full type hints on all new functions

### Previous Story Intelligence (Story 5.2)

**Files Created/Modified in Story 5.2:**
- `src/yolo_developer/agents/analyst/types.py` - Added scope_notes, implementation_hints, confidence
- `src/yolo_developer/agents/analyst/node.py` - Added vague term detection, enhanced parsing
- `src/yolo_developer/agents/prompts/analyst.py` - Enhanced prompts with transformation rules
- Tests added in `tests/unit/agents/analyst/` and `tests/integration/`

**Key Patterns from Story 5.2:**

```python
# CrystallizedRequirement pattern to follow for new type
@dataclass(frozen=True)
class CrystallizedRequirement:
    id: str
    original_text: str
    refined_text: str
    category: str  # "functional", "non-functional", "constraint"
    testable: bool
    scope_notes: str | None = None
    implementation_hints: tuple[str, ...] = ()
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {...}
```

**Vague term detection pattern (from node.py lines 73-110):**

```python
VAGUE_TERMS: frozenset[str] = frozenset([
    "fast", "quick", "slow", "efficient", ...
])

def _detect_vague_terms(text: str) -> set[str]:
    """Detect vague terms in requirement text."""
    if not text:
        return set()
    text_lower = text.lower()
    detected: set[str] = set()
    for term in VAGUE_TERMS:
        if "-" in term:
            if term in text_lower:
                detected.add(term)
        else:
            pattern = rf"\b{re.escape(term)}\b"
            if re.search(pattern, text_lower):
                detected.add(term)
    return detected
```

**LLM Integration Pattern (from node.py lines 223-262):**

```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def _call_llm(prompt: str, system: str) -> str:
    from litellm import acompletion
    from yolo_developer.config import load_config
    config = load_config()
    model = config.llm.cheap_model
    response = await acompletion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""
```

### Existing Code to Reuse (CRITICAL)

**From `agents/analyst/types.py` - Follow pattern for new IdentifiedGap:**
```python
from enum import Enum

class GapType(str, Enum):
    """Type of identified gap."""
    EDGE_CASE = "edge_case"
    IMPLIED_REQUIREMENT = "implied_requirement"
    PATTERN_SUGGESTION = "pattern_suggestion"

class Severity(str, Enum):
    """Severity level for gaps."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass(frozen=True)
class IdentifiedGap:
    """A gap or missing requirement identified during analysis."""
    id: str
    description: str
    gap_type: GapType
    severity: Severity
    source_requirements: tuple[str, ...]  # IDs of related requirements
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "gap_type": self.gap_type.value,
            "severity": self.severity.value,
            "source_requirements": list(self.source_requirements),
            "rationale": self.rationale,
        }
```

**From `agents/analyst/types.py` - Extend AnalystOutput:**
```python
@dataclass(frozen=True)
class AnalystOutput:
    requirements: tuple[CrystallizedRequirement, ...]
    identified_gaps: tuple[str, ...]  # Keep for backward compatibility
    contradictions: tuple[str, ...]
    # NEW: Add structured gaps
    structured_gaps: tuple[IdentifiedGap, ...] = ()
```

### Anti-Patterns to Avoid

- **DO NOT** break existing tests - maintain backward compatibility
- **DO NOT** change `identified_gaps` type (keep as tuple[str, ...] for compatibility)
- **DO NOT** skip severity assessment - this is core functionality
- **DO NOT** hardcode domain patterns without extensibility
- **DO NOT** generate gaps without traceability to source requirements
- **DO NOT** use mutable collections in frozen dataclasses

### Project Structure Notes

**Files to Modify:**
```
src/yolo_developer/agents/analyst/
├── types.py            # Add GapType, Severity, IdentifiedGap; extend AnalystOutput
├── node.py             # Add gap analysis functions, integrate into analyst_node
└── prompts/analyst.py  # Enhance prompts for gap analysis

tests/unit/agents/analyst/
├── test_types.py       # Add tests for new gap types
└── test_node.py        # Add tests for gap detection functions

tests/integration/
└── test_analyst_integration.py  # Add gap analysis flow tests
```

### Domain Pattern Knowledge Base

```python
# Pattern knowledge base for common software domains
DOMAIN_PATTERNS: dict[str, list[str]] = {
    "authentication": [
        "User registration",
        "Password reset/recovery",
        "Session management",
        "Logout functionality",
        "Multi-factor authentication",
        "Account lockout after failed attempts",
        "Password strength validation",
    ],
    "authorization": [
        "Role-based access control",
        "Permission checking",
        "Access denied handling",
        "Admin vs user separation",
    ],
    "crud": [
        "Create operation",
        "Read/retrieve operation",
        "Update operation",
        "Delete operation",
        "List/pagination",
        "Filtering and search",
    ],
    "api": [
        "Error response handling",
        "Input validation",
        "Rate limiting",
        "API versioning",
        "Authentication/authorization headers",
        "Request/response logging",
    ],
    "data": [
        "Data validation",
        "Backup and recovery",
        "Data migration handling",
        "Concurrent access handling",
        "Transaction management",
    ],
}
```

### Edge Case Categories

```python
# Edge case categories to check for each requirement
EDGE_CASE_CATEGORIES: dict[str, list[str]] = {
    "input_validation": [
        "Empty/null input handling",
        "Maximum length exceeded",
        "Invalid format handling",
        "Special characters handling",
        "Unicode/encoding handling",
    ],
    "boundary_conditions": [
        "Zero values",
        "Negative values",
        "Maximum integer overflow",
        "Date boundary handling (leap years, etc.)",
        "Time zone handling",
    ],
    "error_conditions": [
        "Network failure handling",
        "Database connection loss",
        "External service timeout",
        "Concurrent modification conflicts",
        "Disk space exhaustion",
    ],
    "state_transitions": [
        "Duplicate submission prevention",
        "Invalid state transition handling",
        "Rollback on partial failure",
        "Recovery from interrupted operations",
    ],
}
```

### Severity Assessment Rules

```python
# Rules for determining gap severity
SEVERITY_RULES = {
    # Critical: Security, data integrity, or core functionality at risk
    "security_gap": Severity.CRITICAL,
    "data_integrity_gap": Severity.CRITICAL,
    "core_functionality_missing": Severity.CRITICAL,

    # High: Major feature gaps or integration issues
    "missing_integration": Severity.HIGH,
    "major_edge_case": Severity.HIGH,
    "authentication_gap": Severity.HIGH,

    # Medium: UX gaps or minor edge cases
    "ux_improvement": Severity.MEDIUM,
    "minor_edge_case": Severity.MEDIUM,
    "validation_gap": Severity.MEDIUM,

    # Low: Nice-to-have or optimization opportunities
    "optimization_opportunity": Severity.LOW,
    "documentation_gap": Severity.LOW,
    "nice_to_have": Severity.LOW,
}
```

### Dependencies

**Depends On:**
- Story 5.1 (Create Analyst Agent Node) - Complete
- Story 5.2 (Requirement Crystallization) - Complete
- `orchestrator/state.py` - YoloState, create_agent_message
- `orchestrator/context.py` - Decision

**Downstream Dependencies:**
- Story 5.4 (Requirement Categorization) - uses gaps for context
- Story 5.5 (Implementability Validation) - validates requirements plus gaps
- Story 5.6 (Contradiction Flagging) - may surface contradictions in gaps
- Story 5.7 (Escalation to PM) - escalates unresolved gaps

### External Dependencies

- **litellm** (installed) - LLM abstraction layer
- **tenacity** (installed) - Retry logic
- **structlog** (installed) - Structured logging
- No new dependencies required

### Testing Strategy

**Unit Tests:**
```python
import pytest
from yolo_developer.agents.analyst.types import (
    IdentifiedGap, GapType, Severity
)

def test_identified_gap_creation() -> None:
    """Test IdentifiedGap dataclass creation."""
    gap = IdentifiedGap(
        id="gap-001",
        description="Missing error handling for invalid input",
        gap_type=GapType.EDGE_CASE,
        severity=Severity.HIGH,
        source_requirements=("req-001", "req-002"),
        rationale="Input validation requires error response"
    )
    assert gap.gap_type == GapType.EDGE_CASE
    assert gap.severity == Severity.HIGH
    assert len(gap.source_requirements) == 2

def test_identified_gap_to_dict() -> None:
    """Test IdentifiedGap serialization."""
    gap = IdentifiedGap(
        id="gap-001",
        description="Missing logout",
        gap_type=GapType.IMPLIED_REQUIREMENT,
        severity=Severity.MEDIUM,
        source_requirements=("req-001",),
        rationale="Login implies logout needed"
    )
    d = gap.to_dict()
    assert d["gap_type"] == "implied_requirement"
    assert d["severity"] == "medium"

def test_edge_case_detection() -> None:
    """Test edge case identification from requirements."""
    from yolo_developer.agents.analyst.node import _identify_edge_cases

    requirements = [
        CrystallizedRequirement(
            id="req-001",
            original_text="User can submit form",
            refined_text="User submits form with validation",
            category="functional",
            testable=True,
        )
    ]
    edge_cases = _identify_edge_cases(requirements)
    assert any("empty" in gap.description.lower() for gap in edge_cases)
```

**Integration Tests:**
```python
import pytest
from yolo_developer.agents.analyst import analyst_node

@pytest.mark.asyncio
async def test_gap_analysis_produces_structured_gaps() -> None:
    """Test that gap analysis produces structured IdentifiedGap objects."""
    state: YoloState = {
        "messages": [HumanMessage(content="Build user login system")],
        "current_agent": "analyst",
        "handoff_context": None,
        "decisions": [],
    }
    result = await analyst_node(state)

    output_data = result["messages"][0].additional_kwargs["metadata"]["output"]
    structured_gaps = output_data.get("structured_gaps", [])

    # Should identify logout as implied requirement
    assert any(
        "logout" in gap["description"].lower()
        for gap in structured_gaps
    )
```

### Commit Message Pattern

```
feat: Implement missing requirement identification with severity assessment (Story 5.3)

- Add GapType and Severity enums for gap classification
- Add IdentifiedGap dataclass with source traceability
- Implement _identify_edge_cases() for boundary condition detection
- Implement _identify_implied_requirements() with domain rules
- Implement _suggest_from_patterns() for domain pattern matching
- Implement _assess_gap_severity() for impact-based severity
- Extend AnalystOutput with structured_gaps field
- Update LLM prompts with gap analysis instructions
- Integrate gap analysis into analyst_node
- Add comprehensive unit tests for all gap functions
- Add integration tests for full gap analysis flow

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

### References

- [Source: architecture.md#ADR-001] - TypedDict for internal state
- [Source: architecture.md#ADR-003] - LiteLLM SDK integration
- [Source: architecture.md#ADR-005] - LangGraph patterns
- [Source: epics.md#Epic-5] - Analyst Agent epic context
- [Source: epics.md#Story-5.3] - Missing Requirement Identification
- [Source: epics.md#FR37] - Identify missing requirements capability
- [Source: agents/analyst/node.py] - Current analyst_node implementation
- [Source: agents/analyst/types.py] - CrystallizedRequirement, AnalystOutput
- [Source: agents/prompts/analyst.py] - Current prompt templates

### Files to Consult (MUST READ Before Implementation)

| File | Purpose | Key Lines |
|------|---------|-----------|
| `agents/analyst/types.py` | Existing types to extend | Full file (170 lines) |
| `agents/analyst/node.py` | Current node to enhance | Full file (403 lines) |
| `agents/prompts/analyst.py` | Prompts to update for gap analysis | Full file |
| `orchestrator/context.py` | Decision dataclass for audit | 60-91 |
| `tests/unit/agents/analyst/test_types.py` | Existing tests to maintain compatibility | Full file |
| `tests/unit/agents/analyst/test_node.py` | Existing tests to not break | Full file |

### Success Criteria

1. All identified gaps have severity classification
2. Edge cases are traced to source requirements
3. Implied requirements include rationale
4. Pattern suggestions reference domain knowledge
5. Gaps are sorted by severity for prioritization
6. All existing tests continue to pass (backward compatibility)
7. New tests cover all new functionality
8. Gap analysis integrates seamlessly with crystallization output

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A - No debugging issues encountered

### Completion Notes List

- All 10 tasks completed successfully
- 111 unit and integration tests pass (87 unit + 24 integration)
- Backward compatibility maintained with existing AnalystOutput
- Gap analysis integrates with both LLM and placeholder modes
- mypy strict and ruff linting pass

### File List

| File | Action | Description |
|------|--------|-------------|
| `src/yolo_developer/agents/analyst/types.py` | Modified | Added GapType, Severity enums and IdentifiedGap dataclass; extended AnalystOutput with structured_gaps field |
| `src/yolo_developer/agents/analyst/node.py` | Modified | Added gap analysis functions: _identify_edge_cases, _identify_implied_requirements, _suggest_from_patterns, _enhance_with_gap_analysis; updated _parse_llm_response and analyst_node |
| `src/yolo_developer/agents/analyst/__init__.py` | Modified | Exported new types: GapType, Severity, IdentifiedGap |
| `src/yolo_developer/agents/prompts/analyst.py` | Modified | Enhanced system and user prompts with gap analysis instructions |
| `tests/unit/agents/analyst/test_types.py` | Modified | Added tests for GapType, Severity, IdentifiedGap, and AnalystOutput with structured_gaps |
| `tests/unit/agents/analyst/test_node.py` | Modified | Added tests for edge case detection, implied requirements, pattern suggestions, and _enhance_with_gap_analysis |
| `tests/integration/test_analyst_integration.py` | Modified | Added integration tests for gap analysis flow |

