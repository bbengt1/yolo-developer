# Story 5.2: Requirement Crystallization

Status: done

## Story

As a developer,
I want vague requirements transformed into specific statements,
So that implementation scope is clear and bounded.

## Acceptance Criteria

1. **AC1: Vague to Specific Transformation**
   - **Given** a seed document with high-level requirements
   - **When** the Analyst processes them
   - **Then** each vague statement becomes specific, implementable requirements
   - **And** the transformation preserves the original intent
   - **And** the refined text is measurable and testable

2. **AC2: Scope Boundary Definition**
   - **Given** crystallized requirements
   - **When** the output is produced
   - **Then** scope boundaries are clearly defined
   - **And** in-scope vs out-of-scope items are distinguishable
   - **And** edge cases are identified or noted for clarification

3. **AC3: Implementation Approach Hints**
   - **Given** a specific crystallized requirement
   - **When** the analyst completes processing
   - **Then** implementation approach hints are provided where applicable
   - **And** hints suggest architectural patterns, libraries, or techniques
   - **And** hints reference project conventions from memory/context

4. **AC4: Transformation Audit Trail**
   - **Given** a requirement is crystallized
   - **When** the transformation completes
   - **Then** the transformation is logged for audit
   - **And** both original and refined text are preserved
   - **And** transformation rationale is recorded in decisions

## Tasks / Subtasks

- [x] Task 1: Enhance CrystallizedRequirement Type (AC: 2, 3)
  - [x] Add `scope_notes: str | None` field for boundary clarification
  - [x] Add `implementation_hints: tuple[str, ...]` field for dev guidance
  - [x] Add `confidence: float` field for crystallization confidence (0.0-1.0)
  - [x] Update `to_dict()` to serialize new fields
  - [x] Maintain backward compatibility with existing tests

- [x] Task 2: Implement Vague Term Detection (AC: 1)
  - [x] Create `_detect_vague_terms()` helper function
  - [x] Define vague term patterns: "fast", "easy", "simple", "should", "might", etc.
  - [x] Return list of detected vague terms with positions in text
  - [x] Add unit tests for vague term detection

- [x] Task 3: Implement Requirement Refinement Logic (AC: 1, 2)
  - [x] Enhance LLM prompt to produce specific, measurable requirements
  - [x] Add `REFINEMENT_SYSTEM_PROMPT` to prompts/analyst.py
  - [x] Include examples of vague → specific transformations
  - [x] Parse scope boundaries from LLM response
  - [x] Extract in-scope/out-of-scope items

- [x] Task 4: Implement Implementation Hint Generation (AC: 3)
  - [x] Add hint generation to LLM prompt
  - [x] Hints should reference project patterns from architecture
  - [x] Hints should suggest relevant ADRs and conventions
  - [x] Store hints in CrystallizedRequirement.implementation_hints
  - [x] Add unit tests for hint generation

- [x] Task 5: Enhance Audit Logging (AC: 4)
  - [x] Log transformation details with structlog
  - [x] Include original_text, refined_text, confidence in log entry
  - [x] Log scope boundaries defined
  - [x] Log implementation hints generated
  - [x] Ensure decisions capture transformation rationale

- [x] Task 6: Update Prompt Templates (AC: 1, 2, 3)
  - [x] Update ANALYST_SYSTEM_PROMPT with refinement instructions
  - [x] Update ANALYST_USER_PROMPT_TEMPLATE with structured output format
  - [x] Add examples section with vague → specific transformations
  - [x] Define JSON schema for enhanced output format
  - [x] Include scope boundary extraction instructions

- [x] Task 7: Update Node Function Logic (AC: all)
  - [x] Modify `_crystallize_requirements()` to use enhanced prompts
  - [x] Update `_parse_llm_response()` for new fields
  - [x] Handle backward compatibility with old response format
  - [x] Add confidence scoring based on transformation quality
  - [x] Integrate scope boundary extraction

- [x] Task 8: Write Unit Tests (AC: all)
  - [x] Test vague term detection with various inputs
  - [x] Test refinement produces measurable requirements
  - [x] Test scope boundary parsing
  - [x] Test implementation hint extraction
  - [x] Test confidence scoring
  - [x] Test new CrystallizedRequirement fields
  - [x] Test audit logging content

- [x] Task 9: Write Integration Tests (AC: all)
  - [x] Test full crystallization flow with real-like seed content
  - [x] Test audit trail captures all transformations
  - [x] Test scope boundaries are included in output
  - [x] Test backward compatibility with existing tests
  - [x] Test with _USE_LLM flag enabled (mock LLM)

## Dev Notes

### Architecture Compliance

- **ADR-001 (TypedDict State):** Continue using frozen dataclasses for types
- **ADR-003 (LiteLLM):** Use LiteLLM SDK for LLM calls with model tiering
- **ADR-005 (LangGraph):** Maintain node pattern returning state updates
- **ADR-006 (Quality Gates):** Testability gate already applied from Story 5.1
- **FR36:** Analyst Agent can crystallize vague requirements into specific, implementable statements
- [Source: architecture.md#ADR-001] - TypedDict for internal state
- [Source: architecture.md#ADR-003] - LiteLLM SDK integration
- [Source: epics.md#Story-5.2] - Requirement Crystallization requirements

### Technical Requirements

- **Immutable Types:** Use frozen dataclasses, extend existing pattern
- **Pure Functions:** Transformation logic should be side-effect free
- **Async Required:** All LLM calls continue to use async/await
- **Structured Logging:** Enhanced logging with transformation details
- **Type Annotations:** Full type hints on all new functions

### Previous Story Intelligence (Story 5.1)

**Files Created/Modified in Story 5.1:**
- `src/yolo_developer/agents/analyst/__init__.py` - Public API exports
- `src/yolo_developer/agents/analyst/node.py` - analyst_node with @quality_gate
- `src/yolo_developer/agents/analyst/types.py` - CrystallizedRequirement, AnalystOutput
- `src/yolo_developer/agents/prompts/analyst.py` - ANALYST_SYSTEM_PROMPT, ANALYST_USER_PROMPT_TEMPLATE
- Tests in `tests/unit/agents/analyst/` and `tests/integration/`

**Key Patterns from Story 5.1:**

```python
# Existing CrystallizedRequirement pattern to extend
@dataclass(frozen=True)
class CrystallizedRequirement:
    id: str
    original_text: str
    refined_text: str
    category: str  # "functional", "non-functional", "constraint"
    testable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "original_text": self.original_text,
            "refined_text": self.refined_text,
            "category": self.category,
            "testable": self.testable,
        }
```

**LLM Integration Pattern (from node.py lines 162-200):**

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

**From `agents/analyst/types.py` - Extend CrystallizedRequirement:**
```python
# Current implementation to extend (add new optional fields)
@dataclass(frozen=True)
class CrystallizedRequirement:
    id: str
    original_text: str
    refined_text: str
    category: str
    testable: bool
    # NEW FIELDS TO ADD:
    scope_notes: str | None = None
    implementation_hints: tuple[str, ...] = ()
    confidence: float = 1.0
```

**From `agents/prompts/analyst.py` - Current prompts to enhance:**
```python
# Read the existing prompts and enhance them for crystallization
# Add vague term detection and transformation examples
```

**From `agents/analyst/node.py` - _parse_llm_response pattern:**
```python
def _parse_llm_response(response: str) -> AnalystOutput:
    try:
        data = json.loads(response)
        requirements = tuple(
            CrystallizedRequirement(
                id=req.get("id", f"req-{i:03d}"),
                original_text=req.get("original_text", ""),
                refined_text=req.get("refined_text", ""),
                category=req.get("category", "functional"),
                testable=req.get("testable", True),
                # NEW: Add parsing for new fields
            )
            for i, req in enumerate(data.get("requirements", []), start=1)
        )
        return AnalystOutput(...)
```

### Anti-Patterns to Avoid

- **DO NOT** break existing tests - maintain backward compatibility
- **DO NOT** change function signatures without updating callers
- **DO NOT** remove existing fields from dataclasses
- **DO NOT** skip vague term detection - this is core functionality
- **DO NOT** hardcode vague terms - make them configurable
- **DO NOT** produce vague "refined" text - must be measurable

### Project Structure Notes

**Files to Modify:**
```
src/yolo_developer/agents/analyst/
├── types.py            # Add scope_notes, implementation_hints, confidence
├── node.py             # Enhance _crystallize_requirements, add vague detection
└── prompts/analyst.py  # Enhance prompts for crystallization

tests/unit/agents/analyst/
├── test_types.py       # Add tests for new fields
└── test_node.py        # Add tests for vague detection, refinement

tests/integration/
└── test_analyst_integration.py  # Add crystallization flow tests
```

### Vague Term Patterns to Detect

```python
VAGUE_TERMS = [
    # Quantifier vagueness
    "fast", "quick", "slow", "efficient", "performant",
    "scalable", "responsive", "real-time",
    # Ease vagueness
    "easy", "simple", "straightforward", "intuitive",
    "user-friendly", "seamless",
    # Certainty vagueness
    "should", "might", "could", "may", "possibly",
    "probably", "maybe", "sometimes",
    # Scope vagueness
    "etc", "and so on", "and more", "various", "multiple",
    "several", "many", "few", "some",
    # Quality vagueness
    "good", "better", "best", "nice", "beautiful",
    "clean", "modern", "robust",
]
```

### Enhanced Prompt Structure

```python
ANALYST_SYSTEM_PROMPT = """You are a Requirements Analyst expert specializing in
transforming vague requirements into specific, measurable, testable statements.

CRITICAL RULES:
1. Transform vague terms into specific, measurable criteria
2. Define clear scope boundaries (in-scope vs out-of-scope)
3. Provide implementation hints referencing project conventions
4. Preserve the original intent while removing ambiguity
5. Assign confidence scores (0.0-1.0) based on clarity

VAGUE → SPECIFIC TRANSFORMATIONS:
- "fast" → "response time < 200ms for 95th percentile"
- "easy to use" → "user can complete task in < 3 clicks"
- "scalable" → "supports 10,000 concurrent users"
- "should work" → "MUST pass all acceptance criteria"

OUTPUT FORMAT (JSON):
{
  "requirements": [
    {
      "id": "req-001",
      "original_text": "The system should be fast",
      "refined_text": "API response time MUST be < 200ms at 95th percentile",
      "category": "non-functional",
      "testable": true,
      "scope_notes": "Applies to all GET endpoints; POST excluded",
      "implementation_hints": ["Use async handlers", "Add response caching"],
      "confidence": 0.9
    }
  ],
  "identified_gaps": [],
  "contradictions": []
}
"""
```

### Dependencies

**Depends On:**
- Story 5.1 (Create Analyst Agent Node) - ✅ Complete
- Epic 1 (Config) - ✅ Complete - for LLM model configuration
- `orchestrator/state.py` - YoloState, create_agent_message
- `orchestrator/context.py` - Decision

**Downstream Dependencies:**
- Story 5.3 (Missing Requirement Identification) - uses crystallized output
- Story 5.4 (Requirement Categorization) - depends on crystallized requirements
- Story 5.5 (Implementability Validation) - validates crystallized requirements

### External Dependencies

- **litellm** (installed) - LLM abstraction layer
- **tenacity** (installed) - Retry logic
- **structlog** (installed) - Structured logging
- No new dependencies required

### Testing Strategy

**Unit Tests:**
```python
import pytest
from yolo_developer.agents.analyst.types import CrystallizedRequirement

def test_crystallized_requirement_new_fields() -> None:
    """Test new optional fields have correct defaults."""
    req = CrystallizedRequirement(
        id="req-001",
        original_text="System should be fast",
        refined_text="Response time < 200ms",
        category="non-functional",
        testable=True,
    )
    assert req.scope_notes is None
    assert req.implementation_hints == ()
    assert req.confidence == 1.0

def test_crystallized_requirement_with_all_fields() -> None:
    """Test all fields can be set."""
    req = CrystallizedRequirement(
        id="req-001",
        original_text="System should be fast",
        refined_text="Response time < 200ms",
        category="non-functional",
        testable=True,
        scope_notes="GET endpoints only",
        implementation_hints=("Use caching", "Async handlers"),
        confidence=0.85,
    )
    assert req.scope_notes == "GET endpoints only"
    assert len(req.implementation_hints) == 2
    assert req.confidence == 0.85

def test_vague_term_detection() -> None:
    """Test vague terms are detected in text."""
    from yolo_developer.agents.analyst.node import _detect_vague_terms

    text = "The system should be fast and easy to use"
    vague = _detect_vague_terms(text)
    assert "should" in vague
    assert "fast" in vague
    assert "easy" in vague
```

**Integration Tests:**
```python
import pytest
from yolo_developer.agents.analyst import analyst_node

@pytest.mark.asyncio
async def test_crystallization_transforms_vague_requirements() -> None:
    """Test that vague requirements become specific."""
    state: YoloState = {
        "messages": [HumanMessage(content="Build a fast, scalable API")],
        "current_agent": "analyst",
        "handoff_context": None,
        "decisions": [],
    }
    result = await analyst_node(state)

    # Check output contains crystallized requirements
    output_data = result["messages"][0].additional_kwargs["metadata"]["output"]
    requirements = output_data["requirements"]

    # Verify vague terms are replaced with specific criteria
    for req in requirements:
        assert "fast" not in req["refined_text"].lower()
        assert "scalable" not in req["refined_text"].lower()
```

### Commit Message Pattern

```
feat: Implement requirement crystallization with vague term detection (Story 5.2)

- Add scope_notes, implementation_hints, confidence to CrystallizedRequirement
- Implement _detect_vague_terms() for identifying ambiguous language
- Enhance LLM prompts for specific, measurable requirement generation
- Add scope boundary extraction to crystallization output
- Implement implementation hint generation based on project patterns
- Enhance audit logging with transformation details
- Add unit tests for new fields and vague detection
- Add integration tests for full crystallization flow

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

### References

- [Source: architecture.md#ADR-001] - TypedDict for internal state
- [Source: architecture.md#ADR-003] - LiteLLM SDK integration
- [Source: architecture.md#ADR-005] - LangGraph patterns
- [Source: epics.md#Epic-5] - Analyst Agent epic context
- [Source: epics.md#Story-5.2] - Requirement Crystallization
- [Source: epics.md#FR36] - Crystallize vague requirements capability
- [Source: agents/analyst/node.py] - Current analyst_node implementation
- [Source: agents/analyst/types.py] - CrystallizedRequirement, AnalystOutput
- [Source: agents/prompts/analyst.py] - Current prompt templates

### Files to Consult (MUST READ Before Implementation)

| File | Purpose | Key Lines |
|------|---------|-----------|
| `agents/analyst/types.py` | Current CrystallizedRequirement to extend | Full file |
| `agents/analyst/node.py` | Current node implementation to enhance | 162-290 |
| `agents/prompts/analyst.py` | Current prompts to update | Full file |
| `orchestrator/context.py` | Decision dataclass for audit | 60-91 |
| `tests/unit/agents/analyst/test_types.py` | Existing tests to maintain compatibility | Full file |
| `tests/unit/agents/analyst/test_node.py` | Existing tests to not break | Full file |

### Success Criteria

1. All vague terms in seed content are detected and logged
2. Refined requirements contain specific, measurable criteria
3. Scope boundaries are defined for each requirement
4. Implementation hints reference project conventions
5. Confidence scores reflect transformation quality
6. All existing tests continue to pass (backward compatibility)
7. New tests cover all new functionality

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- All 63 tests passing (after code review fixes)
- Type checking clean (mypy)
- Linting clean (ruff)

### Completion Notes List

1. Enhanced CrystallizedRequirement with scope_notes, implementation_hints, and confidence fields
2. Implemented vague term detection using regex word boundaries with 30+ vague patterns
3. Updated prompts with transformation rules and confidence scoring guidance
4. Enhanced _parse_llm_response to handle new fields with backward compatibility
5. Updated _crystallize_requirements placeholder to demonstrate vague detection
6. Added comprehensive unit tests (11 vague detection, 4 parsing, 7 placeholder tests)
7. Added 6 integration tests for full crystallization flow
8. All existing tests continue to pass (backward compatibility maintained)

### Code Review Fixes Applied

- **M1:** Added sprint-status.yaml to File List documentation
- **M2:** Moved `import re` from inside loop to top of node.py (performance fix)
- **M3:** Added test for confidence clamping to minimum 0.3
- **M4:** Added test for REFINEMENT_ prompt aliases
- **L1:** Updated module docstring from "(Story 5.1)" to "(Story 5.1, 5.2)"
- **L2:** Fixed weak OR assertion to assert both vague terms detected

### File List

- `src/yolo_developer/agents/analyst/types.py` - Added scope_notes, implementation_hints, confidence fields
- `src/yolo_developer/agents/analyst/node.py` - Added VAGUE_TERMS, _detect_vague_terms(), updated parsing/crystallization
- `src/yolo_developer/agents/prompts/analyst.py` - Enhanced prompts with transformation rules
- `tests/unit/agents/analyst/test_types.py` - Added TestCrystallizedRequirementEnhanced (8 tests)
- `tests/unit/agents/analyst/test_node.py` - Added TestVagueTermDetection (11), TestLLMResponseParsing (4), TestCrystallizeRequirementsPlaceholder (7), TestPromptAliases (1)
- `tests/integration/test_analyst_integration.py` - Added TestCrystallizationIntegration (6 tests)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Updated story status to review
