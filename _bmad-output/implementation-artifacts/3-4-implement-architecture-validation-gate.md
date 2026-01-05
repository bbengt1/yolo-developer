# Story 3.4: Implement Architecture Validation Gate

Status: done

## Story

As a system user,
I want architectural decisions validated against principles,
So that designs follow established best practices.

## Acceptance Criteria

1. **AC1: 12-Factor Compliance Check**
   - **Given** the Architect agent produces a design
   - **When** the architecture gate evaluates it
   - **Then** 12-Factor principle compliance is checked
   - **And** violations are identified with specific principle references
   - **And** each violation includes remediation guidance

2. **AC2: Tech Stack Constraint Validation**
   - **Given** a project has configured tech stack constraints
   - **When** the architecture gate evaluates a design
   - **Then** tech stack constraint violations are flagged
   - **And** unsupported technologies are identified
   - **And** version compatibility issues are detected

3. **AC3: Security Anti-Pattern Detection**
   - **Given** architectural decisions include component designs
   - **When** the architecture gate evaluates them
   - **Then** security anti-patterns are detected
   - **And** patterns like hardcoded secrets, SQL injection vectors, XSS risks are flagged
   - **And** each security issue includes severity and remediation

4. **AC4: Compliance Score Calculation**
   - **Given** a design has been evaluated against all criteria
   - **When** the compliance score is calculated
   - **Then** a score between 0-100 is returned
   - **And** the score breakdown shows contributing factors
   - **And** scores below configured threshold fail the gate

5. **AC5: Gate Evaluator Registration**
   - **Given** the architecture validation gate evaluator is implemented
   - **When** the gates module is loaded
   - **Then** the evaluator is registered with name "architecture_validation"
   - **And** the evaluator is available via `get_evaluator("architecture_validation")`
   - **And** the evaluator follows the GateEvaluator protocol

6. **AC6: State Integration**
   - **Given** architectural decisions exist in state under `architecture` key
   - **When** the architecture validation gate is applied via @quality_gate("architecture_validation")
   - **Then** the gate reads design from `state["architecture"]`
   - **And** ADRs are evaluated from `state["architecture"]["adrs"]` if present
   - **And** the gate result includes which specific decisions failed

## Tasks / Subtasks

- [x] Task 1: Define Architecture Validation Types (AC: 1, 2, 3, 4)
  - [x] Create `src/yolo_developer/gates/gates/architecture_validation.py` module
  - [x] Define `ArchitectureIssue` dataclass (decision_id, issue_type, description, severity, principle)
  - [x] Define `TWELVE_FACTOR_PRINCIPLES` constant with principle names and descriptions
  - [x] Define `SECURITY_ANTI_PATTERNS` constant with common anti-patterns
  - [x] Export types from `gates/gates/__init__.py`

- [x] Task 2: Implement 12-Factor Compliance Detection (AC: 1)
  - [x] Create `check_twelve_factor_compliance(architecture: dict) -> list[ArchitectureIssue]` function
  - [x] Check for codebase (single deployable unit)
  - [x] Check for dependencies (explicit declaration)
  - [x] Check for config (environment separation)
  - [x] Check for backing services (attached resources)
  - [x] Check for build/release/run separation
  - [x] Check for processes (stateless)
  - [x] Check for port binding (self-contained)
  - [x] Check for concurrency (process model)
  - [x] Check for disposability (fast startup/shutdown)
  - [x] Check for dev/prod parity
  - [x] Check for logs (event streams)
  - [x] Check for admin processes (one-off tasks)

- [x] Task 3: Implement Tech Stack Constraint Validation (AC: 2)
  - [x] Create `validate_tech_stack(architecture: dict, constraints: dict) -> list[ArchitectureIssue]` function
  - [x] Check allowed languages against design
  - [x] Check allowed frameworks against design
  - [x] Check allowed databases against design
  - [x] Check version compatibility where specified
  - [x] Flag unsupported technologies with suggestions

- [x] Task 4: Implement Security Anti-Pattern Detection (AC: 3)
  - [x] Create `detect_security_anti_patterns(architecture: dict) -> list[ArchitectureIssue]` function
  - [x] Check for hardcoded secrets patterns
  - [x] Check for SQL injection vectors (string concatenation in queries)
  - [x] Check for XSS risks (unescaped output)
  - [x] Check for insecure communication (HTTP vs HTTPS)
  - [x] Check for missing authentication/authorization
  - [x] Check for exposed sensitive endpoints
  - [x] Assign severity levels (critical, high, medium, low)

- [x] Task 5: Implement Compliance Score Calculation (AC: 4)
  - [x] Create `calculate_compliance_score(issues: list[ArchitectureIssue]) -> tuple[int, dict]` function
  - [x] Weight issues by severity (critical=25, high=15, medium=5, low=1)
  - [x] Calculate score as 100 - weighted_deductions
  - [x] Return (score, breakdown_dict) tuple
  - [x] Cap minimum score at 0

- [x] Task 6: Implement Architecture Validation Evaluator (AC: 1, 2, 3, 4, 5, 6)
  - [x] Create async `architecture_validation_evaluator(context: GateContext) -> GateResult` function
  - [x] Extract architecture from `context.state["architecture"]`
  - [x] Extract tech stack constraints from `context.state.get("config", {}).get("tech_stack", {})`
  - [x] Run all validation checks
  - [x] Calculate compliance score
  - [x] Return GateResult with passed=True only if score >= threshold
  - [x] Include all issues and score in result metadata

- [x] Task 7: Implement Failure Report Generation (AC: 1, 2, 3, 4)
  - [x] Create `generate_architecture_report(issues: list[ArchitectureIssue], score: int, breakdown: dict) -> str` function
  - [x] Format issues by category (12-Factor, tech stack, security)
  - [x] Include severity levels and affected decisions
  - [x] Include compliance score with breakdown
  - [x] Include remediation suggestions for each issue type

- [x] Task 8: Register Evaluator (AC: 5)
  - [x] Register `architecture_validation_evaluator` in module initialization
  - [x] Use `register_evaluator("architecture_validation", architecture_validation_evaluator)`
  - [x] Update `gates/gates/__init__.py` to auto-register on import
  - [x] Verify registration in `gates/__init__.py` exports

- [x] Task 9: Write Unit Tests (AC: all)
  - [x] Create `tests/unit/gates/test_architecture_validation.py`
  - [x] Test 12-Factor compliance detection with various violations
  - [x] Test tech stack constraint validation
  - [x] Test security anti-pattern detection
  - [x] Test compliance score calculation
  - [x] Test full evaluator with passing designs
  - [x] Test full evaluator with failing designs
  - [x] Test report generation format
  - [x] Test input validation (missing architecture key)

- [x] Task 10: Write Integration Tests (AC: 5, 6)
  - [x] Create `tests/integration/test_architecture_validation_gate.py`
  - [x] Test gate decorator integration with architecture_validation evaluator
  - [x] Test state reading from `state["architecture"]`
  - [x] Test gate blocking behavior with low compliance scores
  - [x] Test gate passing behavior with compliant designs
  - [x] Test configuration threshold integration

## Dev Notes

### Architecture Compliance

- **ADR-006 (Quality Gate Pattern):** Decorator-based gates per architecture specification
- **FR21:** System can evaluate architectural decisions against defined principles
- **Epic 3 Goal:** Validate artifacts at every agent boundary and block low-quality handoffs

### Technical Requirements

- **Evaluator Protocol:** Must implement `GateEvaluator` protocol from Story 3.1
- **Async Pattern:** Evaluator must be async function per architecture
- **State Access:** Read architecture from `state["architecture"]` key
- **Structured Logging:** Use structlog for all log messages

### 12-Factor Principles Reference

```python
TWELVE_FACTOR_PRINCIPLES = {
    "codebase": "One codebase tracked in revision control, many deploys",
    "dependencies": "Explicitly declare and isolate dependencies",
    "config": "Store config in the environment",
    "backing_services": "Treat backing services as attached resources",
    "build_release_run": "Strictly separate build and run stages",
    "processes": "Execute the app as one or more stateless processes",
    "port_binding": "Export services via port binding",
    "concurrency": "Scale out via the process model",
    "disposability": "Maximize robustness with fast startup and graceful shutdown",
    "dev_prod_parity": "Keep development, staging, and production as similar as possible",
    "logs": "Treat logs as event streams",
    "admin_processes": "Run admin/management tasks as one-off processes",
}
```

### Security Anti-Patterns to Detect

```python
SECURITY_ANTI_PATTERNS = {
    "hardcoded_secrets": {
        "patterns": ["password=", "api_key=", "secret=", "token="],
        "severity": "critical",
        "remediation": "Use environment variables or secrets manager"
    },
    "sql_injection": {
        "patterns": ["string concatenation", "f-string in query", "format string in SQL"],
        "severity": "critical",
        "remediation": "Use parameterized queries or ORM"
    },
    "missing_auth": {
        "patterns": ["no authentication", "public endpoint", "unprotected route"],
        "severity": "high",
        "remediation": "Add authentication middleware"
    },
    "insecure_transport": {
        "patterns": ["http://", "no TLS", "plain text"],
        "severity": "high",
        "remediation": "Use HTTPS/TLS for all communications"
    },
    "xss_risk": {
        "patterns": ["innerHTML", "dangerouslySetInnerHTML", "unescaped output"],
        "severity": "high",
        "remediation": "Sanitize and escape user input"
    },
}
```

### Expected Architecture State Structure

```python
# Architecture is expected to be a dict in state
# Minimum structure for validation:
Architecture = TypedDict("Architecture", {
    "decisions": list,          # List of architectural decisions
    "adrs": list,               # Optional: Architecture Decision Records
    "tech_stack": dict,         # Technologies used
    "components": list,         # System components
    "security": dict,           # Security considerations
})

ArchitecturalDecision = TypedDict("ArchitecturalDecision", {
    "id": str,                  # Decision identifier
    "title": str,               # Decision title
    "description": str,         # Detailed description
    "rationale": str,           # Why this decision was made
    "technologies": list,       # Technologies involved
})
```

### Compliance Score Thresholds

- **90-100:** Excellent - Gate passes with no warnings
- **70-89:** Good - Gate passes with warnings
- **50-69:** Fair - Gate fails (blocking)
- **0-49:** Poor - Gate fails (critical blocking)

Default threshold: 70 (configurable via `quality.architecture_threshold`)

### Severity Levels

- **critical:** Security vulnerabilities, major 12-Factor violations (gate fails)
- **high:** Significant issues affecting maintainability (gate fails if multiple)
- **medium:** Best practice deviations (warning)
- **low:** Minor suggestions (informational)

### File Structure

```
src/yolo_developer/gates/
├── __init__.py                  # UPDATE: Export architecture_validation gate
├── types.py                     # From Story 3.1
├── decorator.py                 # From Story 3.1
├── evaluators.py                # From Story 3.1
└── gates/
    ├── __init__.py              # UPDATE: Add architecture_validation exports
    ├── testability.py           # From Story 3.2
    ├── ac_measurability.py      # From Story 3.3
    └── architecture_validation.py  # NEW: Architecture validation implementation
```

### Previous Story Intelligence (from Story 3.3)

**Patterns to Apply:**
1. Use frozen dataclasses for issue types (immutable)
2. Evaluator is async callable: `async def evaluator(ctx: GateContext) -> GateResult`
3. Register via `register_evaluator(gate_name, evaluator)`
4. State is accessible via `context.state`
5. Use `GateResult.to_dict()` for state serialization
6. Add autouse fixture in tests to re-register evaluator after `clear_evaluators()` calls
7. Validate input types before processing (architecture must be dict)
8. Pre-sort constants at module level for performance (from code review feedback)

**Key Files to Reference:**
- `src/yolo_developer/gates/types.py` - GateResult, GateContext dataclasses
- `src/yolo_developer/gates/evaluators.py` - GateEvaluator protocol, registration functions
- `src/yolo_developer/gates/decorator.py` - @quality_gate decorator usage
- `src/yolo_developer/gates/gates/testability.py` - Pattern for detection functions
- `src/yolo_developer/gates/gates/ac_measurability.py` - Latest gate implementation pattern
- `tests/unit/gates/test_ac_measurability.py` - Test patterns including autouse fixture

### Testing Standards

- Use pytest-asyncio for async tests
- Create mock architecture data in test fixtures
- Test both passing and failing scenarios
- Test edge cases (empty architecture, missing keys)
- Verify structured logging output
- Add autouse fixture to ensure evaluator registration

### References

- [Source: architecture.md#ADR-006] - Quality Gate Pattern
- [Source: epics.md#Story-3.4] - Implement Architecture Validation Gate requirements
- [Source: prd.md#FR21] - System can evaluate architectural decisions against defined principles
- [Story 3.1 Implementation] - Gate decorator framework
- [Story 3.2 Implementation] - Testability gate pattern
- [Story 3.3 Implementation] - AC measurability gate (latest pattern)

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

None - implementation completed without errors requiring debug logs.

### Completion Notes List

- Implemented architecture validation gate following TDD (red-green-refactor) approach
- Created 38 unit tests and 12 integration tests (50 new tests total)
- All 893 tests passing after implementation
- Used frozen dataclass pattern from Story 3.3 for ArchitectureIssue
- Implemented all 12-Factor principle checks with keyword detection
- Security anti-patterns use compiled regex patterns for efficiency
- Compliance score calculation: 100 - weighted_deductions (critical=25, high=15, medium=5, low=1)
- Default threshold of 70 is configurable via `config.quality.architecture_threshold`
- Evaluator auto-registered on module import via `register_evaluator()`
- Integration tests include autouse fixture for evaluator registration resilience

### File List

**Created:**
- `src/yolo_developer/gates/gates/architecture_validation.py` - Main implementation (~750 lines)
- `tests/unit/gates/test_architecture_validation.py` - Unit tests
- `tests/integration/test_architecture_validation_gate.py` - Integration tests

**Modified:**
- `src/yolo_developer/gates/gates/__init__.py` - Added exports for architecture_validation module
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Updated story status
