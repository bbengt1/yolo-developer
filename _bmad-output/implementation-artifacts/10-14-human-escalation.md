# Story 10.14: Human Escalation

Status: done

## Story

As a developer,
I want unresolvable issues escalated to me,
so that I can make decisions the system cannot.

## Acceptance Criteria

1. **Given** the system detects an unresolvable issue (circular logic persists after intervention, critical conflict, system error)
   **When** escalation is triggered
   **Then** the SM creates an `EscalationRequest` with:
   - Issue summary and context
   - History of attempted resolutions
   - Available options (if applicable)
   - Recommended action (if determinable)

2. **Given** an escalation is created
   **When** the request is presented to the user
   **Then** the interface shows:
   - Clear problem description
   - Relevant context (agent exchanges, decisions, errors)
   - Action options with explanations
   - Default/recommended option highlighted

3. **Given** the user provides a decision
   **When** the response is received
   **Then** the system:
   - Validates the response format
   - Integrates the decision into state
   - Routes to appropriate agent for continuation
   - Updates escalation status to resolved

4. **Given** any escalation event
   **When** the escalation occurs
   **Then** full audit trail is maintained:
   - Escalation trigger and reason
   - Context snapshot at escalation time
   - User decision and rationale (if provided)
   - Resolution outcome

5. **Given** escalation is triggered
   **When** the user doesn't respond within timeout
   **Then** the system:
   - Uses safe default action (if available)
   - Logs timeout with context
   - Continues with conservative behavior

## Tasks / Subtasks

- [x] Task 1: Create type definitions (AC: #1, #2, #3, #4)
  - [x] 1.1 Create `human_escalation_types.py` with:
    - `EscalationTrigger` Literal type (circular_logic, conflict_unresolved, gate_blocked, system_error, agent_stuck, user_requested)
    - `EscalationStatus` Literal type (pending, presented, resolved, timed_out, cancelled)
    - `EscalationRequest` frozen dataclass (request_id, trigger, agent, summary, context, options, recommended_option, created_at)
    - `EscalationOption` frozen dataclass (option_id, label, description, action, is_recommended)
    - `EscalationResponse` frozen dataclass (request_id, selected_option, user_rationale, responded_at)
    - `EscalationResult` frozen dataclass (request, response, status, resolution_action, integration_success, duration_ms)
    - `EscalationConfig` frozen dataclass (timeout_seconds, default_action, log_escalations, max_pending)
  - [x] 1.2 Add validation in `__post_init__` methods with warning logging
  - [x] 1.3 Add `to_dict()` methods for serialization
  - [x] 1.4 Export constants: `DEFAULT_ESCALATION_TIMEOUT_SECONDS`, `VALID_ESCALATION_TRIGGERS`, `VALID_ESCALATION_STATUSES`

- [x] Task 2: Implement escalation detection (AC: #1)
  - [x] 2.1 Create `human_escalation.py` module
  - [x] 2.2 Implement `should_escalate(state: YoloState, cycle_analysis: CycleAnalysis | None, mediation_result: MediationResult | None, health_status: HealthStatus | None) -> tuple[bool, EscalationTrigger | None]`
  - [x] 2.3 Implement `_check_circular_escalation(cycle_analysis)` - escalate if escalation_triggered is True
  - [x] 2.4 Implement `_check_conflict_escalation(mediation_result)` - escalate if escalations_triggered non-empty
  - [x] 2.5 Implement `_check_health_escalation(health_status)` - escalate if severity is critical
  - [x] 2.6 Implement `_check_gate_blocked_escalation(state)` - escalate if gate blocked with no recovery path

- [x] Task 3: Implement escalation request creation (AC: #1, #2)
  - [x] 3.1 Implement `create_escalation_request(state: YoloState, trigger: EscalationTrigger, context: dict[str, Any]) -> EscalationRequest`
  - [x] 3.2 Implement `_build_escalation_summary(state, trigger)` - creates human-readable summary
  - [x] 3.3 Implement `_build_escalation_context(state, trigger)` - gathers relevant context (exchanges, decisions, errors)
  - [x] 3.4 Implement `_build_escalation_options(state, trigger)` - creates available action options
  - [x] 3.5 Implement `_determine_recommended_option(options, state, trigger)` - selects best default option

- [x] Task 4: Implement response integration (AC: #3, #5)
  - [x] 4.1 Implement `integrate_escalation_response(state: YoloState, request: EscalationRequest, response: EscalationResponse) -> EscalationResult`
  - [x] 4.2 Implement `_validate_response(request, response)` - validates response matches request options
  - [x] 4.3 Implement `_apply_resolution_action(state, request, response)` - applies user decision to state
  - [x] 4.4 Implement `_determine_next_agent(resolution_action)` - determines routing after resolution
  - [x] 4.5 Implement `handle_escalation_timeout(request: EscalationRequest, config: EscalationConfig) -> EscalationResult` - handles timeout with default action

- [x] Task 5: Implement main orchestration function (AC: #1, #2, #3, #4)
  - [x] 5.1 Implement `async def manage_human_escalation(state: YoloState, cycle_analysis: CycleAnalysis | None = None, mediation_result: MediationResult | None = None, health_status: HealthStatus | None = None, config: EscalationConfig | None = None) -> EscalationResult | None`
  - [x] 5.2 Orchestrate: check → create request → (present → integrate) or return None
  - [x] 5.3 Add structured logging with structlog for all escalation events
  - [x] 5.4 Ensure proper error handling per ADR-007

- [x] Task 6: Integrate with SM node (AC: #1, #4)
  - [x] 6.1 Update `sm_node` in `node.py` to call `manage_human_escalation` when escalation is detected
  - [x] 6.2 Update state with escalation_request when pending
  - [x] 6.3 Add escalation to SMOutput rationale
  - [x] 6.4 Ensure escalation integrates with existing circular_logic, conflict, and health checks

- [x] Task 7: Update __init__.py exports (AC: all)
  - [x] 7.1 Add imports from `human_escalation` and `human_escalation_types`
  - [x] 7.2 Update `__all__` list with new exports (alphabetically sorted)
  - [x] 7.3 Update module docstring with human escalation references

- [x] Task 8: Write comprehensive tests (AC: all)
  - [x] 8.1 Test type validation in `human_escalation_types.py` (valid/invalid values, edge cases)
  - [x] 8.2 Test `should_escalate` with various trigger combinations
  - [x] 8.3 Test `create_escalation_request` produces valid requests with options
  - [x] 8.4 Test `integrate_escalation_response` with valid and invalid responses
  - [x] 8.5 Test `handle_escalation_timeout` applies default action correctly
  - [x] 8.6 Test `manage_human_escalation` full flow
  - [x] 8.7 Test SM node integration with escalation
  - [x] 8.8 Test logging output with structlog capture

## Dev Notes

### Architecture Patterns

- **Type Definitions**: Create `human_escalation_types.py` (frozen dataclasses per ADR-001)
- **Implementation**: Create `human_escalation.py` (async functions per ADR-005)
- **Logging**: Use `structlog.get_logger(__name__)` pattern
- **State**: YoloState TypedDict for graph state, frozen dataclasses for internal
- **Error Handling**: Per ADR-007 - retry with backoff, SM-coordinated recovery

### Integration Points

From existing code analysis:
- `circular_detection.py:CycleAnalysis.escalation_triggered` - already flags when escalation needed
- `conflict_mediation.py:MediationResult.escalations_triggered` - tuple of conflict IDs requiring escalation
- `health.py:HealthStatus.severity` - "critical" triggers escalation
- `emergency.py:escalate_emergency()` - existing escalation infrastructure (extend, don't duplicate)
- `node.py:sm_node()` - already checks `should_escalate` flag, needs to create EscalationRequest

### Key Constants from Related Modules

```python
# From circular_detection_types.py
InterventionStrategy = Literal["inject_context", "break_cycle", "escalate_human", "wait_and_retry"]

# From emergency_types.py
EmergencyType = Literal["health_degraded", "circular_logic", "gate_blocked", "agent_stuck", "system_error"]

# From conflict_types.py
MediationResult.escalations_triggered: tuple[str, ...]  # conflict IDs needing human intervention
```

### State Fields to Use/Add

```python
# Existing in YoloState (per architecture.md)
escalate_to_human: bool  # Flag indicating escalation needed

# New fields to add (or use existing escalation infrastructure)
escalation_request: EscalationRequest | None  # Pending request for user
escalation_response: EscalationResponse | None  # User's decision
```

### Project Structure Notes

- Module location: `src/yolo_developer/agents/sm/human_escalation.py`
- Types location: `src/yolo_developer/agents/sm/human_escalation_types.py`
- Tests location: `tests/unit/agents/sm/test_human_escalation.py`
- Test types: `tests/unit/agents/sm/test_human_escalation_types.py`

### References

- [Source: _bmad-output/planning-artifacts/prd.md] FR70: SM Agent can escalate to human when circular logic persists
- [Source: _bmad-output/planning-artifacts/architecture.md#YoloState] escalate_to_human: bool field
- [Source: src/yolo_developer/agents/sm/circular_detection.py:505] "Critical circular logic detected - human intervention required"
- [Source: src/yolo_developer/agents/sm/circular_detection_types.py:269] escalation_triggered: Whether escalation was triggered
- [Source: src/yolo_developer/agents/sm/conflict_types.py:300] escalations_triggered: Conflict IDs requiring escalation
- [Source: src/yolo_developer/agents/sm/emergency.py:750] _create_escalation_record for audit trail
- [Source: _bmad-output/implementation-artifacts/10-13-context-injection.md] Pattern reference for story structure

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A

### Completion Notes List

1. Created `human_escalation_types.py` with all type definitions per ADR-001 (frozen dataclasses):
   - EscalationTrigger Literal type (6 trigger types)
   - EscalationStatus Literal type (5 status values)
   - EscalationOption, EscalationRequest, EscalationResponse, EscalationResult, EscalationConfig dataclasses
   - All with validation in `__post_init__` and `to_dict()` methods

2. Created `human_escalation.py` with the implementation:
   - `should_escalate()`: Detects escalation triggers from circular detection, conflict mediation, health status, gate blocked, and user request
   - `create_escalation_request()`: Builds EscalationRequest with options and recommended option
   - `integrate_escalation_response()`: Processes user decision and validates response
   - `handle_escalation_timeout()`: Handles timeout with default action
   - `manage_human_escalation()`: Main orchestration function (async)

3. Integrated with SM node in `node.py`:
   - Added call to `manage_human_escalation()` after conflict mediation
   - Added `escalation_result` to state updates and SMOutput
   - Added logging for human escalation events

4. Updated `types.py` to add `escalation_result` field to SMOutput

5. Updated `__init__.py` with all exports (alphabetically sorted)

6. Created comprehensive test suite (96 tests total):
   - 39 tests in `test_human_escalation_types.py` for type validation
   - 43 tests in `test_human_escalation.py` for implementation:
     - 15 tests in `TestShouldEscalate` (all trigger types including gate_blocked, user_requested)
     - 5 tests in `TestCreateEscalationRequest`
     - 4 tests in `TestIntegrateEscalationResponse`
     - 3 tests in `TestHandleEscalationTimeout`
     - 4 tests in `TestLoggingOutput`
     - 12 tests in `TestManageHumanEscalation` (full orchestration flow)
   - 14 tests in `test_node.py::TestSMNodeHumanEscalation` for SM node integration

7. All tests pass, linting passes, mypy passes

8. Code Review Fixes Applied:
   - Added missing tests for `manage_human_escalation` async function
   - Added tests for `gate_blocked` and `user_requested` escalation triggers
   - Enhanced `_build_escalation_context()` to include history of attempted resolutions per AC #1
   - Removed unused `_determine_next_agent()` function
   - Updated test count documentation

### File List

- `src/yolo_developer/agents/sm/human_escalation_types.py` (new)
- `src/yolo_developer/agents/sm/human_escalation.py` (new)
- `src/yolo_developer/agents/sm/node.py` (modified)
- `src/yolo_developer/agents/sm/types.py` (modified)
- `src/yolo_developer/agents/sm/__init__.py` (modified)
- `tests/unit/agents/sm/test_human_escalation_types.py` (new)
- `tests/unit/agents/sm/test_human_escalation.py` (new)
- `tests/unit/agents/sm/test_node.py` (modified - added TestSMNodeHumanEscalation)
