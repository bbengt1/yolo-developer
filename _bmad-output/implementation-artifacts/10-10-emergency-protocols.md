# Story 10.10: Emergency Protocols

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want emergency protocols when system health degrades,
So that failures are handled gracefully.

## Acceptance Criteria

1. **Given** system health falling below thresholds
   **When** emergency is triggered
   **Then** appropriate protocol activates

2. **Given** an emergency protocol activation
   **When** the protocol executes
   **Then** current state is checkpointed

3. **Given** a checkpointed state
   **When** recovery is evaluated
   **Then** recovery options are presented

4. **Given** recovery options
   **When** recovery cannot proceed
   **Then** escalation occurs if needed

## Tasks / Subtasks

- [x] Task 1: Create emergency protocol types module (AC: #1, #2, #3, #4)
  - [x] 1.1: Create `src/yolo_developer/agents/sm/emergency_types.py` module
  - [x] 1.2: Define `EmergencyType` Literal type (health_degraded, circular_logic, gate_blocked, agent_stuck, system_error)
  - [x] 1.3: Define `ProtocolStatus` Literal type (pending, active, checkpointed, recovering, resolved, escalated)
  - [x] 1.4: Define `RecoveryAction` Literal type (retry, rollback, skip, escalate, terminate)
  - [x] 1.5: Define `EmergencyTrigger` frozen dataclass (emergency_type, severity, source_agent, trigger_reason, health_status, detected_at)
  - [x] 1.6: Define `Checkpoint` frozen dataclass (checkpoint_id, state_snapshot, created_at, trigger_type, metadata)
  - [x] 1.7: Define `RecoveryOption` frozen dataclass (action, description, confidence, risks, estimated_impact)
  - [x] 1.8: Define `EmergencyProtocol` frozen dataclass (protocol_id, trigger, status, checkpoint, recovery_options, selected_action, escalation_reason, created_at, resolved_at)
  - [x] 1.9: Define `EmergencyConfig` frozen dataclass (auto_checkpoint, max_recovery_attempts, escalation_threshold, enable_auto_recovery)
  - [x] 1.10: Add `to_dict()` method to all dataclasses for serialization
  - [x] 1.11: Define constants: DEFAULT_MAX_RECOVERY_ATTEMPTS, DEFAULT_ESCALATION_THRESHOLD, VALID_EMERGENCY_TYPES, VALID_PROTOCOL_STATUSES, VALID_RECOVERY_ACTIONS

- [x] Task 2: Implement emergency detection functions (AC: #1)
  - [x] 2.1: Create `src/yolo_developer/agents/sm/emergency.py` module
  - [x] 2.2: Implement `_check_health_degradation()` using HealthStatus from health monitoring
  - [x] 2.3: Implement `_check_circular_logic()` using CycleAnalysis from circular detection
  - [x] 2.4: Implement `_check_gate_blocked()` checking for repeated gate failures
  - [x] 2.5: Implement `_check_agent_stuck()` checking idle time thresholds
  - [x] 2.6: Implement `_check_system_error()` checking for unrecoverable errors
  - [x] 2.7: Implement `_detect_emergency()` orchestrating all detection checks
  - [x] 2.8: Implement `_create_emergency_trigger()` from detection results

- [x] Task 3: Implement checkpointing functions (AC: #2)
  - [x] 3.1: Implement `_capture_state_snapshot()` to serialize relevant state
  - [x] 3.2: Implement `_create_checkpoint()` to create Checkpoint from snapshot
  - [x] 3.3: Implement `_store_checkpoint()` to persist checkpoint (in-memory for MVP)
  - [x] 3.4: Implement `_retrieve_checkpoint()` to load checkpoint by ID
  - [x] 3.5: Implement `checkpoint_state()` main checkpointing entry point

- [x] Task 4: Implement recovery evaluation functions (AC: #3)
  - [x] 4.1: Implement `_evaluate_retry_option()` assessing retry viability
  - [x] 4.2: Implement `_evaluate_rollback_option()` assessing rollback viability
  - [x] 4.3: Implement `_evaluate_skip_option()` assessing skip viability
  - [x] 4.4: Implement `_evaluate_escalate_option()` when human intervention needed
  - [x] 4.5: Implement `_evaluate_terminate_option()` for graceful termination
  - [x] 4.6: ~~Implement `_calculate_recovery_confidence()`~~ (REMOVED: confidence calculation integrated into individual `_evaluate_*_option()` functions during code review)
  - [x] 4.7: Implement `_generate_recovery_options()` combining all evaluations
  - [x] 4.8: Implement `_select_best_recovery()` auto-selecting highest confidence option

- [x] Task 5: Implement escalation functions (AC: #4)
  - [x] 5.1: Implement `_should_escalate()` checking escalation criteria
  - [x] 5.2: Implement `_create_escalation_record()` documenting escalation reason
  - [x] 5.3: Implement `_notify_escalation()` logging escalation for human review
  - [x] 5.4: Implement `escalate_emergency()` main escalation entry point

- [x] Task 6: Implement main emergency protocol function (AC: all)
  - [x] 6.1: Implement async `trigger_emergency_protocol()` main entry function
  - [x] 6.2: Orchestrate: detect -> checkpoint -> evaluate_recovery -> execute_or_escalate
  - [x] 6.3: Return `EmergencyProtocol` with full protocol outcome
  - [x] 6.4: Make emergency handling configurable via `EmergencyConfig`
  - [x] 6.5: Handle nested failures gracefully (emergency during emergency)
  - [x] 6.6: Log all protocol steps with structlog

- [x] Task 7: Integrate with SM node (AC: all)
  - [x] 7.1: Update `types.py` to add `emergency_protocol` field to SMOutput
  - [x] 7.2: Update `node.py` to call `trigger_emergency_protocol()` when health status is critical
  - [x] 7.3: Export emergency functions from SM `__init__.py`

- [x] Task 8: Write comprehensive tests (AC: all)
  - [x] 8.1: Create `tests/unit/agents/sm/test_emergency_types.py`
  - [x] 8.2: Create `tests/unit/agents/sm/test_emergency.py`
  - [x] 8.3: Test emergency type detection for all types (health_degraded, circular_logic, gate_blocked, agent_stuck, system_error)
  - [x] 8.4: Test checkpoint creation and retrieval
  - [x] 8.5: Test recovery option evaluation for all action types
  - [x] 8.6: Test escalation criteria and triggering
  - [x] 8.7: Test full protocol flow from detection to resolution
  - [x] 8.8: Test configuration options
  - [x] 8.9: Add integration tests in test_node.py for SM node integration

## Dev Notes

### Architecture Requirements

This story implements:
- **FR17**: SM Agent can trigger emergency protocols when system health degrades
- **FR71**: SM Agent can coordinate rollback operations as emergency sprints
- **FR70**: SM Agent can escalate to human when circular logic persists
- **ADR-007**: Retry with exponential backoff + SM-coordinated recovery

Per the architecture document and ADR-001/ADR-005/ADR-007:
- State management uses TypedDict internally with Pydantic at boundaries
- LangGraph message passing with typed state transitions
- SM is the control plane for orchestration decisions
- All operations should be async
- Return state updates, never mutate input state
- Use frozen dataclasses for immutable types
- Checkpoint-based state recovery for resilience

**Key Concept**: Emergency protocols provide a safety net when the orchestration system encounters problems that cannot be resolved through normal operations. When health degrades, circular logic persists, or agents get stuck, the emergency protocol captures state, evaluates recovery options, and either auto-recovers or escalates to human intervention.

### Related FRs

- **FR17**: SM Agent can trigger emergency protocols when system health degrades (PRIMARY)
- **FR71**: SM Agent can coordinate rollback operations as emergency sprints (PRIMARY)
- **FR70**: SM Agent can escalate to human when circular logic persists (PRIMARY)
- **FR11**: SM Agent can monitor agent activity and health metrics
- **FR12**: SM Agent can detect circular logic between agents (>3 exchanges)
- **FR67**: SM Agent can detect agent churn rate and idle time

### Existing Infrastructure to Use

**Health Types** (`agents/sm/health_types.py` - Story 10.5):

```python
# These types detect when emergencies should trigger:
@dataclass(frozen=True)
class HealthStatus:
    """Overall system health status."""
    status: HealthSeverity  # healthy, warning, degraded, critical
    metrics: HealthMetrics
    alerts: tuple[HealthAlert, ...]
    summary: str
    is_healthy: bool

# Use is_healthy=False and status="critical" to trigger emergency
```

**Circular Detection** (`agents/sm/circular_detection_types.py` - Story 10.6):

```python
# Detect circular logic patterns:
@dataclass(frozen=True)
class CycleAnalysis:
    """Analysis result for circular logic detection."""
    has_circular_pattern: bool
    cycles_detected: tuple[CircularPattern, ...]
    severity: CycleSeverity
    intervention_recommended: InterventionStrategy
```

**State Module** (`orchestrator/state.py` - Already Implemented):

```python
class YoloState(TypedDict):
    """Main state for YOLO Developer orchestration."""
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: str
    handoff_context: HandoffContext | None
    decisions: list[Decision]
```

**SM Types** (`agents/sm/types.py` - Story 10.2):

```python
@dataclass(frozen=True)
class SMOutput:
    """Complete output from SM agent processing."""
    routing_decision: RoutingDecision
    routing_rationale: str
    # ... existing fields ...
    health_status: dict[str, Any] | None = None
    sprint_progress: dict[str, Any] | None = None
    # ADD: emergency_protocol: dict[str, Any] | None = None
```

### Emergency Protocol Data Model

Per existing patterns and new requirements:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

EmergencyType = Literal["health_degraded", "circular_logic", "gate_blocked", "agent_stuck", "system_error"]
"""Type of emergency detected.

Values:
    health_degraded: System health fell below critical thresholds
    circular_logic: Agents in ping-pong loop (>3 exchanges per FR12)
    gate_blocked: Repeated gate failures blocking progress
    agent_stuck: Agent idle beyond threshold (stuck processing)
    system_error: Unrecoverable error in orchestration
"""

ProtocolStatus = Literal["pending", "active", "checkpointed", "recovering", "resolved", "escalated"]
"""Current status of the emergency protocol.

Values:
    pending: Emergency detected, protocol not yet started
    active: Protocol actively executing
    checkpointed: State has been checkpointed
    recovering: Recovery action in progress
    resolved: Emergency resolved successfully
    escalated: Escalated to human intervention
"""

RecoveryAction = Literal["retry", "rollback", "skip", "escalate", "terminate"]
"""Recovery action to take.

Values:
    retry: Retry the failed operation
    rollback: Rollback to checkpoint and try different path
    skip: Skip the problematic step and continue
    escalate: Escalate to human for manual intervention
    terminate: Gracefully terminate the current sprint
"""

# Constants
DEFAULT_MAX_RECOVERY_ATTEMPTS: int = 3
"""Maximum automatic recovery attempts before escalation."""

DEFAULT_ESCALATION_THRESHOLD: float = 0.5
"""Confidence threshold below which to escalate (0.0-1.0)."""

VALID_EMERGENCY_TYPES: frozenset[str] = frozenset({
    "health_degraded", "circular_logic", "gate_blocked", "agent_stuck", "system_error"
})

VALID_PROTOCOL_STATUSES: frozenset[str] = frozenset({
    "pending", "active", "checkpointed", "recovering", "resolved", "escalated"
})

VALID_RECOVERY_ACTIONS: frozenset[str] = frozenset({
    "retry", "rollback", "skip", "escalate", "terminate"
})

@dataclass(frozen=True)
class EmergencyTrigger:
    """Information about what triggered the emergency.

    Captures the conditions that led to emergency protocol activation.
    """
    emergency_type: EmergencyType
    severity: str  # "warning", "critical"
    source_agent: str | None
    trigger_reason: str
    health_status: dict[str, Any] | None  # Snapshot of HealthStatus
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "emergency_type": self.emergency_type,
            "severity": self.severity,
            "source_agent": self.source_agent,
            "trigger_reason": self.trigger_reason,
            "health_status": self.health_status,
            "detected_at": self.detected_at,
        }

@dataclass(frozen=True)
class Checkpoint:
    """Captured state checkpoint for recovery.

    Stores a snapshot of state that can be used for rollback.
    """
    checkpoint_id: str
    state_snapshot: dict[str, Any]
    created_at: str
    trigger_type: EmergencyType
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "state_snapshot": self.state_snapshot,
            "created_at": self.created_at,
            "trigger_type": self.trigger_type,
            "metadata": dict(self.metadata),
        }

@dataclass(frozen=True)
class RecoveryOption:
    """A potential recovery action with evaluation.

    Represents one possible way to recover from the emergency.
    """
    action: RecoveryAction
    description: str
    confidence: float  # 0.0-1.0
    risks: tuple[str, ...]
    estimated_impact: str  # "minimal", "moderate", "significant"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "description": self.description,
            "confidence": self.confidence,
            "risks": list(self.risks),
            "estimated_impact": self.estimated_impact,
        }

@dataclass(frozen=True)
class EmergencyProtocol:
    """Complete emergency protocol execution record.

    Tracks the full lifecycle of an emergency from detection to resolution.
    """
    protocol_id: str
    trigger: EmergencyTrigger
    status: ProtocolStatus
    checkpoint: Checkpoint | None
    recovery_options: tuple[RecoveryOption, ...]
    selected_action: RecoveryAction | None
    escalation_reason: str | None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "trigger": self.trigger.to_dict(),
            "status": self.status,
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
            "recovery_options": [o.to_dict() for o in self.recovery_options],
            "selected_action": self.selected_action,
            "escalation_reason": self.escalation_reason,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }

@dataclass(frozen=True)
class EmergencyConfig:
    """Configuration for emergency protocol behavior.

    Controls automatic recovery and escalation thresholds.
    """
    auto_checkpoint: bool = True
    max_recovery_attempts: int = DEFAULT_MAX_RECOVERY_ATTEMPTS
    escalation_threshold: float = DEFAULT_ESCALATION_THRESHOLD
    enable_auto_recovery: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "auto_checkpoint": self.auto_checkpoint,
            "max_recovery_attempts": self.max_recovery_attempts,
            "escalation_threshold": self.escalation_threshold,
            "enable_auto_recovery": self.enable_auto_recovery,
        }
```

### Main Emergency Protocol Function

```python
import uuid
import structlog

from yolo_developer.agents.sm.emergency_types import (
    Checkpoint,
    EmergencyConfig,
    EmergencyProtocol,
    EmergencyTrigger,
    RecoveryOption,
)
from yolo_developer.agents.sm.health import monitor_health
from yolo_developer.agents.sm.health_types import HealthStatus
from yolo_developer.orchestrator.state import YoloState

logger = structlog.get_logger(__name__)

async def trigger_emergency_protocol(
    state: YoloState,
    health_status: HealthStatus | None = None,
    config: EmergencyConfig | None = None,
) -> EmergencyProtocol:
    """Trigger emergency protocol when system health degrades (FR17, FR71).

    Main entry point for emergency handling. Detects the emergency type,
    checkpoints state, evaluates recovery options, and either auto-recovers
    or escalates to human intervention.

    Args:
        state: Current orchestration state
        health_status: Health status that triggered emergency (optional)
        config: Emergency protocol configuration

    Returns:
        EmergencyProtocol with full protocol outcome
    """
    config = config or EmergencyConfig()
    protocol_id = f"emergency-{uuid.uuid4().hex[:8]}"

    logger.warning(
        "emergency_protocol_triggered",
        protocol_id=protocol_id,
        current_agent=state.get("current_agent"),
    )

    # Step 1: Detect emergency type and create trigger
    trigger = await _detect_emergency(state, health_status)

    # Step 2: Checkpoint state if configured
    checkpoint = None
    if config.auto_checkpoint:
        checkpoint = await checkpoint_state(state, trigger.emergency_type)
        logger.info("state_checkpointed", checkpoint_id=checkpoint.checkpoint_id)

    # Step 3: Evaluate recovery options
    recovery_options = await _generate_recovery_options(
        state=state,
        trigger=trigger,
        checkpoint=checkpoint,
        config=config,
    )

    # Step 4: Select action or escalate
    selected_action = None
    escalation_reason = None
    status: ProtocolStatus = "checkpointed"

    if config.enable_auto_recovery and recovery_options:
        best_option = _select_best_recovery(recovery_options)
        if best_option and best_option.confidence >= config.escalation_threshold:
            selected_action = best_option.action
            status = "recovering"
            logger.info(
                "recovery_action_selected",
                action=selected_action,
                confidence=best_option.confidence,
            )
        else:
            escalation_reason = "No recovery option meets confidence threshold"
            status = "escalated"
    else:
        escalation_reason = "Auto-recovery disabled or no options available"
        status = "escalated"

    # Step 5: Handle escalation if needed
    if status == "escalated":
        await _notify_escalation(protocol_id, trigger, escalation_reason or "Unknown")

    result = EmergencyProtocol(
        protocol_id=protocol_id,
        trigger=trigger,
        status=status,
        checkpoint=checkpoint,
        recovery_options=tuple(recovery_options),
        selected_action=selected_action,
        escalation_reason=escalation_reason,
    )

    logger.warning(
        "emergency_protocol_complete",
        protocol_id=protocol_id,
        status=status,
        selected_action=selected_action,
        escalated=status == "escalated",
    )

    return result
```

### Integration with SM Node

Update sm_node() to trigger emergency protocol when health is critical:

```python
# In node.py - add emergency protocol trigger

from yolo_developer.agents.sm.emergency import trigger_emergency_protocol, EmergencyProtocol

async def sm_node(state: YoloState) -> dict[str, Any]:
    """SM agent node with emergency protocol support (FR17)."""

    # ... existing analysis, routing, health monitoring ...

    # Step X: Check for emergency conditions and trigger protocol
    emergency_protocol: EmergencyProtocol | None = None
    if health_status and not health_status.is_healthy and health_status.status == "critical":
        try:
            emergency_protocol = await trigger_emergency_protocol(
                state=state,
                health_status=health_status,
            )
            logger.warning(
                "emergency_protocol_activated",
                protocol_id=emergency_protocol.protocol_id,
                status=emergency_protocol.status,
            )
        except Exception as e:
            logger.error("emergency_protocol_failed", error=str(e))

    # ... rest of output creation ...

    # Include emergency protocol in output
    output = SMOutput(
        # ... existing fields ...
        emergency_protocol=emergency_protocol.to_dict() if emergency_protocol else None,
    )
```

### Testing Strategy

**Unit Tests:**
- Test each emergency type detection (type definitions and detection logic)
- Test checkpoint creation and retrieval
- Test recovery option evaluation for all action types
- Test escalation criteria
- Test configuration options
- Test protocol status transitions

**Integration Tests:**
- Test full emergency protocol flow with realistic state
- Test SM node integration triggering emergency on critical health
- Test recovery action execution
- Test escalation path

### Previous Story Intelligence

From **Story 10.9** (Sprint Progress Tracking):
- Used frozen dataclasses with `to_dict()` serialization
- Created separate types module (`progress_types.py`) for clarity
- Exported all new types and functions from `__init__.py`
- Used structlog for consistent logging format
- All functions are async
- Graceful degradation on failure (never block main workflow)

From **Story 10.5** (Health Monitoring):
- Pattern for detecting anomalies and generating alerts
- HealthStatus with is_healthy and status fields we'll use to trigger emergencies
- Demonstrates threshold-based alerting

From **Story 10.6** (Circular Logic Detection):
- CycleAnalysis with has_circular_pattern we'll use for circular_logic emergency type
- Pattern for detecting problematic agent interactions

**Key Pattern to Follow:**
```python
# New module structure
src/yolo_developer/agents/sm/
├── emergency.py          # Main emergency protocol logic (NEW)
├── emergency_types.py    # Types only (NEW)
├── node.py               # Updated with emergency protocol trigger
├── types.py              # Add emergency_protocol to SMOutput
└── __init__.py           # Export new types and functions
```

### Git Intelligence

Recent commits show consistent patterns:
- Latest: Story 10.9 sprint progress tracking with code review fixes
- `ea593af`: Story 10.8 agent handoff management with code review fixes
- `35752b6`: Story 10.6 circular logic detection with code review fixes
- `f16eff2`: Story 10.5 health monitoring with code review fixes

Commit message pattern: `feat: Implement <description> with code review fixes (Story X.Y)`

### Project Structure Notes

**New file locations:**
- `src/yolo_developer/agents/sm/emergency.py` - Main emergency protocol module (NEW)
- `src/yolo_developer/agents/sm/emergency_types.py` - Type definitions (NEW)
- `tests/unit/agents/sm/test_emergency.py` - Emergency protocol tests (NEW)
- `tests/unit/agents/sm/test_emergency_types.py` - Types tests (NEW)

**Files to modify:**
- `src/yolo_developer/agents/sm/__init__.py` - Export emergency functions
- `src/yolo_developer/agents/sm/types.py` - Add `emergency_protocol` field to SMOutput
- `src/yolo_developer/agents/sm/node.py` - Integrate emergency protocol triggering

### Implementation Patterns

Per architecture document:

1. **Async-first**: `trigger_emergency_protocol()` must be async
2. **State updates via dict**: Return dict updates, don't mutate state
3. **Structured logging**: Use structlog with key-value format
4. **Type annotations**: Full type hints on all functions
5. **Immutable outputs**: Use frozen dataclasses for types
6. **snake_case**: All state dictionary keys use snake_case
7. **Graceful degradation**: If emergency protocol fails, log but don't crash
8. **Checkpoint-based recovery**: Per ADR-007

```python
# CORRECT pattern for emergency module
from __future__ import annotations

import structlog

from yolo_developer.agents.sm.emergency_types import (
    EmergencyConfig,
    EmergencyProtocol,
    EmergencyTrigger,
    RecoveryOption,
)
from yolo_developer.agents.sm.health_types import HealthStatus
from yolo_developer.orchestrator.state import YoloState

logger = structlog.get_logger(__name__)

async def trigger_emergency_protocol(
    state: YoloState,
    health_status: HealthStatus | None = None,
    config: EmergencyConfig | None = None,
) -> EmergencyProtocol:
    """Trigger emergency protocol when system health degrades (FR17, FR71).

    Args:
        state: Current orchestration state
        health_status: Health status that triggered emergency
        config: Emergency protocol configuration

    Returns:
        EmergencyProtocol with full protocol outcome
    """
    logger.warning(
        "emergency_protocol_triggered",
        current_agent=state.get("current_agent"),
    )

    # ... implementation ...

    logger.warning(
        "emergency_protocol_complete",
        status=result.status,
        escalated=result.status == "escalated",
    )

    return result
```

### Dependencies

**Internal dependencies:**
- `yolo_developer.agents.sm.types` - SMOutput (to be modified)
- `yolo_developer.agents.sm.node` - sm_node function (to be modified)
- `yolo_developer.agents.sm.health` - monitor_health, HealthStatus
- `yolo_developer.agents.sm.health_types` - HealthStatus, HealthSeverity
- `yolo_developer.agents.sm.circular_detection` - detect_circular_logic
- `yolo_developer.agents.sm.circular_detection_types` - CycleAnalysis
- `yolo_developer.orchestrator.state` - YoloState
- `structlog` - logging
- `uuid` - protocol ID generation

**No new external dependencies needed.**

### References

- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-007]
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-001]
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-005]
- [Source: _bmad-output/planning-artifacts/epics.md#Story-10.10]
- [Source: _bmad-output/planning-artifacts/epics.md#FR17]
- [Source: _bmad-output/planning-artifacts/epics.md#FR70]
- [Source: _bmad-output/planning-artifacts/epics.md#FR71]
- [Source: src/yolo_developer/agents/sm/health_types.py - HealthStatus]
- [Source: src/yolo_developer/agents/sm/health.py - monitor_health]
- [Source: src/yolo_developer/agents/sm/circular_detection_types.py - CycleAnalysis]
- [Source: src/yolo_developer/agents/sm/types.py - SMOutput]
- [Source: src/yolo_developer/orchestrator/state.py - YoloState]
- [Source: _bmad-output/implementation-artifacts/10-9-sprint-progress-tracking.md - pattern reference]
- [Source: _bmad-output/implementation-artifacts/10-5-health-monitoring.md - health detection pattern]

---

## Senior Developer Review

### Implementation Summary

Story 10.10 Emergency Protocols has been fully implemented with all tasks complete. The implementation provides a robust emergency protocol system that detects, checkpoints, evaluates recovery options, and escalates when needed.

### Files Created

1. **`src/yolo_developer/agents/sm/emergency_types.py`** (381 lines)
   - EmergencyType, ProtocolStatus, RecoveryAction Literal types
   - EmergencyTrigger, Checkpoint, RecoveryOption, EmergencyProtocol, EmergencyConfig frozen dataclasses
   - Constants: DEFAULT_MAX_RECOVERY_ATTEMPTS, DEFAULT_ESCALATION_THRESHOLD, VALID_* frozensets

2. **`src/yolo_developer/agents/sm/emergency.py`** (900 lines)
   - Detection functions: `_check_health_degradation()`, `_check_circular_logic()`, `_check_gate_blocked()`, `_check_agent_stuck()`, `_check_system_error()`, `_detect_emergency()`, `_create_emergency_trigger()`
   - Checkpointing: `_capture_state_snapshot()`, `_create_checkpoint()`, `_store_checkpoint()`, `_retrieve_checkpoint()`, `checkpoint_state()`, `_clear_checkpoint_store()`
   - Recovery evaluation: `_evaluate_retry_option()`, `_evaluate_rollback_option()`, `_evaluate_skip_option()`, `_evaluate_escalate_option()`, `_evaluate_terminate_option()`, `_generate_recovery_options()`, `_select_best_recovery()`
   - Escalation: `_should_escalate()`, `escalate_emergency()`
   - Main entry: `trigger_emergency_protocol()`

3. **`tests/unit/agents/sm/test_emergency_types.py`** (28 tests)
   - Tests all type definitions, constants, and serialization

4. **`tests/unit/agents/sm/test_emergency.py`** (48 tests)
   - Tests detection, checkpointing, recovery evaluation, escalation, and main protocol flow

### Files Modified

1. **`src/yolo_developer/agents/sm/types.py`**
   - Added `emergency_protocol: dict[str, Any] | None = None` field to SMOutput
   - Updated `to_dict()` method

2. **`src/yolo_developer/agents/sm/node.py`**
   - Added imports for emergency module
   - Added Step 6b2: Emergency protocol trigger when health status is critical
   - Added emergency_protocol to processing notes, SMOutput, logger, and return dict

3. **`src/yolo_developer/agents/sm/__init__.py`**
   - Added emergency module exports (17 types, 3 functions, 5 constants)
   - Updated docstring with FR17, FR70, FR71 references

### Test Results

- **76 emergency tests passing** (28 types + 48 emergency)
- **692 total SM module tests passing**
- **mypy**: No issues
- **ruff**: All checks passed

### Architecture Compliance

- ✅ Frozen dataclasses for immutable types (ADR-001)
- ✅ Async-first with async/await (ADR-005)
- ✅ TypedDict state with dict return (ADR-005)
- ✅ Structlog for structured logging
- ✅ In-memory checkpoint storage (MVP implementation)
- ✅ Confidence-based recovery selection
- ✅ Threshold-based escalation

### Key Design Decisions

1. **Emergency Detection Hierarchy**: system_error → health_degraded → circular_logic → gate_blocked → agent_stuck (most critical first)

2. **Recovery Confidence Scoring**: Each recovery option has a confidence score (0.0-1.0); highest confidence option is auto-selected if above escalation threshold

3. **Checkpoint Storage**: In-memory storage for MVP; ready for persistent storage enhancement

4. **Emergency Protocol Status Flow**: pending → active → checkpointed → recovering → resolved/escalated

### Functional Requirements Implemented

- **FR17**: SM Agent can trigger emergency protocols when system health degrades
- **FR70**: SM Agent can escalate to human when circular logic persists
- **FR71**: SM Agent can coordinate rollback operations as emergency sprints

### Ready for Code Review

All acceptance criteria are met, all tests pass, and the implementation follows established patterns from previous stories.
