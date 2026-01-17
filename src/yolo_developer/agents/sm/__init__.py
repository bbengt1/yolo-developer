"""SM (Scrum Master) agent module for orchestration control plane.

The SM agent serves as the control plane for orchestration decisions,
providing centralized routing logic for the multi-agent workflow.

Key Responsibilities:
- Routing decisions: Determines next agent based on state analysis (Story 10.2)
- Circular logic detection: Detects agent ping-pong patterns (>3 exchanges per FR12)
- Enhanced circular detection: Topic-aware, multi-agent cycle detection (Story 10.6)
- Escalation handling: Triggers human intervention when needed
- Gate-blocked recovery: Routes to appropriate agent for recovery
- Sprint planning: Plan sprints by prioritizing and sequencing stories (Story 10.3)
- Task delegation: Delegate tasks to appropriate specialized agents (Story 10.4)
- Health monitoring: Monitor agent activity, idle time, cycle time, churn rate (Story 10.5)
- Conflict mediation: Mediate conflicts between agents with different recommendations (Story 10.7)
- Handoff management: Manage agent handoffs with context preservation (Story 10.8)
- Sprint progress tracking: Track sprint progress and completion estimates (Story 10.9)
- Emergency protocols: Trigger emergency protocols when health degrades (Story 10.10)
- Priority scoring: Calculate weighted priority scores for story selection (Story 10.11)
- Velocity tracking: Track burn-down velocity and cycle time metrics (Story 10.12)
- Context injection: Inject context when agents lack information (Story 10.13)
- Human escalation: Create escalation requests for human intervention (Story 10.14)

Example:
    >>> from yolo_developer.agents.sm import (
    ...     sm_node,
    ...     SMOutput,
    ...     AgentExchange,
    ...     plan_sprint,
    ...     SprintPlan,
    ...     delegate_task,
    ...     DelegationResult,
    ...     monitor_health,
    ...     HealthStatus,
    ...     mediate_conflicts,
    ...     MediationResult,
    ...     manage_handoff,
    ...     HandoffResult,
    ...     trigger_emergency_protocol,
    ...     EmergencyProtocol,
    ...     calculate_priority_score,
    ...     score_stories,
    ...     PriorityFactors,
    ...     PriorityResult,
    ... )
    >>>
    >>> # Run the SM node
    >>> result = await sm_node(state)
    >>> result["sm_output"]["routing_decision"]
    'pm'
    >>>
    >>> # Plan a sprint (Story 10.3)
    >>> stories = [{"story_id": "1-1", "title": "Setup"}]
    >>> plan = await plan_sprint(stories)
    >>> plan.sprint_id
    'sprint-20260112'
    >>>
    >>> # Delegate a task (Story 10.4)
    >>> result = await delegate_task(state, "implementation", "Implement feature")
    >>> result.request.target_agent
    'dev'
    >>>
    >>> # Monitor health (Story 10.5)
    >>> status = await monitor_health(state)
    >>> status.is_healthy
    True
    >>>
    >>> # Mediate conflicts (Story 10.7)
    >>> mediation = await mediate_conflicts(state)
    >>> mediation.success
    True
    >>>
    >>> # Manage handoffs (Story 10.8)
    >>> result = await manage_handoff(state, "analyst", "pm")
    >>> result.success
    True
    >>>
    >>> # Track progress (Story 10.9)
    >>> progress = await track_progress(state, sprint_plan)
    >>> progress.snapshot.progress_percentage
    50.0
    >>>
    >>> # Trigger emergency protocol (Story 10.10)
    >>> protocol = await trigger_emergency_protocol(state, health_status)
    >>> protocol.status
    'resolved'
    >>>
    >>> # Calculate priority scores (Story 10.11)
    >>> factors = PriorityFactors(story_id="1-1", value_score=0.9)
    >>> result = calculate_priority_score(factors, PriorityScoringConfig())
    >>> result.priority_score  # 0.9*0.4 + 0.0*0.3 + 0.5*0.2 + 0.0*0.1 = 0.46
    0.46
    >>>
    >>> # Track velocity (Story 10.12)
    >>> velocity = calculate_sprint_velocity(completed_stories, "sprint-20260116")
    >>> velocity.stories_completed
    5
    >>> metrics = calculate_velocity_metrics([velocity])
    >>> metrics.trend
    'stable'

Architecture:
    The sm_node function is a LangGraph node that:
    - Receives YoloState TypedDict as input
    - Returns state update dict (not full state)
    - Never mutates input state
    - Uses async/await for all I/O
    - Integrates with sm_routing gate (non-blocking)

References:
    - ADR-005: Inter-Agent Communication
    - ADR-007: Error Handling Strategy
    - FR9: SM Agent can plan sprints by prioritizing and sequencing stories
    - FR10: Task delegation
    - FR11: Health monitoring
    - FR12: Circular logic detection (>3 exchanges)
    - FR13: Conflict mediation
    - FR14: System can execute agents in defined sequence
    - FR15: System can handle agent handoffs with context preservation
    - FR16: System can track sprint progress and completion status
    - FR17: SM Agent can trigger emergency protocols when system health degrades
    - FR65: SM Agent can calculate weighted priority scores for story selection
    - FR66: SM Agent can track burn-down velocity and cycle time metrics
    - FR67: SM Agent can detect agent churn rate and idle time
    - FR70: SM Agent can escalate to human when circular logic persists
    - FR71: SM Agent can coordinate rollback operations as emergency sprints
"""

from __future__ import annotations

from yolo_developer.agents.sm.circular_detection import detect_circular_logic
from yolo_developer.agents.sm.circular_detection_types import (
    DEFAULT_EXCHANGE_THRESHOLD,
    DEFAULT_TIME_WINDOW_SECONDS,
    VALID_CYCLE_SEVERITIES,
    VALID_INTERVENTION_STRATEGIES,
    VALID_PATTERN_TYPES,
    CircularLogicConfig,
    CircularPattern,
    CycleAnalysis,
    CycleLog,
    CycleSeverity,
    InterventionStrategy,
    PatternType,
)
from yolo_developer.agents.sm.conflict_mediation import mediate_conflicts
from yolo_developer.agents.sm.conflict_types import (
    DEFAULT_PRINCIPLES_HIERARCHY,
    RESOLUTION_PRINCIPLES,
    VALID_CONFLICT_SEVERITIES,
    VALID_CONFLICT_TYPES,
    VALID_RESOLUTION_STRATEGIES,
    Conflict,
    ConflictMediationConfig,
    ConflictParty,
    ConflictResolution,
    ConflictSeverity,
    ConflictType,
    MediationResult,
    ResolutionStrategy,
)
from yolo_developer.agents.sm.context_injection import (
    detect_context_gap,
    inject_context,
    manage_context_injection,
    retrieve_relevant_context,
)
from yolo_developer.agents.sm.context_injection_types import (
    DEFAULT_LOG_INJECTIONS,
    DEFAULT_MAX_CONTEXT_ITEMS,
    DEFAULT_MAX_CONTEXT_SIZE_BYTES,
    DEFAULT_MIN_RELEVANCE_SCORE,
    LONG_CYCLE_TIME_MULTIPLIER,
    MAX_CONFIDENCE,
    MAX_RELEVANCE,
    MIN_CONFIDENCE,
    MIN_RELEVANCE,
    VALID_CONTEXT_SOURCES,
    VALID_GAP_REASONS,
    ContextGap,
    ContextSource,
    GapReason,
    InjectionConfig,
    InjectionResult,
    RetrievedContext,
)
from yolo_developer.agents.sm.delegation import (
    delegate_task,
    routing_to_task_type,
)
from yolo_developer.agents.sm.delegation_types import (
    AGENT_EXPERTISE,
    DEFAULT_ACKNOWLEDGMENT_TIMEOUT_SECONDS,
    DEFAULT_MAX_RETRY_ATTEMPTS,
    TASK_TO_AGENT,
    VALID_TASK_TYPES,
    DelegationConfig,
    DelegationRequest,
    DelegationResult,
    Priority,
    TaskType,
)
from yolo_developer.agents.sm.emergency import (
    checkpoint_state,
    escalate_emergency,
    trigger_emergency_protocol,
)
from yolo_developer.agents.sm.emergency_types import (
    DEFAULT_ESCALATION_THRESHOLD,
    DEFAULT_MAX_RECOVERY_ATTEMPTS,
    VALID_EMERGENCY_TYPES,
    VALID_PROTOCOL_STATUSES,
    VALID_RECOVERY_ACTIONS,
    Checkpoint,
    EmergencyConfig,
    EmergencyProtocol,
    EmergencyTrigger,
    EmergencyType,
    ProtocolStatus,
    RecoveryAction,
    RecoveryOption,
)
from yolo_developer.agents.sm.handoff import manage_handoff
from yolo_developer.agents.sm.handoff_types import (
    DEFAULT_MAX_CONTEXT_SIZE,
    DEFAULT_TIMEOUT_SECONDS,
    VALID_HANDOFF_STATUSES,
    HandoffConfig,
    HandoffMetrics,
    HandoffRecord,
    HandoffResult,
    HandoffStatus,
)
from yolo_developer.agents.sm.health import monitor_health
from yolo_developer.agents.sm.health_types import (
    DEFAULT_MAX_CHURN_RATE,
    DEFAULT_MAX_CYCLE_TIME_SECONDS,
    DEFAULT_MAX_IDLE_TIME_SECONDS,
    DEFAULT_WARNING_THRESHOLD_RATIO,
    VALID_AGENTS_FOR_HEALTH,
    VALID_ALERT_SEVERITIES,
    VALID_HEALTH_SEVERITIES,
    AgentHealthSnapshot,
    AlertSeverity,
    HealthAlert,
    HealthConfig,
    HealthMetrics,
    HealthSeverity,
    HealthStatus,
)
from yolo_developer.agents.sm.human_escalation import (
    create_escalation_request,
    handle_escalation_timeout,
    integrate_escalation_response,
    manage_human_escalation,
    should_escalate,
)
from yolo_developer.agents.sm.human_escalation_types import (
    DEFAULT_ESCALATION_TIMEOUT_SECONDS,
    DEFAULT_LOG_ESCALATIONS,
    DEFAULT_MAX_PENDING,
    MAX_DURATION_MS,
    MIN_DURATION_MS,
    VALID_ESCALATION_STATUSES,
    VALID_ESCALATION_TRIGGERS,
    EscalationConfig,
    EscalationOption,
    EscalationRequest,
    EscalationResponse,
    EscalationResult,
    EscalationStatus,
    EscalationTrigger,
)
from yolo_developer.agents.sm.node import sm_node
from yolo_developer.agents.sm.planning import (
    CircularDependencyError,
    plan_sprint,
)
from yolo_developer.agents.sm.planning_types import (
    DEFAULT_DEPENDENCY_WEIGHT,
    DEFAULT_MAX_POINTS,
    DEFAULT_MAX_STORIES,
    DEFAULT_TECH_DEBT_WEIGHT,
    DEFAULT_VALUE_WEIGHT,
    DEFAULT_VELOCITY_WEIGHT,
    PlanningConfig,
    SprintPlan,
    SprintStory,
)
from yolo_developer.agents.sm.priority import (
    calculate_dependency_score,
    calculate_dependency_scores,
    calculate_priority_score,
    normalize_results,
    normalize_scores,
    score_stories,
    update_stories_with_scores,
)
from yolo_developer.agents.sm.priority_types import (
    DEFAULT_INCLUDE_EXPLANATION,
    DEFAULT_MIN_SCORE_THRESHOLD,
    DEFAULT_NORMALIZE_SCORES,
    MAX_SCORE,
    MIN_SCORE,
    PriorityFactors,
    PriorityResult,
    PriorityScoringConfig,
)
from yolo_developer.agents.sm.progress import (
    get_progress_for_display,
    get_progress_summary,
    get_stories_by_status,
    track_progress,
)
from yolo_developer.agents.sm.progress_types import (
    DEFAULT_ESTIMATE_CONFIDENCE_THRESHOLD,
    VALID_STORY_STATUSES,
    CompletionEstimate,
    ProgressConfig,
    SprintProgress,
    SprintProgressSnapshot,
    StoryProgress,
    StoryStatus,
)
from yolo_developer.agents.sm.types import (
    CIRCULAR_LOGIC_THRESHOLD,
    NATURAL_SUCCESSOR,
    VALID_AGENTS,
    AgentExchange,
    EscalationReason,
    RoutingDecision,
    SMOutput,
)
from yolo_developer.agents.sm.velocity import (
    calculate_sprint_velocity,
    calculate_velocity_metrics,
    forecast_velocity,
    get_velocity_trend,
    track_sprint_velocity,
)
from yolo_developer.agents.sm.velocity_types import (
    CONFIDENCE_DECIMAL_PLACES,
    DEFAULT_MIN_SPRINTS_FOR_FORECAST,
    DEFAULT_MIN_SPRINTS_FOR_TREND,
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_TREND_THRESHOLD,
    VALID_TRENDS,
    SprintVelocity,
    VelocityConfig,
    VelocityForecast,
    VelocityMetrics,
    VelocityTrend,
)

__all__ = [
    # Delegation (Story 10.4)
    "AGENT_EXPERTISE",
    # Core Types (Story 10.2)
    "CIRCULAR_LOGIC_THRESHOLD",
    # Velocity Tracking (Story 10.12)
    "CONFIDENCE_DECIMAL_PLACES",
    "DEFAULT_ACKNOWLEDGMENT_TIMEOUT_SECONDS",
    # Planning (Story 10.3)
    "DEFAULT_DEPENDENCY_WEIGHT",
    # Emergency Protocols (Story 10.10)
    "DEFAULT_ESCALATION_THRESHOLD",
    # Human Escalation (Story 10.14)
    "DEFAULT_ESCALATION_TIMEOUT_SECONDS",
    # Progress Tracking (Story 10.9)
    "DEFAULT_ESTIMATE_CONFIDENCE_THRESHOLD",
    # Circular Detection (Story 10.6)
    "DEFAULT_EXCHANGE_THRESHOLD",
    # Priority Scoring (Story 10.11)
    "DEFAULT_INCLUDE_EXPLANATION",
    # Context Injection (Story 10.13)
    "DEFAULT_LOG_ESCALATIONS",
    "DEFAULT_LOG_INJECTIONS",
    # Health Monitoring (Story 10.5)
    "DEFAULT_MAX_CHURN_RATE",
    "DEFAULT_MAX_CONTEXT_ITEMS",
    "DEFAULT_MAX_CONTEXT_SIZE",
    "DEFAULT_MAX_CONTEXT_SIZE_BYTES",
    "DEFAULT_MAX_CYCLE_TIME_SECONDS",
    "DEFAULT_MAX_IDLE_TIME_SECONDS",
    "DEFAULT_MAX_PENDING",
    "DEFAULT_MAX_POINTS",
    "DEFAULT_MAX_RECOVERY_ATTEMPTS",
    "DEFAULT_MAX_RETRY_ATTEMPTS",
    "DEFAULT_MAX_STORIES",
    "DEFAULT_MIN_RELEVANCE_SCORE",
    "DEFAULT_MIN_SCORE_THRESHOLD",
    "DEFAULT_MIN_SPRINTS_FOR_FORECAST",
    "DEFAULT_MIN_SPRINTS_FOR_TREND",
    "DEFAULT_NORMALIZE_SCORES",
    # Conflict Mediation (Story 10.7)
    "DEFAULT_PRINCIPLES_HIERARCHY",
    "DEFAULT_ROLLING_WINDOW",
    "DEFAULT_TECH_DEBT_WEIGHT",
    # Handoff Management (Story 10.8)
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_TIME_WINDOW_SECONDS",
    "DEFAULT_TREND_THRESHOLD",
    "DEFAULT_VALUE_WEIGHT",
    "DEFAULT_VELOCITY_WEIGHT",
    "DEFAULT_WARNING_THRESHOLD_RATIO",
    "LONG_CYCLE_TIME_MULTIPLIER",
    "MAX_CONFIDENCE",
    "MAX_DURATION_MS",
    "MAX_RELEVANCE",
    "MAX_SCORE",
    "MIN_CONFIDENCE",
    "MIN_DURATION_MS",
    "MIN_RELEVANCE",
    "MIN_SCORE",
    "NATURAL_SUCCESSOR",
    "RESOLUTION_PRINCIPLES",
    "TASK_TO_AGENT",
    "VALID_AGENTS",
    "VALID_AGENTS_FOR_HEALTH",
    "VALID_ALERT_SEVERITIES",
    "VALID_CONFLICT_SEVERITIES",
    "VALID_CONFLICT_TYPES",
    "VALID_CONTEXT_SOURCES",
    "VALID_CYCLE_SEVERITIES",
    "VALID_EMERGENCY_TYPES",
    "VALID_ESCALATION_STATUSES",
    "VALID_ESCALATION_TRIGGERS",
    "VALID_GAP_REASONS",
    "VALID_HANDOFF_STATUSES",
    "VALID_HEALTH_SEVERITIES",
    "VALID_INTERVENTION_STRATEGIES",
    "VALID_PATTERN_TYPES",
    "VALID_PROTOCOL_STATUSES",
    "VALID_RECOVERY_ACTIONS",
    "VALID_RESOLUTION_STRATEGIES",
    "VALID_STORY_STATUSES",
    "VALID_TASK_TYPES",
    "VALID_TRENDS",
    "AgentExchange",
    "AgentHealthSnapshot",
    "AlertSeverity",
    "Checkpoint",
    "CircularDependencyError",
    "CircularLogicConfig",
    "CircularPattern",
    "CompletionEstimate",
    "Conflict",
    "ConflictMediationConfig",
    "ConflictParty",
    "ConflictResolution",
    "ConflictSeverity",
    "ConflictType",
    "ContextGap",
    "ContextSource",
    "CycleAnalysis",
    "CycleLog",
    "CycleSeverity",
    "DelegationConfig",
    "DelegationRequest",
    "DelegationResult",
    "EmergencyConfig",
    "EmergencyProtocol",
    "EmergencyTrigger",
    "EmergencyType",
    "EscalationConfig",
    "EscalationOption",
    "EscalationReason",
    "EscalationRequest",
    "EscalationResponse",
    "EscalationResult",
    "EscalationStatus",
    "EscalationTrigger",
    "GapReason",
    "HandoffConfig",
    "HandoffMetrics",
    "HandoffRecord",
    "HandoffResult",
    "HandoffStatus",
    "HealthAlert",
    "HealthConfig",
    "HealthMetrics",
    "HealthSeverity",
    "HealthStatus",
    "InjectionConfig",
    "InjectionResult",
    "InterventionStrategy",
    "MediationResult",
    "PatternType",
    "PlanningConfig",
    "Priority",
    "PriorityFactors",
    "PriorityResult",
    "PriorityScoringConfig",
    "ProgressConfig",
    "ProtocolStatus",
    "RecoveryAction",
    "RecoveryOption",
    "ResolutionStrategy",
    "RetrievedContext",
    "RoutingDecision",
    "SMOutput",
    "SprintPlan",
    "SprintProgress",
    "SprintProgressSnapshot",
    "SprintStory",
    "SprintVelocity",
    "StoryProgress",
    "StoryStatus",
    "TaskType",
    "VelocityConfig",
    "VelocityForecast",
    "VelocityMetrics",
    "VelocityTrend",
    "calculate_dependency_score",
    "calculate_dependency_scores",
    "calculate_priority_score",
    "calculate_sprint_velocity",
    "calculate_velocity_metrics",
    "checkpoint_state",
    "create_escalation_request",
    "delegate_task",
    "detect_circular_logic",
    "detect_context_gap",
    "escalate_emergency",
    "forecast_velocity",
    "get_progress_for_display",
    "get_progress_summary",
    "get_stories_by_status",
    "get_velocity_trend",
    "handle_escalation_timeout",
    "inject_context",
    "integrate_escalation_response",
    "manage_context_injection",
    "manage_handoff",
    "manage_human_escalation",
    "mediate_conflicts",
    "monitor_health",
    "normalize_results",
    "normalize_scores",
    "plan_sprint",
    "retrieve_relevant_context",
    "routing_to_task_type",
    "score_stories",
    "should_escalate",
    "sm_node",
    "track_progress",
    "track_sprint_velocity",
    "trigger_emergency_protocol",
    "update_stories_with_scores",
]
