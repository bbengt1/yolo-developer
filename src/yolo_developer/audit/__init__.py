"""Audit trail, decision logging, and requirement traceability module.

This module provides functionality for:
- Logging and tracking agent decisions (FR81: Story 11.1)
- Tracing requirements to code for full traceability (FR82: Story 11.2)

Components:
    Decision Logging (Story 11.1):
        - Type definitions: Decision, AgentIdentity, DecisionContext, etc.
        - Store protocol: DecisionStore for pluggable storage backends
        - In-memory store: InMemoryDecisionStore for testing and single-session use
        - Logger: DecisionLogger for high-level decision logging with auto-generated IDs

    Requirement Traceability (Story 11.2):
        - Traceability types: TraceableArtifact, TraceLink, ArtifactType, LinkType
        - Store protocol: TraceabilityStore for pluggable traceability backends
        - In-memory store: InMemoryTraceabilityStore for testing
        - Service: TraceabilityService for creating and navigating trace chains

Example (Decision Logging):
    >>> from yolo_developer.audit import (
    ...     DecisionLogger,
    ...     DecisionContext,
    ...     InMemoryDecisionStore,
    ...     get_logger,
    ... )
    >>>
    >>> # Create store and logger
    >>> store = InMemoryDecisionStore()
    >>> logger = get_logger(store)
    >>>
    >>> # Log a decision
    >>> decision_id = await logger.log(
    ...     agent_name="analyst",
    ...     agent_type="analyst",
    ...     decision_type="requirement_analysis",
    ...     content="OAuth2 authentication required",
    ...     rationale="Industry standard security",
    ...     context=DecisionContext(sprint_id="sprint-1"),
    ... )
    >>>
    >>> # Retrieve decisions
    >>> decisions = await store.get_decisions()

Example (Requirement Traceability):
    >>> from yolo_developer.audit import (
    ...     TraceabilityService,
    ...     InMemoryTraceabilityStore,
    ...     get_traceability_service,
    ... )
    >>>
    >>> # Create store and service
    >>> store = InMemoryTraceabilityStore()
    >>> service = get_traceability_service(store)
    >>>
    >>> # Create trace chain
    >>> await service.trace_requirement("FR82", "Traceability", "Description")
    >>> await service.trace_story("story-1", "FR82", "Story Name", "Description")
    >>> await service.trace_code("code-1", "design-1", "Code Name", "Description")
    >>>
    >>> # Navigate trace chain
    >>> requirement = await service.get_requirement_for_code("code-1")

References:
    - FR81: System can log all agent decisions with rationale
    - FR82: System can generate decision traceability from requirement to code
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

# Logger
from yolo_developer.audit.logger import (
    DecisionLogger,
    get_logger,
)

# In-memory implementation
from yolo_developer.audit.memory_store import InMemoryDecisionStore

# Store protocol and filters
from yolo_developer.audit.store import (
    DecisionFilters,
    DecisionStore,
)

# Traceability service
from yolo_developer.audit.traceability import (
    TraceabilityService,
    get_traceability_service,
)

# Traceability in-memory implementation
from yolo_developer.audit.traceability_memory_store import InMemoryTraceabilityStore

# Traceability store protocol
from yolo_developer.audit.traceability_store import TraceabilityStore

# Traceability types
from yolo_developer.audit.traceability_types import (
    VALID_ARTIFACT_TYPES,
    VALID_LINK_TYPES,
    ArtifactType,
    LinkType,
    TraceableArtifact,
    TraceLink,
)

# Types
from yolo_developer.audit.types import (
    VALID_DECISION_SEVERITIES,
    VALID_DECISION_TYPES,
    AgentIdentity,
    Decision,
    DecisionContext,
    DecisionSeverity,
    DecisionType,
)

__all__ = [
    "VALID_ARTIFACT_TYPES",
    "VALID_DECISION_SEVERITIES",
    "VALID_DECISION_TYPES",
    "VALID_LINK_TYPES",
    "AgentIdentity",
    "ArtifactType",
    "Decision",
    "DecisionContext",
    "DecisionFilters",
    "DecisionLogger",
    "DecisionSeverity",
    "DecisionStore",
    "DecisionType",
    "InMemoryDecisionStore",
    "InMemoryTraceabilityStore",
    "LinkType",
    "TraceLink",
    "TraceabilityService",
    "TraceabilityStore",
    "TraceableArtifact",
    "get_logger",
    "get_traceability_service",
]
