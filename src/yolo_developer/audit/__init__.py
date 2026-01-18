"""Audit trail, decision logging, and requirement traceability module.

This module provides functionality for:
- Logging and tracking agent decisions (FR81: Story 11.1)
- Tracing requirements to code for full traceability (FR82: Story 11.2)
- Viewing audit trail in human-readable format (FR83: Story 11.3)

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

    Human-Readable View (Story 11.3):
        - Formatter types: FormatterStyle, ColorScheme, FormatOptions
        - Formatter protocol: AuditFormatter for pluggable output formats
        - Rich formatter: RichAuditFormatter for terminal output with colors
        - Plain formatter: PlainAuditFormatter for ASCII text output
        - View service: AuditViewService for viewing decisions and traces

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

Example (Human-Readable View):
    >>> from yolo_developer.audit import (
    ...     AuditViewService,
    ...     InMemoryDecisionStore,
    ...     InMemoryTraceabilityStore,
    ...     get_audit_view_service,
    ... )
    >>>
    >>> # Create stores and view service
    >>> decision_store = InMemoryDecisionStore()
    >>> traceability_store = InMemoryTraceabilityStore()
    >>> service = get_audit_view_service(decision_store, traceability_store)
    >>>
    >>> # View decisions
    >>> output = await service.view_decisions()
    >>> print(output)

References:
    - FR81: System can log all agent decisions with rationale
    - FR82: System can generate decision traceability from requirement to code
    - FR83: Users can view audit trail in human-readable format
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

# Formatter protocol
from yolo_developer.audit.formatter_protocol import AuditFormatter

# Formatter types
from yolo_developer.audit.formatter_types import (
    DEFAULT_COLOR_SCHEME,
    DEFAULT_FORMAT_OPTIONS,
    VALID_FORMATTER_STYLES,
    ColorScheme,
    FormatOptions,
    FormatterStyle,
)

# Logger
from yolo_developer.audit.logger import (
    DecisionLogger,
    get_logger,
)

# In-memory implementation
from yolo_developer.audit.memory_store import InMemoryDecisionStore

# Plain text formatter
from yolo_developer.audit.plain_formatter import PlainAuditFormatter

# Rich terminal formatter
from yolo_developer.audit.rich_formatter import RichAuditFormatter

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

# View service
from yolo_developer.audit.view import (
    AuditViewService,
    get_audit_view_service,
)

__all__ = [
    # Constants
    "DEFAULT_COLOR_SCHEME",
    "DEFAULT_FORMAT_OPTIONS",
    "VALID_ARTIFACT_TYPES",
    "VALID_DECISION_SEVERITIES",
    "VALID_DECISION_TYPES",
    "VALID_FORMATTER_STYLES",
    "VALID_LINK_TYPES",
    # Types
    "AgentIdentity",
    "ArtifactType",
    "AuditFormatter",
    # Services and Stores
    "AuditViewService",
    "ColorScheme",
    "Decision",
    "DecisionContext",
    "DecisionFilters",
    "DecisionLogger",
    "DecisionSeverity",
    "DecisionStore",
    "DecisionType",
    "FormatOptions",
    "FormatterStyle",
    "InMemoryDecisionStore",
    "InMemoryTraceabilityStore",
    "LinkType",
    "PlainAuditFormatter",
    "RichAuditFormatter",
    "TraceLink",
    "TraceabilityService",
    "TraceabilityStore",
    "TraceableArtifact",
    # Factory functions
    "get_audit_view_service",
    "get_logger",
    "get_traceability_service",
]
