"""Audit trail, decision logging, and requirement traceability module.

This module provides functionality for:
- Logging and tracking agent decisions (FR81: Story 11.1)
- Tracing requirements to code for full traceability (FR82: Story 11.2)
- Viewing audit trail in human-readable format (FR83: Story 11.3)
- Exporting audit trail for compliance reporting (FR84: Story 11.4)
- Correlating decisions across agent boundaries (FR85: Story 11.5)
- Tracking token usage and costs per operation (FR86: Story 11.6)

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

    Audit Export (Story 11.4):
        - Export types: ExportFormat, ExportOptions, RedactionConfig
        - Exporter protocol: AuditExporter for pluggable export formats
        - JSON exporter: JsonAuditExporter for machine-readable output
        - CSV exporter: CsvAuditExporter for spreadsheet-compatible output
        - PDF exporter: PdfAuditExporter for human-readable reports
        - Export service: AuditExportService for high-level export operations

    Cross-Agent Correlation (Story 11.5):
        - Correlation types: DecisionChain, CausalRelation, AgentTransition, TimelineEntry
        - Store protocol: CorrelationStore for pluggable correlation backends
        - In-memory store: InMemoryCorrelationStore for testing
        - Service: CorrelationService for correlating decisions across agents

    Token/Cost Tracking (Story 11.6):
        - Cost types: TokenUsage, CostRecord, CostAggregation
        - Store protocol: CostStore for pluggable cost storage backends
        - In-memory store: InMemoryCostStore for testing and single-session use
        - Service: CostTrackingService for recording and querying LLM costs
        - LiteLLM utilities: extract_token_usage, extract_cost, calculate_cost_if_missing

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

Example (Audit Export):
    >>> from yolo_developer.audit import (
    ...     AuditExportService,
    ...     InMemoryDecisionStore,
    ...     InMemoryTraceabilityStore,
    ...     get_audit_export_service,
    ... )
    >>>
    >>> # Create stores and export service
    >>> decision_store = InMemoryDecisionStore()
    >>> traceability_store = InMemoryTraceabilityStore()
    >>> service = get_audit_export_service(decision_store, traceability_store)
    >>>
    >>> # Export as JSON
    >>> json_bytes = await service.export(format="json")
    >>>
    >>> # Export to file with redaction
    >>> from yolo_developer.audit import ExportOptions, RedactionConfig
    >>> options = ExportOptions(redaction_config=RedactionConfig(redact_metadata=True))
    >>> await service.export_to_file("/path/to/audit.pdf", options=options)

Example (Cross-Agent Correlation):
    >>> from yolo_developer.audit import (
    ...     CorrelationService,
    ...     InMemoryDecisionStore,
    ...     InMemoryCorrelationStore,
    ...     get_correlation_service,
    ... )
    >>>
    >>> # Create stores and correlation service
    >>> decision_store = InMemoryDecisionStore()
    >>> correlation_store = InMemoryCorrelationStore()
    >>> service = get_correlation_service(decision_store, correlation_store)
    >>>
    >>> # Correlate decisions
    >>> chain = await service.correlate_decisions(["dec-001", "dec-002"])
    >>>
    >>> # Record agent transition
    >>> await service.record_transition("analyst", "pm", "dec-002")
    >>>
    >>> # Get timeline view
    >>> timeline = await service.get_timeline_view(session_id="session-123")

Example (Token/Cost Tracking):
    >>> from yolo_developer.audit import (
    ...     CostTrackingService,
    ...     InMemoryCostStore,
    ...     get_cost_tracking_service,
    ...     extract_token_usage,
    ...     extract_cost,
    ... )
    >>>
    >>> # Create store and service
    >>> cost_store = InMemoryCostStore()
    >>> service = get_cost_tracking_service(cost_store)
    >>>
    >>> # Record an LLM call
    >>> record = await service.record_llm_call(
    ...     model="gpt-4o-mini",
    ...     tier="routine",
    ...     prompt_tokens=100,
    ...     completion_tokens=50,
    ...     cost_usd=0.0015,
    ...     agent_name="analyst",
    ...     session_id="session-123",
    ... )
    >>>
    >>> # Get cost breakdown by agent
    >>> agent_costs = await service.get_agent_costs()

References:
    - FR81: System can log all agent decisions with rationale
    - FR82: System can generate decision traceability from requirement to code
    - FR83: Users can view audit trail in human-readable format
    - FR84: System can export audit trail for compliance reporting
    - FR85: System can correlate decisions across agent boundaries
    - FR86: System can track token usage and cost per operation
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

# Correlation service
from yolo_developer.audit.correlation import (
    CorrelationService,
    get_correlation_service,
)

# Correlation in-memory implementation
from yolo_developer.audit.correlation_memory_store import InMemoryCorrelationStore

# Correlation store protocol
from yolo_developer.audit.correlation_store import (
    CorrelationFilters,
    CorrelationStore,
)

# Correlation types
from yolo_developer.audit.correlation_types import (
    VALID_CORRELATION_TYPES,
    AgentTransition,
    CausalRelation,
    CorrelationType,
    DecisionChain,
    TimelineEntry,
)

# Cost tracking in-memory store
from yolo_developer.audit.cost_memory_store import InMemoryCostStore

# Cost tracking service
from yolo_developer.audit.cost_service import (
    CostTrackingService,
    get_cost_tracking_service,
)

# Cost store protocol
from yolo_developer.audit.cost_store import (
    CostFilters,
    CostStore,
)

# Cost tracking types
from yolo_developer.audit.cost_types import (
    VALID_GROUPBY_VALUES,
    VALID_TIER_VALUES,
    CostAggregation,
    CostGroupBy,
    CostRecord,
    TokenUsage,
)

# Cost tracking utilities
from yolo_developer.audit.cost_utils import (
    calculate_cost_if_missing,
    extract_cost,
    extract_token_usage,
)

# CSV exporter
from yolo_developer.audit.csv_exporter import CsvAuditExporter

# Export service
from yolo_developer.audit.export import (
    AuditExportService,
    get_audit_export_service,
)

# Export protocol
from yolo_developer.audit.export_protocol import AuditExporter

# Export types
from yolo_developer.audit.export_types import (
    DEFAULT_EXPORT_OPTIONS,
    DEFAULT_REDACTION_CONFIG,
    VALID_EXPORT_FORMATS,
    ExportFormat,
    ExportOptions,
    RedactionConfig,
)

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

# JSON exporter
from yolo_developer.audit.json_exporter import JsonAuditExporter

# Logger
from yolo_developer.audit.logger import (
    DecisionLogger,
    get_logger,
)

# In-memory implementation
from yolo_developer.audit.memory_store import InMemoryDecisionStore

# PDF exporter
from yolo_developer.audit.pdf_exporter import PdfAuditExporter

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
    "DEFAULT_EXPORT_OPTIONS",
    "DEFAULT_FORMAT_OPTIONS",
    "DEFAULT_REDACTION_CONFIG",
    "VALID_ARTIFACT_TYPES",
    "VALID_CORRELATION_TYPES",
    "VALID_DECISION_SEVERITIES",
    "VALID_DECISION_TYPES",
    "VALID_EXPORT_FORMATS",
    "VALID_FORMATTER_STYLES",
    "VALID_GROUPBY_VALUES",
    "VALID_LINK_TYPES",
    "VALID_TIER_VALUES",
    # Types
    "AgentIdentity",
    "AgentTransition",
    "ArtifactType",
    # Services and Stores
    "AuditExportService",
    "AuditExporter",
    "AuditFormatter",
    "AuditViewService",
    "CausalRelation",
    "ColorScheme",
    "CorrelationFilters",
    "CorrelationService",
    "CorrelationStore",
    "CorrelationType",
    "CostAggregation",
    "CostFilters",
    "CostGroupBy",
    "CostRecord",
    "CostStore",
    "CostTrackingService",
    "CsvAuditExporter",
    "Decision",
    "DecisionChain",
    "DecisionContext",
    "DecisionFilters",
    "DecisionLogger",
    "DecisionSeverity",
    "DecisionStore",
    "DecisionType",
    "ExportFormat",
    "ExportOptions",
    "FormatOptions",
    "FormatterStyle",
    "InMemoryCorrelationStore",
    "InMemoryCostStore",
    "InMemoryDecisionStore",
    "InMemoryTraceabilityStore",
    "JsonAuditExporter",
    "LinkType",
    "PdfAuditExporter",
    "PlainAuditFormatter",
    "RedactionConfig",
    "RichAuditFormatter",
    "TimelineEntry",
    "TokenUsage",
    "TraceLink",
    "TraceabilityService",
    "TraceabilityStore",
    "TraceableArtifact",
    # Factory functions
    "calculate_cost_if_missing",
    "extract_cost",
    "extract_token_usage",
    "get_audit_export_service",
    "get_audit_view_service",
    "get_correlation_service",
    "get_cost_tracking_service",
    "get_logger",
    "get_traceability_service",
]
