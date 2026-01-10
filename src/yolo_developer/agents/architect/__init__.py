"""Architect agent module for design decisions and ADR generation (Story 7.1).

The Architect agent is responsible for:
- Generating design decisions for stories
- Producing Architecture Decision Records (ADRs)
- Evaluating designs against quality attributes
- Identifying technical risks
- Ensuring designs follow 12-Factor principles

Example:
    >>> from yolo_developer.agents.architect import (
    ...     architect_node,
    ...     ArchitectOutput,
    ...     DesignDecision,
    ...     ADR,
    ...     DesignDecisionType,
    ...     ADRStatus,
    ... )
    >>>
    >>> # Create a design decision
    >>> decision = DesignDecision(
    ...     id="design-001",
    ...     story_id="story-001",
    ...     decision_type="pattern",
    ...     description="Use Repository pattern",
    ...     rationale="Clean separation of concerns",
    ...     alternatives_considered=("Active Record",),
    ... )
    >>>
    >>> # Run the architect node
    >>> result = await architect_node(state)

Architecture:
    The architect_node function is a LangGraph node that:
    - Receives YoloState TypedDict as input
    - Returns state update dict (not full state)
    - Never mutates input state
    - Uses async/await for all I/O
    - Integrates with architecture_validation gate (Story 7.1)

References:
    - ADR-001: TypedDict for internal state
    - ADR-005: LangGraph node patterns
    - ADR-006: Quality gate patterns
    - FR49-56: Architect Agent capabilities
"""

from __future__ import annotations

from yolo_developer.agents.architect import types
from yolo_developer.agents.architect.node import architect_node
from yolo_developer.agents.architect.twelve_factor import (
    analyze_twelve_factor,
    analyze_twelve_factor_with_llm,
)
from yolo_developer.agents.architect.types import (
    ADR,
    TWELVE_FACTORS,
    ADRStatus,
    ArchitectOutput,
    DesignDecision,
    DesignDecisionType,
    FactorResult,
    TwelveFactorAnalysis,
)

__all__ = [
    "ADR",
    "TWELVE_FACTORS",
    "ADRStatus",
    "ArchitectOutput",
    "DesignDecision",
    "DesignDecisionType",
    "FactorResult",
    "TwelveFactorAnalysis",
    "analyze_twelve_factor",
    "analyze_twelve_factor_with_llm",
    "architect_node",
    "types",
]
