"""Analyst agent module for requirement crystallization (Story 5.1, 5.2, 5.3, 5.4).

The Analyst agent is responsible for:
- Crystallizing raw requirements from seed content
- Identifying gaps in requirements (with structured analysis)
- Flagging contradictions between requirements
- Categorizing requirements (functional, non-functional, constraint)
- Assessing requirement testability
- Detecting edge cases, implied requirements, and pattern-based suggestions

Example:
    >>> from yolo_developer.agents.analyst import (
    ...     analyst_node,
    ...     AnalystOutput,
    ...     CrystallizedRequirement,
    ...     IdentifiedGap,
    ...     GapType,
    ...     Severity,
    ... )
    >>>
    >>> # Create a crystallized requirement
    >>> req = CrystallizedRequirement(
    ...     id="req-001",
    ...     original_text="Fast system",
    ...     refined_text="API response < 200ms",
    ...     category="non-functional",
    ...     testable=True,
    ... )
    >>>
    >>> # Create an identified gap
    >>> gap = IdentifiedGap(
    ...     id="gap-001",
    ...     description="Missing logout functionality",
    ...     gap_type=GapType.IMPLIED_REQUIREMENT,
    ...     severity=Severity.HIGH,
    ...     source_requirements=("req-001",),
    ...     rationale="Login implies logout needed",
    ... )
    >>>
    >>> # Run the analyst node
    >>> result = await analyst_node(state)

Architecture:
    The analyst_node function is a LangGraph node that:
    - Receives YoloState TypedDict as input
    - Returns state update dict (not full state)
    - Never mutates input state
    - Uses async/await for all I/O

References:
    - ADR-001: TypedDict for internal state
    - ADR-005: LangGraph node patterns
    - FR36-41: Analyst Agent capabilities
"""

from __future__ import annotations

from yolo_developer.agents.analyst.node import analyst_node
from yolo_developer.agents.analyst.types import (
    AnalystOutput,
    CategorizationResult,
    ConstraintSubCategory,
    CrystallizedRequirement,
    FunctionalSubCategory,
    GapType,
    IdentifiedGap,
    NonFunctionalSubCategory,
    RequirementCategory,
    Severity,
)

__all__ = [
    "AnalystOutput",
    "CategorizationResult",
    "ConstraintSubCategory",
    "CrystallizedRequirement",
    "FunctionalSubCategory",
    "GapType",
    "IdentifiedGap",
    "NonFunctionalSubCategory",
    "RequirementCategory",
    "Severity",
    "analyst_node",
]
