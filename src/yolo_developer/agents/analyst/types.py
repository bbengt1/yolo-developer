"""Type definitions for Analyst agent (Story 5.1).

This module provides the data types used by the Analyst agent:

- CrystallizedRequirement: A refined, categorized requirement with testability
- AnalystOutput: Complete output from analyst processing

All types are frozen dataclasses (immutable) per ADR-001 for internal state.

Example:
    >>> from yolo_developer.agents.analyst.types import (
    ...     CrystallizedRequirement,
    ...     AnalystOutput,
    ... )
    >>>
    >>> req = CrystallizedRequirement(
    ...     id="req-001",
    ...     original_text="System should be fast",
    ...     refined_text="Response time < 200ms at 95th percentile",
    ...     category="non-functional",
    ...     testable=True,
    ... )
    >>> req.to_dict()
    {'id': 'req-001', ...}

Security Note:
    These types are used for internal state only. Validation of user input
    should happen at system boundaries using Pydantic models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CrystallizedRequirement:
    """A refined, categorized requirement extracted from seed content.

    Immutable dataclass representing a single requirement that has been
    processed by the Analyst agent. Each requirement is categorized and
    assessed for testability.

    Attributes:
        id: Unique identifier for this requirement (e.g., "req-001").
        original_text: The original text from the seed document.
        refined_text: The clarified, refined requirement text.
        category: Category of the requirement:
            - "functional": Feature or behavior
            - "non-functional": Quality attribute (performance, security, etc.)
            - "constraint": Technical or business constraint
        testable: Whether this requirement can be objectively tested.

    Example:
        >>> req = CrystallizedRequirement(
        ...     id="req-001",
        ...     original_text="The system should be fast",
        ...     refined_text="API response time < 200ms for 95th percentile",
        ...     category="non-functional",
        ...     testable=True,
        ... )
        >>> req.category
        'non-functional'
    """

    id: str
    original_text: str
    refined_text: str
    category: str
    testable: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all requirement fields.

        Example:
            >>> req = CrystallizedRequirement(
            ...     id="req-001",
            ...     original_text="orig",
            ...     refined_text="ref",
            ...     category="functional",
            ...     testable=True,
            ... )
            >>> req.to_dict()
            {'id': 'req-001', 'original_text': 'orig', ...}
        """
        return {
            "id": self.id,
            "original_text": self.original_text,
            "refined_text": self.refined_text,
            "category": self.category,
            "testable": self.testable,
        }


@dataclass(frozen=True)
class AnalystOutput:
    """Complete output from Analyst agent processing.

    Immutable dataclass containing all results from analyzing seed content:
    crystallized requirements, identified gaps, and contradictions.

    Attributes:
        requirements: Tuple of CrystallizedRequirement objects.
        identified_gaps: Tuple of strings describing missing information.
        contradictions: Tuple of strings describing conflicting requirements.

    Example:
        >>> req = CrystallizedRequirement(
        ...     id="req-001",
        ...     original_text="orig",
        ...     refined_text="ref",
        ...     category="functional",
        ...     testable=True,
        ... )
        >>> output = AnalystOutput(
        ...     requirements=(req,),
        ...     identified_gaps=("Missing auth details",),
        ...     contradictions=(),
        ... )
        >>> len(output.requirements)
        1
    """

    requirements: tuple[CrystallizedRequirement, ...]
    identified_gaps: tuple[str, ...]
    contradictions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Serializes nested CrystallizedRequirement objects as well.

        Returns:
            Dictionary with all output fields.

        Example:
            >>> output = AnalystOutput(
            ...     requirements=(),
            ...     identified_gaps=("gap1",),
            ...     contradictions=(),
            ... )
            >>> d = output.to_dict()
            >>> d["identified_gaps"]
            ['gap1']
        """
        return {
            "requirements": [r.to_dict() for r in self.requirements],
            "identified_gaps": list(self.identified_gaps),
            "contradictions": list(self.contradictions),
        }
