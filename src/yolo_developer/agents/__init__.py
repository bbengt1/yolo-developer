"""Agents package for YOLO Developer (Story 5.1+, 6.1+).

This package contains the agent implementations for the multi-agent
orchestration system. Each agent is implemented as a LangGraph node
that follows the pattern of:
- Receiving YoloState TypedDict
- Returning state update dict (not mutating input)
- Using async/await for all I/O

Available Agents:
    analyst_node: Requirement crystallization and analysis
    pm_node: Story transformation and acceptance criteria generation

Types:
    CrystallizedRequirement: A refined requirement with category and testability
    AnalystOutput: Complete output from analyst processing
    Story: A user story with acceptance criteria
    PMOutput: Complete output from PM processing

Example:
    >>> from yolo_developer.agents import analyst_node, pm_node
    >>> from yolo_developer.orchestrator.state import YoloState
    >>>
    >>> state: YoloState = {
    ...     "messages": [HumanMessage(content="Build an app")],
    ...     "current_agent": "analyst",
    ...     "handoff_context": None,
    ...     "decisions": [],
    ... }
    >>> result = await analyst_node(state)
    >>> result["messages"]  # New messages

Architecture:
    Agents are decorated with @quality_gate to enforce quality
    checks at handoff boundaries. See ADR-006 for gate details.
"""

from __future__ import annotations

from yolo_developer.agents.analyst import (
    AnalystOutput,
    CrystallizedRequirement,
    analyst_node,
)
from yolo_developer.agents.pm import (
    AcceptanceCriterion,
    PMOutput,
    Story,
    StoryPriority,
    StoryStatus,
    pm_node,
)

__all__ = [
    "AcceptanceCriterion",
    "AnalystOutput",
    "CrystallizedRequirement",
    "PMOutput",
    "Story",
    "StoryPriority",
    "StoryStatus",
    "analyst_node",
    "pm_node",
]
