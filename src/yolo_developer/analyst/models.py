from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SessionPhase(str, Enum):
    DISCOVERY = "discovery"
    USE_CASES = "use_cases"
    REQUIREMENTS = "requirements"
    CONSTRAINTS = "constraints"
    EDGE_CASES = "edge_cases"
    VALIDATION = "validation"
    REFINEMENT = "refinement"
    COMPLETE = "complete"


class QuestionType(str, Enum):
    OPEN_ENDED = "open_ended"
    MULTIPLE_CHOICE = "multiple_choice"
    YES_NO = "yes_no"
    SCALE = "scale"
    LIST = "list"
    CONFIRMATION = "confirmation"


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    type: QuestionType
    options: list[str] | None = None
    context: str | None = None
    follow_ups: list[str] = field(default_factory=list)
    required: bool = True
    phase: SessionPhase = SessionPhase.DISCOVERY


@dataclass(frozen=True)
class Answer:
    question_id: str
    response: Any
    timestamp: datetime
    clarifications: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExtractedRequirement:
    id: str
    description: str
    type: str
    priority: str
    source_answers: list[str]
    confidence: float
    needs_clarification: bool = False


@dataclass
class GatheringSession:
    id: str
    project_name: str
    phase: SessionPhase
    started_at: datetime
    updated_at: datetime
    questions_asked: list[Question]
    answers: list[Answer]
    extracted_requirements: list[ExtractedRequirement]
    context: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SessionResponse:
    session: GatheringSession
    new_requirements: list[ExtractedRequirement]
    next_questions: list[Question]
    phase_changed: bool
    is_complete: bool


@dataclass(frozen=True)
class SessionProgress:
    session_id: str
    phase: SessionPhase
    phase_progress: float
    questions_asked: int
    questions_answered: int
    requirements_extracted: int
    estimated_completion: int


@dataclass(frozen=True)
class SessionSummary:
    id: str
    project_name: str
    phase: SessionPhase
    started_at: datetime
    requirements_count: int
