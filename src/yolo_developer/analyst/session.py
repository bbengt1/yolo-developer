from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from yolo_developer.analyst.elicitation import RequirementsElicitor
from yolo_developer.analyst.export import RequirementsExporter
from yolo_developer.analyst.models import (
    Answer,
    ExtractedRequirement,
    GatheringSession,
    Question,
    QuestionType,
    SessionPhase,
    SessionProgress,
    SessionResponse,
    SessionSummary,
)
from yolo_developer.analyst.validation import CompletenessValidator
from yolo_developer.config import load_config


class SessionManager:
    """Manage interactive requirements gathering sessions."""

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self.elicitor = RequirementsElicitor()
        self.validator = CompletenessValidator()
        self.exporter = RequirementsExporter()
        self.storage_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls) -> "SessionManager":
        config = load_config()
        storage_path = Path(config.analyst.gathering.storage_path)
        return cls(storage_path)

    def start_session(
        self,
        project_name: str,
        initial_description: str | None = None,
        project_type: str | None = None,
    ) -> GatheringSession:
        now = datetime.now(timezone.utc)
        session = GatheringSession(
            id=now.strftime("%Y%m%d%H%M%S"),
            project_name=project_name,
            phase=SessionPhase.DISCOVERY,
            started_at=now,
            updated_at=now,
            questions_asked=[],
            answers=[],
            extracted_requirements=[],
            context={},
            metadata={},
        )
        if initial_description:
            session.context["initial_description"] = initial_description
        if project_type:
            session.context["project_type"] = project_type
        session.questions_asked.extend(self.elicitor.generate_opening(session))
        self._save_session(session)
        return session

    def process_response(self, session_id: str, response: str) -> SessionResponse:
        session = self._load_session(session_id)
        current_question = self.get_current_question(session_id)
        if current_question is None:
            return SessionResponse(
                session=session,
                new_requirements=[],
                next_questions=[],
                phase_changed=False,
                is_complete=session.phase == SessionPhase.COMPLETE,
            )
        answer = Answer(
            question_id=current_question.id,
            response=response,
            timestamp=datetime.now(timezone.utc),
        )
        session.answers.append(answer)
        new_reqs = self.elicitor.extract_requirements(session, current_question, answer)
        session.extracted_requirements.extend(new_reqs)
        phase_changed = False
        if self._should_advance_phase(session):
            session.phase = _next_phase(session.phase)
            phase_changed = True
        next_questions = []
        if session.phase != SessionPhase.COMPLETE:
            next_questions = self.elicitor.generate_next_questions(session)
            session.questions_asked.extend(next_questions)
        session.updated_at = datetime.now(timezone.utc)
        self._save_session(session)
        return SessionResponse(
            session=session,
            new_requirements=new_reqs,
            next_questions=next_questions,
            phase_changed=phase_changed,
            is_complete=session.phase == SessionPhase.COMPLETE,
        )

    def get_current_question(self, session_id: str) -> Question | None:
        session = self._load_session(session_id)
        answered = {answer.question_id for answer in session.answers}
        for question in session.questions_asked:
            if question.id not in answered:
                return question
        return None

    def get_progress(self, session_id: str) -> SessionProgress:
        session = self._load_session(session_id)
        phases = [phase for phase in SessionPhase if phase != SessionPhase.COMPLETE]
        phase_index = phases.index(session.phase) if session.phase in phases else len(phases)
        phase_progress = phase_index / max(len(phases), 1)
        return SessionProgress(
            session_id=session.id,
            phase=session.phase,
            phase_progress=phase_progress,
            questions_asked=len(session.questions_asked),
            questions_answered=len(session.answers),
            requirements_extracted=len(session.extracted_requirements),
            estimated_completion=max(0, len(phases) - phase_index),
        )

    def export_requirements(self, session_id: str, format: str = "markdown") -> str:
        session = self._load_session(session_id)
        validation = self.validator.validate(session)
        metadata = {
            "project_name": session.project_name,
            "session_id": session.id,
            "gathered_at": session.started_at.isoformat(),
            "warnings": validation.warnings,
        }
        return self.exporter.export(
            session.extracted_requirements,
            format=format,
            metadata=metadata,
        )

    def resume_session(self, session_id: str) -> GatheringSession:
        return self._load_session(session_id)

    def list_sessions(self) -> list[SessionSummary]:
        sessions: list[SessionSummary] = []
        for path in sorted(self.storage_path.glob("*.json")):
            session = self._load_session(path.stem)
            sessions.append(
                SessionSummary(
                    id=session.id,
                    project_name=session.project_name,
                    phase=session.phase,
                    started_at=session.started_at,
                    requirements_count=len(session.extracted_requirements),
                )
            )
        return sorted(sessions, key=lambda item: item.started_at, reverse=True)

    def _save_session(self, session: GatheringSession) -> None:
        data = _serialize_session(session)
        path = self.storage_path / f"{session.id}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_session(self, session_id: str) -> GatheringSession:
        path = self.storage_path / f"{session_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return _deserialize_session(data)

    def _should_advance_phase(self, session: GatheringSession) -> bool:
        if session.phase == SessionPhase.REFINEMENT:
            return True
        return len(session.answers) >= len(session.questions_asked)


def _serialize_session(session: GatheringSession) -> dict:
    def _serialize_datetime(value: datetime) -> str:
        return value.isoformat()

    payload = asdict(session)
    payload["started_at"] = _serialize_datetime(session.started_at)
    payload["updated_at"] = _serialize_datetime(session.updated_at)
    payload["phase"] = session.phase.value
    for question in payload["questions_asked"]:
        question["type"] = question["type"].value
        question["phase"] = question["phase"].value
    for answer in payload["answers"]:
        answer["timestamp"] = answer["timestamp"].isoformat()
    return payload


def _deserialize_session(payload: dict) -> GatheringSession:
    started_at = datetime.fromisoformat(payload["started_at"])
    updated_at = datetime.fromisoformat(payload["updated_at"])
    questions = []
    for question in payload["questions_asked"]:
        questions.append(
            Question(
                id=question["id"],
                text=question["text"],
                type=_parse_question_type(question["type"]),
                options=question.get("options"),
                context=question.get("context"),
                follow_ups=question.get("follow_ups", []),
                required=question.get("required", True),
                phase=SessionPhase(question.get("phase", SessionPhase.DISCOVERY.value)),
            )
        )
    answers = []
    for answer in payload["answers"]:
        answers.append(
            Answer(
                question_id=answer["question_id"],
                response=answer["response"],
                timestamp=datetime.fromisoformat(answer["timestamp"]),
                clarifications=answer.get("clarifications", []),
            )
        )
    requirements = []
    for req in payload["extracted_requirements"]:
        requirements.append(
            ExtractedRequirement(
                id=req["id"],
                description=req["description"],
                type=req["type"],
                priority=req["priority"],
                source_answers=req.get("source_answers", []),
                confidence=req.get("confidence", 0.0),
                needs_clarification=req.get("needs_clarification", False),
            )
        )
    return GatheringSession(
        id=payload["id"],
        project_name=payload["project_name"],
        phase=SessionPhase(payload["phase"]),
        started_at=started_at,
        updated_at=updated_at,
        questions_asked=questions,
        answers=answers,
        extracted_requirements=requirements,
        context=payload.get("context", {}),
        metadata=payload.get("metadata", {}),
    )


def _parse_question_type(value: str) -> QuestionType:
    try:
        return QuestionType(value)
    except ValueError:
        return QuestionType.OPEN_ENDED


def _next_phase(current: SessionPhase) -> SessionPhase:
    order = [
        SessionPhase.DISCOVERY,
        SessionPhase.USE_CASES,
        SessionPhase.REQUIREMENTS,
        SessionPhase.CONSTRAINTS,
        SessionPhase.EDGE_CASES,
        SessionPhase.VALIDATION,
        SessionPhase.REFINEMENT,
        SessionPhase.COMPLETE,
    ]
    try:
        idx = order.index(current)
    except ValueError:
        return SessionPhase.COMPLETE
    return order[min(idx + 1, len(order) - 1)]
