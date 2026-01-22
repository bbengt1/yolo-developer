from __future__ import annotations

import re
import uuid

from yolo_developer.analyst.models import (
    Answer,
    ExtractedRequirement,
    GatheringSession,
    Question,
    QuestionType,
    SessionPhase,
)


class RequirementsElicitor:
    """Generate questions and extract requirements from responses."""

    def generate_opening(self, session: GatheringSession) -> list[Question]:
        if "initial_description" not in session.context:
            return [
                Question(
                    id=self._gen_id(),
                    text="What would you like to build? Please describe your idea briefly.",
                    type=QuestionType.OPEN_ENDED,
                    phase=SessionPhase.DISCOVERY,
                )
            ]
        return [
            Question(
                id=self._gen_id(),
                text="Who are the primary users for this project?",
                type=QuestionType.OPEN_ENDED,
                phase=SessionPhase.DISCOVERY,
            ),
            Question(
                id=self._gen_id(),
                text="What problem does this solve for those users?",
                type=QuestionType.OPEN_ENDED,
                phase=SessionPhase.DISCOVERY,
            ),
        ]

    def generate_next_questions(self, session: GatheringSession) -> list[Question]:
        phase = session.phase
        if phase == SessionPhase.DISCOVERY:
            return [
                Question(
                    id=self._gen_id(),
                    text="Describe the primary workflow a user should complete.",
                    type=QuestionType.OPEN_ENDED,
                    phase=SessionPhase.USE_CASES,
                )
            ]
        if phase == SessionPhase.USE_CASES:
            return [
                Question(
                    id=self._gen_id(),
                    text="List the main features or actions users need.",
                    type=QuestionType.LIST,
                    phase=SessionPhase.REQUIREMENTS,
                )
            ]
        if phase == SessionPhase.REQUIREMENTS:
            return [
                Question(
                    id=self._gen_id(),
                    text="Are there any required technologies, integrations, or constraints?",
                    type=QuestionType.OPEN_ENDED,
                    phase=SessionPhase.CONSTRAINTS,
                )
            ]
        if phase == SessionPhase.CONSTRAINTS:
            return [
                Question(
                    id=self._gen_id(),
                    text="What edge cases or failure scenarios should be handled?",
                    type=QuestionType.OPEN_ENDED,
                    phase=SessionPhase.EDGE_CASES,
                )
            ]
        if phase == SessionPhase.EDGE_CASES:
            return [
                Question(
                    id=self._gen_id(),
                    text="Is anything missing or unclear in the requirements so far?",
                    type=QuestionType.OPEN_ENDED,
                    phase=SessionPhase.VALIDATION,
                )
            ]
        if phase == SessionPhase.VALIDATION:
            return [
                Question(
                    id=self._gen_id(),
                    text="Confirm: Are these requirements complete and accurate?",
                    type=QuestionType.CONFIRMATION,
                    options=["yes", "no"],
                    phase=SessionPhase.REFINEMENT,
                )
            ]
        return []

    def extract_requirements(
        self,
        session: GatheringSession,
        question: Question,
        answer: Answer,
    ) -> list[ExtractedRequirement]:
        response = str(answer.response).strip()
        if not response:
            return []
        candidates = _split_candidate_requirements(response)
        extracted = []
        for idx, sentence in enumerate(candidates, start=1):
            req_type = _detect_requirement_type(sentence)
            extracted.append(
                ExtractedRequirement(
                    id=f"req-{session.id}-{len(session.extracted_requirements) + idx:03d}",
                    description=sentence,
                    type=req_type,
                    priority="medium",
                    source_answers=[answer.question_id],
                    confidence=0.7,
                )
            )
        return extracted

    def _gen_id(self) -> str:
        return uuid.uuid4().hex[:8]


def _split_candidate_requirements(response: str) -> list[str]:
    lines = [line.strip("- ").strip() for line in response.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    sentences = re.split(r"[.!?]+", response)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _detect_requirement_type(text: str) -> str:
    lowered = text.lower()
    if any(keyword in lowered for keyword in ["must", "cannot", "only", "requires"]):
        return "constraint"
    if any(keyword in lowered for keyword in ["performance", "latency", "throughput", "secure"]):
        return "non-functional"
    return "functional"
