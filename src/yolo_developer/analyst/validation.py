from __future__ import annotations

from dataclasses import dataclass

from yolo_developer.analyst.models import GatheringSession


@dataclass(frozen=True)
class ValidationResult:
    is_complete: bool
    warnings: list[str]


class CompletenessValidator:
    """Minimal validation for gathered requirements."""

    def validate(self, session: GatheringSession) -> ValidationResult:
        warnings: list[str] = []
        if not session.extracted_requirements:
            warnings.append("No requirements extracted yet.")
        if session.phase != session.phase.COMPLETE and len(session.answers) < 3:
            warnings.append("Session may be incomplete.")
        return ValidationResult(is_complete=not warnings, warnings=warnings)
