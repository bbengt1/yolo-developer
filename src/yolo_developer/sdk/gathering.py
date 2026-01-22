from __future__ import annotations

from yolo_developer.analyst import SessionManager
from yolo_developer.analyst.models import Question


class GatheringClient:
    """SDK client for interactive requirements gathering."""

    def __init__(self) -> None:
        self._manager = SessionManager.from_config()
        self._current_session_id: str | None = None

    async def start_session(
        self,
        project_name: str,
        initial_description: str | None = None,
        project_type: str | None = None,
    ):
        session = self._manager.start_session(project_name, initial_description, project_type)
        self._current_session_id = session.id
        return session

    async def get_next_question(self) -> Question | None:
        if not self._current_session_id:
            raise ValueError("No active session")
        return self._manager.get_current_question(self._current_session_id)

    async def submit_response(self, response: str):
        if not self._current_session_id:
            raise ValueError("No active session")
        return self._manager.process_response(self._current_session_id, response)

    async def get_progress(self):
        if not self._current_session_id:
            raise ValueError("No active session")
        return self._manager.get_progress(self._current_session_id)

    async def export_requirements(self, format: str = "markdown") -> str:
        if not self._current_session_id:
            raise ValueError("No active session")
        return self._manager.export_requirements(self._current_session_id, format=format)
