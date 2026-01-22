from __future__ import annotations

from pathlib import Path

from yolo_developer.analyst.session import SessionManager


def test_session_lifecycle(tmp_path: Path) -> None:
    manager = SessionManager(storage_path=tmp_path)
    session = manager.start_session("demo", "Build a task app", "web")
    assert session.project_name == "demo"

    question = manager.get_current_question(session.id)
    assert question is not None

    response = manager.process_response(session.id, "Users can add tasks.")
    assert response.session.extracted_requirements

    progress = manager.get_progress(session.id)
    assert progress.requirements_extracted >= 1

    exported = manager.export_requirements(session.id, format="markdown")
    assert "Requirements" in exported


def test_list_sessions(tmp_path: Path) -> None:
    manager = SessionManager(storage_path=tmp_path)
    first = manager.start_session("alpha")
    second = manager.start_session("beta")
    sessions = manager.list_sessions()
    ids = {session.id for session in sessions}
    assert first.id in ids
    assert second.id in ids
