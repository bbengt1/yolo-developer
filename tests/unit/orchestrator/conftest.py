"""Shared fixtures for orchestrator unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from yolo_developer.orchestrator.graph import Checkpointer
from yolo_developer.orchestrator.session import SessionManager
from yolo_developer.orchestrator.state import YoloState


@pytest.fixture
def temp_sessions_dir(tmp_path: Path) -> Path:
    """Create a temporary sessions directory."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    return sessions_dir


@pytest.fixture
def manager(temp_sessions_dir: Path) -> SessionManager:
    """Create a SessionManager with temporary directory."""
    return SessionManager(sessions_dir=temp_sessions_dir)


@pytest.fixture
def checkpointer(manager: SessionManager) -> Checkpointer:
    """Create a Checkpointer for testing."""
    return Checkpointer(manager)


@pytest.fixture
def sample_state() -> YoloState:
    """Create a sample YoloState for testing."""
    return {
        "messages": [HumanMessage(content="Test message")],
        "current_agent": "analyst",
        "handoff_context": None,
        "decisions": [],
    }
