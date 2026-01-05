"""Unit tests for session persistence module.

Tests cover:
- SessionMetadata dataclass creation and immutability
- SessionState dataclass creation
- serialize_state and deserialize_state functions
- SessionManager save, load, list operations
- Error handling for missing and corrupted sessions
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from yolo_developer.orchestrator.context import Decision, HandoffContext
from yolo_developer.orchestrator.graph import Checkpointer
from yolo_developer.orchestrator.session import (
    SessionLoadError,
    SessionManager,
    SessionMetadata,
    SessionNotFoundError,
    SessionState,
    deserialize_state,
    serialize_state,
)
from yolo_developer.orchestrator.state import YoloState


class TestSessionMetadata:
    """Tests for the SessionMetadata dataclass."""

    def test_metadata_creation_with_required_fields(self) -> None:
        """SessionMetadata should store all required fields."""
        now = datetime.now(timezone.utc)
        metadata = SessionMetadata(
            session_id="session-abc123",
            created_at=now,
            last_checkpoint=now,
            current_agent="analyst",
        )
        assert metadata.session_id == "session-abc123"
        assert metadata.created_at == now
        assert metadata.last_checkpoint == now
        assert metadata.current_agent == "analyst"

    def test_metadata_has_default_progress_fields(self) -> None:
        """SessionMetadata should have default values for progress fields."""
        now = datetime.now(timezone.utc)
        metadata = SessionMetadata(
            session_id="test",
            created_at=now,
            last_checkpoint=now,
            current_agent="pm",
        )
        assert metadata.stories_completed == 0
        assert metadata.stories_total == 0

    def test_metadata_with_progress_fields(self) -> None:
        """SessionMetadata should store progress fields."""
        now = datetime.now(timezone.utc)
        metadata = SessionMetadata(
            session_id="test",
            created_at=now,
            last_checkpoint=now,
            current_agent="dev",
            stories_completed=3,
            stories_total=5,
        )
        assert metadata.stories_completed == 3
        assert metadata.stories_total == 5

    def test_metadata_is_frozen(self) -> None:
        """SessionMetadata should be immutable."""
        now = datetime.now(timezone.utc)
        metadata = SessionMetadata(
            session_id="test",
            created_at=now,
            last_checkpoint=now,
            current_agent="analyst",
        )
        with pytest.raises(AttributeError):
            metadata.session_id = "changed"  # type: ignore[misc]


class TestSessionState:
    """Tests for the SessionState dataclass."""

    def test_state_creation(self) -> None:
        """SessionState should store metadata and state_data."""
        now = datetime.now(timezone.utc)
        metadata = SessionMetadata(
            session_id="test",
            created_at=now,
            last_checkpoint=now,
            current_agent="analyst",
        )
        state_data = {"current_agent": "analyst", "messages": []}

        session = SessionState(metadata=metadata, state_data=state_data)

        assert session.metadata.session_id == "test"
        assert session.state_data["current_agent"] == "analyst"

    def test_state_is_frozen(self) -> None:
        """SessionState should be immutable."""
        now = datetime.now(timezone.utc)
        metadata = SessionMetadata(
            session_id="test",
            created_at=now,
            last_checkpoint=now,
            current_agent="analyst",
        )
        session = SessionState(metadata=metadata, state_data={})

        with pytest.raises(AttributeError):
            session.metadata = metadata  # type: ignore[misc]


class TestSerializeState:
    """Tests for serialize_state function."""

    def test_serializes_empty_state(self) -> None:
        """serialize_state should handle empty state."""
        state: YoloState = {
            "messages": [],
            "current_agent": "",
            "handoff_context": None,
            "decisions": [],
        }
        result = serialize_state(state)

        assert result["messages"] == []
        assert result["current_agent"] == ""
        assert result["handoff_context"] is None
        assert result["decisions"] == []

    def test_serializes_messages(self) -> None:
        """serialize_state should serialize BaseMessage objects."""
        state: YoloState = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there"),
            ],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }
        result = serialize_state(state)

        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][0], dict)
        # LangChain messages_to_dict wraps content in 'data' key with 'type' at top level
        assert "type" in result["messages"][0]
        assert "data" in result["messages"][0]
        assert result["messages"][0]["data"]["content"] == "Hello"

    def test_serializes_decisions(self) -> None:
        """serialize_state should serialize Decision objects."""
        fixed_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        state: YoloState = {
            "messages": [],
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [
                Decision(
                    agent="analyst",
                    summary="Test summary",
                    rationale="Test rationale",
                    timestamp=fixed_time,
                    related_artifacts=("req-001",),
                )
            ],
        }
        result = serialize_state(state)

        assert len(result["decisions"]) == 1
        assert result["decisions"][0]["agent"] == "analyst"
        assert result["decisions"][0]["summary"] == "Test summary"
        assert result["decisions"][0]["timestamp"] == "2024-01-01T00:00:00+00:00"
        assert result["decisions"][0]["related_artifacts"] == ["req-001"]

    def test_serializes_handoff_context(self) -> None:
        """serialize_state should serialize HandoffContext."""
        fixed_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        decision = Decision(
            agent="analyst",
            summary="Decision in context",
            rationale="Reason",
            timestamp=fixed_time,
        )
        context = HandoffContext(
            source_agent="analyst",
            target_agent="pm",
            decisions=(decision,),
            memory_refs=("ref-001", "ref-002"),
            timestamp=fixed_time,
        )
        state: YoloState = {
            "messages": [],
            "current_agent": "pm",
            "handoff_context": context,
            "decisions": [],
        }
        result = serialize_state(state)

        assert result["handoff_context"] is not None
        assert result["handoff_context"]["source_agent"] == "analyst"
        assert result["handoff_context"]["target_agent"] == "pm"
        assert len(result["handoff_context"]["decisions"]) == 1
        assert result["handoff_context"]["memory_refs"] == ["ref-001", "ref-002"]

    def test_serialized_state_is_json_compatible(self) -> None:
        """serialize_state output should be JSON serializable."""
        fixed_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        state: YoloState = {
            "messages": [HumanMessage(content="Test")],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [
                Decision(
                    agent="pm",
                    summary="Test",
                    rationale="Test",
                    timestamp=fixed_time,
                )
            ],
        }
        result = serialize_state(state)

        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)


class TestDeserializeState:
    """Tests for deserialize_state function."""

    def test_deserializes_empty_state(self) -> None:
        """deserialize_state should handle empty data."""
        data: dict[str, Any] = {
            "messages": [],
            "current_agent": "",
            "handoff_context": None,
            "decisions": [],
        }
        result = deserialize_state(data)

        assert result["messages"] == []
        assert result["current_agent"] == ""
        assert result["handoff_context"] is None
        assert result["decisions"] == []

    def test_deserializes_decisions(self) -> None:
        """deserialize_state should reconstruct Decision objects."""
        data: dict[str, Any] = {
            "messages": [],
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [
                {
                    "agent": "analyst",
                    "summary": "Test summary",
                    "rationale": "Test rationale",
                    "timestamp": "2024-01-01T00:00:00+00:00",
                    "related_artifacts": ["req-001"],
                }
            ],
        }
        result = deserialize_state(data)

        assert len(result["decisions"]) == 1
        decision = result["decisions"][0]
        assert isinstance(decision, Decision)
        assert decision.agent == "analyst"
        assert decision.summary == "Test summary"
        assert decision.related_artifacts == ("req-001",)

    def test_deserializes_handoff_context(self) -> None:
        """deserialize_state should reconstruct HandoffContext."""
        data: dict[str, Any] = {
            "messages": [],
            "current_agent": "pm",
            "handoff_context": {
                "source_agent": "analyst",
                "target_agent": "pm",
                "decisions": [
                    {
                        "agent": "analyst",
                        "summary": "Test",
                        "rationale": "Reason",
                        "timestamp": "2024-01-01T00:00:00+00:00",
                        "related_artifacts": [],
                    }
                ],
                "memory_refs": ["ref-001"],
                "timestamp": "2024-01-01T00:00:00+00:00",
            },
            "decisions": [],
        }
        result = deserialize_state(data)

        ctx = result["handoff_context"]
        assert isinstance(ctx, HandoffContext)
        assert ctx.source_agent == "analyst"
        assert ctx.target_agent == "pm"
        assert len(ctx.decisions) == 1
        assert ctx.memory_refs == ("ref-001",)

    def test_handles_missing_optional_fields(self) -> None:
        """deserialize_state should handle missing optional fields."""
        data: dict[str, Any] = {
            "current_agent": "analyst",
        }
        result = deserialize_state(data)

        assert result["messages"] == []
        assert result["decisions"] == []
        assert result["handoff_context"] is None


class TestSerializationRoundTrip:
    """Tests for serialize/deserialize round-trip preservation."""

    def test_round_trip_preserves_messages(self) -> None:
        """Messages should survive serialization round-trip."""
        state: YoloState = {
            "messages": [
                HumanMessage(content="User input"),
                AIMessage(content="Agent response"),
            ],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        serialized = serialize_state(state)
        restored = deserialize_state(serialized)

        assert len(restored["messages"]) == 2
        assert restored["messages"][0].content == "User input"
        assert restored["messages"][1].content == "Agent response"

    def test_round_trip_preserves_decisions(self) -> None:
        """Decisions should survive serialization round-trip."""
        fixed_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        original_decision = Decision(
            agent="architect",
            summary="Chose PostgreSQL",
            rationale="ACID compliance required",
            timestamp=fixed_time,
            related_artifacts=("req-001", "req-002"),
        )
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [original_decision],
        }

        serialized = serialize_state(state)
        restored = deserialize_state(serialized)

        assert len(restored["decisions"]) == 1
        restored_decision = restored["decisions"][0]
        assert restored_decision.agent == "architect"
        assert restored_decision.summary == "Chose PostgreSQL"
        assert restored_decision.timestamp == fixed_time
        assert restored_decision.related_artifacts == ("req-001", "req-002")

    def test_round_trip_preserves_handoff_context(self) -> None:
        """HandoffContext should survive serialization round-trip."""
        fixed_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        decision = Decision(
            agent="analyst",
            summary="Test",
            rationale="Reason",
            timestamp=fixed_time,
        )
        context = HandoffContext(
            source_agent="analyst",
            target_agent="pm",
            decisions=(decision,),
            memory_refs=("ref-001", "ref-002"),
            timestamp=fixed_time,
        )
        state: YoloState = {
            "messages": [],
            "current_agent": "pm",
            "handoff_context": context,
            "decisions": [],
        }

        serialized = serialize_state(state)
        restored = deserialize_state(serialized)

        restored_ctx = restored["handoff_context"]
        assert restored_ctx is not None
        assert restored_ctx.source_agent == "analyst"
        assert restored_ctx.target_agent == "pm"
        assert len(restored_ctx.decisions) == 1
        assert restored_ctx.memory_refs == ("ref-001", "ref-002")
        assert restored_ctx.timestamp == fixed_time


class TestSessionManager:
    """Tests for SessionManager class."""

    @pytest.mark.asyncio
    async def test_save_session_creates_file(
        self,
        manager: SessionManager,
        sample_state: YoloState,
        temp_sessions_dir: Path,
    ) -> None:
        """save_session should create a session file."""
        session_id = await manager.save_session(sample_state)

        session_path = temp_sessions_dir / f"{session_id}.json"
        assert session_path.exists()

    @pytest.mark.asyncio
    async def test_save_session_returns_session_id(
        self,
        manager: SessionManager,
        sample_state: YoloState,
    ) -> None:
        """save_session should return a session ID."""
        session_id = await manager.save_session(sample_state)

        assert session_id.startswith("session-")
        assert len(session_id) > 8

    @pytest.mark.asyncio
    async def test_save_session_with_custom_id(
        self,
        manager: SessionManager,
        sample_state: YoloState,
    ) -> None:
        """save_session should use provided session ID."""
        session_id = await manager.save_session(sample_state, session_id="custom-session")

        assert session_id == "custom-session"

    @pytest.mark.asyncio
    async def test_save_session_updates_active_pointer(
        self,
        manager: SessionManager,
        sample_state: YoloState,
        temp_sessions_dir: Path,
    ) -> None:
        """save_session should update the active session pointer."""
        session_id = await manager.save_session(sample_state)

        active_file = temp_sessions_dir / "_active.json"
        assert active_file.exists()

        with open(active_file) as f:
            data = json.load(f)
        assert data["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_load_session_restores_state(
        self,
        manager: SessionManager,
        sample_state: YoloState,
    ) -> None:
        """load_session should restore the saved state."""
        session_id = await manager.save_session(sample_state)

        restored_state, metadata = await manager.load_session(session_id)

        assert restored_state["current_agent"] == "analyst"
        assert len(restored_state["messages"]) == 1
        assert metadata.session_id == session_id

    @pytest.mark.asyncio
    async def test_load_session_restores_metadata(
        self,
        manager: SessionManager,
        sample_state: YoloState,
    ) -> None:
        """load_session should restore metadata with timestamps."""
        session_id = await manager.save_session(sample_state)

        _, metadata = await manager.load_session(session_id)

        assert metadata.session_id == session_id
        assert isinstance(metadata.created_at, datetime)
        assert isinstance(metadata.last_checkpoint, datetime)
        assert metadata.current_agent == "analyst"

    @pytest.mark.asyncio
    async def test_load_session_restores_story_progress(
        self,
        manager: SessionManager,
        sample_state: YoloState,
    ) -> None:
        """load_session should restore stories_completed and stories_total."""
        session_id = await manager.save_session(
            sample_state,
            stories_completed=3,
            stories_total=8,
        )

        _, metadata = await manager.load_session(session_id)

        assert metadata.stories_completed == 3
        assert metadata.stories_total == 8

    @pytest.mark.asyncio
    async def test_list_sessions_includes_story_progress(
        self,
        manager: SessionManager,
        sample_state: YoloState,
    ) -> None:
        """list_sessions should include story progress in metadata."""
        await manager.save_session(
            sample_state,
            session_id="session-with-progress",
            stories_completed=5,
            stories_total=10,
        )

        sessions = await manager.list_sessions()

        assert len(sessions) == 1
        assert sessions[0].stories_completed == 5
        assert sessions[0].stories_total == 10

    @pytest.mark.asyncio
    async def test_load_session_raises_for_missing(
        self,
        manager: SessionManager,
    ) -> None:
        """load_session should raise SessionNotFoundError for missing sessions."""
        with pytest.raises(SessionNotFoundError) as exc_info:
            await manager.load_session("nonexistent-session")

        assert exc_info.value.session_id == "nonexistent-session"

    @pytest.mark.asyncio
    async def test_load_session_raises_for_corrupted_json(
        self,
        manager: SessionManager,
        temp_sessions_dir: Path,
    ) -> None:
        """load_session should raise SessionLoadError for corrupted JSON."""
        # Create a corrupted session file
        session_path = temp_sessions_dir / "session-corrupted.json"
        session_path.write_text("{ invalid json }")

        with pytest.raises(SessionLoadError) as exc_info:
            await manager.load_session("session-corrupted")

        assert exc_info.value.session_id == "session-corrupted"
        assert "corrupted" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_load_session_raises_for_missing_fields(
        self,
        manager: SessionManager,
        temp_sessions_dir: Path,
    ) -> None:
        """load_session should raise SessionLoadError for missing required fields."""
        # Create a session file missing required fields
        session_path = temp_sessions_dir / "session-incomplete.json"
        session_path.write_text('{"session_id": "session-incomplete"}')

        with pytest.raises(SessionLoadError) as exc_info:
            await manager.load_session("session-incomplete")

        assert exc_info.value.session_id == "session-incomplete"
        assert "missing" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_list_sessions_returns_all_sessions(
        self,
        manager: SessionManager,
        sample_state: YoloState,
    ) -> None:
        """list_sessions should return all saved sessions."""
        await manager.save_session(sample_state, session_id="session-001")
        await manager.save_session(sample_state, session_id="session-002")
        await manager.save_session(sample_state, session_id="session-003")

        sessions = await manager.list_sessions()

        assert len(sessions) == 3
        session_ids = [s.session_id for s in sessions]
        assert "session-001" in session_ids
        assert "session-002" in session_ids
        assert "session-003" in session_ids

    @pytest.mark.asyncio
    async def test_list_sessions_sorted_by_last_checkpoint(
        self,
        manager: SessionManager,
        sample_state: YoloState,
    ) -> None:
        """list_sessions should return sessions sorted by last_checkpoint (newest first)."""
        # Save sessions in order (each save has later timestamp)
        await manager.save_session(sample_state, session_id="session-first")
        await manager.save_session(sample_state, session_id="session-second")
        await manager.save_session(sample_state, session_id="session-third")

        sessions = await manager.list_sessions()

        # Most recent should be first
        assert sessions[0].session_id == "session-third"
        assert sessions[-1].session_id == "session-first"

    @pytest.mark.asyncio
    async def test_list_sessions_returns_empty_for_no_sessions(
        self,
        manager: SessionManager,
    ) -> None:
        """list_sessions should return empty list when no sessions exist."""
        sessions = await manager.list_sessions()
        assert sessions == []

    @pytest.mark.asyncio
    async def test_list_sessions_skips_corrupted_files(
        self,
        manager: SessionManager,
        sample_state: YoloState,
        temp_sessions_dir: Path,
    ) -> None:
        """list_sessions should skip corrupted session files."""
        await manager.save_session(sample_state, session_id="session-valid")

        # Create a corrupted session file
        corrupted_path = temp_sessions_dir / "session-corrupted.json"
        corrupted_path.write_text("{ invalid json }")

        sessions = await manager.list_sessions()

        assert len(sessions) == 1
        assert sessions[0].session_id == "session-valid"

    @pytest.mark.asyncio
    async def test_get_active_session_id_returns_latest(
        self,
        manager: SessionManager,
        sample_state: YoloState,
    ) -> None:
        """get_active_session_id should return the most recently saved session."""
        await manager.save_session(sample_state, session_id="session-first")
        await manager.save_session(sample_state, session_id="session-second")

        active_id = await manager.get_active_session_id()

        assert active_id == "session-second"

    @pytest.mark.asyncio
    async def test_get_active_session_id_returns_none_when_empty(
        self,
        manager: SessionManager,
    ) -> None:
        """get_active_session_id should return None when no sessions exist."""
        active_id = await manager.get_active_session_id()
        assert active_id is None

    @pytest.mark.asyncio
    async def test_delete_session_removes_file(
        self,
        manager: SessionManager,
        sample_state: YoloState,
        temp_sessions_dir: Path,
    ) -> None:
        """delete_session should remove the session file."""
        session_id = await manager.save_session(sample_state)
        session_path = temp_sessions_dir / f"{session_id}.json"
        assert session_path.exists()

        deleted = await manager.delete_session(session_id)

        assert deleted is True
        assert not session_path.exists()

    @pytest.mark.asyncio
    async def test_delete_session_returns_false_for_missing(
        self,
        manager: SessionManager,
    ) -> None:
        """delete_session should return False for non-existent sessions."""
        deleted = await manager.delete_session("nonexistent-session")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_delete_session_clears_active_pointer(
        self,
        manager: SessionManager,
        sample_state: YoloState,
        temp_sessions_dir: Path,
    ) -> None:
        """delete_session should clear active pointer if deleting active session."""
        session_id = await manager.save_session(sample_state)
        active_file = temp_sessions_dir / "_active.json"
        assert active_file.exists()

        await manager.delete_session(session_id)

        assert not active_file.exists()

    @pytest.mark.asyncio
    async def test_save_preserves_created_at_on_update(
        self,
        manager: SessionManager,
        sample_state: YoloState,
    ) -> None:
        """save_session should preserve created_at when updating."""
        session_id = await manager.save_session(sample_state)

        # Load to get original created_at
        _, original_metadata = await manager.load_session(session_id)
        original_created = original_metadata.created_at

        # Save again (update)
        await manager.save_session(sample_state, session_id=session_id)

        # Load again and verify created_at preserved
        _, updated_metadata = await manager.load_session(session_id)

        assert updated_metadata.created_at == original_created
        assert updated_metadata.last_checkpoint > original_metadata.last_checkpoint


class TestSessionLoadError:
    """Tests for SessionLoadError exception."""

    def test_error_stores_session_id(self) -> None:
        """SessionLoadError should store the session ID."""
        error = SessionLoadError(
            message="Test error",
            session_id="test-session",
        )
        assert error.session_id == "test-session"

    def test_error_stores_cause(self) -> None:
        """SessionLoadError should store the underlying cause."""
        cause = ValueError("underlying error")
        error = SessionLoadError(
            message="Test error",
            session_id="test-session",
            cause=cause,
        )
        assert error.cause is cause

    def test_error_has_descriptive_message(self) -> None:
        """SessionLoadError should have a descriptive message."""
        error = SessionLoadError(
            message="Session file corrupted",
            session_id="test-session",
        )
        assert "corrupted" in str(error).lower()


class TestSessionNotFoundError:
    """Tests for SessionNotFoundError exception."""

    def test_is_subclass_of_session_load_error(self) -> None:
        """SessionNotFoundError should be a subclass of SessionLoadError."""
        error = SessionNotFoundError(
            message="Not found",
            session_id="test-session",
        )
        assert isinstance(error, SessionLoadError)

    def test_error_stores_session_id(self) -> None:
        """SessionNotFoundError should store the session ID."""
        error = SessionNotFoundError(
            message="Not found",
            session_id="missing-session",
        )
        assert error.session_id == "missing-session"


class TestCheckpointer:
    """Tests for Checkpointer class."""

    @pytest.mark.asyncio
    async def test_checkpoint_saves_state(
        self,
        checkpointer: Checkpointer,
        sample_state: YoloState,
    ) -> None:
        """checkpoint should save the state and return session ID."""
        session_id = await checkpointer.checkpoint(sample_state)

        assert session_id is not None
        assert session_id.startswith("session-")
        assert checkpointer.session_id == session_id

    @pytest.mark.asyncio
    async def test_checkpoint_updates_session_id(
        self,
        checkpointer: Checkpointer,
        sample_state: YoloState,
    ) -> None:
        """checkpoint should update the checkpointer's session_id."""
        assert checkpointer.session_id is None

        session_id = await checkpointer.checkpoint(sample_state)

        assert checkpointer.session_id == session_id

    @pytest.mark.asyncio
    async def test_checkpoint_reuses_session_id(
        self,
        checkpointer: Checkpointer,
        sample_state: YoloState,
    ) -> None:
        """Subsequent checkpoints should update the same session."""
        first_id = await checkpointer.checkpoint(sample_state)

        # Modify state and checkpoint again
        sample_state["current_agent"] = "pm"
        second_id = await checkpointer.checkpoint(sample_state)

        assert first_id == second_id

    @pytest.mark.asyncio
    async def test_resume_loads_state(
        self,
        checkpointer: Checkpointer,
        sample_state: YoloState,
    ) -> None:
        """resume should load the checkpointed state."""
        await checkpointer.checkpoint(sample_state)

        state, metadata = await checkpointer.resume()

        assert state["current_agent"] == "analyst"
        assert metadata.current_agent == "analyst"

    @pytest.mark.asyncio
    async def test_resume_uses_active_session(
        self,
        manager: SessionManager,
        sample_state: YoloState,
    ) -> None:
        """resume should use active session when session_id not set."""
        # Save a session first
        session_id = await manager.save_session(sample_state)

        # Create new checkpointer without session_id
        checkpointer = Checkpointer(manager)
        assert checkpointer.session_id is None

        _state, metadata = await checkpointer.resume()

        assert metadata.session_id == session_id
        assert checkpointer.session_id == session_id

    @pytest.mark.asyncio
    async def test_resume_raises_when_no_session(
        self,
        manager: SessionManager,
    ) -> None:
        """resume should raise SessionNotFoundError when no session exists."""
        checkpointer = Checkpointer(manager)

        with pytest.raises(SessionNotFoundError):
            await checkpointer.resume()

    @pytest.mark.asyncio
    async def test_checkpointer_with_initial_session_id(
        self,
        manager: SessionManager,
        sample_state: YoloState,
    ) -> None:
        """Checkpointer should accept initial session_id for resuming."""
        # Save initial session
        session_id = await manager.save_session(sample_state, session_id="session-test")

        # Create checkpointer with known session_id
        checkpointer = Checkpointer(manager, session_id=session_id)

        _state, metadata = await checkpointer.resume()

        assert metadata.session_id == "session-test"
