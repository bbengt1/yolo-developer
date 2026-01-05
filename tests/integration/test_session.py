"""Integration tests for session persistence.

Tests cover:
- Full session save/load with realistic orchestrator state
- Resume mid-sprint with agent position preserved
- Handoff context preserved through session resume
- Multiple sessions isolated correctly
- Auto-checkpointing with wrap_node integration
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from yolo_developer.orchestrator import (
    Checkpointer,
    Decision,
    SessionManager,
    SessionNotFoundError,
    YoloState,
    create_agent_message,
)
from yolo_developer.orchestrator.graph import wrap_node


class TestFullSessionSaveLoad:
    """Tests for full session save and load cycle."""

    @pytest.fixture
    def temp_sessions_dir(self, tmp_path: Path) -> Path:
        """Create a temporary sessions directory."""
        sessions_dir = tmp_path / ".yolo" / "sessions"
        sessions_dir.mkdir(parents=True)
        return sessions_dir

    @pytest.fixture
    def manager(self, temp_sessions_dir: Path) -> SessionManager:
        """Create a SessionManager with temporary directory."""
        return SessionManager(sessions_dir=temp_sessions_dir)

    @pytest.mark.asyncio
    async def test_save_and_load_full_orchestrator_state(
        self,
        manager: SessionManager,
    ) -> None:
        """Should save and restore complete orchestrator state."""
        # Create a realistic mid-sprint state
        state: YoloState = {
            "messages": [
                HumanMessage(content="Build a user authentication system"),
                AIMessage(
                    content="Analysis: Requirements identified",
                    additional_kwargs={"agent": "analyst"},
                ),
                AIMessage(
                    content="Story: As a user, I can log in",
                    additional_kwargs={"agent": "pm"},
                ),
            ],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [
                Decision(
                    agent="analyst",
                    summary="Security is priority",
                    rationale="User data protection required",
                ),
                Decision(
                    agent="pm",
                    summary="OAuth integration needed",
                    rationale="Modern auth requirement",
                ),
            ],
        }

        # Save session
        session_id = await manager.save_session(state)

        # Load session in a new context (simulating CLI restart)
        restored_state, _metadata = await manager.load_session(session_id)

        # Verify full state restoration
        assert len(restored_state["messages"]) == 3
        assert restored_state["current_agent"] == "architect"
        assert len(restored_state["decisions"]) == 2

        # Verify message content preserved
        assert restored_state["messages"][0].content == "Build a user authentication system"
        assert "Requirements identified" in restored_state["messages"][1].content

        # Verify decisions preserved
        decision_summaries = [d.summary for d in restored_state["decisions"]]
        assert "Security is priority" in decision_summaries
        assert "OAuth integration needed" in decision_summaries

    @pytest.mark.asyncio
    async def test_resume_preserves_agent_position(
        self,
        manager: SessionManager,
    ) -> None:
        """Resuming should restore the exact agent position."""
        # State interrupted mid-architect work
        state: YoloState = {
            "messages": [create_agent_message("Design in progress", agent="architect")],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
        }

        session_id = await manager.save_session(state)
        restored_state, metadata = await manager.load_session(session_id)

        # Should resume at architect
        assert restored_state["current_agent"] == "architect"
        assert metadata.current_agent == "architect"

    @pytest.mark.asyncio
    async def test_resume_preserves_handoff_context(
        self,
        manager: SessionManager,
    ) -> None:
        """Resuming should restore handoff context for receiving agent."""
        from yolo_developer.orchestrator import HandoffContext

        # State with pending handoff context
        handoff_ctx = HandoffContext(
            source_agent="pm",
            target_agent="architect",
            decisions=(
                Decision(
                    agent="pm",
                    summary="API-first design",
                    rationale="Mobile app planned",
                ),
            ),
            memory_refs=("req-001", "story-001"),
        )

        state: YoloState = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": handoff_ctx,
            "decisions": [],
        }

        session_id = await manager.save_session(state)
        restored_state, _ = await manager.load_session(session_id)

        # Handoff context should be fully restored
        restored_ctx = restored_state["handoff_context"]
        assert restored_ctx is not None
        assert restored_ctx.source_agent == "pm"
        assert restored_ctx.target_agent == "architect"
        assert len(restored_ctx.decisions) == 1
        assert restored_ctx.decisions[0].summary == "API-first design"
        assert "req-001" in restored_ctx.memory_refs
        assert "story-001" in restored_ctx.memory_refs


class TestMultipleSessionsIsolation:
    """Tests for multiple session isolation."""

    @pytest.fixture
    def temp_sessions_dir(self, tmp_path: Path) -> Path:
        """Create a temporary sessions directory."""
        sessions_dir = tmp_path / ".yolo" / "sessions"
        sessions_dir.mkdir(parents=True)
        return sessions_dir

    @pytest.fixture
    def manager(self, temp_sessions_dir: Path) -> SessionManager:
        """Create a SessionManager with temporary directory."""
        return SessionManager(sessions_dir=temp_sessions_dir)

    @pytest.mark.asyncio
    async def test_multiple_sessions_isolated(
        self,
        manager: SessionManager,
    ) -> None:
        """Multiple sessions should not interfere with each other."""
        # Create two different project sessions
        state_a: YoloState = {
            "messages": [HumanMessage(content="Project A requirements")],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [Decision(agent="analyst", summary="A decision", rationale="A reason")],
        }

        state_b: YoloState = {
            "messages": [HumanMessage(content="Project B requirements")],
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [Decision(agent="pm", summary="B decision", rationale="B reason")],
        }

        # Save both sessions
        session_a = await manager.save_session(state_a, session_id="session-project-a")
        session_b = await manager.save_session(state_b, session_id="session-project-b")

        # Load each independently
        restored_a, _ = await manager.load_session(session_a)
        restored_b, _ = await manager.load_session(session_b)

        # Verify isolation
        assert restored_a["messages"][0].content == "Project A requirements"
        assert restored_b["messages"][0].content == "Project B requirements"

        assert restored_a["current_agent"] == "analyst"
        assert restored_b["current_agent"] == "pm"

        assert restored_a["decisions"][0].summary == "A decision"
        assert restored_b["decisions"][0].summary == "B decision"

    @pytest.mark.asyncio
    async def test_list_sessions_shows_all(
        self,
        manager: SessionManager,
    ) -> None:
        """list_sessions should return all saved sessions."""
        state: YoloState = {
            "messages": [],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        await manager.save_session(state, session_id="session-1")
        await manager.save_session(state, session_id="session-2")
        await manager.save_session(state, session_id="session-3")

        sessions = await manager.list_sessions()

        assert len(sessions) == 3
        session_ids = [s.session_id for s in sessions]
        assert "session-1" in session_ids
        assert "session-2" in session_ids
        assert "session-3" in session_ids


class TestCheckpointerWithWrapNode:
    """Tests for checkpointer integration with wrap_node."""

    @pytest.fixture
    def temp_sessions_dir(self, tmp_path: Path) -> Path:
        """Create a temporary sessions directory."""
        sessions_dir = tmp_path / ".yolo" / "sessions"
        sessions_dir.mkdir(parents=True)
        return sessions_dir

    @pytest.fixture
    def manager(self, temp_sessions_dir: Path) -> SessionManager:
        """Create a SessionManager with temporary directory."""
        return SessionManager(sessions_dir=temp_sessions_dir)

    @pytest.fixture
    def checkpointer(self, manager: SessionManager) -> Checkpointer:
        """Create a Checkpointer for testing."""
        return Checkpointer(manager)

    @pytest.mark.asyncio
    async def test_wrap_node_auto_checkpoints(
        self,
        checkpointer: Checkpointer,
        temp_sessions_dir: Path,
    ) -> None:
        """wrap_node with checkpointer should auto-checkpoint after node completion."""

        async def analyst_node(state: YoloState) -> dict[str, Any]:
            return {
                "messages": [create_agent_message("Analysis done", agent="analyst")],
                "decisions": [
                    Decision(agent="analyst", summary="Test decision", rationale="Testing")
                ],
            }

        wrapped = wrap_node(
            analyst_node,
            agent_name="analyst",
            target_agent="pm",
            checkpointer=checkpointer,
        )

        state: YoloState = {
            "messages": [HumanMessage(content="Start")],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        # Execute wrapped node (should trigger checkpoint)
        await wrapped(state)

        # Verify checkpoint was created
        assert checkpointer.session_id is not None
        session_path = temp_sessions_dir / f"{checkpointer.session_id}.json"
        assert session_path.exists()

    @pytest.mark.asyncio
    async def test_checkpoint_contains_updated_state(
        self,
        checkpointer: Checkpointer,
        manager: SessionManager,
    ) -> None:
        """Checkpoint should contain the state after node execution."""

        async def pm_node(state: YoloState) -> dict[str, Any]:
            return {
                "messages": [create_agent_message("Story created", agent="pm")],
                "decisions": [
                    Decision(agent="pm", summary="User story added", rationale="Requirements")
                ],
            }

        wrapped = wrap_node(
            pm_node,
            agent_name="pm",
            target_agent="architect",
            checkpointer=checkpointer,
        )

        initial_state: YoloState = {
            "messages": [HumanMessage(content="Initial request")],
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [],
        }

        await wrapped(initial_state)

        # Load the checkpointed state
        restored_state, _metadata = await manager.load_session(checkpointer.session_id)

        # Should have both initial and new messages
        assert len(restored_state["messages"]) == 2
        assert restored_state["messages"][0].content == "Initial request"
        assert "Story created" in restored_state["messages"][1].content

        # Should have updated current_agent
        assert restored_state["current_agent"] == "architect"

        # Should have the decision
        assert len(restored_state["decisions"]) == 1
        assert restored_state["decisions"][0].summary == "User story added"

    @pytest.mark.asyncio
    async def test_multi_node_chain_checkpoints_accumulate(
        self,
        manager: SessionManager,
    ) -> None:
        """Multiple nodes with checkpointing should accumulate state correctly."""
        checkpointer = Checkpointer(manager)

        async def agent1(state: YoloState) -> dict[str, Any]:
            return {
                "messages": [create_agent_message("Step 1", agent="agent1")],
                "decisions": [Decision(agent="agent1", summary="D1", rationale="R1")],
            }

        async def agent2(state: YoloState) -> dict[str, Any]:
            return {
                "messages": [create_agent_message("Step 2", agent="agent2")],
                "decisions": [Decision(agent="agent2", summary="D2", rationale="R2")],
            }

        wrapped1 = wrap_node(agent1, "agent1", "agent2", checkpointer)
        wrapped2 = wrap_node(agent2, "agent2", "done", checkpointer)

        # Run first node
        state: YoloState = {
            "messages": [],
            "current_agent": "agent1",
            "handoff_context": None,
            "decisions": [],
        }

        result1 = await wrapped1(state)

        # Update state for second node
        state["messages"] = state["messages"] + result1.get("messages", [])
        state["handoff_context"] = result1["handoff_context"]
        state["current_agent"] = result1["current_agent"]
        state["decisions"] = list(state["decisions"]) + [
            d for d in result1.get("decisions", []) if isinstance(d, Decision)
        ]

        # Run second node
        await wrapped2(state)

        # Load final checkpoint
        final_state, _ = await manager.load_session(checkpointer.session_id)

        # Should have messages from both nodes
        assert len(final_state["messages"]) == 2

        # Should have decisions from both nodes
        assert len(final_state["decisions"]) == 2
        summaries = [d.summary for d in final_state["decisions"]]
        assert "D1" in summaries
        assert "D2" in summaries

        # Current agent should be final target
        assert final_state["current_agent"] == "done"


class TestCheckpointerResume:
    """Tests for resuming from checkpointed sessions."""

    @pytest.fixture
    def temp_sessions_dir(self, tmp_path: Path) -> Path:
        """Create a temporary sessions directory."""
        sessions_dir = tmp_path / ".yolo" / "sessions"
        sessions_dir.mkdir(parents=True)
        return sessions_dir

    @pytest.fixture
    def manager(self, temp_sessions_dir: Path) -> SessionManager:
        """Create a SessionManager with temporary directory."""
        return SessionManager(sessions_dir=temp_sessions_dir)

    @pytest.mark.asyncio
    async def test_resume_continues_from_checkpoint(
        self,
        manager: SessionManager,
    ) -> None:
        """Should be able to resume work from a checkpointed session."""
        # First run: checkpoint state mid-work
        checkpointer1 = Checkpointer(manager)

        state: YoloState = {
            "messages": [
                HumanMessage(content="Build feature X"),
                AIMessage(content="Analyzing", additional_kwargs={"agent": "analyst"}),
            ],
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [
                Decision(agent="analyst", summary="Feasible", rationale="Resources available")
            ],
        }

        session_id = await checkpointer1.checkpoint(state)

        # Simulate CLI restart - new checkpointer with same session
        checkpointer2 = Checkpointer(manager, session_id=session_id)
        restored_state, _metadata = await checkpointer2.resume()

        # Should resume exactly where we left off
        assert restored_state["current_agent"] == "pm"
        assert len(restored_state["messages"]) == 2
        assert len(restored_state["decisions"]) == 1

        # Can continue checkpointing
        restored_state["current_agent"] = "architect"
        restored_state["messages"] = [
            *list(restored_state["messages"]),
            AIMessage(content="Story written", additional_kwargs={"agent": "pm"}),
        ]

        await checkpointer2.checkpoint(restored_state)

        # Verify continuation was saved
        final_state, _ = await manager.load_session(session_id)
        assert final_state["current_agent"] == "architect"
        assert len(final_state["messages"]) == 3

    @pytest.mark.asyncio
    async def test_resume_uses_active_session(
        self,
        manager: SessionManager,
    ) -> None:
        """Resume without session_id should use the active session."""
        # Save a session (becomes active)
        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
        }
        saved_id = await manager.save_session(state)

        # New checkpointer without session_id
        checkpointer = Checkpointer(manager)
        restored_state, metadata = await checkpointer.resume()

        assert metadata.session_id == saved_id
        assert restored_state["current_agent"] == "dev"

    @pytest.mark.asyncio
    async def test_resume_raises_when_no_session(
        self,
        manager: SessionManager,
    ) -> None:
        """Resume should raise clear error when no session exists."""
        checkpointer = Checkpointer(manager)

        with pytest.raises(SessionNotFoundError) as exc_info:
            await checkpointer.resume()

        assert "no active session" in str(exc_info.value).lower()


class TestMemoryStoreAvailabilityAfterResume:
    """Tests for AC3: Memory stores remain available after session resume."""

    @pytest.fixture
    def temp_sessions_dir(self, tmp_path: Path) -> Path:
        """Create a temporary sessions directory."""
        sessions_dir = tmp_path / ".yolo" / "sessions"
        sessions_dir.mkdir(parents=True)
        return sessions_dir

    @pytest.fixture
    def temp_memory_dir(self, tmp_path: Path) -> Path:
        """Create a temporary memory directory."""
        memory_dir = tmp_path / ".yolo" / "memory"
        memory_dir.mkdir(parents=True)
        return memory_dir

    @pytest.fixture
    def manager(self, temp_sessions_dir: Path) -> SessionManager:
        """Create a SessionManager with temporary directory."""
        return SessionManager(sessions_dir=temp_sessions_dir)

    @pytest.mark.asyncio
    async def test_chromadb_available_after_resume(
        self,
        manager: SessionManager,
        temp_memory_dir: Path,
    ) -> None:
        """ChromaDB memory store should remain queryable after session resume."""
        from yolo_developer.memory import ChromaMemory

        # Create ChromaMemory with persistent storage
        memory = ChromaMemory(persist_directory=temp_memory_dir)

        # Store some embeddings before session save
        await memory.store_embedding(
            key="req-001",
            content="User authentication via OAuth2",
            metadata={"type": "requirement", "source": "prd.md"},
        )
        await memory.store_embedding(
            key="req-002",
            content="Password must be at least 8 characters",
            metadata={"type": "requirement", "source": "prd.md"},
        )

        # Create and save session state
        state: YoloState = {
            "messages": [HumanMessage(content="Implement auth system")],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [
                Decision(agent="analyst", summary="OAuth2 required", rationale="Security")
            ],
        }
        session_id = await manager.save_session(state)

        # Simulate CLI restart - create new memory instance pointing to same directory
        memory_after_resume = ChromaMemory(persist_directory=temp_memory_dir)

        # Resume session
        restored_state, _ = await manager.load_session(session_id)

        # Verify ChromaDB is still accessible and contains our data
        results = await memory_after_resume.search_similar("OAuth authentication", k=2)

        assert len(results) >= 1
        assert any(r.key == "req-001" for r in results)
        assert restored_state["current_agent"] == "architect"

    @pytest.mark.asyncio
    async def test_json_graph_store_available_after_resume(
        self,
        manager: SessionManager,
        temp_memory_dir: Path,
    ) -> None:
        """JSONGraphStore should remain queryable after session resume."""
        from yolo_developer.memory import JSONGraphStore

        graph_path = temp_memory_dir / "graph.json"

        # Create JSONGraphStore with persistent storage
        graph = JSONGraphStore(persist_path=graph_path)

        # Store some relationships before session save
        await graph.store_relationship("story-001", "req-001", "implements")
        await graph.store_relationship("story-001", "req-002", "implements")
        await graph.store_relationship("story-002", "story-001", "depends_on")

        # Create and save session state
        state: YoloState = {
            "messages": [HumanMessage(content="Build feature")],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
        }
        session_id = await manager.save_session(state)

        # Simulate CLI restart - create new graph instance pointing to same file
        graph_after_resume = JSONGraphStore(persist_path=graph_path)

        # Resume session
        restored_state, _ = await manager.load_session(session_id)

        # Verify JSONGraphStore is still accessible and contains our data
        relationships = await graph_after_resume.get_relationships("story-001")

        assert len(relationships) == 2
        targets = [r.target for r in relationships]
        assert "req-001" in targets
        assert "req-002" in targets
        assert restored_state["current_agent"] == "dev"

    @pytest.mark.asyncio
    async def test_both_memory_stores_available_after_checkpointer_resume(
        self,
        manager: SessionManager,
        temp_memory_dir: Path,
    ) -> None:
        """Both ChromaDB and JSONGraphStore should work after Checkpointer.resume()."""
        from yolo_developer.memory import ChromaMemory, JSONGraphStore

        # Set up persistent memory stores
        memory = ChromaMemory(persist_directory=temp_memory_dir)
        graph = JSONGraphStore(persist_path=temp_memory_dir / "graph.json")

        # Populate both stores
        await memory.store_embedding(
            key="design-001",
            content="REST API with JWT tokens",
            metadata={"type": "decision", "agent": "architect"},
        )
        await graph.store_relationship("design-001", "req-001", "addresses")

        # Create checkpoint
        checkpointer = Checkpointer(manager)
        state: YoloState = {
            "messages": [create_agent_message("Architecture designed", agent="architect")],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [
                Decision(agent="architect", summary="REST + JWT", rationale="Standard approach")
            ],
        }
        session_id = await checkpointer.checkpoint(state)

        # Simulate full CLI restart with new instances
        new_checkpointer = Checkpointer(manager, session_id=session_id)
        new_memory = ChromaMemory(persist_directory=temp_memory_dir)
        new_graph = JSONGraphStore(persist_path=temp_memory_dir / "graph.json")

        # Resume session
        restored_state, metadata = await new_checkpointer.resume()

        # Verify both stores are accessible
        memory_results = await new_memory.search_similar("JWT authentication", k=1)
        graph_results = await new_graph.get_relationships("design-001")

        # Session state restored
        assert restored_state["current_agent"] == "dev"
        assert metadata.session_id == session_id

        # Memory stores have data
        assert len(memory_results) >= 1
        assert memory_results[0].key == "design-001"
        assert len(graph_results) == 1
        assert graph_results[0].target == "req-001"
