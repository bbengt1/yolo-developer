"""Unit tests for orchestrator workflow module (Story 10.1).

Tests cover:
- Task 1: Workflow module structure (WorkflowConfig, agent registration)
- Task 2: StateGraph construction (build_workflow, nodes, edges)
- Task 3: Conditional routing functions
- Task 4: Checkpointing integration
- Task 5: Workflow execution interface
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:

    from yolo_developer.orchestrator.state import YoloState


# =============================================================================
# Task 1: Workflow Module Structure Tests
# =============================================================================


class TestWorkflowConfig:
    """Tests for WorkflowConfig dataclass."""

    def test_workflow_config_exists(self) -> None:
        """WorkflowConfig class should exist in workflow module."""
        from yolo_developer.orchestrator.workflow import WorkflowConfig

        assert WorkflowConfig is not None

    def test_workflow_config_default_values(self) -> None:
        """WorkflowConfig should have sensible defaults."""
        from yolo_developer.orchestrator.workflow import WorkflowConfig

        config = WorkflowConfig()
        assert config.entry_point == "analyst"
        assert config.enable_checkpointing is True

    def test_workflow_config_custom_values(self) -> None:
        """WorkflowConfig should accept custom values."""
        from yolo_developer.orchestrator.workflow import WorkflowConfig

        config = WorkflowConfig(
            entry_point="pm",
            enable_checkpointing=False,
        )
        assert config.entry_point == "pm"
        assert config.enable_checkpointing is False

    def test_workflow_config_is_frozen(self) -> None:
        """WorkflowConfig should be immutable (frozen dataclass)."""
        from yolo_developer.orchestrator.workflow import WorkflowConfig

        config = WorkflowConfig()
        with pytest.raises(AttributeError):
            config.entry_point = "dev"  # type: ignore[misc]


class TestAgentNodeRegistry:
    """Tests for agent node registration system."""

    def test_get_default_agent_nodes_returns_dict(self) -> None:
        """get_default_agent_nodes should return a dict of agent name to node."""
        from yolo_developer.orchestrator.workflow import get_default_agent_nodes

        nodes = get_default_agent_nodes()
        assert isinstance(nodes, dict)

    def test_get_default_agent_nodes_contains_all_agents(self) -> None:
        """get_default_agent_nodes should include all required agents."""
        from yolo_developer.orchestrator.workflow import get_default_agent_nodes

        nodes = get_default_agent_nodes()
        expected_agents = {"analyst", "pm", "architect", "dev", "tea"}
        assert set(nodes.keys()) == expected_agents

    def test_get_default_agent_nodes_values_are_callable(self) -> None:
        """Each agent node should be a callable function."""
        from yolo_developer.orchestrator.workflow import get_default_agent_nodes

        nodes = get_default_agent_nodes()
        for agent_name, node_fn in nodes.items():
            assert callable(node_fn), f"{agent_name} node should be callable"


# =============================================================================
# Task 2: StateGraph Construction Tests
# =============================================================================


class TestBuildWorkflow:
    """Tests for build_workflow function."""

    def test_build_workflow_exists(self) -> None:
        """build_workflow function should exist."""
        from yolo_developer.orchestrator.workflow import build_workflow

        assert callable(build_workflow)

    def test_build_workflow_returns_compiled_graph(self) -> None:
        """build_workflow should return a compiled StateGraph."""
        from yolo_developer.orchestrator.workflow import build_workflow

        graph = build_workflow()
        # Compiled graphs have ainvoke and astream methods
        assert hasattr(graph, "ainvoke")
        assert hasattr(graph, "astream")

    def test_build_workflow_with_custom_config(self) -> None:
        """build_workflow should accept custom WorkflowConfig."""
        from yolo_developer.orchestrator.workflow import (
            WorkflowConfig,
            build_workflow,
        )

        config = WorkflowConfig(entry_point="analyst")
        graph = build_workflow(config=config)
        assert graph is not None

    def test_build_workflow_with_custom_nodes(self) -> None:
        """build_workflow should accept custom node functions."""
        from yolo_developer.orchestrator.workflow import build_workflow

        async def custom_analyst(state: YoloState) -> dict[str, Any]:
            return {"messages": []}

        async def custom_pm(state: YoloState) -> dict[str, Any]:
            return {"messages": []}

        async def custom_architect(state: YoloState) -> dict[str, Any]:
            return {"messages": []}

        async def custom_dev(state: YoloState) -> dict[str, Any]:
            return {"messages": []}

        async def custom_tea(state: YoloState) -> dict[str, Any]:
            return {"messages": []}

        # Must provide all required nodes for graph to compile
        custom_nodes = {
            "analyst": custom_analyst,
            "pm": custom_pm,
            "architect": custom_architect,
            "dev": custom_dev,
            "tea": custom_tea,
        }
        graph = build_workflow(nodes=custom_nodes)
        assert graph is not None


class TestGraphNodes:
    """Tests for graph node configuration."""

    def test_graph_has_analyst_node(self) -> None:
        """Graph should have analyst node."""
        from yolo_developer.orchestrator.workflow import build_workflow

        graph = build_workflow()
        # Check nodes in the graph structure
        assert "analyst" in graph.nodes

    def test_graph_has_pm_node(self) -> None:
        """Graph should have pm node."""
        from yolo_developer.orchestrator.workflow import build_workflow

        graph = build_workflow()
        assert "pm" in graph.nodes

    def test_graph_has_architect_node(self) -> None:
        """Graph should have architect node."""
        from yolo_developer.orchestrator.workflow import build_workflow

        graph = build_workflow()
        assert "architect" in graph.nodes

    def test_graph_has_dev_node(self) -> None:
        """Graph should have dev node."""
        from yolo_developer.orchestrator.workflow import build_workflow

        graph = build_workflow()
        assert "dev" in graph.nodes

    def test_graph_has_tea_node(self) -> None:
        """Graph should have tea node."""
        from yolo_developer.orchestrator.workflow import build_workflow

        graph = build_workflow()
        assert "tea" in graph.nodes


# =============================================================================
# Task 3: Conditional Routing Tests
# =============================================================================


class TestRouteAfterAnalyst:
    """Tests for route_after_analyst routing function."""

    def test_route_after_analyst_exists(self) -> None:
        """route_after_analyst function should exist."""
        from yolo_developer.orchestrator.workflow import route_after_analyst

        assert callable(route_after_analyst)

    def test_route_after_analyst_returns_pm_by_default(self) -> None:
        """route_after_analyst should return 'pm' for normal flow."""
        from yolo_developer.orchestrator.workflow import route_after_analyst

        state: YoloState = {
            "messages": [],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }
        result = route_after_analyst(state)
        assert result == "pm"

    def test_route_after_analyst_escalates_when_flagged(self) -> None:
        """route_after_analyst should return 'escalate' when escalate_to_human is set."""
        from yolo_developer.orchestrator.workflow import route_after_analyst

        state: YoloState = {
            "messages": [],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
            "escalate_to_human": True,  # type: ignore[typeddict-unknown-key]
        }
        result = route_after_analyst(state)
        assert result == "escalate"


class TestRouteAfterPm:
    """Tests for route_after_pm routing function."""

    def test_route_after_pm_exists(self) -> None:
        """route_after_pm function should exist."""
        from yolo_developer.orchestrator.workflow import route_after_pm

        assert callable(route_after_pm)

    def test_route_after_pm_returns_architect_when_needed(self) -> None:
        """route_after_pm should return 'architect' when needs_architecture is True."""
        from yolo_developer.orchestrator.workflow import route_after_pm

        state: YoloState = {
            "messages": [],
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [],
            "needs_architecture": True,  # type: ignore[typeddict-unknown-key]
        }
        result = route_after_pm(state)
        assert result == "architect"

    def test_route_after_pm_returns_dev_by_default(self) -> None:
        """route_after_pm should return 'dev' when no architecture needed."""
        from yolo_developer.orchestrator.workflow import route_after_pm

        state: YoloState = {
            "messages": [],
            "current_agent": "pm",
            "handoff_context": None,
            "decisions": [],
        }
        result = route_after_pm(state)
        assert result == "dev"


class TestRouteAfterArchitect:
    """Tests for route_after_architect routing function."""

    def test_route_after_architect_exists(self) -> None:
        """route_after_architect function should exist."""
        from yolo_developer.orchestrator.workflow import route_after_architect

        assert callable(route_after_architect)

    def test_route_after_architect_returns_dev(self) -> None:
        """route_after_architect should always return 'dev'."""
        from yolo_developer.orchestrator.workflow import route_after_architect

        state: YoloState = {
            "messages": [],
            "current_agent": "architect",
            "handoff_context": None,
            "decisions": [],
        }
        result = route_after_architect(state)
        assert result == "dev"


class TestRouteAfterDev:
    """Tests for route_after_dev routing function."""

    def test_route_after_dev_exists(self) -> None:
        """route_after_dev function should exist."""
        from yolo_developer.orchestrator.workflow import route_after_dev

        assert callable(route_after_dev)

    def test_route_after_dev_returns_tea(self) -> None:
        """route_after_dev should always return 'tea'."""
        from yolo_developer.orchestrator.workflow import route_after_dev

        state: YoloState = {
            "messages": [],
            "current_agent": "dev",
            "handoff_context": None,
            "decisions": [],
        }
        result = route_after_dev(state)
        assert result == "tea"


class TestRouteAfterTea:
    """Tests for route_after_tea routing function."""

    def test_route_after_tea_exists(self) -> None:
        """route_after_tea function should exist."""
        from yolo_developer.orchestrator.workflow import route_after_tea

        assert callable(route_after_tea)

    def test_route_after_tea_returns_end_when_ready(self) -> None:
        """route_after_tea should return END when deployment_ready is True."""
        from yolo_developer.orchestrator.workflow import route_after_tea

        state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
            "deployment_ready": True,  # type: ignore[typeddict-unknown-key]
        }
        result = route_after_tea(state)
        assert result == "__end__"

    def test_route_after_tea_returns_dev_when_blocked(self) -> None:
        """route_after_tea should return 'dev' when gate_blocked is True."""
        from yolo_developer.orchestrator.workflow import route_after_tea

        state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
            "gate_blocked": True,  # type: ignore[typeddict-unknown-key]
        }
        result = route_after_tea(state)
        assert result == "dev"

    def test_route_after_tea_returns_end_by_default(self) -> None:
        """route_after_tea should return END by default."""
        from yolo_developer.orchestrator.workflow import route_after_tea

        state: YoloState = {
            "messages": [],
            "current_agent": "tea",
            "handoff_context": None,
            "decisions": [],
        }
        result = route_after_tea(state)
        assert result == "__end__"


# =============================================================================
# Task 4: Checkpointing Integration Tests
# =============================================================================


class TestCheckpointingIntegration:
    """Tests for checkpointing integration."""

    def test_create_workflow_with_checkpointing_exists(self) -> None:
        """create_workflow_with_checkpointing function should exist."""
        from yolo_developer.orchestrator.workflow import (
            create_workflow_with_checkpointing,
        )

        assert callable(create_workflow_with_checkpointing)

    def test_create_workflow_with_memory_saver(self) -> None:
        """create_workflow_with_checkpointing should work with MemorySaver."""
        from langgraph.checkpoint.memory import MemorySaver

        from yolo_developer.orchestrator.workflow import (
            create_workflow_with_checkpointing,
        )

        checkpointer = MemorySaver()
        graph = create_workflow_with_checkpointing(checkpointer=checkpointer)
        assert graph is not None
        assert hasattr(graph, "ainvoke")

    def test_build_workflow_checkpointing_disabled(self) -> None:
        """build_workflow with checkpointing disabled should still work."""
        from yolo_developer.orchestrator.workflow import (
            WorkflowConfig,
            build_workflow,
        )

        config = WorkflowConfig(enable_checkpointing=False)
        graph = build_workflow(config=config)
        assert graph is not None

    @pytest.mark.asyncio
    async def test_run_workflow_auto_creates_checkpointer_when_enabled(self) -> None:
        """run_workflow should auto-create MemorySaver when enable_checkpointing=True."""
        from yolo_developer.orchestrator.workflow import (
            WorkflowConfig,
            run_workflow,
        )

        async def mock_analyst(state: YoloState) -> dict[str, Any]:
            return {"messages": [], "deployment_ready": True}

        async def mock_pm(state: YoloState) -> dict[str, Any]:
            return {"messages": [], "deployment_ready": True}

        async def mock_architect(state: YoloState) -> dict[str, Any]:
            return {"messages": [], "deployment_ready": True}

        async def mock_dev(state: YoloState) -> dict[str, Any]:
            return {"messages": [], "deployment_ready": True}

        async def mock_tea(state: YoloState) -> dict[str, Any]:
            return {"messages": [], "deployment_ready": True}

        mock_nodes = {
            "analyst": mock_analyst,
            "pm": mock_pm,
            "architect": mock_architect,
            "dev": mock_dev,
            "tea": mock_tea,
        }

        initial_state: YoloState = {
            "messages": [],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        # enable_checkpointing=True (default) should work without explicit checkpointer
        config = WorkflowConfig(enable_checkpointing=True)
        result = await run_workflow(
            initial_state=initial_state,
            config=config,
            nodes=mock_nodes,
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_run_workflow_no_checkpointer_when_disabled(self) -> None:
        """run_workflow should not create checkpointer when enable_checkpointing=False."""
        from yolo_developer.orchestrator.workflow import (
            WorkflowConfig,
            run_workflow,
        )

        async def mock_analyst(state: YoloState) -> dict[str, Any]:
            return {"messages": [], "deployment_ready": True}

        async def mock_pm(state: YoloState) -> dict[str, Any]:
            return {"messages": [], "deployment_ready": True}

        async def mock_architect(state: YoloState) -> dict[str, Any]:
            return {"messages": [], "deployment_ready": True}

        async def mock_dev(state: YoloState) -> dict[str, Any]:
            return {"messages": [], "deployment_ready": True}

        async def mock_tea(state: YoloState) -> dict[str, Any]:
            return {"messages": [], "deployment_ready": True}

        mock_nodes = {
            "analyst": mock_analyst,
            "pm": mock_pm,
            "architect": mock_architect,
            "dev": mock_dev,
            "tea": mock_tea,
        }

        initial_state: YoloState = {
            "messages": [],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        # enable_checkpointing=False should work without checkpointer
        config = WorkflowConfig(enable_checkpointing=False)
        result = await run_workflow(
            initial_state=initial_state,
            config=config,
            nodes=mock_nodes,
        )
        assert result is not None


# =============================================================================
# Task 5: Workflow Execution Interface Tests
# =============================================================================


class TestRunWorkflow:
    """Tests for run_workflow execution function."""

    def test_run_workflow_exists(self) -> None:
        """run_workflow function should exist."""
        from yolo_developer.orchestrator.workflow import run_workflow

        assert callable(run_workflow)

    @pytest.mark.asyncio
    async def test_run_workflow_with_mocked_agents(self) -> None:
        """run_workflow should execute with mocked agent nodes."""
        from yolo_developer.orchestrator.workflow import (
            run_workflow,
        )

        # Create mock nodes that immediately complete
        async def mock_analyst(state: YoloState) -> dict[str, Any]:
            return {
                "messages": [],
                "current_agent": "pm",
                "deployment_ready": True,  # Skip to end
            }

        async def mock_pm(state: YoloState) -> dict[str, Any]:
            return {"messages": [], "current_agent": "dev", "deployment_ready": True}

        async def mock_architect(state: YoloState) -> dict[str, Any]:
            return {"messages": [], "current_agent": "dev", "deployment_ready": True}

        async def mock_dev(state: YoloState) -> dict[str, Any]:
            return {"messages": [], "current_agent": "tea", "deployment_ready": True}

        async def mock_tea(state: YoloState) -> dict[str, Any]:
            return {"messages": [], "current_agent": "tea", "deployment_ready": True}

        mock_nodes = {
            "analyst": mock_analyst,
            "pm": mock_pm,
            "architect": mock_architect,
            "dev": mock_dev,
            "tea": mock_tea,
        }

        initial_state: YoloState = {
            "messages": [],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        result = await run_workflow(
            initial_state=initial_state,
            nodes=mock_nodes,
        )
        assert result is not None
        assert "messages" in result


class TestStreamWorkflow:
    """Tests for stream_workflow streaming function."""

    def test_stream_workflow_exists(self) -> None:
        """stream_workflow function should exist."""
        from yolo_developer.orchestrator.workflow import stream_workflow

        assert callable(stream_workflow)

    @pytest.mark.asyncio
    async def test_stream_workflow_yields_events(self) -> None:
        """stream_workflow should yield events during execution."""
        from yolo_developer.orchestrator.workflow import stream_workflow

        # Create mock nodes
        async def mock_analyst(state: YoloState) -> dict[str, Any]:
            return {"messages": [], "deployment_ready": True}

        async def mock_pm(state: YoloState) -> dict[str, Any]:
            return {"messages": [], "deployment_ready": True}

        async def mock_architect(state: YoloState) -> dict[str, Any]:
            return {"messages": [], "deployment_ready": True}

        async def mock_dev(state: YoloState) -> dict[str, Any]:
            return {"messages": [], "deployment_ready": True}

        async def mock_tea(state: YoloState) -> dict[str, Any]:
            return {"messages": [], "deployment_ready": True}

        mock_nodes = {
            "analyst": mock_analyst,
            "pm": mock_pm,
            "architect": mock_architect,
            "dev": mock_dev,
            "tea": mock_tea,
        }

        initial_state: YoloState = {
            "messages": [],
            "current_agent": "analyst",
            "handoff_context": None,
            "decisions": [],
        }

        events = []
        async for event in stream_workflow(
            initial_state=initial_state,
            nodes=mock_nodes,
        ):
            events.append(event)

        assert len(events) > 0


class TestCreateInitialState:
    """Tests for initial state creation helper."""

    def test_create_initial_state_exists(self) -> None:
        """create_initial_state function should exist."""
        from yolo_developer.orchestrator.workflow import create_initial_state

        assert callable(create_initial_state)

    def test_create_initial_state_returns_valid_yolostate(self) -> None:
        """create_initial_state should return a valid YoloState."""
        from yolo_developer.orchestrator.workflow import create_initial_state

        state = create_initial_state()
        assert "messages" in state
        assert "current_agent" in state
        assert "handoff_context" in state
        assert "decisions" in state

    def test_create_initial_state_default_agent(self) -> None:
        """create_initial_state should default to analyst."""
        from yolo_developer.orchestrator.workflow import create_initial_state

        state = create_initial_state()
        assert state["current_agent"] == "analyst"

    def test_create_initial_state_with_custom_agent(self) -> None:
        """create_initial_state should accept custom starting agent."""
        from yolo_developer.orchestrator.workflow import create_initial_state

        state = create_initial_state(starting_agent="pm")
        assert state["current_agent"] == "pm"

    def test_create_initial_state_with_initial_messages(self) -> None:
        """create_initial_state should accept initial messages."""
        from langchain_core.messages import HumanMessage

        from yolo_developer.orchestrator.workflow import create_initial_state

        messages = [HumanMessage(content="Build an app")]
        state = create_initial_state(messages=messages)
        assert len(state["messages"]) == 1
        assert state["messages"][0].content == "Build an app"
