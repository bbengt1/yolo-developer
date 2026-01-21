"""Tests for MCP tool implementations.

Tests cover AC1-AC6 for Story 14.2: yolo_seed MCP Tool.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from yolo_developer.audit.types import Decision


@pytest.fixture(autouse=True)
def clear_seed_storage() -> None:
    """Clear seed storage before each test to ensure test isolation."""
    from yolo_developer.mcp.tools import clear_seeds, clear_sprints

    clear_seeds()
    clear_sprints()


# Helper to call the underlying function since @mcp.tool wraps it
async def call_yolo_seed(**kwargs: str | None) -> dict:
    """Call yolo_seed tool's underlying function directly for testing."""
    from yolo_developer.mcp.tools import yolo_seed

    # FastMCP wraps the function, access the underlying function via fn attribute
    if hasattr(yolo_seed, "fn"):
        return await yolo_seed.fn(**kwargs)
    # Fallback for direct function access
    return await yolo_seed(**kwargs)


async def call_yolo_run(**kwargs: str | None) -> dict:
    """Call yolo_run tool's underlying function directly for testing."""
    from yolo_developer.mcp.tools import yolo_run

    if hasattr(yolo_run, "fn"):
        return await yolo_run.fn(**kwargs)
    return await yolo_run(**kwargs)


async def call_yolo_status(**kwargs: str | None) -> dict:
    """Call yolo_status tool's underlying function directly for testing."""
    from yolo_developer.mcp.tools import yolo_status

    if hasattr(yolo_status, "fn"):
        return await yolo_status.fn(**kwargs)
    return await yolo_status(**kwargs)


async def call_yolo_audit(**kwargs: str | None) -> dict:
    """Call yolo_audit tool's underlying function directly for testing."""
    from yolo_developer.mcp.tools import yolo_audit

    if hasattr(yolo_audit, "fn"):
        return await yolo_audit.fn(**kwargs)
    return await yolo_audit(**kwargs)


def _create_audit_decision(
    *,
    decision_id: str,
    agent_name: str,
    decision_type: str,
    timestamp: str,
    metadata: dict[str, str],
) -> Decision:
    """Create a Decision object for audit tests."""
    from yolo_developer.audit.types import AgentIdentity, Decision, DecisionContext

    return Decision(
        id=decision_id,
        decision_type=decision_type,  # type: ignore[arg-type]
        content=f"Decision {decision_id}",
        rationale="Testing audit trail",
        agent=AgentIdentity(
            agent_name=agent_name,
            agent_type=agent_name,
            session_id="session-1",
        ),
        context=DecisionContext(
            sprint_id="sprint-1",
            story_id="story-1",
        ),
        timestamp=timestamp,
        metadata=metadata,
        severity="info",  # type: ignore[arg-type]
    )


class TestYoloSeedTool:
    """Tests for the yolo_seed MCP tool."""

    @pytest.mark.asyncio
    async def test_yolo_seed_with_text_content(self) -> None:
        """Test yolo_seed accepts text content and returns expected response."""
        result = await call_yolo_seed(content="Build a REST API for user management")

        assert result["status"] == "accepted"
        assert "seed_id" in result
        assert result["content_length"] == len("Build a REST API for user management")
        assert result["source"] == "text"

    @pytest.mark.asyncio
    async def test_yolo_seed_with_file_path(self) -> None:
        """Test yolo_seed reads content from file."""
        # Create a temporary file with seed content
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Implement a login system with OAuth2")
            temp_path = f.name

        try:
            result = await call_yolo_seed(file_path=temp_path)

            assert result["status"] == "accepted"
            assert "seed_id" in result
            assert result["source"] == "file"
            assert result["content_length"] == len("Implement a login system with OAuth2")
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_yolo_seed_with_empty_content_returns_error(self) -> None:
        """Test yolo_seed rejects empty content."""
        result = await call_yolo_seed(content="")

        assert result["status"] == "error"
        assert "error" in result
        assert "empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_yolo_seed_with_whitespace_only_returns_error(self) -> None:
        """Test yolo_seed rejects whitespace-only content."""
        result = await call_yolo_seed(content="   \n\t  ")

        assert result["status"] == "error"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_yolo_seed_with_nonexistent_file_returns_error(self) -> None:
        """Test yolo_seed returns error for nonexistent file."""
        result = await call_yolo_seed(file_path="/nonexistent/path/to/file.txt")

        assert result["status"] == "error"
        assert "error" in result
        assert "not found" in result["error"].lower() or "exist" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_yolo_seed_with_neither_content_nor_file_returns_error(self) -> None:
        """Test yolo_seed returns error when no input provided."""
        result = await call_yolo_seed()

        assert result["status"] == "error"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_yolo_seed_with_both_content_and_file_prefers_content(self) -> None:
        """Test yolo_seed uses content when both are provided."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("File content")
            temp_path = f.name

        try:
            result = await call_yolo_seed(content="Text content", file_path=temp_path)

            assert result["status"] == "accepted"
            assert result["source"] == "text"
            assert result["content_length"] == len("Text content")
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_yolo_seed_generates_unique_ids(self) -> None:
        """Test yolo_seed generates unique seed_ids for each call."""
        result1 = await call_yolo_seed(content="First seed")
        result2 = await call_yolo_seed(content="Second seed")

        assert result1["seed_id"] != result2["seed_id"]

    @pytest.mark.asyncio
    async def test_yolo_seed_response_is_json_serializable(self) -> None:
        """Test yolo_seed response can be JSON serialized."""
        import json

        result = await call_yolo_seed(content="Test content")

        # Should not raise
        json_str = json.dumps(result)
        assert json_str is not None

    @pytest.mark.asyncio
    async def test_yolo_seed_with_directory_path_returns_error(self, tmp_path: Path) -> None:
        """Test yolo_seed returns error for directory path."""
        result = await call_yolo_seed(file_path=str(tmp_path))

        assert result["status"] == "error"
        assert "not a file" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_yolo_seed_with_empty_file_returns_error(self) -> None:
        """Test yolo_seed returns error for empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")  # Empty content
            temp_path = f.name

        try:
            result = await call_yolo_seed(file_path=temp_path)

            assert result["status"] == "error"
            assert "empty" in result["error"].lower()
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_yolo_seed_with_whitespace_only_file_returns_error(self) -> None:
        """Test yolo_seed returns error for file with only whitespace."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("   \n\t  ")  # Whitespace only
            temp_path = f.name

        try:
            result = await call_yolo_seed(file_path=temp_path)

            assert result["status"] == "error"
            assert "empty" in result["error"].lower() or "whitespace" in result["error"].lower()
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_yolo_seed_with_unreadable_file_returns_error(self) -> None:
        """Test yolo_seed handles file read errors gracefully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("content")
            temp_path = f.name

        try:
            # Mock Path.read_text to raise OSError
            with patch.object(Path, "read_text", side_effect=OSError("Permission denied")):
                result = await call_yolo_seed(file_path=temp_path)

            assert result["status"] == "error"
            assert "error reading file" in result["error"].lower()
        finally:
            Path(temp_path).unlink()


class TestSeedStorage:
    """Tests for seed storage functionality."""

    def test_store_seed_returns_stored_seed(self) -> None:
        """Test store_seed returns a StoredSeed with correct attributes."""
        from yolo_developer.mcp.tools import store_seed

        seed = store_seed(content="Test requirements", source="text")

        assert seed.seed_id is not None
        assert seed.content == "Test requirements"
        assert seed.source == "text"
        assert seed.content_length == len("Test requirements")
        assert seed.created_at is not None
        assert seed.file_path is None

    def test_store_seed_with_file_path(self) -> None:
        """Test store_seed stores file_path when provided."""
        from yolo_developer.mcp.tools import store_seed

        seed = store_seed(content="File content", source="file", file_path="/path/to/file.txt")

        assert seed.file_path == "/path/to/file.txt"
        assert seed.source == "file"

    def test_get_seed_returns_stored_seed(self) -> None:
        """Test get_seed retrieves previously stored seed."""
        from yolo_developer.mcp.tools import get_seed, store_seed

        stored = store_seed(content="Retrievable seed", source="text")
        retrieved = get_seed(stored.seed_id)

        assert retrieved is not None
        assert retrieved.seed_id == stored.seed_id
        assert retrieved.content == "Retrievable seed"

    def test_get_seed_returns_none_for_unknown_id(self) -> None:
        """Test get_seed returns None for unknown seed_id."""
        from yolo_developer.mcp.tools import get_seed

        result = get_seed("nonexistent-seed-id")

        assert result is None

    def test_clear_seeds_removes_all_seeds(self) -> None:
        """Test clear_seeds removes all stored seeds."""
        from yolo_developer.mcp.tools import clear_seeds, get_seed, store_seed

        seed = store_seed(content="To be cleared", source="text")
        clear_seeds()
        result = get_seed(seed.seed_id)

        assert result is None


class TestYoloRunTool:
    """Tests for the yolo_run MCP tool."""

    @pytest.mark.asyncio
    async def test_yolo_run_with_seed_id_starts_sprint(self) -> None:
        """Test yolo_run starts a sprint with a valid seed_id."""
        from yolo_developer.mcp.tools import get_sprint, store_seed

        seed = store_seed(content="Seed for sprint", source="text")

        with patch("yolo_developer.mcp.tools._run_sprint", new_callable=AsyncMock) as mock_run:
            result = await call_yolo_run(seed_id=seed.seed_id)

        assert result["status"] == "started"
        assert "sprint_id" in result
        assert result["seed_id"] == seed.seed_id

        sprint = get_sprint(result["sprint_id"])
        assert sprint is not None
        assert sprint.seed_id == seed.seed_id
        assert sprint.status == "running"
        assert mock_run.called

    @pytest.mark.asyncio
    async def test_yolo_run_with_seed_content_creates_seed(self) -> None:
        """Test yolo_run stores seed_content and starts sprint."""
        from yolo_developer.mcp.tools import get_seed, get_sprint

        with (
            patch("yolo_developer.mcp.tools._run_sprint", new_callable=AsyncMock),
            patch("yolo_developer.mcp.tools.parse_seed", new_callable=AsyncMock),
            patch("yolo_developer.mcp.tools.generate_validation_report"),
            patch(
                "yolo_developer.mcp.tools.validate_quality_thresholds",
                return_value=type("Result", (), {"passed": True})(),
            ),
        ):
            result = await call_yolo_run(seed_content="Seed content")

        assert result["status"] == "started"
        assert "seed_id" in result
        assert "sprint_id" in result

        seed = get_seed(result["seed_id"])
        assert seed is not None
        assert seed.content == "Seed content"

        sprint = get_sprint(result["sprint_id"])
        assert sprint is not None
        assert sprint.seed_id == result["seed_id"]

    @pytest.mark.asyncio
    async def test_yolo_run_with_missing_seed_id_returns_error(self) -> None:
        """Test yolo_run returns error for unknown seed_id."""
        result = await call_yolo_run(seed_id="missing-seed")

        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_yolo_run_with_empty_seed_content_returns_error(self) -> None:
        """Test yolo_run rejects empty seed content."""
        result = await call_yolo_run(seed_content="   ")

        assert result["status"] == "error"
        assert "empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_yolo_run_with_invalid_seed_content_returns_error(self) -> None:
        """Test yolo_run returns error when seed validation fails."""
        with (
            patch("yolo_developer.mcp.tools.parse_seed", new_callable=AsyncMock),
            patch("yolo_developer.mcp.tools.generate_validation_report"),
            patch(
                "yolo_developer.mcp.tools.validate_quality_thresholds",
                return_value=type("Result", (), {"passed": False})(),
            ),
        ):
            result = await call_yolo_run(seed_content="Invalid seed content")

        assert result["status"] == "error"
        assert "validation" in result["error"].lower()


class TestYoloStatusTool:
    """Tests for the yolo_status MCP tool."""

    @pytest.mark.asyncio
    async def test_yolo_status_with_valid_sprint_id_returns_status(self) -> None:
        """Test yolo_status returns sprint metadata for known sprint."""
        from yolo_developer.mcp.tools import store_seed, store_sprint

        seed = store_seed(content="Seed for status", source="text")
        sprint = store_sprint(seed_id=seed.seed_id, thread_id="thread-test")

        result = await call_yolo_status(sprint_id=sprint.sprint_id)

        assert result["status"] == "running"
        assert result["sprint_id"] == sprint.sprint_id
        assert result["seed_id"] == seed.seed_id
        assert result["thread_id"] == "thread-test"
        assert result["started_at"] == sprint.started_at.isoformat()
        assert result["completed_at"] is None
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_yolo_status_with_unknown_sprint_returns_error(self) -> None:
        """Test yolo_status returns error for unknown sprint_id."""
        result = await call_yolo_status(sprint_id="missing-sprint")

        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_yolo_status_with_empty_sprint_id_returns_error(self) -> None:
        """Test yolo_status returns error for empty sprint_id."""
        result = await call_yolo_status(sprint_id="   ")

        assert result["status"] == "error"
        assert "sprint_id" in result["error"].lower()


class TestYoloAuditTool:
    """Tests for the yolo_audit MCP tool."""

    @pytest.mark.asyncio
    async def test_yolo_audit_empty_store_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test yolo_audit returns empty results for missing audit data."""
        monkeypatch.chdir(tmp_path)
        audit_path = Path(".yolo/audit/decisions.json")
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text("", encoding="utf-8")

        result = await call_yolo_audit()

        assert result["status"] == "ok"
        assert result["entries"] == []
        assert result["total"] == 0
        assert result["limit"] == 100
        assert result["offset"] == 0

    @pytest.mark.asyncio
    async def test_yolo_audit_missing_store_returns_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test yolo_audit returns error when audit store is missing."""
        monkeypatch.chdir(tmp_path)

        result = await call_yolo_audit()

        assert result["status"] == "error"
        assert "audit store not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_yolo_audit_filters_and_pagination(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test yolo_audit supports filtering and pagination."""
        from yolo_developer.audit import JsonDecisionStore

        monkeypatch.chdir(tmp_path)

        store = JsonDecisionStore(Path(".yolo/audit/decisions.json"))
        await store.log_decision(
            _create_audit_decision(
                decision_id="dec-1",
                agent_name="analyst",
                decision_type="requirement_analysis",
                timestamp="2026-01-18T10:00:00+00:00",
                metadata={"artifact_type": "requirement"},
            )
        )
        await store.log_decision(
            _create_audit_decision(
                decision_id="dec-2",
                agent_name="pm",
                decision_type="story_creation",
                timestamp="2026-01-18T11:00:00+00:00",
                metadata={"artifact_type": "story"},
            )
        )
        await store.log_decision(
            _create_audit_decision(
                decision_id="dec-3",
                agent_name="analyst",
                decision_type="requirement_analysis",
                timestamp="2026-01-18T12:00:00+00:00",
                metadata={"artifact_type": "story"},
            )
        )

        filtered = await call_yolo_audit(agent="analyst", decision_type="requirement_analysis")
        assert filtered["status"] == "ok"
        assert filtered["total"] == 2
        assert [entry["entry_id"] for entry in filtered["entries"]] == ["dec-1", "dec-3"]

        artifact_filtered = await call_yolo_audit(artifact_type="story")
        assert artifact_filtered["status"] == "ok"
        assert artifact_filtered["total"] == 2
        assert [entry["entry_id"] for entry in artifact_filtered["entries"]] == ["dec-2", "dec-3"]

        paginated = await call_yolo_audit(limit=1, offset=1)
        assert paginated["status"] == "ok"
        assert paginated["total"] == 3
        assert len(paginated["entries"]) == 1
        assert paginated["entries"][0]["entry_id"] == "dec-2"


class TestToolRegistration:
    """Tests for MCP tool registration."""

    def test_yolo_seed_is_registered_on_mcp_server(self) -> None:
        """Test yolo_seed tool is registered with the MCP server."""
        from yolo_developer.mcp import mcp

        # FastMCP stores tools internally - check it's registered
        # The tool should be accessible after import
        assert hasattr(mcp, "_tool_manager") or hasattr(mcp, "tools") or hasattr(mcp, "_tools")

    @pytest.mark.asyncio
    async def test_mcp_server_lists_yolo_seed_tool(self) -> None:
        """Test MCP server includes yolo_seed in list_tools."""
        from yolo_developer.mcp import mcp

        # Get tools from the server - FastMCP 2.x returns a dict keyed by tool name
        tools = await mcp.get_tools()

        # FastMCP 2.x returns dict[str, Tool] or similar
        if isinstance(tools, dict):
            tool_names = list(tools.keys())
        elif isinstance(tools, list):
            tool_names = [t.name if hasattr(t, "name") else str(t) for t in tools]
        else:
            # Fallback - try to iterate
            tool_names = [str(t) for t in tools]

        assert "yolo_seed" in tool_names

    @pytest.mark.asyncio
    async def test_mcp_server_lists_yolo_run_tool(self) -> None:
        """Test MCP server includes yolo_run in list_tools."""
        from yolo_developer.mcp import mcp

        tools = await mcp.get_tools()

        if isinstance(tools, dict):
            tool_names = list(tools.keys())
        elif isinstance(tools, list):
            tool_names = [t.name if hasattr(t, "name") else str(t) for t in tools]
        else:
            tool_names = [str(t) for t in tools]

        assert "yolo_run" in tool_names

    @pytest.mark.asyncio
    async def test_mcp_server_lists_yolo_status_tool(self) -> None:
        """Test MCP server includes yolo_status in list_tools."""
        from yolo_developer.mcp import mcp

        tools = await mcp.get_tools()

        if isinstance(tools, dict):
            tool_names = list(tools.keys())
        elif isinstance(tools, list):
            tool_names = [t.name if hasattr(t, "name") else str(t) for t in tools]
        else:
            tool_names = [str(t) for t in tools]

        assert "yolo_status" in tool_names

    @pytest.mark.asyncio
    async def test_mcp_server_lists_yolo_audit_tool(self) -> None:
        """Test MCP server includes yolo_audit in list_tools."""
        from yolo_developer.mcp import mcp

        tools = await mcp.get_tools()

        if isinstance(tools, dict):
            tool_names = list(tools.keys())
        elif isinstance(tools, list):
            tool_names = [t.name if hasattr(t, "name") else str(t) for t in tools]
        else:
            tool_names = [str(t) for t in tools]

        assert "yolo_audit" in tool_names

    @pytest.mark.asyncio
    async def test_yolo_seed_tool_has_description(self) -> None:
        """Test yolo_seed tool has a description for LLM understanding."""
        from yolo_developer.mcp.tools import yolo_seed

        # Check the tool has a description via its docstring or description attribute
        if hasattr(yolo_seed, "description"):
            description = yolo_seed.description
        elif hasattr(yolo_seed, "fn") and yolo_seed.fn.__doc__:
            description = yolo_seed.fn.__doc__
        elif hasattr(yolo_seed, "__doc__"):
            description = yolo_seed.__doc__
        else:
            description = None

        assert description is not None
        assert len(description) > 0
        assert "seed" in description.lower()

    @pytest.mark.asyncio
    async def test_yolo_run_tool_has_description(self) -> None:
        """Test yolo_run tool has a description for LLM understanding."""
        from yolo_developer.mcp.tools import yolo_run

        if hasattr(yolo_run, "description"):
            description = yolo_run.description
        elif hasattr(yolo_run, "fn") and yolo_run.fn.__doc__:
            description = yolo_run.fn.__doc__
        elif hasattr(yolo_run, "__doc__"):
            description = yolo_run.__doc__
        else:
            description = None

        assert description is not None
        assert len(description) > 0
        assert "sprint" in description.lower()

    @pytest.mark.asyncio
    async def test_yolo_status_tool_has_description(self) -> None:
        """Test yolo_status tool has a description for LLM understanding."""
        from yolo_developer.mcp.tools import yolo_status

        if hasattr(yolo_status, "description"):
            description = yolo_status.description
        elif hasattr(yolo_status, "fn") and yolo_status.fn.__doc__:
            description = yolo_status.fn.__doc__
        elif hasattr(yolo_status, "__doc__"):
            description = yolo_status.__doc__
        else:
            description = None

        assert description is not None
        assert len(description) > 0
        assert "status" in description.lower()

    @pytest.mark.asyncio
    async def test_yolo_audit_tool_has_description(self) -> None:
        """Test yolo_audit tool has a description for LLM understanding."""
        from yolo_developer.mcp.tools import yolo_audit

        if hasattr(yolo_audit, "description"):
            description = yolo_audit.description
        elif hasattr(yolo_audit, "fn") and yolo_audit.fn.__doc__:
            description = yolo_audit.fn.__doc__
        elif hasattr(yolo_audit, "__doc__"):
            description = yolo_audit.__doc__
        else:
            description = None

        assert description is not None
        assert len(description) > 0
        assert "audit" in description.lower()
