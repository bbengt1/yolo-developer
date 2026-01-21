"""Integration tests for the MCP server with FastMCP Client.

Tests cover:
- Story 14.1 AC6: FastMCP testing patterns with mocked clients
- Story 14.2 AC7: Integration tests for yolo_seed tool
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import Client

from yolo_developer.mcp import mcp
from yolo_developer.mcp.tools import (
    clear_seeds,
    clear_sprints,
    get_seed,
    get_sprint,
    store_seed,
    store_sprint,
)


@pytest.fixture(autouse=True)
def clear_seed_storage() -> None:
    """Clear seed storage before each test."""
    clear_seeds()
    clear_sprints()


@pytest.mark.asyncio
async def test_server_responds_to_client() -> None:
    """Test server can handle client requests via FastMCP Client."""
    async with Client(mcp) as client:
        tools = await client.list_tools()
        assert isinstance(tools, list)


@pytest.mark.asyncio
async def test_server_lists_yolo_seed_tool() -> None:
    """Test server includes yolo_seed in available tools."""
    async with Client(mcp) as client:
        tools = await client.list_tools()
        tool_names = [t.name for t in tools]
        assert "yolo_seed" in tool_names


@pytest.mark.asyncio
async def test_server_lists_yolo_run_tool() -> None:
    """Test server includes yolo_run in available tools."""
    async with Client(mcp) as client:
        tools = await client.list_tools()
        tool_names = [t.name for t in tools]
        assert "yolo_run" in tool_names


@pytest.mark.asyncio
async def test_server_lists_yolo_status_tool() -> None:
    """Test server includes yolo_status in available tools."""
    async with Client(mcp) as client:
        tools = await client.list_tools()
        tool_names = [t.name for t in tools]
        assert "yolo_status" in tool_names


@pytest.mark.asyncio
async def test_client_can_ping_server() -> None:
    """Test client can establish connection with server."""
    async with Client(mcp) as client:
        # If we get here without error, connection works
        # List resources as a basic connectivity check
        resources = await client.list_resources()
        assert isinstance(resources, list)


class TestYoloSeedIntegration:
    """Integration tests for yolo_seed MCP tool via FastMCP Client."""

    @pytest.mark.asyncio
    async def test_yolo_seed_via_mcp_client_with_text(self) -> None:
        """Test yolo_seed through FastMCP Client with text content."""
        async with Client(mcp) as client:
            result = await client.call_tool("yolo_seed", {"content": "Build a REST API"})

            # FastMCP returns tool result content
            assert_result_accepted(result)

    @pytest.mark.asyncio
    async def test_yolo_seed_via_mcp_client_with_file(self) -> None:
        """Test yolo_seed through FastMCP Client with file input."""
        # Create a temporary file
        file_content = "File-based seed requirements"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(file_content)
            temp_path = f.name

        try:
            async with Client(mcp) as client:
                result = await client.call_tool("yolo_seed", {"file_path": temp_path})

                assert_result_accepted(result, source="file")

                # Verify seed was stored correctly (Issue 7 fix)
                seed_id = extract_seed_id(result)
                assert seed_id is not None
                stored_seed = get_seed(seed_id)
                assert stored_seed is not None
                assert stored_seed.source == "file"
                assert stored_seed.file_path == temp_path
                assert stored_seed.content == file_content
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_yolo_seed_full_flow_seed_and_retrieve(self) -> None:
        """Test full flow: seed via MCP client, then retrieve stored seed."""
        seed_content = "Complete feature: User authentication with OAuth2"

        async with Client(mcp) as client:
            # Seed via MCP
            result = await client.call_tool("yolo_seed", {"content": seed_content})

            # Extract seed_id from result
            seed_id = extract_seed_id(result)
            assert seed_id is not None

            # Verify seed was stored correctly
            stored_seed = get_seed(seed_id)
            assert stored_seed is not None
            assert stored_seed.content == seed_content
            assert stored_seed.source == "text"

    @pytest.mark.asyncio
    async def test_yolo_seed_validation_error_via_client(self) -> None:
        """Test yolo_seed returns error for invalid input via client."""
        async with Client(mcp) as client:
            result = await client.call_tool("yolo_seed", {"content": ""})

            # Should return error status
            assert_result_error(result)

    @pytest.mark.asyncio
    async def test_yolo_seed_tool_has_parameters(self) -> None:
        """Test yolo_seed tool exposes parameters for MCP clients."""
        async with Client(mcp) as client:
            tools = await client.list_tools()
            yolo_seed_tool = next((t for t in tools if t.name == "yolo_seed"), None)

            assert yolo_seed_tool is not None
            # Tool should have input schema for parameters
            assert yolo_seed_tool.inputSchema is not None


class TestYoloRunIntegration:
    """Integration tests for yolo_run MCP tool via FastMCP Client."""

    @pytest.mark.asyncio
    async def test_yolo_run_via_client_with_seed_id(self) -> None:
        """Test yolo_run starts sprint via MCP client."""
        seed = store_seed(content="Seed for MCP run", source="text")

        with patch("yolo_developer.mcp.tools._run_sprint", new_callable=AsyncMock):
            async with Client(mcp) as client:
                result = await client.call_tool("yolo_run", {"seed_id": seed.seed_id})

        assert_result_started(result)
        sprint_id = extract_sprint_id(result)
        assert sprint_id is not None
        sprint = get_sprint(sprint_id)
        assert sprint is not None
        assert sprint.seed_id == seed.seed_id

    @pytest.mark.asyncio
    async def test_yolo_run_with_seed_content_rejects_invalid_seed(self) -> None:
        """Test yolo_run rejects invalid seed content via MCP client."""
        with (
            patch("yolo_developer.mcp.tools.parse_seed", new_callable=AsyncMock),
            patch("yolo_developer.mcp.tools.generate_validation_report"),
            patch(
                "yolo_developer.mcp.tools.validate_quality_thresholds",
                return_value=type("Result", (), {"passed": False})(),
            ),
        ):
            async with Client(mcp) as client:
                result = await client.call_tool(
                    "yolo_run",
                    {"seed_content": "Invalid seed content"},
                )

        assert_result_error(result)

    @pytest.mark.asyncio
    async def test_yolo_run_missing_seed_returns_error(self) -> None:
        """Test yolo_run returns error for unknown seed_id via client."""
        async with Client(mcp) as client:
            result = await client.call_tool("yolo_run", {"seed_id": "missing-seed"})

        assert_result_error(result)


class TestYoloStatusIntegration:
    """Integration tests for yolo_status MCP tool via FastMCP Client."""

    @pytest.mark.asyncio
    async def test_yolo_status_via_client_returns_status(self) -> None:
        """Test yolo_status returns sprint metadata via MCP client."""
        seed = store_seed(content="Seed for status", source="text")
        sprint = get_sprint(
            store_sprint(
                seed_id=seed.seed_id,
                thread_id="thread-integration",
            ).sprint_id
        )
        assert sprint is not None

        async with Client(mcp) as client:
            result = await client.call_tool(
                "yolo_status",
                {"sprint_id": sprint.sprint_id},
            )

        data = extract_result_data(result)
        assert data.get("status") == "running"
        assert data.get("sprint_id") == sprint.sprint_id
        assert data.get("seed_id") == seed.seed_id
        assert data.get("thread_id") == "thread-integration"
        assert data.get("started_at") == sprint.started_at.isoformat()
        assert data.get("completed_at") is None
        assert data.get("error") is None

    @pytest.mark.asyncio
    async def test_yolo_status_unknown_sprint_returns_error(self) -> None:
        """Test yolo_status returns error for unknown sprint_id via client."""
        async with Client(mcp) as client:
            result = await client.call_tool("yolo_status", {"sprint_id": "missing-sprint"})

        assert_result_error(result)


def assert_result_accepted(result: Any, source: str = "text") -> None:
    """Assert that the tool result indicates accepted status."""
    # FastMCP wraps tool results - extract content
    if hasattr(result, "content"):
        # Single content item
        content = result.content
        if isinstance(content, list) and len(content) > 0:
            content = content[0]
        if hasattr(content, "text"):
            import json

            data = json.loads(content.text)
        else:
            data = content
    elif isinstance(result, dict):
        data = result
    elif isinstance(result, list) and len(result) > 0:
        # List of content items
        content = result[0]
        if hasattr(content, "text"):
            import json

            data = json.loads(content.text)
        else:
            data = content
    else:
        raise AssertionError(f"Unexpected result type: {type(result)}")

    assert data.get("status") == "accepted", f"Expected accepted status, got: {data}"
    assert "seed_id" in data, f"Expected seed_id in result: {data}"
    assert data.get("source") == source, f"Expected source={source}, got: {data}"


def assert_result_error(result: Any) -> None:
    """Assert that the tool result indicates error status."""
    if hasattr(result, "content"):
        content = result.content
        if isinstance(content, list) and len(content) > 0:
            content = content[0]
        if hasattr(content, "text"):
            import json

            data = json.loads(content.text)
        else:
            data = content
    elif isinstance(result, dict):
        data = result
    elif isinstance(result, list) and len(result) > 0:
        content = result[0]
        if hasattr(content, "text"):
            import json

            data = json.loads(content.text)
        else:
            data = content
    else:
        raise AssertionError(f"Unexpected result type: {type(result)}")

    assert data.get("status") == "error", f"Expected error status, got: {data}"
    assert "error" in data, f"Expected error message in result: {data}"


def assert_result_started(result: Any) -> None:
    """Assert that the tool result indicates sprint started."""
    if hasattr(result, "content"):
        content = result.content
        if isinstance(content, list) and len(content) > 0:
            content = content[0]
        if hasattr(content, "text"):
            import json

            data = json.loads(content.text)
        else:
            data = content
    elif isinstance(result, dict):
        data = result
    elif isinstance(result, list) and len(result) > 0:
        content = result[0]
        if hasattr(content, "text"):
            import json

            data = json.loads(content.text)
        else:
            data = content
    else:
        raise AssertionError(f"Unexpected result type: {type(result)}")

    assert data.get("status") == "started", f"Expected started status, got: {data}"
    assert "sprint_id" in data, f"Expected sprint_id in result: {data}"


def extract_result_data(result: Any) -> dict[str, Any]:
    """Extract tool result data from FastMCP response."""
    if hasattr(result, "content"):
        content = result.content
        if isinstance(content, list) and len(content) > 0:
            content = content[0]
        if hasattr(content, "text"):
            import json

            return json.loads(content.text)
        if isinstance(content, dict):
            return content
        return {"content": content}
    if isinstance(result, dict):
        return result
    if isinstance(result, list) and len(result) > 0:
        content = result[0]
        if hasattr(content, "text"):
            import json

            return json.loads(content.text)
        if isinstance(content, dict):
            return content
    return {}


def extract_seed_id(result: Any) -> str | None:
    """Extract seed_id from tool result."""
    if hasattr(result, "content"):
        content = result.content
        if isinstance(content, list) and len(content) > 0:
            content = content[0]
        if hasattr(content, "text"):
            import json

            data = json.loads(content.text)
        else:
            data = content
    elif isinstance(result, dict):
        data = result
    elif isinstance(result, list) and len(result) > 0:
        content = result[0]
        if hasattr(content, "text"):
            import json

            data = json.loads(content.text)
        else:
            data = content
    else:
        return None

    return data.get("seed_id")


def extract_sprint_id(result: Any) -> str | None:
    """Extract sprint_id from tool result."""
    if hasattr(result, "content"):
        content = result.content
        if isinstance(content, list) and len(content) > 0:
            content = content[0]
        if hasattr(content, "text"):
            import json

            data = json.loads(content.text)
        else:
            data = content
    elif isinstance(result, dict):
        data = result
    elif isinstance(result, list) and len(result) > 0:
        content = result[0]
        if hasattr(content, "text"):
            import json

            data = json.loads(content.text)
        else:
            data = content
    else:
        return None

    return data.get("sprint_id")
