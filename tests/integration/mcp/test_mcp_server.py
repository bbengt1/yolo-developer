"""Integration tests for the MCP server with FastMCP Client.

Tests cover AC6: FastMCP testing patterns with mocked clients.
"""

from __future__ import annotations

import pytest
from fastmcp import Client

from yolo_developer.mcp import mcp


@pytest.mark.asyncio
async def test_server_responds_to_client() -> None:
    """Test server can handle client requests via FastMCP Client."""
    async with Client(mcp) as client:
        tools = await client.list_tools()
        # No tools registered yet in this story (tools added in 14.2-14.5)
        assert isinstance(tools, list)


@pytest.mark.asyncio
async def test_server_returns_empty_tools_list() -> None:
    """Test server returns empty tools list when no tools registered."""
    async with Client(mcp) as client:
        tools = await client.list_tools()
        # Story 14.1 sets up server only - no tools yet
        assert len(tools) == 0


@pytest.mark.asyncio
async def test_client_can_ping_server() -> None:
    """Test client can establish connection with server."""
    async with Client(mcp) as client:
        # If we get here without error, connection works
        # List resources as a basic connectivity check
        resources = await client.list_resources()
        assert isinstance(resources, list)
