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
async def test_server_lists_yolo_audit_tool() -> None:
    """Test server includes yolo_audit in available tools."""
    async with Client(mcp) as client:
        tools = await client.list_tools()
        tool_names = [t.name for t in tools]
        assert "yolo_audit" in tool_names


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


class TestYoloAuditIntegration:
    """Integration tests for yolo_audit MCP tool via FastMCP Client."""

    @pytest.mark.asyncio
    async def test_yolo_audit_via_client_returns_entries(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test yolo_audit returns audit entries via MCP client."""
        from yolo_developer.audit import JsonDecisionStore
        from yolo_developer.audit.types import AgentIdentity, Decision, DecisionContext

        monkeypatch.chdir(tmp_path)

        store = JsonDecisionStore(Path(".yolo/audit/decisions.json"))
        await store.log_decision(
            Decision(
                id="dec-100",
                decision_type="requirement_analysis",  # type: ignore[arg-type]
                content="Requirement noted",
                rationale="Needed for test",
                agent=AgentIdentity(
                    agent_name="analyst",
                    agent_type="analyst",
                    session_id="session-99",
                ),
                context=DecisionContext(
                    sprint_id="sprint-1",
                    story_id="story-1",
                ),
                timestamp="2026-01-18T10:00:00+00:00",
                metadata={"artifact_type": "requirement"},
                severity="info",  # type: ignore[arg-type]
            )
        )

        async with Client(mcp) as client:
            result = await client.call_tool("yolo_audit", {"agent": "analyst"})

        data = extract_result_data(result)
        assert data.get("status") == "ok"
        assert data.get("total") == 1
        assert isinstance(data.get("entries"), list)
        assert data["entries"][0]["entry_id"] == "dec-100"


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


class TestMcpToolDiscovery:
    """Tests for MCP tool discovery and parameter validation (Story 14.6 AC1)."""

    @pytest.mark.asyncio
    async def test_all_tools_discoverable_via_list_tools(self) -> None:
        """Test all YOLO tools are discoverable via MCP protocol."""
        async with Client(mcp) as client:
            tools = await client.list_tools()
            tool_names = [t.name for t in tools]

            # All expected tools should be present
            expected_tools = ["yolo_seed", "yolo_run", "yolo_status", "yolo_audit"]
            for tool_name in expected_tools:
                assert tool_name in tool_names, f"Tool {tool_name} not discoverable"

    @pytest.mark.asyncio
    async def test_yolo_seed_has_valid_input_schema(self) -> None:
        """Test yolo_seed tool has properly typed parameters per MCP spec."""
        async with Client(mcp) as client:
            tools = await client.list_tools()
            tool = next((t for t in tools if t.name == "yolo_seed"), None)

            assert tool is not None
            assert tool.inputSchema is not None

            schema = tool.inputSchema
            # Should have properties defined
            assert "properties" in schema or schema.get("type") == "object"

            if "properties" in schema:
                props = schema["properties"]
                # content should be string or null
                if "content" in props:
                    assert props["content"].get("type") in ["string", None] or (
                        "anyOf" in props["content"]
                    )
                # file_path should be string or null
                if "file_path" in props:
                    assert props["file_path"].get("type") in ["string", None] or (
                        "anyOf" in props["file_path"]
                    )

    @pytest.mark.asyncio
    async def test_yolo_run_has_valid_input_schema(self) -> None:
        """Test yolo_run tool has properly typed parameters per MCP spec."""
        async with Client(mcp) as client:
            tools = await client.list_tools()
            tool = next((t for t in tools if t.name == "yolo_run"), None)

            assert tool is not None
            assert tool.inputSchema is not None

            schema = tool.inputSchema
            assert "properties" in schema or schema.get("type") == "object"

            if "properties" in schema:
                props = schema["properties"]
                # seed_id and seed_content should be string types
                for param in ["seed_id", "seed_content"]:
                    if param in props:
                        assert props[param].get("type") in ["string", None] or (
                            "anyOf" in props[param]
                        )

    @pytest.mark.asyncio
    async def test_yolo_status_has_valid_input_schema(self) -> None:
        """Test yolo_status tool has properly typed parameters per MCP spec."""
        async with Client(mcp) as client:
            tools = await client.list_tools()
            tool = next((t for t in tools if t.name == "yolo_status"), None)

            assert tool is not None
            assert tool.inputSchema is not None

            schema = tool.inputSchema
            assert "properties" in schema or schema.get("type") == "object"

            if "properties" in schema:
                props = schema["properties"]
                # sprint_id is required string
                assert "sprint_id" in props
                assert props["sprint_id"].get("type") == "string"

    @pytest.mark.asyncio
    async def test_yolo_audit_has_valid_input_schema(self) -> None:
        """Test yolo_audit tool has properly typed parameters per MCP spec."""
        async with Client(mcp) as client:
            tools = await client.list_tools()
            tool = next((t for t in tools if t.name == "yolo_audit"), None)

            assert tool is not None
            assert tool.inputSchema is not None

            schema = tool.inputSchema
            assert "properties" in schema or schema.get("type") == "object"

            if "properties" in schema:
                props = schema["properties"]
                # Check expected optional parameters
                optional_params = [
                    "agent",
                    "decision_type",
                    "artifact_type",
                    "start_time",
                    "end_time",
                ]
                for param in optional_params:
                    if param in props:
                        assert props[param].get("type") in ["string", None] or (
                            "anyOf" in props[param]
                        )
                # Check numeric parameters
                for param in ["limit", "offset"]:
                    if param in props:
                        assert props[param].get("type") == "integer"

    @pytest.mark.asyncio
    async def test_all_tools_have_descriptions(self) -> None:
        """Test all tools have non-empty descriptions for LLM understanding."""
        async with Client(mcp) as client:
            tools = await client.list_tools()

            for tool in tools:
                if tool.name.startswith("yolo_"):
                    assert tool.description is not None, (
                        f"Tool {tool.name} has no description"
                    )
                    assert len(tool.description) > 10, (
                        f"Tool {tool.name} description too short"
                    )


class TestMcpFullWorkflow:
    """End-to-end integration tests for MCP workflow (Story 14.6 AC all)."""

    @pytest.mark.asyncio
    async def test_full_workflow_seed_run_status(self) -> None:
        """Test complete workflow: seed -> run -> status."""
        async with Client(mcp) as client:
            # Step 1: Seed requirements
            seed_result = await client.call_tool(
                "yolo_seed", {"content": "Build a complete user authentication system"}
            )
            seed_data = extract_result_data(seed_result)
            assert seed_data["status"] == "accepted"
            seed_id = seed_data["seed_id"]

            # Step 2: Run sprint with seed_id
            with patch("yolo_developer.mcp.tools._run_sprint", new_callable=AsyncMock):
                run_result = await client.call_tool("yolo_run", {"seed_id": seed_id})
            run_data = extract_result_data(run_result)
            assert run_data["status"] == "started"
            sprint_id = run_data["sprint_id"]
            assert run_data["seed_id"] == seed_id

            # Step 3: Check status
            status_result = await client.call_tool(
                "yolo_status", {"sprint_id": sprint_id}
            )
            status_data = extract_result_data(status_result)
            assert status_data["status"] == "running"
            assert status_data["sprint_id"] == sprint_id
            assert status_data["seed_id"] == seed_id

    @pytest.mark.asyncio
    async def test_concurrent_seed_requests(self) -> None:
        """Test concurrent MCP requests for thread safety."""
        import asyncio

        async with Client(mcp) as client:
            # Create multiple concurrent seed requests
            tasks = [
                client.call_tool("yolo_seed", {"content": f"Requirement {i}"})
                for i in range(5)
            ]

            # Execute all concurrently
            results = await asyncio.gather(*tasks)

            # All should succeed with unique seed_ids
            seed_ids = set()
            for result in results:
                data = extract_result_data(result)
                assert data["status"] == "accepted"
                seed_ids.add(data["seed_id"])

            # All seed_ids should be unique
            assert len(seed_ids) == 5

    @pytest.mark.asyncio
    async def test_concurrent_status_requests(self) -> None:
        """Test concurrent status queries for thread safety."""
        import asyncio

        # Create a sprint first
        seed = store_seed(content="Test seed", source="text")
        sprint = store_sprint(seed_id=seed.seed_id, thread_id="thread-concurrent")

        async with Client(mcp) as client:
            # Query status multiple times concurrently
            tasks = [
                client.call_tool("yolo_status", {"sprint_id": sprint.sprint_id})
                for _ in range(10)
            ]

            results = await asyncio.gather(*tasks)

            # All should return consistent data
            for result in results:
                data = extract_result_data(result)
                assert data["status"] == "running"
                assert data["sprint_id"] == sprint.sprint_id
                assert data["seed_id"] == seed.seed_id

    @pytest.mark.asyncio
    async def test_workflow_with_inline_seed_content(self) -> None:
        """Test workflow using inline seed_content instead of seed_id."""
        async with Client(mcp) as client:
            # Use seed_content directly with yolo_run
            with (
                patch("yolo_developer.mcp.tools.parse_seed", new_callable=AsyncMock),
                patch("yolo_developer.mcp.tools.generate_validation_report") as mock_report,
                patch("yolo_developer.mcp.tools.validate_quality_thresholds") as mock_validate,
                patch("yolo_developer.mcp.tools._run_sprint", new_callable=AsyncMock),
            ):
                # Setup mocks to pass validation
                mock_report.return_value = type("Report", (), {"quality_metrics": {}})()
                mock_validate.return_value = type("Result", (), {"passed": True})()

                run_result = await client.call_tool(
                    "yolo_run", {"seed_content": "Build a REST API"}
                )

            run_data = extract_result_data(run_result)
            assert run_data["status"] == "started"
            assert "sprint_id" in run_data
            assert "seed_id" in run_data


class TestMcpProviderCompatibility:
    """Integration tests for multi-provider compatibility (Story 14.6)."""

    @pytest.mark.asyncio
    async def test_tool_names_follow_mcp_convention(self) -> None:
        """Test tool names follow MCP naming conventions (snake_case)."""
        import re

        async with Client(mcp) as client:
            tools = await client.list_tools()

            for tool in tools:
                if tool.name.startswith("yolo_"):
                    # MCP tool names should be snake_case
                    assert re.match(r"^[a-z][a-z0-9_]*$", tool.name), (
                        f"Tool name '{tool.name}' doesn't follow snake_case convention"
                    )

    @pytest.mark.asyncio
    async def test_all_tool_descriptions_are_provider_agnostic(self) -> None:
        """Test tool descriptions don't contain provider-specific terms."""
        provider_terms = ["claude", "anthropic", "openai", "gpt", "codex"]

        async with Client(mcp) as client:
            tools = await client.list_tools()

            for tool in tools:
                if tool.name.startswith("yolo_") and tool.description:
                    desc_lower = tool.description.lower()
                    for term in provider_terms:
                        assert term not in desc_lower, (
                            f"Tool '{tool.name}' description contains "
                            f"provider-specific term '{term}'"
                        )

    @pytest.mark.asyncio
    async def test_tool_responses_are_json_serializable(self) -> None:
        """Test all tool responses can be serialized to JSON."""
        import json

        async with Client(mcp) as client:
            # Test yolo_seed
            result = await client.call_tool(
                "yolo_seed", {"content": "Test requirements"}
            )
            data = extract_result_data(result)
            json.dumps(data)  # Should not raise

            # Test yolo_run with seed_id
            seed_id = data["seed_id"]
            with patch("yolo_developer.mcp.tools._run_sprint", new_callable=AsyncMock):
                result = await client.call_tool("yolo_run", {"seed_id": seed_id})
            data = extract_result_data(result)
            json.dumps(data)

            # Test yolo_status
            sprint_id = data["sprint_id"]
            result = await client.call_tool("yolo_status", {"sprint_id": sprint_id})
            data = extract_result_data(result)
            json.dumps(data)

    @pytest.mark.asyncio
    async def test_error_responses_are_mcp_compliant(self) -> None:
        """Test error responses follow MCP error format."""
        async with Client(mcp) as client:
            # Trigger various errors
            error_tests = [
                ("yolo_seed", {}),  # Missing required parameter
                ("yolo_seed", {"content": ""}),  # Empty content
                ("yolo_run", {}),  # Missing parameter
                ("yolo_run", {"seed_id": "nonexistent"}),  # Not found
                ("yolo_status", {"sprint_id": "nonexistent"}),  # Not found
            ]

            for tool_name, args in error_tests:
                result = await client.call_tool(tool_name, args)
                data = extract_result_data(result)

                # All errors should have status and error fields
                assert "status" in data, f"Missing status for {tool_name}"
                assert data["status"] == "error", f"Expected error for {tool_name}"
                assert "error" in data, f"Missing error message for {tool_name}"
                assert isinstance(data["error"], str), (
                    f"Error message not string for {tool_name}"
                )
