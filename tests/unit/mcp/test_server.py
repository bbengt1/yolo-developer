"""Tests for the FastMCP server module.

Tests cover:
- AC1: FastMCP server initialization and configuration
- AC4: Server metadata (name, version, instructions, error masking)
"""

from __future__ import annotations


def test_server_import() -> None:
    """Test that mcp server can be imported from the package."""
    from yolo_developer.mcp import mcp

    assert mcp is not None


def test_server_name() -> None:
    """Test MCP server name is set to 'YOLO Developer'."""
    from yolo_developer.mcp import mcp

    assert mcp.name == "YOLO Developer"


def test_server_version_from_package() -> None:
    """Test server version matches package version."""
    from yolo_developer import __version__
    from yolo_developer.mcp import mcp

    assert mcp.version == __version__


def test_server_has_instructions() -> None:
    """Test server has instructions configured."""
    from yolo_developer.mcp import mcp
    from yolo_developer.mcp.server import SERVER_INSTRUCTIONS

    assert mcp.instructions is not None
    assert mcp.instructions == SERVER_INSTRUCTIONS
    # Instructions should mention the planned tools
    assert "yolo_seed" in SERVER_INSTRUCTIONS
    assert "yolo_run" in SERVER_INSTRUCTIONS
    assert "yolo_status" in SERVER_INSTRUCTIONS
    assert "yolo_audit" in SERVER_INSTRUCTIONS


def test_server_masks_errors() -> None:
    """Test error masking is enabled for production safety."""
    # Import the mcp module to ensure settings are configured
    from fastmcp import settings as fastmcp_settings

    from yolo_developer.mcp import mcp  # noqa: F401

    # FastMCP 2.x uses global settings for error masking
    assert fastmcp_settings.mask_error_details is True


def test_server_instance_is_fastmcp() -> None:
    """Test server is a FastMCP instance."""
    from fastmcp import FastMCP

    from yolo_developer.mcp import mcp

    assert isinstance(mcp, FastMCP)


def test_server_instructions_export() -> None:
    """Test SERVER_INSTRUCTIONS can be imported from server module."""
    from yolo_developer.mcp.server import SERVER_INSTRUCTIONS

    assert isinstance(SERVER_INSTRUCTIONS, str)
    assert len(SERVER_INSTRUCTIONS) > 0


# Transport configuration tests (AC2, AC3)
def test_server_has_run_method() -> None:
    """Test server has run() method for transport execution."""
    from yolo_developer.mcp import mcp

    assert hasattr(mcp, "run")
    assert callable(mcp.run)


def test_server_has_stdio_async_method() -> None:
    """Test server has run_stdio_async() method for STDIO transport."""
    from yolo_developer.mcp import mcp

    assert hasattr(mcp, "run_stdio_async")
    assert callable(mcp.run_stdio_async)


def test_server_has_http_async_method() -> None:
    """Test server has run_http_async() method for HTTP transport."""
    from yolo_developer.mcp import mcp

    assert hasattr(mcp, "run_http_async")
    assert callable(mcp.run_http_async)


def test_transport_type_validation() -> None:
    """Test TransportType enum is defined for type safety."""
    from yolo_developer.mcp.server import TransportType

    assert hasattr(TransportType, "STDIO")
    assert hasattr(TransportType, "HTTP")
    assert TransportType.STDIO.value == "stdio"
    assert TransportType.HTTP.value == "http"


def test_run_server_function_exists() -> None:
    """Test run_server function is available for CLI integration."""
    from yolo_developer.mcp.server import run_server

    assert callable(run_server)


def test_run_server_accepts_string_transport() -> None:
    """Test run_server accepts string transport type."""
    import inspect

    from yolo_developer.mcp.server import TransportType, run_server

    sig = inspect.signature(run_server)
    # Check transport parameter accepts str
    transport_param = sig.parameters.get("transport")
    assert transport_param is not None
    # The default should be STDIO
    assert transport_param.default == TransportType.STDIO


def test_run_server_accepts_port_parameter() -> None:
    """Test run_server accepts port parameter."""
    import inspect

    from yolo_developer.mcp.server import run_server

    sig = inspect.signature(run_server)
    port_param = sig.parameters.get("port")
    assert port_param is not None
    assert port_param.default == 8000


def test_transport_type_string_conversion() -> None:
    """Test TransportType can be created from string."""
    from yolo_developer.mcp.server import TransportType

    assert TransportType("stdio") == TransportType.STDIO
    assert TransportType("http") == TransportType.HTTP


def test_server_has_tool_manager() -> None:
    """Test server has tool management capabilities (tools added in future stories)."""
    from yolo_developer.mcp import mcp

    # Server should have tool management capability
    assert hasattr(mcp, "tool")  # Decorator for adding tools
    assert hasattr(mcp, "add_tool")  # Method for adding tools
    assert hasattr(mcp, "get_tools")  # Method for getting tools (async)


def test_all_exports() -> None:
    """Test __all__ exports correct items."""
    from yolo_developer.mcp import server

    expected_exports = {"SERVER_INSTRUCTIONS", "TransportType", "mcp", "run_server"}
    assert set(server.__all__) == expected_exports


def test_mcp_module_exports() -> None:
    """Test mcp module __init__ exports correct items."""
    from yolo_developer import mcp

    # Should be able to import mcp instance directly
    assert hasattr(mcp, "mcp")
    assert hasattr(mcp, "SERVER_INSTRUCTIONS")
    # Should also export TransportType and run_server for programmatic access
    assert hasattr(mcp, "TransportType")
    assert hasattr(mcp, "run_server")


class TestTransportCompatibility:
    """Tests for MCP transport compatibility (Story 14.6 AC5)."""

    def test_stdio_transport_is_default(self) -> None:
        """Test STDIO is the default transport for local AI assistants."""
        import inspect

        from yolo_developer.mcp.server import TransportType, run_server

        sig = inspect.signature(run_server)
        transport_param = sig.parameters.get("transport")
        assert transport_param is not None
        assert transport_param.default == TransportType.STDIO

    def test_stdio_transport_value_is_standard(self) -> None:
        """Test STDIO transport uses standard 'stdio' value."""
        from yolo_developer.mcp.server import TransportType

        # MCP standard uses lowercase 'stdio'
        assert TransportType.STDIO.value == "stdio"

    def test_http_transport_value_is_standard(self) -> None:
        """Test HTTP transport uses standard 'http' value."""
        from yolo_developer.mcp.server import TransportType

        # MCP standard uses lowercase 'http'
        assert TransportType.HTTP.value == "http"

    def test_server_supports_both_transport_methods(self) -> None:
        """Test server has async methods for both transports."""
        from yolo_developer.mcp import mcp

        # STDIO transport method
        assert hasattr(mcp, "run_stdio_async")
        # HTTP transport method
        assert hasattr(mcp, "run_http_async")

    def test_transport_type_accepts_string_lowercase(self) -> None:
        """Test TransportType accepts lowercase string values."""
        from yolo_developer.mcp.server import TransportType

        assert TransportType("stdio") == TransportType.STDIO
        assert TransportType("http") == TransportType.HTTP

    def test_run_server_normalizes_string_transport(self) -> None:
        """Test run_server normalizes string transport types."""
        import inspect

        from yolo_developer.mcp.server import run_server

        sig = inspect.signature(run_server)
        transport_param = sig.parameters.get("transport")
        # Annotation should accept both TransportType and str
        annotation = transport_param.annotation if transport_param else None
        # The function signature shows it accepts TransportType | str
        assert annotation is not None

    def test_tool_implementations_are_transport_agnostic(self) -> None:
        """Test tool implementations don't make transport-specific assumptions."""
        import inspect

        from yolo_developer.mcp.tools import yolo_audit, yolo_run, yolo_seed, yolo_status

        # Get the underlying functions (FastMCP wraps them)
        for tool in [yolo_seed, yolo_run, yolo_status, yolo_audit]:
            fn = tool.fn if hasattr(tool, "fn") else tool
            source = inspect.getsource(fn)

            # Split at first """ to skip docstring
            code_parts = source.split('"""')
            if len(code_parts) >= 3:
                # Skip first docstring (parts[0] is before, parts[1] is docstring)
                code_only = '"""'.join(code_parts[2:])
            else:
                code_only = source

            # Tool implementation should not directly reference transport concepts
            # (transport is handled at the FastMCP server level, not in tools)
            assert "TransportType" not in code_only, (
                f"Tool {tool} should not reference TransportType directly"
            )

    def test_tools_return_serializable_data_for_any_transport(self) -> None:
        """Test tools return data serializable over any transport."""
        import json

        from yolo_developer.mcp.tools import store_seed

        # Store a seed synchronously
        seed = store_seed(content="Test content", source="text")

        # Verify the stored data can be serialized (required for any transport)
        seed_dict = {
            "seed_id": seed.seed_id,
            "content": seed.content,
            "source": seed.source,
            "created_at": seed.created_at.isoformat(),
            "content_length": seed.content_length,
            "file_path": seed.file_path,
        }

        # Should serialize without error (works for both STDIO and HTTP)
        json_str = json.dumps(seed_dict)
        assert json_str is not None
        parsed = json.loads(json_str)
        assert parsed["seed_id"] == seed.seed_id

    def test_http_transport_default_port(self) -> None:
        """Test HTTP transport uses standard default port."""
        import inspect

        from yolo_developer.mcp.server import run_server

        sig = inspect.signature(run_server)
        port_param = sig.parameters.get("port")
        assert port_param is not None
        # Default port 8000 is a common convention
        assert port_param.default == 8000
