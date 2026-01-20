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
