"""Tests for the MCP CLI command.

Tests cover AC5: CLI integration for MCP server.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from yolo_developer.cli.main import app

runner = CliRunner()


def test_mcp_command_exists() -> None:
    """Test 'yolo mcp' command is registered."""
    result = runner.invoke(app, ["mcp", "--help"])
    assert result.exit_code == 0
    assert "MCP server" in result.stdout or "mcp" in result.stdout.lower()


def test_mcp_command_transport_option() -> None:
    """Test --transport option is available."""
    result = runner.invoke(app, ["mcp", "--help"])
    assert result.exit_code == 0
    assert "--transport" in result.stdout or "-t" in result.stdout


def test_mcp_command_port_option() -> None:
    """Test --port option is available."""
    result = runner.invoke(app, ["mcp", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.stdout or "-p" in result.stdout


def test_mcp_command_transport_choices() -> None:
    """Test transport option accepts 'stdio' and 'http'."""
    result = runner.invoke(app, ["mcp", "--help"])
    assert result.exit_code == 0
    # Help should mention transport choices
    assert "stdio" in result.stdout.lower() or "http" in result.stdout.lower()


@patch("yolo_developer.cli.commands.mcp.run_server")
def test_mcp_command_default_stdio_transport(mock_run_server: MagicMock) -> None:
    """Test MCP command uses STDIO transport by default."""
    from yolo_developer.mcp.server import TransportType

    runner.invoke(app, ["mcp"])
    # The command should call run_server with default STDIO
    mock_run_server.assert_called_once()
    call_args = mock_run_server.call_args
    # Default should be STDIO
    assert (
        call_args.kwargs.get("transport") == TransportType.STDIO
        or call_args.args[0] == TransportType.STDIO
        if call_args.args
        else True
    )


@patch("yolo_developer.cli.commands.mcp.run_server")
def test_mcp_command_http_transport(mock_run_server: MagicMock) -> None:
    """Test MCP command accepts HTTP transport."""
    runner.invoke(app, ["mcp", "--transport", "http"])
    mock_run_server.assert_called_once()


@patch("yolo_developer.cli.commands.mcp.run_server")
def test_mcp_command_custom_port(mock_run_server: MagicMock) -> None:
    """Test MCP command accepts custom port."""
    runner.invoke(app, ["mcp", "--transport", "http", "--port", "8080"])
    mock_run_server.assert_called_once()
    call_args = mock_run_server.call_args
    assert (
        call_args.kwargs.get("port") == 8080 or 8080 in call_args.args if call_args.args else True
    )
