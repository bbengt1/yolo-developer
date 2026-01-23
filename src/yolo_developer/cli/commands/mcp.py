"""MCP server CLI command.

This module provides the 'yolo mcp' command for starting the MCP server
with different transport protocols.

Example:
    $ yolo mcp                           # Start with STDIO (default)
    $ yolo mcp --transport http          # Start with HTTP transport
    $ yolo mcp --transport http --port 8080  # HTTP on custom port
"""

from __future__ import annotations

from typing import Literal

import typer

from yolo_developer.cli.commands.integrate import app as integrate_app
from yolo_developer.mcp.server import TransportType, run_server

# Create a Typer app for the mcp command
# This allows it to be registered as a subcommand
app = typer.Typer(help="MCP server commands")
app.add_typer(integrate_app, name="integrate")


def mcp_command(
    transport: Literal["stdio", "http"] = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="Transport protocol to use. 'stdio' for Claude Desktop, 'http' for remote access.",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port for HTTP transport (only used when transport is 'http').",
    ),
) -> None:
    """Start the YOLO Developer MCP server.

    The MCP server exposes YOLO Developer functionality to external tools
    like Claude Code and other MCP-compatible clients.

    Transports:
      - stdio: Standard I/O transport for Claude Desktop integration (default)
      - http: HTTP transport for remote access on specified port

    Examples:
        yolo mcp                           # Start with STDIO (default)
        yolo mcp --transport http          # HTTP on default port 8000
        yolo mcp -t http -p 8080           # HTTP on custom port 8080
    """
    transport_type = TransportType(transport)
    run_server(transport=transport_type, port=port)


# Register the command on the app
app.callback(invoke_without_command=True)(mcp_command)

__all__ = ["app", "mcp_command"]
