"""MCP (Model Context Protocol) server for YOLO Developer.

This module exposes YOLO Developer functionality to external tools
via the MCP protocol, enabling integration with Claude Code and
other MCP-compatible clients.

Example:
    >>> from yolo_developer.mcp import mcp, run_server, TransportType
    >>> mcp.run()  # Start with STDIO transport (default)
    >>> run_server()  # Start with STDIO transport (type-safe wrapper)
    >>> run_server(TransportType.HTTP, port=8080)  # HTTP transport

Note:
    Use `run_server()` for type-safe transport selection via the TransportType enum.
    Use `mcp.run()` directly for quick testing or when passing transport as string.

The server is configured with:
    - Name: "YOLO Developer"
    - Version: Package version from yolo_developer.__version__
    - Instructions: Description of available tools
    - Error masking: Enabled for production safety

Tools available:
    - yolo_seed: Provide seed requirements for development (Story 14.2)
    - yolo_run: Execute autonomous sprint (Story 14.3)
    - yolo_status: Query sprint status (Story 14.4)
    - yolo_audit: Access audit trail (Story 14.5)
"""

from __future__ import annotations

from yolo_developer.mcp.server import (
    SERVER_INSTRUCTIONS,
    TransportType,
    mcp,
    run_server,
)
from yolo_developer.mcp.tools import (
    StoredSeed,
    StoredSprint,
    clear_seeds,
    clear_sprints,
    get_seed,
    get_sprint,
    store_seed,
    store_sprint,
    yolo_audit,
    yolo_run,
    yolo_seed,
    yolo_status,
)

# Public API exports:
# - mcp: FastMCP server instance for direct use
# - run_server: Type-safe wrapper for starting the server
# - TransportType: Enum for transport protocol selection
# - SERVER_INSTRUCTIONS: Server instructions string for reference
# - yolo_seed: MCP tool for providing seed requirements
# - StoredSeed: Dataclass for stored seed objects
# - store_seed, get_seed, clear_seeds: Seed storage functions
__all__ = [
    "SERVER_INSTRUCTIONS",
    "StoredSeed",
    "StoredSprint",
    "TransportType",
    "clear_seeds",
    "clear_sprints",
    "get_seed",
    "get_sprint",
    "mcp",
    "run_server",
    "store_seed",
    "store_sprint",
    "yolo_audit",
    "yolo_run",
    "yolo_seed",
    "yolo_status",
]
