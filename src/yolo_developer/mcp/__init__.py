"""MCP (Model Context Protocol) server for YOLO Developer.

This module exposes YOLO Developer functionality to MCP-compatible
AI assistants and tools via the standard MCP protocol.

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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from yolo_developer.mcp.server import (  # noqa: F401
        SERVER_INSTRUCTIONS,
        TransportType,
        mcp,
        run_server,
    )
    from yolo_developer.mcp.tools import (  # noqa: F401
        StoredSeed,
        StoredSprint,
        clear_seeds,
        clear_sprints,
        get_seed,
        get_sprint,
        store_seed,
        store_sprint,
        yolo_audit,
        yolo_import_issue,
        yolo_import_issues,
        yolo_preview_import,
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
    "yolo_git_commit",
    "yolo_issue_create",
    "yolo_import_issue",
    "yolo_import_issues",
    "yolo_preview_import",
    "yolo_run",
    "yolo_pr_create",
    "yolo_pr_respond",
    "yolo_release_create",
    "yolo_seed",
    "yolo_status",
]


def __getattr__(name: str) -> Any:
    if name in {"SERVER_INSTRUCTIONS", "TransportType", "mcp", "run_server"}:
        from yolo_developer.mcp import server

        return getattr(server, name)
    if name in {
        "StoredSeed",
        "StoredSprint",
        "clear_seeds",
        "clear_sprints",
        "get_seed",
        "get_sprint",
        "store_seed",
        "store_sprint",
        "yolo_audit",
        "yolo_git_commit",
        "yolo_issue_create",
        "yolo_import_issue",
        "yolo_import_issues",
        "yolo_preview_import",
        "yolo_run",
        "yolo_pr_create",
        "yolo_pr_respond",
        "yolo_release_create",
        "yolo_seed",
        "yolo_status",
    }:
        from yolo_developer.mcp import tools

        return getattr(tools, name)
    raise AttributeError(f"module 'yolo_developer.mcp' has no attribute {name}")
