"""Entry point for running YOLO Developer MCP server as a module.

Usage:
    python -m yolo_developer.mcp              # STDIO transport (default)
    python -m yolo_developer.mcp --http       # HTTP transport on port 8000
    python -m yolo_developer.mcp --http 8080  # HTTP transport on custom port

This enables MCP-compatible AI assistants to invoke YOLO Developer tools
via the standard MCP protocol.
"""

from __future__ import annotations

import sys

from yolo_developer.mcp.server import TransportType, run_server


def main() -> None:
    """Run the MCP server with transport specified via command-line args."""
    if "--http" in sys.argv:
        # Extract port if specified after --http
        idx = sys.argv.index("--http")
        if idx + 1 < len(sys.argv) and sys.argv[idx + 1].isdigit():
            port = int(sys.argv[idx + 1])
        else:
            port = 8000
        run_server(TransportType.HTTP, port=port)
    else:
        # Default to STDIO transport
        run_server(TransportType.STDIO)


if __name__ == "__main__":
    main()
