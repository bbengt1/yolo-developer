"""MCP tool implementations for YOLO Developer.

This module provides the MCP tools that expose YOLO Developer functionality
to external clients via the FastMCP server.

Tools:
    - yolo_seed: Provide seed requirements for autonomous development

Example:
    >>> from yolo_developer.mcp import mcp
    >>> # yolo_seed tool is automatically registered when this module is imported
    >>> # MCP clients can call: await client.call_tool("yolo_seed", {"content": "..."})

References:
    - Story 14.2: yolo_seed MCP Tool
    - ADR-004: FastMCP 2.x for MCP server implementation
    - FR112-FR117: MCP protocol integration requirements
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from yolo_developer.mcp.server import mcp


@dataclass
class StoredSeed:
    """A stored seed with metadata.

    Attributes:
        seed_id: Unique identifier for the seed (UUID).
        content: The seed requirements content.
        source: Source type - either "text" or "file".
        created_at: Timestamp when the seed was created.
        content_length: Length of the content in characters.
        file_path: Original file path if source is "file".
    """

    seed_id: str
    content: str
    source: Literal["text", "file"]
    created_at: datetime
    content_length: int
    file_path: str | None = None


# In-memory storage for seeds
# This is a simple MVP implementation; can be replaced with persistent storage later
_seeds: dict[str, StoredSeed] = {}
_seeds_lock = threading.Lock()  # Thread safety for concurrent MCP requests


def store_seed(
    content: str,
    source: Literal["text", "file"],
    file_path: str | None = None,
) -> StoredSeed:
    """Store a seed and return the stored seed object.

    This function is thread-safe for concurrent MCP requests.

    Args:
        content: The seed requirements content.
        source: Source type - "text" for direct content, "file" for file-based.
        file_path: Original file path if source is "file".

    Returns:
        StoredSeed: The stored seed with generated ID and metadata.

    Example:
        >>> seed = store_seed("Build a REST API", source="text")
        >>> print(seed.seed_id)
        '550e8400-e29b-41d4-a716-446655440000'
    """
    seed_id = str(uuid.uuid4())
    seed = StoredSeed(
        seed_id=seed_id,
        content=content,
        source=source,
        created_at=datetime.now(timezone.utc),
        content_length=len(content),
        file_path=file_path,
    )
    with _seeds_lock:
        _seeds[seed_id] = seed
    return seed


def get_seed(seed_id: str) -> StoredSeed | None:
    """Retrieve a seed by its ID.

    This function is thread-safe for concurrent MCP requests.

    Args:
        seed_id: The unique identifier of the seed.

    Returns:
        StoredSeed if found, None otherwise.

    Example:
        >>> seed = get_seed("550e8400-e29b-41d4-a716-446655440000")
        >>> if seed:
        ...     print(seed.content)
    """
    with _seeds_lock:
        return _seeds.get(seed_id)


def clear_seeds() -> None:
    """Clear all stored seeds.

    This function is primarily intended for testing to reset state between tests.
    It is thread-safe for concurrent MCP requests.

    Note:
        This is exposed in the public API for test fixtures. Production code
        should generally not call this function.
    """
    with _seeds_lock:
        _seeds.clear()


@mcp.tool
async def yolo_seed(
    content: str | None = None,
    file_path: str | None = None,
) -> dict[str, Any]:
    """Provide seed requirements for autonomous development.

    Provide EITHER content (as text) OR file_path (to read from).
    If both are provided, content takes precedence.

    Args:
        content: Seed requirements as plain text.
        file_path: Path to a file containing seed requirements.

    Returns:
        dict: Response with status, seed_id (if successful), and metadata.
            Success: {"status": "accepted", "seed_id": "...", "content_length": N, "source": "text"|"file"}
            Error: {"status": "error", "error": "Error message"}

    Example:
        >>> result = await yolo_seed(content="Build a REST API for user management")
        >>> print(result["seed_id"])
    """
    # Validate input: at least one of content or file_path must be provided
    if content is None and file_path is None:
        return {
            "status": "error",
            "error": "Either content or file_path must be provided",
        }

    # If content is provided, use it (takes precedence over file_path)
    if content is not None:
        # Validate content is not empty or whitespace-only
        if not content.strip():
            return {
                "status": "error",
                "error": "Content cannot be empty or whitespace-only",
            }

        seed = store_seed(content=content, source="text")
        return {
            "status": "accepted",
            "seed_id": seed.seed_id,
            "content_length": seed.content_length,
            "source": "text",
        }

    # file_path is provided (and content is None)
    assert file_path is not None  # For type checker

    # Validate file exists
    path = Path(file_path)
    if not path.exists():
        return {
            "status": "error",
            "error": f"File not found: {file_path}",
        }

    if not path.is_file():
        return {
            "status": "error",
            "error": f"Path is not a file: {file_path}",
        }

    # Read file content
    try:
        file_content = path.read_text(encoding="utf-8")
    except OSError as e:
        return {
            "status": "error",
            "error": f"Error reading file: {e}",
        }

    # Validate file content is not empty
    if not file_content.strip():
        return {
            "status": "error",
            "error": "File content is empty or whitespace-only",
        }

    seed = store_seed(content=file_content, source="file", file_path=file_path)
    return {
        "status": "accepted",
        "seed_id": seed.seed_id,
        "content_length": seed.content_length,
        "source": "file",
    }


__all__ = ["StoredSeed", "clear_seeds", "get_seed", "store_seed", "yolo_seed"]
