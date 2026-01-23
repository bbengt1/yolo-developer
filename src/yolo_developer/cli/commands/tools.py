"""YOLO tools command implementation (Issue #17).

This module provides the yolo tools command which manages external CLI tool
integrations like Claude Code and Aider.

Subcommands:
- `yolo tools` - Show tool status (default)
- `yolo tools status` - Show detailed tool status

Examples:
    yolo tools                  # Show tool status
    yolo tools status           # Same as above
    yolo tools status --json    # Output as JSON
"""

from __future__ import annotations

import json
import shutil

import structlog
import typer
from rich.table import Table

from yolo_developer.cli.display import (
    console,
    info_panel,
    warning_panel,
)

logger = structlog.get_logger(__name__)

# Create the tools subcommand app
tools_app = typer.Typer(
    name="tools",
    help="Manage external CLI tool integrations.",
    no_args_is_help=False,
)


def _load_tools_config() -> tuple[bool, dict[str, dict[str, object]] | None, str | None]:
    """Load tools configuration from yolo.yaml.

    Returns:
        Tuple of (success, config_dict, error_message).
    """
    try:
        from yolo_developer.config import load_config

        config = load_config()
        tools_config = {
            "claude_code": {
                "enabled": config.tools.claude_code.enabled,
                "path": config.tools.claude_code.path,
                "timeout": config.tools.claude_code.timeout,
                "output_format": config.tools.claude_code.output_format,
            },
            "aider": {
                "enabled": config.tools.aider.enabled,
                "path": config.tools.aider.path,
                "timeout": config.tools.aider.timeout,
                "output_format": config.tools.aider.output_format,
            },
        }
        return True, tools_config, None
    except Exception as e:
        return False, None, str(e)


def _check_tool_availability(tool_name: str, config: dict[str, object]) -> dict[str, object]:
    """Check if a tool binary is available.

    Args:
        tool_name: Name of the tool (e.g., "claude_code").
        config: Tool configuration dict.

    Returns:
        Dict with availability information.
    """
    binary_map = {
        "claude_code": "claude",
        "aider": "aider",
    }

    binary_name = binary_map.get(tool_name, tool_name)
    custom_path = config.get("path")

    if custom_path:
        # Check if custom path exists
        import os

        binary_path = custom_path
        is_available = os.path.isfile(str(custom_path)) and os.access(str(custom_path), os.X_OK)
    else:
        # Check PATH
        binary_path = shutil.which(binary_name)
        is_available = binary_path is not None

    return {
        "binary_name": binary_name,
        "binary_path": binary_path,
        "is_available": is_available,
        "using_custom_path": custom_path is not None,
    }


def show_tools_status(json_output: bool = False) -> None:
    """Show external CLI tool status.

    Args:
        json_output: Output as JSON instead of Rich table.
    """
    logger.debug("show_tools_status_invoked", json_output=json_output)

    # Load configuration
    success, tools_config, error = _load_tools_config()

    if not success:
        if json_output:
            print(json.dumps({"error": error, "status": "config_error"}))
        else:
            warning_panel(
                f"Could not load configuration: {error}\n\n"
                "Run 'yolo init' to create a project with configuration.",
                title="Configuration Error",
            )
        raise typer.Exit(code=1)

    assert tools_config is not None  # for type checker

    # Build status info for each tool
    tools_status: list[dict[str, object]] = []

    for tool_name, config in tools_config.items():
        availability = _check_tool_availability(tool_name, config)

        status_entry = {
            "name": tool_name,
            "enabled": config["enabled"],
            "binary": availability["binary_name"],
            "available": availability["is_available"],
            "path": availability["binary_path"],
            "timeout": config["timeout"],
            "output_format": config["output_format"],
        }
        tools_status.append(status_entry)

    # Output
    if json_output:
        print(json.dumps({"tools": tools_status, "status": "success"}, indent=2))
    else:
        _display_tools_table(tools_status)


def _display_tools_table(tools_status: list[dict[str, object]]) -> None:
    """Display tools status as a Rich table.

    Args:
        tools_status: List of tool status dictionaries.
    """
    table = Table(title="External CLI Tools", show_header=True, header_style="bold cyan")

    table.add_column("Tool", style="cyan")
    table.add_column("Enabled", justify="center")
    table.add_column("Available", justify="center")
    table.add_column("Binary")
    table.add_column("Timeout", justify="right")

    for tool in tools_status:
        enabled = "[green]Yes[/green]" if tool["enabled"] else "[dim]No[/dim]"
        available = "[green]Yes[/green]" if tool["available"] else "[red]No[/red]"
        binary = str(tool["path"]) if tool["path"] else f"[dim]{tool['binary']} (not found)[/dim]"
        timeout = f"{tool['timeout']}s"

        table.add_row(
            str(tool["name"]),
            enabled,
            available,
            binary,
            timeout,
        )

    console.print(table)
    console.print()

    # Show summary and hints
    enabled_tools = [t for t in tools_status if t["enabled"]]

    if not enabled_tools:
        info_panel(
            "No tools are enabled. To enable a tool, add to yolo.yaml:\n\n"
            "  tools:\n"
            "    claude_code:\n"
            "      enabled: true\n\n"
            "Or set environment variable: YOLO_TOOLS__CLAUDE_CODE__ENABLED=true",
            title="Tip",
        )
    elif enabled_tools and not all(t["available"] for t in enabled_tools):
        missing = [t["name"] for t in enabled_tools if not t["available"]]
        info_panel(
            f"Some enabled tools are not installed: {', '.join(missing)}\n\n"
            "Install the missing tool binaries to use them.",
            title="Missing Tools",
        )


@tools_app.callback(invoke_without_command=True)
def tools_callback(
    ctx: typer.Context,
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON.",
    ),
) -> None:
    """Manage external CLI tool integrations.

    Shows tool status when called without a subcommand.
    """
    if ctx.invoked_subcommand is None:
        show_tools_status(json_output=json_output)


@tools_app.command("status")
def tools_status_command(
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON.",
    ),
) -> None:
    """Show external CLI tool status and availability."""
    show_tools_status(json_output=json_output)
