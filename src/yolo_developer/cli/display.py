"""Rich display utilities for CLI commands (Story 12.1).

This module provides reusable Rich formatting utilities for consistent
output styling across all CLI commands.

The utilities include:
- Panel functions for success, error, info, and warning messages
- A coming_soon function for placeholder commands
- A create_table factory for structured data display

Example:
    >>> from yolo_developer.cli.display import success_panel, create_table
    >>>
    >>> success_panel("Operation completed successfully!")
    >>> table = create_table("Results", [("Name", "cyan"), ("Value", "green")])
    >>> table.add_row("test", "passed")
    >>> console.print(table)

References:
    - FR98-FR105: CLI command requirements
    - ADR-009: Typer + Rich framework selection
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def success_panel(message: str, title: str = "Success") -> None:
    """Display a success panel with green styling.

    Args:
        message: The success message to display.
        title: The panel title. Defaults to "Success".

    Example:
        >>> success_panel("Project initialized successfully!")
    """
    console.print(Panel(f"[green]{message}[/green]", title=title, border_style="green"))


def error_panel(message: str, title: str = "Error") -> None:
    """Display an error panel with red styling.

    Args:
        message: The error message to display.
        title: The panel title. Defaults to "Error".

    Example:
        >>> error_panel("Failed to read configuration file.")
    """
    console.print(Panel(f"[red]{message}[/red]", title=title, border_style="red"))


def info_panel(message: str, title: str = "Info") -> None:
    """Display an info panel with blue styling.

    Args:
        message: The informational message to display.
        title: The panel title. Defaults to "Info".

    Example:
        >>> info_panel("Processing 5 seed documents...")
    """
    console.print(Panel(f"[blue]{message}[/blue]", title=title, border_style="blue"))


def warning_panel(message: str, title: str = "Warning") -> None:
    """Display a warning panel with yellow styling.

    Args:
        message: The warning message to display.
        title: The panel title. Defaults to "Warning".

    Example:
        >>> warning_panel("Configuration file not found, using defaults.")
    """
    console.print(Panel(f"[yellow]{message}[/yellow]", title=title, border_style="yellow"))


def coming_soon(command: str) -> None:
    """Display a 'coming soon' message for unimplemented commands.

    Args:
        command: The name of the unimplemented command.

    Example:
        >>> coming_soon("run")
        # Displays: "The 'run' command is not yet implemented."
    """
    console.print(
        Panel(
            f"[yellow]The '{command}' command is not yet implemented.[/yellow]\n\n"
            f"[dim]This command will be available in a future release.[/dim]",
            title="Coming Soon",
            border_style="yellow",
        )
    )


def create_table(title: str, columns: list[tuple[str, str]]) -> Table:
    """Create a styled table with given columns.

    Args:
        title: The table title.
        columns: List of (column_name, style) tuples.

    Returns:
        A configured Rich Table ready for adding rows.

    Example:
        >>> table = create_table("Results", [("Name", "cyan"), ("Status", "green")])
        >>> table.add_row("test_feature", "passed")
        >>> console.print(table)
    """
    table = Table(title=title, show_header=True, header_style="bold")
    for name, style in columns:
        table.add_column(name, style=style)
    return table
