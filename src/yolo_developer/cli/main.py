"""Main CLI entry point for YOLO Developer."""

from __future__ import annotations

import typer
from rich.console import Console

from yolo_developer.cli.commands.init import init_command

app = typer.Typer(
    name="yolo",
    help="YOLO Developer - Autonomous multi-agent AI development system",
    no_args_is_help=True,
)
console = Console()


@app.command("init")
def init(
    path: str | None = typer.Argument(
        None,
        help="Directory to initialize the project in. Defaults to current directory.",
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        "-n",
        help="Project name. Defaults to directory name.",
    ),
    author: str | None = typer.Option(
        None,
        "--author",
        "-a",
        help="Author name for pyproject.toml.",
    ),
    email: str | None = typer.Option(
        None,
        "--email",
        "-e",
        help="Author email for pyproject.toml.",
    ),
) -> None:
    """Initialize a new YOLO Developer project.

    Creates a new Python project with all required dependencies for
    autonomous multi-agent development using the BMad Method.
    """
    init_command(path=path, name=name, author=author, email=email)


@app.command("version")
def version() -> None:
    """Show YOLO Developer version."""
    from yolo_developer import __version__

    console.print(f"YOLO Developer v{__version__}")


if __name__ == "__main__":
    app()
