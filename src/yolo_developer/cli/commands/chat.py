"""CLI command for interactive chat sessions."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterable

import typer
from rich.console import Console
from rich.panel import Panel

from yolo_developer.cli.display import error_panel
from yolo_developer.config import ConfigurationError, YoloConfig, load_config
from yolo_developer.llm.router import (
    LLMConfigurationError,
    LLMProviderError,
    LLMRouter,
)

console = Console()

_EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}


def read_piped_input() -> str | None:
    """Return piped stdin content if available."""
    if sys.stdin is None or sys.stdin.isatty():
        return None
    data = sys.stdin.read()
    if not data:
        return None
    stripped = data.strip()
    return stripped or None


def resolve_prompt(
    args: Iterable[str] | None,
    stdin_text: str | None,
) -> tuple[str | None, bool]:
    """Resolve prompt content and interactive flag from args/stdin."""
    prompt = " ".join(args).strip() if args else None
    if prompt:
        return prompt, False
    if stdin_text:
        return stdin_text, False
    return None, True


def chat_command(prompt: str | None, interactive: bool = True) -> None:
    """Run interactive or one-shot chat."""
    if not interactive and not prompt:
        error_panel("No prompt provided for non-interactive chat.")
        raise typer.Exit(code=1)

    try:
        config = load_config()
    except ConfigurationError:
        config = YoloConfig(project_name="chat")

    router = LLMRouter(config.llm)

    if not interactive and prompt:
        _run_one_shot(router, prompt)
        return

    _run_interactive(router)


def _run_one_shot(router: LLMRouter, prompt: str) -> None:
    try:
        response = asyncio.run(
            router.call(messages=[{"role": "user", "content": prompt}], tier="routine")
        )
    except (LLMConfigurationError, LLMProviderError) as exc:
        error_panel(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(response)


def _run_interactive(router: LLMRouter) -> None:
    console.clear()
    console.print(
        Panel(
            "Interactive session started. Type /exit or press Ctrl+D to quit.",
            title="YOLO Chat",
            border_style="cyan",
        )
    )
    messages: list[dict[str, str]] = []

    while True:
        try:
            user_input = console.input("[bold cyan]you>[/] ")
        except EOFError:
            console.print("[dim]Session ended.[/dim]")
            return
        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped in _EXIT_COMMANDS:
            console.print("[dim]Session ended.[/dim]")
            return

        messages.append({"role": "user", "content": user_input})
        try:
            response = asyncio.run(router.call(messages=messages, tier="routine"))
        except (LLMConfigurationError, LLMProviderError) as exc:
            error_panel(str(exc))
            return
        console.print(f"[bold green]yolo>[/] {response}")
        messages.append({"role": "assistant", "content": response})
