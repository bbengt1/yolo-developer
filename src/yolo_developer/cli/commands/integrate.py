"""CLI commands for integrating YOLO MCP with external clients."""

from __future__ import annotations

from pathlib import Path
import json
import os
import platform
import shutil
from typing import Any, Literal

import click
import typer
from typer.core import TyperGroup

from yolo_developer.cli.display import error_panel, info_panel, success_panel, warning_panel

ClientName = Literal["codex", "claude-code", "cursor", "vscode"]

MCP_SERVER_NAME = "yolo-developer"
DEFAULT_CLAUDE_CONFIG_NAME = "claude_desktop_config.json"

class IntegrateCLIGroup(TyperGroup):
    """Parse args without treating the client name as a subcommand."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if not args and self.no_args_is_help and not ctx.resilient_parsing:
            raise click.NoArgsIsHelpError(ctx)
        rest = click.Command.parse_args(self, ctx, args)
        if self.chain:
            ctx._protected_args = rest
            ctx.args = []
        elif rest:
            cmd = self.get_command(ctx, rest[0])
            if cmd is None:
                ctx._protected_args = []
                ctx.args = rest
                return ctx.args
            ctx._protected_args, ctx.args = rest[:1], rest[1:]
        return ctx.args


app = typer.Typer(
    name="integrate",
    help="Integrate YOLO MCP with external clients",
    invoke_without_command=True,
    cls=IntegrateCLIGroup,
    context_settings={"allow_interspersed_args": True},
)


def _default_config_paths(
    client: ClientName,
    *,
    system: str | None = None,
    home: Path | None = None,
    appdata: Path | None = None,
) -> list[Path]:
    system_name = (system or platform.system()).lower()
    home_dir = home or Path.home()
    appdata_dir = appdata or Path(os.environ.get("APPDATA", home_dir))

    if client == "claude-code":
        if system_name == "darwin":
            return [home_dir / ".claude" / DEFAULT_CLAUDE_CONFIG_NAME]
        if system_name == "windows":
            return [appdata_dir / "Claude" / DEFAULT_CLAUDE_CONFIG_NAME]
        return [home_dir / ".config" / "Claude" / DEFAULT_CLAUDE_CONFIG_NAME]

    if client == "codex":
        if system_name == "windows":
            return [appdata_dir / "Codex" / "config.json"]
        return [
            home_dir / ".codex" / "config.json",
            home_dir / ".config" / "codex" / "config.json",
        ]

    if client == "cursor":
        if system_name == "darwin":
            return [home_dir / "Library" / "Application Support" / "Cursor" / "User" / "settings.json"]
        if system_name == "windows":
            return [appdata_dir / "Cursor" / "User" / "settings.json"]
        return [home_dir / ".config" / "Cursor" / "User" / "settings.json"]

    if client == "vscode":
        if system_name == "darwin":
            return [home_dir / "Library" / "Application Support" / "Code" / "User" / "settings.json"]
        if system_name == "windows":
            return [appdata_dir / "Code" / "User" / "settings.json"]
        return [home_dir / ".config" / "Code" / "User" / "settings.json"]

    raise ValueError(f"Unsupported client: {client}")


def _resolve_config_path(client: ClientName, config_path: Path | None) -> Path:
    if config_path is not None:
        return config_path
    candidates = _default_config_paths(client)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_mcp_command(project_dir: Path) -> tuple[str, list[str], str | None]:
    yolo_mcp = shutil.which("yolo-mcp")
    if yolo_mcp:
        return yolo_mcp, [], None
    uv = shutil.which("uv")
    if uv:
        return uv, ["run", "--directory", str(project_dir), "yolo-mcp"], str(project_dir)
    raise RuntimeError("Neither `yolo-mcp` nor `uv` was found on PATH.")


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return {}
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        sanitized = _strip_jsonc(content)
        data = json.loads(sanitized)
        warning_panel(
            f"Config at {path} appears to be JSONC. Comments will be removed when writing."
        )
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return data


def _strip_jsonc(content: str) -> str:
    return _strip_trailing_commas(_strip_jsonc_comments(content))


def _strip_jsonc_comments(content: str) -> str:
    result: list[str] = []
    in_string = False
    escape = False
    in_single_comment = False
    in_multi_comment = False
    i = 0
    length = len(content)

    while i < length:
        ch = content[i]
        nxt = content[i + 1] if i + 1 < length else ""

        if in_single_comment:
            if ch == "\n":
                in_single_comment = False
                result.append(ch)
            i += 1
            continue

        if in_multi_comment:
            if ch == "*" and nxt == "/":
                in_multi_comment = False
                i += 2
                continue
            i += 1
            continue

        if in_string:
            result.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_single_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_multi_comment = True
            i += 2
            continue

        if ch == "\"":
            in_string = True
        result.append(ch)
        i += 1

    return "".join(result)


def _strip_trailing_commas(content: str) -> str:
    result: list[str] = []
    in_string = False
    escape = False
    i = 0
    length = len(content)

    while i < length:
        ch = content[i]

        if in_string:
            result.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            i += 1
            continue

        if ch == "\"":
            in_string = True
            result.append(ch)
            i += 1
            continue

        if ch == ",":
            j = i + 1
            while j < length and content[j].isspace():
                j += 1
            if j < length and content[j] in ("}", "]"):
                i += 1
                continue

        result.append(ch)
        i += 1

    return "".join(result)


def _apply_mcp_entry(
    config: dict[str, Any],
    entry: dict[str, Any],
    *,
    force: bool,
) -> tuple[dict[str, Any], bool]:
    mcp_servers = config.get("mcpServers")
    if mcp_servers is None:
        mcp_servers = {}
    if not isinstance(mcp_servers, dict):
        raise ValueError("Existing mcpServers entry must be a JSON object.")
    existing = mcp_servers.get(MCP_SERVER_NAME)
    if existing is not None and not force and existing != entry:
        raise ValueError(
            f"mcpServers.{MCP_SERVER_NAME} already exists. Use --force to overwrite."
        )
    if existing == entry:
        config["mcpServers"] = mcp_servers
        return config, False
    mcp_servers[MCP_SERVER_NAME] = entry
    config["mcpServers"] = mcp_servers
    return config, True


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


@app.callback()
def integrate(
    client: ClientName = typer.Argument(..., help="Client to integrate."),
    config_path: Path | None = typer.Option(
        None,
        "--config-path",
        "-c",
        help="Override default config path.",
    ),
    project_dir: Path | None = typer.Option(
        None,
        "--project-dir",
        "-p",
        help="Project directory to use for `uv run` fallback.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show resulting JSON without writing.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing MCP entry.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt.",
    ),
) -> None:
    """Integrate YOLO MCP server with a CLI client configuration."""
    path = _resolve_config_path(client, config_path)
    effective_project_dir = project_dir or Path.cwd()
    try:
        command, args, cwd = _resolve_mcp_command(effective_project_dir)
    except RuntimeError as exc:
        error_panel(str(exc))
        raise typer.Exit(code=1) from exc

    entry: dict[str, Any] = {"command": command, "args": args}
    if cwd is not None:
        entry["cwd"] = cwd

    try:
        config = _load_config(path)
        updated, changed = _apply_mcp_entry(config, entry, force=force)
    except (ValueError, json.JSONDecodeError) as exc:
        error_panel(str(exc))
        raise typer.Exit(code=1) from exc

    if dry_run:
        info_panel(f"Dry run: {path}")
        typer.echo(json.dumps(updated, indent=2))
        return

    if not yes:
        confirm = typer.confirm(f"Update {path} with YOLO MCP entry?", default=True)
        if not confirm:
            warning_panel("Aborted.")
            raise typer.Exit(code=1)

    _write_config(path, updated)

    if changed:
        success_panel(f"Updated {path}")
    else:
        info_panel(f"No changes needed in {path}")
    typer.echo("Restart your client and re-open MCP tools to load YOLO.")


__all__ = [
    "ClientName",
    "DEFAULT_CLAUDE_CONFIG_NAME",
    "MCP_SERVER_NAME",
    "_apply_mcp_entry",
    "_default_config_paths",
    "_load_config",
    "_resolve_config_path",
    "_resolve_mcp_command",
]
