"""Tests for MCP integration CLI helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from typer.testing import CliRunner

from yolo_developer.cli.commands import integrate as integrate_commands

runner = CliRunner()


def test_default_paths_claude_macos(tmp_path: Path) -> None:
    paths = integrate_commands._default_config_paths(
        "claude-code",
        system="Darwin",
        home=tmp_path,
    )

    assert paths == [tmp_path / ".claude" / integrate_commands.DEFAULT_CLAUDE_CONFIG_NAME]


def test_default_paths_codex_windows(tmp_path: Path) -> None:
    appdata = tmp_path / "AppData" / "Roaming"
    paths = integrate_commands._default_config_paths(
        "codex",
        system="Windows",
        home=tmp_path,
        appdata=appdata,
    )

    assert paths == [appdata / "Codex" / "config.json"]


def test_resolve_config_path_prefers_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    second.write_text("{}", encoding="utf-8")

    def fake_paths(_client: integrate_commands.ClientName) -> list[Path]:
        return [first, second]

    monkeypatch.setattr(integrate_commands, "_default_config_paths", fake_paths)

    resolved = integrate_commands._resolve_config_path("codex", None)

    assert resolved == second


def test_apply_mcp_entry_adds_entry() -> None:
    config = {}
    entry = {"command": "yolo-mcp", "args": []}

    updated, changed = integrate_commands._apply_mcp_entry(config, entry, force=False)

    assert changed is True
    assert updated["mcpServers"][integrate_commands.MCP_SERVER_NAME] == entry


def test_apply_mcp_entry_no_change_when_same() -> None:
    entry = {"command": "yolo-mcp", "args": []}
    config = {"mcpServers": {integrate_commands.MCP_SERVER_NAME: entry}}

    updated, changed = integrate_commands._apply_mcp_entry(config, entry, force=False)

    assert changed is False
    assert updated["mcpServers"][integrate_commands.MCP_SERVER_NAME] == entry


def test_apply_mcp_entry_requires_force_for_overwrite() -> None:
    config = {"mcpServers": {integrate_commands.MCP_SERVER_NAME: {"command": "old"}}}
    entry = {"command": "yolo-mcp", "args": []}

    with pytest.raises(ValueError, match="already exists"):
        integrate_commands._apply_mcp_entry(config, entry, force=False)


def test_integrate_dry_run_does_not_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "settings.json"

    def fake_mcp_command(_project_dir: Path) -> tuple[str, list[str], str | None]:
        return "yolo-mcp", [], None

    monkeypatch.setattr(integrate_commands, "_resolve_mcp_command", fake_mcp_command)

    result = runner.invoke(
        integrate_commands.app,
        [
            "codex",
            "--config-path",
            str(config_path),
            "--dry-run",
            "--yes",
        ],
    )

    assert result.exit_code == 0
    assert not config_path.exists()
