"""CLI commands for YOLO Developer."""

from __future__ import annotations

from typing import Any

__all__ = [
    "config_command",
    "init_command",
    "logs_command",
    "run_command",
    "scan_command",
    "seed_command",
    "status_command",
    "tune_command",
]


def __getattr__(name: str) -> Any:
    if name == "config_command":
        from yolo_developer.cli.commands.config import config_command

        return config_command
    if name == "init_command":
        from yolo_developer.cli.commands.init import init_command

        return init_command
    if name == "logs_command":
        from yolo_developer.cli.commands.logs import logs_command

        return logs_command
    if name == "run_command":
        from yolo_developer.cli.commands.run import run_command

        return run_command
    if name == "scan_command":
        from yolo_developer.cli.commands.scan import scan_command

        return scan_command
    if name == "seed_command":
        from yolo_developer.cli.commands.seed import seed_command

        return seed_command
    if name == "status_command":
        from yolo_developer.cli.commands.status import status_command

        return status_command
    if name == "tune_command":
        from yolo_developer.cli.commands.tune import tune_command

        return tune_command
    raise AttributeError(f"module 'yolo_developer.cli.commands' has no attribute {name}")
