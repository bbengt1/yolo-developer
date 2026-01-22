"""YOLO Developer - Autonomous multi-agent AI development system using BMad Method.

This package provides the core YOLO Developer system with three external entry points:

1. CLI: Command-line interface for interactive use
   >>> yolo init my-project
   >>> yolo seed requirements.md
   >>> yolo run

2. SDK: Python API for programmatic access
   >>> from yolo_developer import YoloClient
   >>> client = YoloClient()
   >>> result = client.init(project_name="my-project")

3. MCP: Model Context Protocol for external tool integration

Example:
    >>> from yolo_developer import YoloClient, YoloConfig
    >>> config = YoloConfig(project_name="my-app")
    >>> client = YoloClient(config=config)
    >>> client.init()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from yolo_developer.version import __version__

if TYPE_CHECKING:
    from yolo_developer.sdk import (  # noqa: F401
        AuditEntry,
        ClientNotInitializedError,
        GatheringClient,
        InitResult,
        ProjectNotFoundError,
        RunResult,
        SDKError,
        SeedResult,
        SeedValidationError,
        StatusResult,
        WorkflowExecutionError,
        YoloClient,
    )
    from yolo_developer.config import YoloConfig  # noqa: F401

__all__ = [
    # SDK Result Types
    "AuditEntry",
    # SDK Exceptions
    "ClientNotInitializedError",
    "InitResult",
    "GatheringClient",
    "ProjectNotFoundError",
    "RunResult",
    "SDKError",
    "SeedResult",
    "SeedValidationError",
    "StatusResult",
    "WorkflowExecutionError",
    # SDK Client
    "YoloClient",
    # Configuration
    "YoloConfig",
    # Version
    "__version__",
]


def __getattr__(name: str) -> Any:
    if name == "YoloConfig":
        from yolo_developer import config

        return getattr(config, name)
    if name in {
        "AuditEntry",
        "ClientNotInitializedError",
        "GatheringClient",
        "InitResult",
        "ProjectNotFoundError",
        "RunResult",
        "SDKError",
        "SeedResult",
        "SeedValidationError",
        "StatusResult",
        "WorkflowExecutionError",
        "YoloClient",
    }:
        from yolo_developer import sdk

        return getattr(sdk, name)
    raise AttributeError(f"module 'yolo_developer' has no attribute {name}")
