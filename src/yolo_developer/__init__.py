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

from yolo_developer.config import YoloConfig
from yolo_developer.sdk import (
    AuditEntry,
    ClientNotInitializedError,
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

__version__ = "0.1.0"

__all__ = [
    # SDK Result Types
    "AuditEntry",
    # SDK Exceptions
    "ClientNotInitializedError",
    "InitResult",
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
