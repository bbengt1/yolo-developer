"""Python SDK for YOLO Developer (Story 13.1).

This module provides the YoloClient class for programmatic access to
all YOLO Developer functionality including project initialization,
seed processing, workflow execution, and audit access.

The SDK is one of three external entry points to YOLO Developer:
- CLI: Command-line interface for interactive use
- SDK: Python API for programmatic access (this module)
- MCP: Model Context Protocol for external tool integration

Example:
    >>> from yolo_developer import YoloClient
    >>>
    >>> # Initialize with default config
    >>> client = YoloClient()
    >>>
    >>> # Or with custom config
    >>> from yolo_developer.config import YoloConfig
    >>> config = YoloConfig(project_name="my-project")
    >>> client = YoloClient(config=config)
    >>>
    >>> # Initialize a new project
    >>> init_result = client.init(project_name="my-app")
    >>>
    >>> # Process a seed document
    >>> seed_result = client.seed(content="Build a REST API with authentication")
    >>>
    >>> # Run workflow (async)
    >>> result = await client.run_async(seed_id=seed_result.seed_id)
    >>>
    >>> # Check status
    >>> status = client.status()
    >>> print(f"Project: {status.project_name}, Status: {status.workflow_status}")
    >>>
    >>> # Get audit trail
    >>> entries = client.get_audit(agent_filter="analyst")

References:
    - FR106: Developers can initialize projects programmatically via SDK
    - FR107: Developers can provide seeds and execute runs via SDK
    - FR108: Developers can access audit trail data via SDK
    - FR109: Developers can configure all project settings via SDK
    - ADR-009: SDK as programmatic API layer
"""

from __future__ import annotations

from yolo_developer.sdk.client import YoloClient
from yolo_developer.sdk.exceptions import (
    ClientNotInitializedError,
    ProjectNotFoundError,
    SDKError,
    SeedValidationError,
    WorkflowExecutionError,
)
from yolo_developer.sdk.types import (
    AuditEntry,
    InitResult,
    RunResult,
    SeedResult,
    StatusResult,
)

__all__ = [
    "AuditEntry",
    "ClientNotInitializedError",
    "InitResult",
    "ProjectNotFoundError",
    "RunResult",
    "SDKError",
    "SeedResult",
    "SeedValidationError",
    "StatusResult",
    "WorkflowExecutionError",
    "YoloClient",
]
