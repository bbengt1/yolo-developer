"""YoloClient class for programmatic SDK access (Stories 13.1, 13.2, 13.3, 13.5).

This module provides the main YoloClient class that serves as the
primary entry point for programmatic access to YOLO Developer.

Example:
    >>> from yolo_developer import YoloClient
    >>>
    >>> # Initialize with default config
    >>> client = YoloClient()
    >>>
    >>> # Or from config file
    >>> client = YoloClient.from_config_file("./yolo.yaml")
    >>>
    >>> # Or with custom config
    >>> from yolo_developer.config import YoloConfig
    >>> config = YoloConfig(project_name="my-project")
    >>> client = YoloClient(config=config)

References:
    - FR106: Developers can initialize projects programmatically via SDK
    - FR107: Developers can provide seeds and execute runs via SDK
    - FR108: Developers can access audit trail data via SDK
    - FR109: Developers can configure all project settings via SDK
    - FR110: Developers can extend agent behavior via SDK hooks
    - AC1: YoloClient instantiation with configuration
    - AC2: Full functionality access
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import structlog

from yolo_developer.config import ConfigurationError, YoloConfig, load_config
from yolo_developer.sdk.exceptions import (
    ClientNotInitializedError,
    ConfigurationAPIError,
    ProjectNotFoundError,
    SDKError,
    SeedValidationError,
    WorkflowExecutionError,
)
from yolo_developer.sdk.types import (
    AuditEntry,
    ConfigSaveResult,
    ConfigUpdateResult,
    ConfigValidationIssue,
    ConfigValidationResult,
    HookRegistration,
    HookResult,
    InitResult,
    PostHook,
    PreHook,
    RunResult,
    SeedResult,
    StatusResult,
)

if TYPE_CHECKING:
    from datetime import datetime

logger = structlog.get_logger(__name__)

_T = TypeVar("_T")


def _run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run a coroutine synchronously with proper event loop handling.

    This helper handles the deprecated asyncio.get_event_loop() behavior
    in Python 3.10+ by creating a new event loop if one doesn't exist.

    Args:
        coro: The coroutine to run.

    Returns:
        The result of the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, use asyncio.run() which creates one
        return asyncio.run(coro)
    else:
        # Already in an async context, can't use asyncio.run()
        # This should rarely happen for SDK sync methods
        return loop.run_until_complete(coro)


class YoloClient:
    """Python SDK client for YOLO Developer.

    Provides programmatic access to all YOLO Developer functionality including
    project initialization, seed processing, workflow execution, and audit access.

    The client supports both synchronous and asynchronous operations. Sync methods
    wrap async operations for convenience, while async methods provide the full
    async API for advanced users.

    Attributes:
        config: The YoloConfig instance used by this client.
        project_path: Path to the project directory.
        is_initialized: Whether the project has been initialized.

    Example:
        >>> from yolo_developer import YoloClient
        >>>
        >>> # Initialize with default config
        >>> client = YoloClient()
        >>>
        >>> # Or with custom config
        >>> client = YoloClient(config=my_config)
        >>>
        >>> # Initialize a project
        >>> result = client.init(project_name="my-app")
        >>>
        >>> # Process a seed
        >>> seed_result = client.seed(content="Build a REST API")
        >>>
        >>> # Run workflow
        >>> run_result = await client.run_async(seed_id=seed_result.seed_id)
    """

    def __init__(
        self,
        config: YoloConfig | None = None,
        *,
        project_path: Path | str | None = None,
    ) -> None:
        """Initialize YoloClient.

        Args:
            config: Optional YoloConfig instance. If not provided, loads from
                default location (./yolo.yaml) or uses defaults.
            project_path: Optional project directory path. Defaults to current directory.

        Raises:
            SDKError: If configuration loading fails.

        Example:
            >>> # Default config
            >>> client = YoloClient()
            >>>
            >>> # Custom config
            >>> from yolo_developer.config import YoloConfig
            >>> config = YoloConfig(project_name="my-project")
            >>> client = YoloClient(config=config)
            >>>
            >>> # Specific project path
            >>> client = YoloClient(project_path="/path/to/project")
        """
        self._project_path = Path(project_path) if project_path else Path.cwd()
        self._hooks: dict[str, HookRegistration] = {}  # hook_id -> registration

        try:
            if config is not None:
                self._config = config
            else:
                # Try to load config from project path
                config_file = self._project_path / "yolo.yaml"
                if config_file.exists():
                    self._config = load_config(config_file)
                else:
                    # Use default config
                    self._config = YoloConfig(project_name="untitled")
        except ConfigurationError as e:
            raise SDKError(
                f"Failed to load configuration: {e}",
                original_error=e,
                details={"project_path": str(self._project_path)},
            ) from e

        logger.debug(
            "yolo_client_initialized",
            project_path=str(self._project_path),
            project_name=self._config.project_name,
        )

    @classmethod
    def from_config_file(
        cls,
        config_path: Path | str,
        *,
        project_path: Path | str | None = None,
    ) -> YoloClient:
        """Create a YoloClient from a configuration file.

        Args:
            config_path: Path to the configuration file (YAML).
            project_path: Optional project directory path. Defaults to the
                parent directory of the config file.

        Returns:
            A new YoloClient instance configured from the file.

        Raises:
            SDKError: If the configuration file cannot be loaded.

        Example:
            >>> client = YoloClient.from_config_file("./yolo.yaml")
            >>> client = YoloClient.from_config_file("/path/to/config.yaml")
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise SDKError(
                f"Configuration file not found: {config_path}",
                details={"config_path": str(config_path)},
            )

        try:
            config = load_config(config_path)
        except ConfigurationError as e:
            raise SDKError(
                f"Failed to load configuration from {config_path}: {e}",
                original_error=e,
                details={"config_path": str(config_path)},
            ) from e

        # Default project path to config file's directory
        if project_path is None:
            project_path = config_path.parent

        return cls(config=config, project_path=project_path)

    @property
    def config(self) -> YoloConfig:
        """Get the current configuration.

        Returns:
            The YoloConfig instance used by this client.
        """
        return self._config

    @property
    def project_path(self) -> Path:
        """Get the project directory path.

        Returns:
            The Path to the project directory.
        """
        return self._project_path

    @property
    def is_initialized(self) -> bool:
        """Check if the project is initialized.

        Returns:
            True if the project has been initialized, False otherwise.
        """
        # Check for .yolo directory or yolo.yaml file
        yolo_dir = self._project_path / ".yolo"
        yolo_config = self._project_path / "yolo.yaml"
        return yolo_dir.exists() or yolo_config.exists()

    # =========================================================================
    # Configuration API Methods (Story 13.4)
    # =========================================================================

    def update_config(
        self,
        *,
        llm: dict[str, Any] | None = None,
        quality: dict[str, Any] | None = None,
        memory: dict[str, Any] | None = None,
        project_name: str | None = None,
        persist: bool = False,
    ) -> ConfigUpdateResult:
        """Update configuration settings.

        Updates the in-memory configuration with the provided partial settings.
        Only specified fields are updated; unspecified fields retain their values.
        Validation runs before applying changes.

        Args:
            llm: Partial LLM configuration to update (e.g., {"cheap_model": "gpt-4o"}).
            quality: Partial quality configuration to update.
            memory: Partial memory configuration to update.
            project_name: New project name.
            persist: If True, save changes to yolo.yaml after updating.

        Returns:
            ConfigUpdateResult with update status and validation results.

        Raises:
            ConfigurationAPIError: If validation fails with errors.

        Example:
            >>> # Update quality thresholds
            >>> result = client.update_config(
            ...     quality={"test_coverage_threshold": 0.85},
            ...     persist=True
            ... )
            >>> if result.success:
            ...     print("Configuration updated")
            >>>
            >>> # Update multiple sections
            >>> result = client.update_config(
            ...     llm={"cheap_model": "gpt-4o"},
            ...     quality={"confidence_threshold": 0.95}
            ... )
        """
        return _run_sync(
            self.update_config_async(
                llm=llm,
                quality=quality,
                memory=memory,
                project_name=project_name,
                persist=persist,
            )
        )

    async def update_config_async(
        self,
        *,
        llm: dict[str, Any] | None = None,
        quality: dict[str, Any] | None = None,
        memory: dict[str, Any] | None = None,
        project_name: str | None = None,
        persist: bool = False,
    ) -> ConfigUpdateResult:
        """Update configuration settings (async version).

        Updates the in-memory configuration with the provided partial settings.
        Only specified fields are updated; unspecified fields retain their values.
        Validation runs before applying changes.

        Args:
            llm: Partial LLM configuration to update.
            quality: Partial quality configuration to update.
            memory: Partial memory configuration to update.
            project_name: New project name.
            persist: If True, save changes to yolo.yaml after updating.

        Returns:
            ConfigUpdateResult with update status and validation results.

        Raises:
            ConfigurationAPIError: If validation fails with errors.
        """
        from pydantic import ValidationError as PydanticValidationError

        # Track previous values
        previous_values: dict[str, Any] = {}
        new_values: dict[str, Any] = {}

        # Build updates dict
        current_data = self._config.model_dump()

        if project_name is not None:
            previous_values["project_name"] = current_data["project_name"]
            new_values["project_name"] = project_name
            current_data["project_name"] = project_name

        if llm is not None:
            previous_values["llm"] = {
                k: current_data["llm"][k] for k in llm if k in current_data["llm"]
            }
            new_values["llm"] = llm
            current_data["llm"] = {**current_data["llm"], **llm}

        if quality is not None:
            previous_values["quality"] = {
                k: current_data["quality"][k] for k in quality if k in current_data["quality"]
            }
            new_values["quality"] = quality
            current_data["quality"] = {**current_data["quality"], **quality}

        if memory is not None:
            previous_values["memory"] = {
                k: current_data["memory"][k] for k in memory if k in current_data["memory"]
            }
            new_values["memory"] = memory
            current_data["memory"] = {**current_data["memory"], **memory}

        # Validate new configuration
        try:
            new_config = YoloConfig(**current_data)
        except PydanticValidationError as e:
            error_messages = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            raise ConfigurationAPIError(
                f"Configuration validation failed: {'; '.join(error_messages)}",
                validation_errors=error_messages,
                original_error=e,
            ) from e

        # Run additional validation
        validation_result = await self.validate_config_async(config=new_config)

        # Check for fatal errors
        if not validation_result.is_valid:
            error_messages = [issue.message for issue in validation_result.errors]
            raise ConfigurationAPIError(
                f"Configuration validation failed: {'; '.join(error_messages)}",
                validation_errors=error_messages,
            )

        # Apply the new configuration
        self._config = new_config

        # Persist if requested
        persisted = False
        if persist:
            save_result = await self.save_config_async()
            persisted = save_result.success

        logger.info(
            "config_updated",
            updated_fields=list(new_values.keys()),
            persisted=persisted,
        )

        return ConfigUpdateResult(
            success=True,
            previous_values=previous_values,
            new_values=new_values,
            persisted=persisted,
            validation=validation_result,
        )

    def validate_config(
        self,
        *,
        config: YoloConfig | None = None,
    ) -> ConfigValidationResult:
        """Validate configuration settings.

        Runs all validation checks on the configuration and returns
        a result with any errors or warnings found.

        Args:
            config: Configuration to validate. Defaults to current config.

        Returns:
            ConfigValidationResult with validation status and issues.

        Example:
            >>> result = client.validate_config()
            >>> if result.is_valid:
            ...     print("Configuration is valid")
            >>> for issue in result.warnings:
            ...     print(f"Warning: {issue.message}")
        """
        return _run_sync(self.validate_config_async(config=config))

    async def validate_config_async(
        self,
        *,
        config: YoloConfig | None = None,
    ) -> ConfigValidationResult:
        """Validate configuration settings (async version).

        Runs all validation checks on the configuration and returns
        a result with any errors or warnings found.

        Args:
            config: Configuration to validate. Defaults to current config.

        Returns:
            ConfigValidationResult with validation status and issues.
        """
        from yolo_developer.config import validate_config as _validate_config

        config_to_validate = config or self._config
        result = _validate_config(config_to_validate)

        # Convert to SDK types
        issues: list[ConfigValidationIssue] = []

        for error in result.errors:
            issues.append(
                ConfigValidationIssue(
                    field=error.field,
                    message=error.message,
                    severity="error",
                )
            )

        for warning in result.warnings:
            issues.append(
                ConfigValidationIssue(
                    field=warning.field,
                    message=warning.message,
                    severity="warning",
                )
            )

        return ConfigValidationResult(
            is_valid=result.is_valid,
            issues=issues,
        )

    def save_config(self) -> ConfigSaveResult:
        """Save configuration to yolo.yaml.

        Persists the current configuration to the project's yolo.yaml file.
        API keys are excluded from the saved file for security.

        Returns:
            ConfigSaveResult with save status and file path.

        Raises:
            ConfigurationAPIError: If save operation fails.

        Example:
            >>> result = client.save_config()
            >>> if result.success:
            ...     print(f"Config saved to {result.config_path}")
        """
        return _run_sync(self.save_config_async())

    async def save_config_async(self) -> ConfigSaveResult:
        """Save configuration to yolo.yaml (async version).

        Persists the current configuration to the project's yolo.yaml file.
        API keys are excluded from the saved file for security.

        Returns:
            ConfigSaveResult with save status and file path.

        Raises:
            ConfigurationAPIError: If save operation fails.
        """
        from yolo_developer.config import export_config

        config_path = self._project_path / "yolo.yaml"

        try:
            export_config(self._config, config_path)
        except Exception as e:
            raise ConfigurationAPIError(
                f"Failed to save configuration: {e}",
                original_error=e,
                details={"config_path": str(config_path)},
            ) from e

        logger.info(
            "config_saved",
            config_path=str(config_path),
        )

        return ConfigSaveResult(
            success=True,
            config_path=str(config_path),
            secrets_excluded=["openai_api_key", "anthropic_api_key"],
        )

    def init(
        self,
        *,
        project_name: str | None = None,
        force: bool = False,
    ) -> InitResult:
        """Initialize a new YOLO Developer project.

        Creates the necessary project structure and configuration files
        for a new YOLO Developer project.

        Args:
            project_name: Name for the project. Defaults to directory name.
            force: If True, reinitialize even if already initialized.

        Returns:
            InitResult containing initialization details.

        Raises:
            SDKError: If initialization fails.
            ClientNotInitializedError: If the project already exists and force=False.

        Example:
            >>> result = client.init(project_name="my-app")
            >>> print(f"Initialized at {result.project_path}")
        """
        return _run_sync(self.init_async(project_name=project_name, force=force))

    async def init_async(
        self,
        *,
        project_name: str | None = None,
        force: bool = False,
    ) -> InitResult:
        """Initialize a new YOLO Developer project (async).

        Creates the necessary project structure and configuration files
        for a new YOLO Developer project.

        Args:
            project_name: Name for the project. Defaults to directory name.
            force: If True, reinitialize even if already initialized.

        Returns:
            InitResult containing initialization details.

        Raises:
            SDKError: If initialization fails.
            ClientNotInitializedError: If the project already exists and force=False.

        Example:
            >>> result = await client.init_async(project_name="my-app")
            >>> print(f"Initialized at {result.project_path}")
        """
        from datetime import datetime, timezone

        if self.is_initialized and not force:
            raise ClientNotInitializedError(
                f"Project already initialized at {self._project_path}. Use force=True to reinitialize.",
                details={"project_path": str(self._project_path)},
            )

        # Use directory name as default project name
        name = project_name or self._project_path.name

        # Create .yolo directory
        yolo_dir = self._project_path / ".yolo"
        directories_created = []

        try:
            if not yolo_dir.exists():
                yolo_dir.mkdir(parents=True)
                directories_created.append(str(yolo_dir))

            # Create subdirectories
            for subdir in ["sessions", "memory", "audit"]:
                subdir_path = yolo_dir / subdir
                if not subdir_path.exists():
                    subdir_path.mkdir()
                    directories_created.append(str(subdir_path))

            # Create config file if not exists
            config_file = self._project_path / "yolo.yaml"
            config_created = False
            if not config_file.exists() or force:
                # Update config with project name
                self._config = YoloConfig(
                    project_name=name,
                    **{k: v for k, v in self._config.model_dump().items() if k != "project_name"},
                )
                # Write comprehensive config matching CLI format
                config_content = f"""# YOLO Developer Configuration
# Generated by YoloClient.init()
# Documentation: https://github.com/anthropics/yolo-developer

# Project name (required)
project_name: {name}

# LLM Provider Configuration
llm:
  # Model for routine, low-complexity tasks
  cheap_model: gpt-4o-mini

  # Model for complex reasoning tasks
  premium_model: claude-sonnet-4-20250514

  # Model for critical decisions requiring highest quality
  best_model: claude-opus-4-5-20251101

  # API keys should be set via environment variables:
  # - YOLO_LLM__OPENAI_API_KEY
  # - YOLO_LLM__ANTHROPIC_API_KEY

# Quality Gate Configuration
quality:
  # Minimum test coverage ratio (0.0-1.0)
  test_coverage_threshold: 0.80

  # Minimum confidence score (0.0-1.0) for deployment approval
  confidence_threshold: 0.90

  # Seed quality thresholds
  seed_thresholds:
    overall: 0.70
    ambiguity: 0.60
    sop: 0.80

  # Path patterns that require 100% test coverage
  critical_paths:
    - orchestrator/
    - gates/
    - agents/

# Memory & Storage Configuration
memory:
  # Directory path for persisting memory data
  persist_path: .yolo/memory

  # Vector store backend (chromadb)
  vector_store_type: chromadb

  # Graph store backend (json for MVP, neo4j optional)
  graph_store_type: json
"""
                config_file.write_text(config_content)
                config_created = True

            logger.info(
                "project_initialized",
                project_path=str(self._project_path),
                project_name=name,
                directories_created=len(directories_created),
            )

            return InitResult(
                project_path=str(self._project_path),
                project_name=name,
                config_created=config_created,
                directories_created=directories_created,
                timestamp=datetime.now(timezone.utc),
            )

        except OSError as e:
            raise SDKError(
                f"Failed to initialize project: {e}",
                original_error=e,
                details={"project_path": str(self._project_path)},
            ) from e

    def seed(
        self,
        content: str,
        *,
        source: str | None = None,
        validate: bool = True,
    ) -> SeedResult:
        """Process a seed document.

        Parses and validates a natural language seed document,
        extracting goals, features, and constraints.

        Args:
            content: The seed document content.
            source: Optional source identifier (e.g., filename).
            validate: If True, perform full validation including ambiguity detection.

        Returns:
            SeedResult containing parsing results and quality metrics.

        Raises:
            SeedValidationError: If validation fails.
            SDKError: If seed processing fails.

        Example:
            >>> result = client.seed(content="Build an e-commerce platform")
            >>> print(f"Found {result.goal_count} goals, quality: {result.quality_score:.0%}")
        """
        return _run_sync(self.seed_async(content=content, source=source, validate=validate))

    async def seed_async(
        self,
        content: str,
        *,
        source: str | None = None,
        validate: bool = True,
    ) -> SeedResult:
        """Process a seed document (async).

        Parses and validates a natural language seed document,
        extracting goals, features, and constraints.

        Args:
            content: The seed document content.
            source: Optional source identifier (e.g., filename).
            validate: If True, perform full validation including ambiguity detection.

        Returns:
            SeedResult containing parsing results and quality metrics.

        Raises:
            SeedValidationError: If validation fails.
            SDKError: If seed processing fails.

        Example:
            >>> result = await client.seed_async(content="Build an e-commerce platform")
            >>> print(f"Found {result.goal_count} goals")
        """
        import uuid
        from datetime import datetime, timezone

        from yolo_developer.seed import parse_seed

        try:
            parse_result = await parse_seed(
                content,
                filename=source,
                detect_ambiguities=validate,
            )

            seed_id = f"seed-{uuid.uuid4().hex[:8]}"

            # Extract ambiguity descriptions
            ambiguities: list[str] = []
            if parse_result.has_ambiguities:
                ambiguities = [amb.description for amb in parse_result.ambiguities]

            # Calculate quality score using config thresholds (matching CLI behavior)
            quality_score = self._calculate_seed_quality_score(parse_result)

            # Determine status based on quality thresholds from config
            status = self._determine_seed_status(quality_score, ambiguities)

            # Collect warnings
            warnings: list[str] = []
            if parse_result.goal_count == 0:
                warnings.append("No goals detected in seed document")
            if parse_result.feature_count == 0:
                warnings.append("No features detected in seed document")

            logger.info(
                "seed_processed",
                seed_id=seed_id,
                goal_count=parse_result.goal_count,
                feature_count=parse_result.feature_count,
                ambiguity_count=len(ambiguities),
                quality_score=quality_score,
                status=status,
            )

            return SeedResult(
                seed_id=seed_id,
                status=status,
                goal_count=parse_result.goal_count,
                feature_count=parse_result.feature_count,
                constraint_count=parse_result.constraint_count,
                ambiguities=ambiguities,
                quality_score=quality_score,
                warnings=warnings,
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            raise SeedValidationError(
                f"Failed to process seed: {e}",
                original_error=e,
                details={"source": source},
            ) from e

    def _calculate_seed_quality_score(self, parse_result: Any) -> float:
        """Calculate quality score for a seed document.

        Uses a heuristic scoring algorithm based on seed content quality:
        - Base score: 1.0
        - Ambiguity penalty: -0.05 per ambiguity (max -0.3)
        - No goals penalty: -0.2
        - No features penalty: -0.1

        Args:
            parse_result: The SeedParseResult from parsing.

        Returns:
            Quality score between 0.0 and 1.0.
        """
        # Base score starts at 1.0
        score = 1.0

        # Penalize for ambiguities
        if parse_result.has_ambiguities:
            ambiguity_count = len(parse_result.ambiguities)
            # Each ambiguity reduces score by 0.05, capped at 0.3 reduction
            score -= min(0.3, ambiguity_count * 0.05)

        # Penalize for missing goals
        if parse_result.goal_count == 0:
            score -= 0.2

        # Penalize for missing features
        if parse_result.feature_count == 0:
            score -= 0.1

        # Ensure score stays in valid range
        return max(0.0, min(1.0, score))

    def _determine_seed_status(
        self,
        quality_score: float,
        ambiguities: list[str],
    ) -> Literal["accepted", "rejected", "pending"]:
        """Determine seed status based on quality score and config thresholds.

        Uses the same thresholds as the CLI seed command.

        Args:
            quality_score: The calculated quality score.
            ambiguities: List of detected ambiguities.

        Returns:
            Status string: 'accepted', 'rejected', or 'pending'.
        """
        # Get thresholds from config
        overall_threshold = self._config.quality.seed_thresholds.overall
        ambiguity_threshold = self._config.quality.seed_thresholds.ambiguity

        # Calculate ambiguity score (inverse of ambiguity rate)
        ambiguity_score = 1.0 - min(1.0, len(ambiguities) * 0.1)

        # Reject if overall quality is below threshold
        if quality_score < overall_threshold:
            return "rejected"

        # Pending if ambiguity score is below threshold (needs clarification)
        if ambiguity_score < ambiguity_threshold:
            return "pending"

        return "accepted"

    def run(
        self,
        *,
        seed_id: str | None = None,
        seed_content: str | None = None,
    ) -> RunResult:
        """Execute a workflow.

        Runs the full YOLO Developer workflow, executing all agents
        to process the seed and generate output.

        Args:
            seed_id: ID of a previously processed seed.
            seed_content: Direct seed content to process and run.

        Returns:
            RunResult containing execution results.

        Raises:
            WorkflowExecutionError: If workflow execution fails.
            ClientNotInitializedError: If project is not initialized.
            SDKError: If run fails for other reasons.

        Example:
            >>> result = client.run(seed_content="Build a REST API")
            >>> print(f"Workflow {result.workflow_id}: {result.status}")
        """
        return _run_sync(self.run_async(seed_id=seed_id, seed_content=seed_content))

    async def run_async(
        self,
        *,
        seed_id: str | None = None,
        seed_content: str | None = None,
    ) -> RunResult:
        """Execute a workflow (async).

        Runs the full YOLO Developer workflow, executing all agents
        to process the seed and generate output.

        Args:
            seed_id: ID of a previously processed seed.
            seed_content: Direct seed content to process and run.

        Returns:
            RunResult containing execution results.

        Raises:
            WorkflowExecutionError: If workflow execution fails.
            ClientNotInitializedError: If project is not initialized.
            SDKError: If run fails for other reasons.

        Example:
            >>> result = await client.run_async(seed_content="Build a REST API")
            >>> print(f"Completed {result.stories_completed} stories")
        """
        import time
        import uuid
        from datetime import datetime, timezone

        if not self.is_initialized:
            raise ClientNotInitializedError(
                "Project not initialized. Call init() first.",
                details={"project_path": str(self._project_path)},
            )

        if seed_id is None and seed_content is None:
            raise SDKError(
                "Either seed_id or seed_content must be provided",
                details={"seed_id": seed_id, "seed_content": bool(seed_content)},
            )

        # seed_id lookup is not yet implemented - requires seed persistence layer
        if seed_id is not None and seed_content is None:
            raise SDKError(
                "seed_id lookup is not yet implemented. Please provide seed_content instead.",
                details={"seed_id": seed_id},
            )

        workflow_id = f"workflow-{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        agents_executed: list[str] = []
        errors: list[str] = []

        try:
            # Import orchestrator components
            from yolo_developer.orchestrator import (
                WorkflowConfig,
                create_initial_state,
                run_workflow,
            )

            # If seed_content provided, process it first
            if seed_content:
                seed_result = await self.seed_async(content=seed_content)
                if seed_result.status == "rejected":
                    raise WorkflowExecutionError(
                        "Seed was rejected",
                        workflow_id=workflow_id,
                        details={"seed_id": seed_result.seed_id},
                    )

            # Create workflow config
            workflow_config = WorkflowConfig(
                entry_point="analyst",
                enable_checkpointing=True,
            )

            # Create initial state
            initial_state = create_initial_state(
                starting_agent="analyst",
            )

            # Execute pre-hooks before workflow starts (Story 13.5)
            # Note: Per-agent hooks require orchestrator-level integration.
            # Currently hooks fire at workflow boundaries (start/end).
            entry_agent = workflow_config.entry_point
            pre_modifications, _pre_results = await self._execute_pre_hooks(
                entry_agent, dict(initial_state)
            )
            if pre_modifications:
                # Merge hook modifications into initial state.
                # Hooks can inject additional context beyond the TypedDict schema.
                state_dict: dict[str, Any] = dict(initial_state)
                state_dict.update(pre_modifications)
                initial_state = state_dict  # type: ignore[assignment]

            # Run workflow
            final_state = await run_workflow(
                initial_state=initial_state,
                config=workflow_config,
            )

            # Execute post-hooks after workflow completes (Story 13.5)
            last_agent = final_state.get("current_agent", entry_agent)
            workflow_output = {
                "decisions": final_state.get("decisions", []),
                "messages": final_state.get("messages", []),
                "current_agent": last_agent,
            }
            post_modifications, _post_results = await self._execute_post_hooks(
                last_agent, dict(initial_state), workflow_output
            )
            if post_modifications:
                # Post-hooks can modify the workflow output
                workflow_output = post_modifications

            # Extract results from final state
            # Note: YoloState has messages, current_agent, handoff_context, decisions
            # Workflow execution details would be derived from these
            decisions_list = final_state.get("decisions", [])
            agents_executed = list({d.agent for d in decisions_list})
            # For now, stories are not tracked in YoloState directly
            # This would need to be expanded when story tracking is implemented
            stories_completed = 0
            stories_total = 0

            duration = time.time() - start_time

            logger.info(
                "workflow_completed",
                workflow_id=workflow_id,
                agents_executed=agents_executed,
                stories_completed=stories_completed,
                duration_seconds=duration,
            )

            return RunResult(
                workflow_id=workflow_id,
                status="completed",
                agents_executed=agents_executed,
                stories_completed=stories_completed,
                stories_total=stories_total,
                duration_seconds=duration,
                artifacts=[],
                errors=errors,
                timestamp=datetime.now(timezone.utc),
            )

        except (ClientNotInitializedError, SDKError):
            raise
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            errors.append(error_msg)

            logger.error(
                "workflow_failed",
                workflow_id=workflow_id,
                error=error_msg,
                duration_seconds=duration,
            )

            raise WorkflowExecutionError(
                f"Workflow execution failed: {e}",
                workflow_id=workflow_id,
                agent=agents_executed[-1] if agents_executed else None,
                original_error=e,
                details={"duration_seconds": duration},
            ) from e

    def status(self) -> StatusResult:
        """Get current project status.

        Returns status information about the current project,
        including initialization state and any running workflows.

        Returns:
            StatusResult containing project status information.

        Raises:
            ProjectNotFoundError: If project path doesn't exist.
            SDKError: If status retrieval fails.

        Example:
            >>> status = client.status()
            >>> if status.is_initialized:
            ...     print(f"Project: {status.project_name}")
        """
        return _run_sync(self.status_async())

    async def status_async(self) -> StatusResult:
        """Get current project status (async).

        Returns status information about the current project,
        including initialization state and any running workflows.

        Returns:
            StatusResult containing project status information.

        Raises:
            ProjectNotFoundError: If project path doesn't exist.
            SDKError: If status retrieval fails.

        Example:
            >>> status = await client.status_async()
            >>> print(f"Workflow status: {status.workflow_status}")
        """
        if not self._project_path.exists():
            raise ProjectNotFoundError(
                f"Project path does not exist: {self._project_path}",
                project_path=str(self._project_path),
            )

        return StatusResult(
            project_name=self._config.project_name,
            project_path=str(self._project_path),
            is_initialized=self.is_initialized,
            current_sprint=None,  # Would query session manager
            active_agent=None,  # Would query orchestrator
            workflow_status="idle",
            last_activity=None,
            stats={},
        )

    def get_audit(
        self,
        *,
        agent_filter: str | None = None,
        decision_type: str | None = None,
        artifact_type: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEntry]:
        """Get audit trail entries.

        Retrieves audit trail entries with optional filtering and pagination.

        Args:
            agent_filter: Filter by agent name (e.g., "analyst", "pm", "dev").
            decision_type: Filter by decision type (e.g., "requirement_analysis").
            artifact_type: Reserved for future artifact filtering. Currently not
                implemented for decision filtering.
            start_time: Filter entries after this time (inclusive).
            end_time: Filter entries before this time (inclusive).
            limit: Maximum number of entries to return (default: 100).
            offset: Number of entries to skip for pagination (default: 0).

        Returns:
            List of AuditEntry objects matching the filters.

        Raises:
            ClientNotInitializedError: If project is not initialized.
            SDKError: If audit retrieval fails.

        Example:
            >>> # Get all entries from analyst agent
            >>> entries = client.get_audit(agent_filter="analyst", limit=50)
            >>> for entry in entries:
            ...     print(f"{entry.agent}: {entry.content}")
            >>>
            >>> # Filter by artifact type
            >>> requirement_entries = client.get_audit(artifact_type="requirement")
            >>>
            >>> # Paginate through results
            >>> page1 = client.get_audit(limit=10, offset=0)
            >>> page2 = client.get_audit(limit=10, offset=10)
        """
        return _run_sync(
            self.get_audit_async(
                agent_filter=agent_filter,
                decision_type=decision_type,
                artifact_type=artifact_type,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
                offset=offset,
            )
        )

    async def get_audit_async(
        self,
        *,
        agent_filter: str | None = None,
        decision_type: str | None = None,
        artifact_type: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEntry]:
        """Get audit trail entries (async).

        Retrieves audit trail entries with optional filtering and pagination.

        Args:
            agent_filter: Filter by agent name (e.g., "analyst", "pm", "dev").
            decision_type: Filter by decision type (e.g., "requirement_analysis").
            artifact_type: Reserved for future artifact filtering. Currently not
                implemented for decision filtering.
            start_time: Filter entries after this time (inclusive).
            end_time: Filter entries before this time (inclusive).
            limit: Maximum number of entries to return (default: 100).
            offset: Number of entries to skip for pagination (default: 0).

        Returns:
            List of AuditEntry objects matching the filters.

        Raises:
            ClientNotInitializedError: If project is not initialized.
            SDKError: If audit retrieval fails.

        Example:
            >>> # Get entries with multiple filters
            >>> entries = await client.get_audit_async(
            ...     agent_filter="analyst",
            ...     decision_type="requirement_analysis",
            ...     artifact_type="requirement",
            ...     limit=20,
            ... )
            >>> print(f"Found {len(entries)} entries")
            >>>
            >>> # Paginate through large result sets
            >>> all_entries = []
            >>> offset = 0
            >>> while True:
            ...     batch = await client.get_audit_async(limit=50, offset=offset)
            ...     if not batch:
            ...         break
            ...     all_entries.extend(batch)
            ...     offset += 50
        """
        if not self.is_initialized:
            raise ClientNotInitializedError(
                "Project not initialized. Call init() first.",
                details={"project_path": str(self._project_path)},
            )

        try:
            from yolo_developer.audit import (
                AuditFilters,
                InMemoryTraceabilityStore,
                get_audit_filter_service,
            )

            # Get or create the audit store for this project
            # Note: Currently uses in-memory store. For persistent storage,
            # implement a file-based DecisionStore that loads from .yolo/audit/
            decision_store = self._get_decision_store()
            traceability_store = InMemoryTraceabilityStore()
            filter_service = get_audit_filter_service(
                decision_store=decision_store,
                traceability_store=traceability_store,
                cost_store=None,
            )

            # Apply filters including decision_type and artifact_type
            filters = AuditFilters(
                agent_name=agent_filter,
                decision_type=decision_type,
                artifact_type=artifact_type,
                start_time=start_time.isoformat() if start_time else None,
                end_time=end_time.isoformat() if end_time else None,
            )

            # Query decisions
            results = await filter_service.filter_all(filters)
            decisions = results.get("decisions", [])

            # Apply pagination (offset, then limit)
            paginated_decisions = decisions[offset : offset + limit]

            # Convert to AuditEntry
            entries = []
            for decision in paginated_decisions:
                entries.append(
                    AuditEntry(
                        entry_id=decision.id,
                        timestamp=decision.timestamp,
                        agent=decision.agent.name,
                        decision_type=decision.decision_type,
                        content=decision.content,
                        rationale=decision.rationale,
                        metadata=decision.metadata,
                    )
                )

            return entries

        except ClientNotInitializedError:
            raise
        except Exception as e:
            raise SDKError(
                f"Failed to retrieve audit entries: {e}",
                original_error=e,
            ) from e

    def _get_decision_store(self) -> Any:
        """Get or create the decision store for this project.

        Returns a DecisionStore instance that persists to .yolo/audit/decisions.json.
        Creates the audit directory if it doesn't exist.

        Returns:
            JsonDecisionStore for persistent storage.

        Note:
            AC5 implementation: Decisions persist across client sessions by
            storing to a JSON file in the project's .yolo/audit directory.
        """
        from yolo_developer.audit import JsonDecisionStore

        # Use persistent JSON file storage (Story 13.3 AC5)
        audit_dir = self._project_path / ".yolo" / "audit"
        decisions_file = audit_dir / "decisions.json"
        return JsonDecisionStore(decisions_file)

    # =========================================================================
    # Hook Registration API (Story 13.5)
    # =========================================================================

    def register_hook(
        self,
        *,
        agent: str,
        phase: Literal["pre", "post"],
        callback: PreHook | PostHook,
    ) -> str:
        """Register a hook for agent execution.

        Hooks extend agent behavior by executing custom code before or after
        agent execution. Multiple hooks can be registered for the same agent
        and phase, and they execute in registration order.

        Args:
            agent: Target agent name (e.g., "analyst", "pm", "dev") or "*" for all agents.
            phase: Execution phase - "pre" (before agent) or "post" (after agent).
            callback: The hook function to execute. For pre-hooks, signature is
                `(agent: str, state: dict) -> dict | None`. For post-hooks,
                signature is `(agent: str, state: dict, output: dict) -> dict | None`.

        Returns:
            Hook ID string that can be used to unregister the hook.

        Example:
            >>> # Pre-hook to inject context
            >>> def inject_context(agent: str, state: dict) -> dict | None:
            ...     return {"custom_context": "my data"}
            >>>
            >>> hook_id = client.register_hook(
            ...     agent="analyst",
            ...     phase="pre",
            ...     callback=inject_context,
            ... )
            >>>
            >>> # Post-hook for all agents
            >>> def log_output(agent: str, state: dict, output: dict) -> dict | None:
            ...     print(f"{agent} completed")
            ...     return None
            >>>
            >>> hook_id = client.register_hook(
            ...     agent="*",
            ...     phase="post",
            ...     callback=log_output,
            ... )
        """
        import uuid
        from datetime import datetime, timezone

        hook_id = f"hook-{uuid.uuid4().hex[:8]}"

        registration = HookRegistration(
            hook_id=hook_id,
            agent=agent,
            phase=phase,
            callback=callback,
            timestamp=datetime.now(timezone.utc),
        )

        self._hooks[hook_id] = registration

        logger.info(
            "hook_registered",
            hook_id=hook_id,
            agent=agent,
            phase=phase,
        )

        return hook_id

    def unregister_hook(self, hook_id: str) -> bool:
        """Unregister a previously registered hook.

        Removes the hook from the registry so it won't fire on subsequent
        agent executions.

        Args:
            hook_id: The hook ID returned from register_hook().

        Returns:
            True if the hook was found and removed, False if not found.

        Example:
            >>> hook_id = client.register_hook(agent="analyst", phase="pre", callback=my_hook)
            >>> # ... later ...
            >>> client.unregister_hook(hook_id)
            True
        """
        if hook_id in self._hooks:
            del self._hooks[hook_id]
            logger.info("hook_unregistered", hook_id=hook_id)
            return True
        return False

    def list_hooks(self) -> list[HookRegistration]:
        """List all registered hooks.

        Returns a list of all currently registered hooks, sorted by registration
        time (oldest first).

        Returns:
            List of HookRegistration objects.

        Example:
            >>> hooks = client.list_hooks()
            >>> for hook in hooks:
            ...     print(f"{hook.hook_id}: {hook.agent} ({hook.phase})")
        """
        return sorted(self._hooks.values(), key=lambda h: h.timestamp)

    # =========================================================================
    # Hook Execution (Story 13.5)
    # =========================================================================

    async def _execute_pre_hooks(
        self,
        agent: str,
        state: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, list[HookResult]]:
        """Execute all matching pre-hooks for an agent.

        Finds all pre-hooks that match the agent (exact match or wildcard "*")
        and executes them in registration order. Hook errors are caught and
        logged but don't block execution.

        Args:
            agent: Name of the agent about to execute.
            state: Current workflow state (passed as read-only snapshot).

        Returns:
            Tuple of (merged modifications dict or None, list of HookResults).
        """
        from datetime import datetime, timezone

        matching_hooks = [
            h
            for h in self.list_hooks()
            if h.phase == "pre" and (h.agent == agent or h.agent == "*")
        ]

        merged_modifications: dict[str, Any] = {}
        results: list[HookResult] = []

        for hook in matching_hooks:
            try:
                # Create read-only snapshot by copying
                state_snapshot = dict(state)
                result = hook.callback(agent, state_snapshot)  # type: ignore[call-arg]

                if result is not None:
                    merged_modifications.update(result)

                results.append(
                    HookResult(
                        hook_id=hook.hook_id,
                        agent=agent,
                        phase="pre",
                        success=True,
                        modifications=result,
                        timestamp=datetime.now(timezone.utc),
                    )
                )

                logger.debug(
                    "pre_hook_executed",
                    hook_id=hook.hook_id,
                    agent=agent,
                    has_modifications=result is not None,
                )

            except Exception as e:
                error_msg = str(e)
                logger.error(
                    "hook_execution_failed",
                    hook_id=hook.hook_id,
                    agent=agent,
                    phase="pre",
                    error=error_msg,
                )

                results.append(
                    HookResult(
                        hook_id=hook.hook_id,
                        agent=agent,
                        phase="pre",
                        success=False,
                        error=error_msg,
                        timestamp=datetime.now(timezone.utc),
                    )
                )

        return merged_modifications if merged_modifications else None, results

    async def _execute_post_hooks(
        self,
        agent: str,
        state: dict[str, Any],
        output: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, list[HookResult]]:
        """Execute all matching post-hooks for an agent.

        Finds all post-hooks that match the agent (exact match or wildcard "*")
        and executes them in registration order. Each hook receives the previous
        hook's modified output (if any). Hook errors are caught and logged but
        don't block execution.

        Args:
            agent: Name of the agent that executed.
            state: Input state the agent received.
            output: Output from the agent.

        Returns:
            Tuple of (final modified output or None, list of HookResults).
        """
        from datetime import datetime, timezone

        matching_hooks = [
            h
            for h in self.list_hooks()
            if h.phase == "post" and (h.agent == agent or h.agent == "*")
        ]

        current_output = output
        final_modifications: dict[str, Any] | None = None
        results: list[HookResult] = []

        for hook in matching_hooks:
            try:
                # Pass current output (which may have been modified by previous hooks)
                result = hook.callback(agent, state, current_output)  # type: ignore[call-arg]

                if result is not None:
                    current_output = result
                    final_modifications = result

                results.append(
                    HookResult(
                        hook_id=hook.hook_id,
                        agent=agent,
                        phase="post",
                        success=True,
                        modifications=result,
                        timestamp=datetime.now(timezone.utc),
                    )
                )

                logger.debug(
                    "post_hook_executed",
                    hook_id=hook.hook_id,
                    agent=agent,
                    has_modifications=result is not None,
                )

            except Exception as e:
                error_msg = str(e)
                logger.error(
                    "hook_execution_failed",
                    hook_id=hook.hook_id,
                    agent=agent,
                    phase="post",
                    error=error_msg,
                )

                results.append(
                    HookResult(
                        hook_id=hook.hook_id,
                        agent=agent,
                        phase="post",
                        success=False,
                        error=error_msg,
                        timestamp=datetime.now(timezone.utc),
                    )
                )

        return final_modifications, results
