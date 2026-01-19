"""YoloClient class for programmatic SDK access (Story 13.1).

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
        self._is_initialized = False

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
                # Write basic config
                config_content = f"""# YOLO Developer Configuration
project_name: {name}
"""
                config_file.write_text(config_content)
                config_created = True

            self._is_initialized = True

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

        Note:
            The quality_score in SeedResult is currently a simplified heuristic
            (0.8 for seeds without ambiguities, 0.6 otherwise). A more sophisticated
            scoring algorithm will be implemented in a future version.

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
            ambiguities = []
            if parse_result.has_ambiguities:
                ambiguities = [amb.description for amb in parse_result.ambiguities]

            # Determine status based on validation
            status: Literal["accepted", "rejected", "pending"] = "accepted"
            if parse_result.has_ambiguities and len(ambiguities) > 5:
                status = "pending"  # Needs clarification

            logger.info(
                "seed_processed",
                seed_id=seed_id,
                goal_count=parse_result.goal_count,
                feature_count=parse_result.feature_count,
                ambiguity_count=len(ambiguities),
                status=status,
            )

            return SeedResult(
                seed_id=seed_id,
                status=status,
                goal_count=parse_result.goal_count,
                feature_count=parse_result.feature_count,
                constraint_count=parse_result.constraint_count,
                ambiguities=ambiguities,
                quality_score=0.8 if not ambiguities else 0.6,  # See docstring Note
                warnings=[],
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            raise SeedValidationError(
                f"Failed to process seed: {e}",
                original_error=e,
                details={"source": source},
            ) from e

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

            # Run workflow
            final_state = await run_workflow(
                initial_state=initial_state,
                config=workflow_config,
            )

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
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Get audit trail entries.

        Retrieves audit trail entries with optional filtering.

        Args:
            agent_filter: Filter by agent name.
            start_time: Filter entries after this time.
            end_time: Filter entries before this time.
            limit: Maximum number of entries to return.

        Returns:
            List of AuditEntry objects.

        Raises:
            ClientNotInitializedError: If project is not initialized.
            SDKError: If audit retrieval fails.

        Example:
            >>> entries = client.get_audit(agent_filter="analyst", limit=50)
            >>> for entry in entries:
            ...     print(f"{entry.agent}: {entry.content}")
        """
        return _run_sync(
            self.get_audit_async(
                agent_filter=agent_filter,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )
        )

    async def get_audit_async(
        self,
        *,
        agent_filter: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Get audit trail entries (async).

        Retrieves audit trail entries with optional filtering.

        Args:
            agent_filter: Filter by agent name.
            start_time: Filter entries after this time.
            end_time: Filter entries before this time.
            limit: Maximum number of entries to return.

        Returns:
            List of AuditEntry objects.

        Raises:
            ClientNotInitializedError: If project is not initialized.
            SDKError: If audit retrieval fails.

        Example:
            >>> entries = await client.get_audit_async(agent_filter="analyst")
            >>> print(f"Found {len(entries)} entries")
        """
        if not self.is_initialized:
            raise ClientNotInitializedError(
                "Project not initialized. Call init() first.",
                details={"project_path": str(self._project_path)},
            )

        try:
            from yolo_developer.audit import (
                AuditFilters,
                InMemoryDecisionStore,
                InMemoryTraceabilityStore,
                get_audit_filter_service,
            )

            # Create filter service (would normally use persistent store)
            decision_store = InMemoryDecisionStore()
            traceability_store = InMemoryTraceabilityStore()
            filter_service = get_audit_filter_service(
                decision_store=decision_store,
                traceability_store=traceability_store,
                cost_store=None,
            )

            # Apply filters
            filters = AuditFilters(
                agent_name=agent_filter,
                start_time=start_time.isoformat() if start_time else None,
                end_time=end_time.isoformat() if end_time else None,
            )

            # Query decisions
            results = await filter_service.filter_all(filters)
            decisions = results.get("decisions", [])[:limit]

            # Convert to AuditEntry
            entries = []
            for decision in decisions:
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

        except Exception as e:
            raise SDKError(
                f"Failed to retrieve audit entries: {e}",
                original_error=e,
            ) from e
