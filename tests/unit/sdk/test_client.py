"""Unit tests for YoloClient class (Story 13.1).

Tests cover:
- Client initialization with various configurations
- from_config_file class method
- Core client methods (init, seed, run, status, get_audit)
- Error handling and exceptions
- Type hints verification
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yolo_developer.config import YoloConfig
from yolo_developer.sdk.client import YoloClient
from yolo_developer.sdk.exceptions import (
    ClientNotInitializedError,
    ProjectNotFoundError,
    SDKError,
    SeedValidationError,
)
from yolo_developer.sdk.types import (
    InitResult,
    RunResult,
    SeedResult,
    StatusResult,
)


class TestYoloClientInitialization:
    """Tests for YoloClient initialization."""

    def test_init_with_default_config(self, tmp_path: Path) -> None:
        """Test YoloClient initialization with default config."""
        with patch("yolo_developer.sdk.client.Path.cwd", return_value=tmp_path):
            client = YoloClient()

        assert client.project_path == tmp_path
        assert client.config is not None
        assert client.config.project_name == "untitled"

    def test_init_with_custom_config(self, tmp_path: Path) -> None:
        """Test YoloClient initialization with custom config."""
        config = YoloConfig(project_name="my-custom-project")
        client = YoloClient(config=config, project_path=tmp_path)

        assert client.config.project_name == "my-custom-project"
        assert client.project_path == tmp_path

    def test_init_with_project_path_string(self, tmp_path: Path) -> None:
        """Test YoloClient initialization with string path."""
        client = YoloClient(project_path=str(tmp_path))

        assert client.project_path == tmp_path

    def test_init_loads_existing_config(self, tmp_path: Path) -> None:
        """Test YoloClient loads config from yolo.yaml if exists."""
        config_file = tmp_path / "yolo.yaml"
        config_file.write_text("project_name: loaded-project\n")

        client = YoloClient(project_path=tmp_path)

        assert client.config.project_name == "loaded-project"

    def test_init_config_error_wrapped(self, tmp_path: Path) -> None:
        """Test that configuration errors are wrapped in SDKError."""
        config_file = tmp_path / "yolo.yaml"
        config_file.write_text("invalid: yaml: content: [")

        with pytest.raises(SDKError) as exc_info:
            YoloClient(project_path=tmp_path)

        assert "configuration" in str(exc_info.value).lower()


class TestYoloClientFromConfigFile:
    """Tests for YoloClient.from_config_file class method."""

    def test_from_config_file_success(self, tmp_path: Path) -> None:
        """Test creating client from config file."""
        config_file = tmp_path / "yolo.yaml"
        config_file.write_text("project_name: file-project\n")

        client = YoloClient.from_config_file(config_file)

        assert client.config.project_name == "file-project"
        assert client.project_path == tmp_path

    def test_from_config_file_custom_project_path(self, tmp_path: Path) -> None:
        """Test from_config_file with custom project path."""
        config_file = tmp_path / "config" / "yolo.yaml"
        config_file.parent.mkdir()
        config_file.write_text("project_name: custom-path-project\n")

        project_dir = tmp_path / "project"
        project_dir.mkdir()

        client = YoloClient.from_config_file(config_file, project_path=project_dir)

        assert client.project_path == project_dir

    def test_from_config_file_with_string_path(self, tmp_path: Path) -> None:
        """Test from_config_file accepts string path."""
        config_file = tmp_path / "yolo.yaml"
        config_file.write_text("project_name: string-path-project\n")

        # Pass string instead of Path
        client = YoloClient.from_config_file(str(config_file))

        assert client.config.project_name == "string-path-project"
        assert client.project_path == tmp_path

    def test_from_config_file_not_found(self, tmp_path: Path) -> None:
        """Test from_config_file with non-existent file."""
        config_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(SDKError) as exc_info:
            YoloClient.from_config_file(config_file)

        assert "not found" in str(exc_info.value).lower()


class TestYoloClientIsInitialized:
    """Tests for is_initialized property."""

    def test_is_initialized_false_empty_dir(self, tmp_path: Path) -> None:
        """Test is_initialized returns False for empty directory."""
        client = YoloClient(project_path=tmp_path)

        assert client.is_initialized is False

    def test_is_initialized_true_yolo_dir(self, tmp_path: Path) -> None:
        """Test is_initialized returns True when .yolo directory exists."""
        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)

        assert client.is_initialized is True

    def test_is_initialized_true_yolo_yaml(self, tmp_path: Path) -> None:
        """Test is_initialized returns True when yolo.yaml exists."""
        (tmp_path / "yolo.yaml").write_text("project_name: test\n")
        client = YoloClient(project_path=tmp_path)

        assert client.is_initialized is True


class TestYoloClientInit:
    """Tests for YoloClient.init() method."""

    def test_init_creates_directories(self, tmp_path: Path) -> None:
        """Test init() creates required directories."""
        client = YoloClient(project_path=tmp_path)
        result = client.init(project_name="test-project")

        assert isinstance(result, InitResult)
        assert result.project_name == "test-project"
        assert (tmp_path / ".yolo").exists()
        assert (tmp_path / ".yolo" / "sessions").exists()
        assert (tmp_path / ".yolo" / "memory").exists()
        assert (tmp_path / ".yolo" / "audit").exists()

    def test_init_creates_config_file(self, tmp_path: Path) -> None:
        """Test init() creates yolo.yaml config file."""
        client = YoloClient(project_path=tmp_path)
        result = client.init(project_name="new-project")

        assert result.config_created is True
        config_file = tmp_path / "yolo.yaml"
        assert config_file.exists()
        content = config_file.read_text()
        assert "new-project" in content

    def test_init_uses_directory_name(self, tmp_path: Path) -> None:
        """Test init() uses directory name as default project name."""
        client = YoloClient(project_path=tmp_path)
        result = client.init()

        assert result.project_name == tmp_path.name

    def test_init_fails_if_already_initialized(self, tmp_path: Path) -> None:
        """Test init() fails if project already initialized."""
        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)

        with pytest.raises(ClientNotInitializedError):
            client.init()

    def test_init_force_reinitialize(self, tmp_path: Path) -> None:
        """Test init() with force=True reinitializes."""
        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)

        result = client.init(project_name="reinit-project", force=True)

        assert result.project_name == "reinit-project"


class TestYoloClientSeed:
    """Tests for YoloClient.seed() method."""

    @pytest.mark.asyncio
    async def test_seed_async_processes_content(self, tmp_path: Path) -> None:
        """Test seed_async() processes seed content."""
        client = YoloClient(project_path=tmp_path)

        with patch("yolo_developer.seed.parse_seed") as mock_parse:
            mock_result = MagicMock()
            mock_result.goal_count = 2
            mock_result.feature_count = 3
            mock_result.constraint_count = 1
            mock_result.has_ambiguities = False
            mock_result.ambiguities = []
            mock_parse.return_value = mock_result

            result = await client.seed_async(content="Build something")

        assert isinstance(result, SeedResult)
        assert result.goal_count == 2
        assert result.feature_count == 3
        assert result.status == "accepted"
        assert result.seed_id.startswith("seed-")

    @pytest.mark.asyncio
    async def test_seed_async_handles_ambiguities(self, tmp_path: Path) -> None:
        """Test seed_async() handles ambiguous content."""
        client = YoloClient(project_path=tmp_path)

        with patch("yolo_developer.seed.parse_seed") as mock_parse:
            mock_ambiguity = MagicMock()
            mock_ambiguity.description = "Unclear requirement"

            mock_result = MagicMock()
            mock_result.goal_count = 1
            mock_result.feature_count = 1
            mock_result.constraint_count = 0
            mock_result.has_ambiguities = True
            mock_result.ambiguities = [mock_ambiguity] * 10  # Many ambiguities
            mock_parse.return_value = mock_result

            result = await client.seed_async(content="Vague requirements")

        assert result.status == "pending"
        assert len(result.ambiguities) == 10

    @pytest.mark.asyncio
    async def test_seed_async_raises_on_error(self, tmp_path: Path) -> None:
        """Test seed_async() raises SeedValidationError on failure."""
        client = YoloClient(project_path=tmp_path)

        with patch("yolo_developer.seed.parse_seed") as mock_parse:
            mock_parse.side_effect = ValueError("Parse error")

            with pytest.raises(SeedValidationError) as exc_info:
                await client.seed_async(content="Invalid content")

            assert "Failed to process seed" in str(exc_info.value)


class TestYoloClientRun:
    """Tests for YoloClient.run() method."""

    @pytest.mark.asyncio
    async def test_run_async_requires_initialization(self, tmp_path: Path) -> None:
        """Test run_async() requires project initialization."""
        client = YoloClient(project_path=tmp_path)

        with pytest.raises(ClientNotInitializedError):
            await client.run_async(seed_content="Build something")

    @pytest.mark.asyncio
    async def test_run_async_requires_seed(self, tmp_path: Path) -> None:
        """Test run_async() requires seed_id or seed_content."""
        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)

        with pytest.raises(SDKError) as exc_info:
            await client.run_async()

        assert "seed_id or seed_content" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_run_async_executes_workflow(self, tmp_path: Path) -> None:
        """Test run_async() executes workflow successfully."""
        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)

        # Create mock decisions with agent attributes
        mock_decision_analyst = MagicMock()
        mock_decision_analyst.agent = "analyst"
        mock_decision_pm = MagicMock()
        mock_decision_pm.agent = "pm"

        # Mock the orchestrator module that gets imported inside run_async
        mock_orchestrator = MagicMock()
        mock_orchestrator.WorkflowConfig = MagicMock()
        mock_orchestrator.create_initial_state = MagicMock(return_value={"decisions": []})
        mock_orchestrator.run_workflow = AsyncMock(
            return_value={
                "decisions": [mock_decision_analyst, mock_decision_pm],
                "messages": [],
                "current_agent": "pm",
                "handoff_context": None,
            }
        )

        with (
            patch("yolo_developer.seed.parse_seed") as mock_parse,
            patch.dict(
                "sys.modules",
                {"yolo_developer.orchestrator": mock_orchestrator},
            ),
        ):
            # Mock seed parsing
            mock_seed_result = MagicMock()
            mock_seed_result.goal_count = 1
            mock_seed_result.feature_count = 1
            mock_seed_result.constraint_count = 0
            mock_seed_result.has_ambiguities = False
            mock_seed_result.ambiguities = []
            mock_parse.return_value = mock_seed_result

            result = await client.run_async(seed_content="Build something")

        assert isinstance(result, RunResult)
        assert result.status == "completed"
        # Agents are extracted from decisions, order may vary (using set)
        assert set(result.agents_executed) == {"analyst", "pm"}
        # Stories not yet tracked in YoloState
        assert result.stories_completed == 0
        assert result.stories_total == 0

    def test_run_sync_executes_workflow(self, tmp_path: Path) -> None:
        """Test run() sync method executes workflow successfully."""
        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)

        # Create mock decisions with agent attributes
        mock_decision_analyst = MagicMock()
        mock_decision_analyst.agent = "analyst"

        # Mock the orchestrator module
        mock_orchestrator = MagicMock()
        mock_orchestrator.WorkflowConfig = MagicMock()
        mock_orchestrator.create_initial_state = MagicMock(return_value={"decisions": []})
        mock_orchestrator.run_workflow = AsyncMock(
            return_value={
                "decisions": [mock_decision_analyst],
                "messages": [],
                "current_agent": "analyst",
                "handoff_context": None,
            }
        )

        with (
            patch("yolo_developer.seed.parse_seed") as mock_parse,
            patch.dict(
                "sys.modules",
                {"yolo_developer.orchestrator": mock_orchestrator},
            ),
        ):
            # Mock seed parsing
            mock_seed_result = MagicMock()
            mock_seed_result.goal_count = 1
            mock_seed_result.feature_count = 1
            mock_seed_result.constraint_count = 0
            mock_seed_result.has_ambiguities = False
            mock_seed_result.ambiguities = []
            mock_parse.return_value = mock_seed_result

            result = client.run(seed_content="Build something")

        assert isinstance(result, RunResult)
        assert result.status == "completed"


class TestYoloClientStatus:
    """Tests for YoloClient.status() method."""

    @pytest.mark.asyncio
    async def test_status_async_returns_status(self, tmp_path: Path) -> None:
        """Test status_async() returns status result."""
        (tmp_path / ".yolo").mkdir()
        config = YoloConfig(project_name="status-test")
        client = YoloClient(config=config, project_path=tmp_path)

        result = await client.status_async()

        assert isinstance(result, StatusResult)
        assert result.project_name == "status-test"
        assert result.is_initialized is True
        assert result.workflow_status == "idle"

    @pytest.mark.asyncio
    async def test_status_async_project_not_found(self, tmp_path: Path) -> None:
        """Test status_async() raises when project path doesn't exist."""
        nonexistent = tmp_path / "nonexistent"
        client = YoloClient(project_path=nonexistent)

        with pytest.raises(ProjectNotFoundError):
            await client.status_async()


class TestYoloClientGetAudit:
    """Tests for YoloClient.get_audit() method."""

    @pytest.mark.asyncio
    async def test_get_audit_async_requires_initialization(self, tmp_path: Path) -> None:
        """Test get_audit_async() requires project initialization."""
        client = YoloClient(project_path=tmp_path)

        with pytest.raises(ClientNotInitializedError):
            await client.get_audit_async()

    @pytest.mark.asyncio
    async def test_get_audit_async_returns_entries(self, tmp_path: Path) -> None:
        """Test get_audit_async() returns audit entries."""
        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)

        with patch("yolo_developer.audit.get_audit_filter_service") as mock_service:
            mock_filter_service = MagicMock()
            mock_filter_service.filter_all = AsyncMock(return_value={"decisions": []})
            mock_service.return_value = mock_filter_service

            entries = await client.get_audit_async()

        assert isinstance(entries, list)


class TestYoloClientTypes:
    """Tests for type annotations on YoloClient."""

    def test_config_property_type(self, tmp_path: Path) -> None:
        """Test config property returns YoloConfig."""
        client = YoloClient(project_path=tmp_path)
        config = client.config

        assert isinstance(config, YoloConfig)

    def test_project_path_property_type(self, tmp_path: Path) -> None:
        """Test project_path property returns Path."""
        client = YoloClient(project_path=tmp_path)
        path = client.project_path

        assert isinstance(path, Path)

    def test_is_initialized_property_type(self, tmp_path: Path) -> None:
        """Test is_initialized property returns bool."""
        client = YoloClient(project_path=tmp_path)
        initialized = client.is_initialized

        assert isinstance(initialized, bool)
