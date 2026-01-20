"""Unit tests for YoloClient class (Stories 13.1, 13.5, 13.6).

Tests cover:
- Client initialization with various configurations
- from_config_file class method
- Core client methods (init, seed, run, status, get_audit)
- Error handling and exceptions
- Type hints verification
- Agent hooks (Story 13.5)
- Event emission (Story 13.6)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yolo_developer.config import YoloConfig
from yolo_developer.sdk.client import YoloClient
from yolo_developer.sdk.exceptions import (
    ClientNotInitializedError,
    ProjectNotFoundError,
    SDKError,
    SeedValidationError,
    WorkflowExecutionError,
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
        """Test init() creates yolo.yaml config file with comprehensive settings."""
        client = YoloClient(project_path=tmp_path)
        result = client.init(project_name="new-project")

        assert result.config_created is True
        config_file = tmp_path / "yolo.yaml"
        assert config_file.exists()
        content = config_file.read_text()
        # Verify comprehensive config matching CLI format
        assert "project_name: new-project" in content
        assert "llm:" in content
        assert "cheap_model: gpt-4o-mini" in content
        assert "premium_model: claude-sonnet-4-20250514" in content
        assert "quality:" in content
        assert "test_coverage_threshold: 0.80" in content
        assert "memory:" in content
        assert "vector_store_type: chromadb" in content

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
        # Quality score should be 1.0 for seeds with goals, features, and no ambiguities
        assert result.quality_score == 1.0

    @pytest.mark.asyncio
    async def test_seed_async_handles_ambiguities(self, tmp_path: Path) -> None:
        """Test seed_async() handles ambiguous content and uses config thresholds."""
        client = YoloClient(project_path=tmp_path)

        with patch("yolo_developer.seed.parse_seed") as mock_parse:
            mock_ambiguity = MagicMock()
            mock_ambiguity.description = "Unclear requirement"

            mock_result = MagicMock()
            mock_result.goal_count = 1
            mock_result.feature_count = 1
            mock_result.constraint_count = 0
            mock_result.has_ambiguities = True
            # 5 ambiguities: ambiguity_score = 1.0 - 0.5 = 0.5 < 0.6 threshold = pending
            mock_result.ambiguities = [mock_ambiguity] * 5
            mock_parse.return_value = mock_result

            result = await client.seed_async(content="Vague requirements")

        assert result.status == "pending"  # Below ambiguity threshold (0.6)
        assert len(result.ambiguities) == 5
        # Quality score reduced by ambiguities (5 * 0.05 = 0.25)
        assert result.quality_score == 0.75

    @pytest.mark.asyncio
    async def test_seed_async_rejected_below_threshold(self, tmp_path: Path) -> None:
        """Test seed_async() rejects seeds below quality threshold."""
        client = YoloClient(project_path=tmp_path)

        with patch("yolo_developer.seed.parse_seed") as mock_parse:
            mock_ambiguity = MagicMock()
            mock_ambiguity.description = "Missing info"

            mock_result = MagicMock()
            mock_result.goal_count = 0  # No goals: -0.2
            mock_result.feature_count = 0  # No features: -0.1
            mock_result.constraint_count = 0
            mock_result.has_ambiguities = True
            mock_result.ambiguities = [mock_ambiguity] * 6  # 6 ambiguities: -0.3
            mock_parse.return_value = mock_result

            result = await client.seed_async(content="Bad seed")

        # Quality score: 1.0 - 0.2 - 0.1 - 0.3 = 0.4 < 0.7 threshold
        assert result.status == "rejected"
        assert abs(result.quality_score - 0.4) < 0.01  # Float comparison with tolerance
        assert any("goals" in w.lower() for w in result.warnings)
        assert any("features" in w.lower() for w in result.warnings)

    @pytest.mark.asyncio
    async def test_seed_async_raises_on_error(self, tmp_path: Path) -> None:
        """Test seed_async() raises SeedValidationError on failure."""
        client = YoloClient(project_path=tmp_path)

        with patch("yolo_developer.seed.parse_seed") as mock_parse:
            mock_parse.side_effect = ValueError("Parse error")

            with pytest.raises(SeedValidationError) as exc_info:
                await client.seed_async(content="Invalid content")

            assert "Failed to process seed" in str(exc_info.value)

    def test_seed_sync_processes_content(self, tmp_path: Path) -> None:
        """Test sync seed() method processes content correctly."""
        client = YoloClient(project_path=tmp_path)

        with patch("yolo_developer.seed.parse_seed") as mock_parse:
            mock_result = MagicMock()
            mock_result.goal_count = 1
            mock_result.feature_count = 2
            mock_result.constraint_count = 0
            mock_result.has_ambiguities = False
            mock_result.ambiguities = []
            mock_parse.return_value = mock_result

            result = client.seed(content="Build a web app")

        assert isinstance(result, SeedResult)
        assert result.status == "accepted"
        assert result.goal_count == 1
        assert result.feature_count == 2


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
    async def test_run_async_seed_id_not_implemented(self, tmp_path: Path) -> None:
        """Test run_async() raises SDKError when only seed_id is provided."""
        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)

        with pytest.raises(SDKError) as exc_info:
            await client.run_async(seed_id="seed-123")

        assert "seed_id lookup is not yet implemented" in str(exc_info.value)
        assert exc_info.value.details is not None
        assert exc_info.value.details.get("seed_id") == "seed-123"

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

    @pytest.mark.asyncio
    async def test_get_audit_async_with_decision_type_filter(self, tmp_path: Path) -> None:
        """Test get_audit_async() accepts decision_type filter."""
        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)

        with patch("yolo_developer.audit.get_audit_filter_service") as mock_service:
            mock_filter_service = MagicMock()
            mock_filter_service.filter_all = AsyncMock(return_value={"decisions": []})
            mock_service.return_value = mock_filter_service

            entries = await client.get_audit_async(decision_type="requirement_analysis")

        assert isinstance(entries, list)

    @pytest.mark.asyncio
    async def test_get_audit_async_with_artifact_type_filter(self, tmp_path: Path) -> None:
        """Test get_audit_async() accepts artifact_type parameter (reserved for future use)."""
        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)

        with patch("yolo_developer.audit.get_audit_filter_service") as mock_service:
            mock_filter_service = MagicMock()
            mock_filter_service.filter_all = AsyncMock(return_value={"decisions": []})
            mock_service.return_value = mock_filter_service

            entries = await client.get_audit_async(artifact_type="requirement")

        assert isinstance(entries, list)

    @pytest.mark.asyncio
    async def test_get_audit_async_with_pagination(self, tmp_path: Path) -> None:
        """Test get_audit_async() supports offset and limit for pagination."""
        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)

        with patch("yolo_developer.audit.get_audit_filter_service") as mock_service:
            mock_filter_service = MagicMock()
            mock_filter_service.filter_all = AsyncMock(return_value={"decisions": []})
            mock_service.return_value = mock_filter_service

            # Test with offset and limit
            entries = await client.get_audit_async(limit=10, offset=5)

        assert isinstance(entries, list)

    @pytest.mark.asyncio
    async def test_get_audit_async_pagination_slices_correctly(self, tmp_path: Path) -> None:
        """Test that pagination correctly slices results."""
        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)

        # Create mock decisions
        mock_decisions = []
        for i in range(20):
            mock_decision = MagicMock()
            mock_decision.id = f"decision-{i}"
            mock_decision.timestamp = datetime.now(timezone.utc)
            mock_decision.agent = MagicMock()
            mock_decision.agent.name = "analyst"
            mock_decision.decision_type = "test"
            mock_decision.content = f"Decision {i}"
            mock_decision.rationale = None
            mock_decision.metadata = {}
            mock_decisions.append(mock_decision)

        with patch("yolo_developer.audit.get_audit_filter_service") as mock_service:
            mock_filter_service = MagicMock()
            mock_filter_service.filter_all = AsyncMock(return_value={"decisions": mock_decisions})
            mock_service.return_value = mock_filter_service

            # Get first page (0-9)
            page1 = await client.get_audit_async(limit=10, offset=0)
            assert len(page1) == 10
            assert page1[0].entry_id == "decision-0"
            assert page1[9].entry_id == "decision-9"

            # Get second page (10-19)
            page2 = await client.get_audit_async(limit=10, offset=10)
            assert len(page2) == 10
            assert page2[0].entry_id == "decision-10"
            assert page2[9].entry_id == "decision-19"

            # Get third page (should be empty)
            page3 = await client.get_audit_async(limit=10, offset=20)
            assert len(page3) == 0

    def test_get_audit_sync_wraps_async(self, tmp_path: Path) -> None:
        """Test get_audit() correctly wraps get_audit_async()."""
        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)

        with patch("yolo_developer.audit.get_audit_filter_service") as mock_service:
            mock_filter_service = MagicMock()
            mock_filter_service.filter_all = AsyncMock(return_value={"decisions": []})
            mock_service.return_value = mock_filter_service

            # Sync method should work
            entries = client.get_audit(agent_filter="analyst", limit=50, offset=10)

        assert isinstance(entries, list)

    @pytest.mark.asyncio
    async def test_get_audit_async_with_all_filters(self, tmp_path: Path) -> None:
        """Test get_audit_async() with all filter parameters."""
        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)
        now = datetime.now(timezone.utc)

        with patch("yolo_developer.audit.get_audit_filter_service") as mock_service:
            mock_filter_service = MagicMock()
            mock_filter_service.filter_all = AsyncMock(return_value={"decisions": []})
            mock_service.return_value = mock_filter_service

            entries = await client.get_audit_async(
                agent_filter="analyst",
                decision_type="requirement_analysis",
                artifact_type="requirement",
                start_time=now,
                end_time=now,
                limit=50,
                offset=5,
            )

        assert isinstance(entries, list)

    @pytest.mark.asyncio
    async def test_get_audit_async_uses_persistent_store(self, tmp_path: Path) -> None:
        """Test that get_audit uses JsonDecisionStore for persistence (AC5)."""
        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)

        # Call _get_decision_store and verify it returns JsonDecisionStore
        store = client._get_decision_store()

        # Verify it's JsonDecisionStore by checking it has the file_path attribute
        assert hasattr(store, "_file_path")
        expected_path = tmp_path / ".yolo" / "audit" / "decisions.json"
        assert store._file_path == expected_path

    @pytest.mark.asyncio
    async def test_get_audit_async_entry_structure(self, tmp_path: Path) -> None:
        """Test AuditEntry structure has all required fields."""
        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)

        # Create a mock decision with all fields
        mock_decision = MagicMock()
        mock_decision.id = "dec-123"
        mock_decision.timestamp = datetime.now(timezone.utc)
        mock_decision.agent = MagicMock()
        mock_decision.agent.name = "analyst"
        mock_decision.decision_type = "requirement_analysis"
        mock_decision.content = "Analyzed requirement"
        mock_decision.rationale = "Industry best practice"
        mock_decision.metadata = {"sprint_id": "sprint-1"}

        with patch("yolo_developer.audit.get_audit_filter_service") as mock_service:
            mock_filter_service = MagicMock()
            mock_filter_service.filter_all = AsyncMock(return_value={"decisions": [mock_decision]})
            mock_service.return_value = mock_filter_service

            entries = await client.get_audit_async()

        assert len(entries) == 1
        entry = entries[0]

        # Verify AuditEntry structure (AC3)
        assert entry.entry_id == "dec-123"
        assert isinstance(entry.timestamp, datetime)
        assert entry.timestamp.tzinfo is not None  # Timezone-aware
        assert entry.agent == "analyst"
        assert entry.decision_type == "requirement_analysis"
        assert entry.content == "Analyzed requirement"
        assert entry.rationale == "Industry best practice"
        assert entry.metadata == {"sprint_id": "sprint-1"}


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


# ============================================================================
# Configuration API Tests (Story 13.4)
# ============================================================================


class TestYoloClientConfigRead:
    """Tests for configuration read access (AC1)."""

    def test_config_returns_yoloconfig(self, tmp_path: Path) -> None:
        """Test client.config returns complete YoloConfig object."""
        config = YoloConfig(project_name="test-project")
        client = YoloClient(config=config, project_path=tmp_path)

        result = client.config

        assert isinstance(result, YoloConfig)
        assert result.project_name == "test-project"

    def test_config_nested_access(self, tmp_path: Path) -> None:
        """Test accessing nested configuration settings."""
        config = YoloConfig(project_name="test-project")
        client = YoloClient(config=config, project_path=tmp_path)

        # Access nested llm settings
        assert client.config.llm.cheap_model == "gpt-4o-mini"
        assert client.config.llm.premium_model == "claude-sonnet-4-20250514"

        # Access nested quality settings
        assert client.config.quality.test_coverage_threshold == 0.80
        assert client.config.quality.confidence_threshold == 0.90

        # Access nested memory settings
        assert client.config.memory.persist_path == ".yolo/memory"
        assert client.config.memory.vector_store_type == "chromadb"

    def test_config_reflects_current_state(self, tmp_path: Path) -> None:
        """Test config reflects current in-memory state."""
        config = YoloConfig(project_name="initial")
        client = YoloClient(config=config, project_path=tmp_path)

        assert client.config.project_name == "initial"

        # Update config
        client.update_config(project_name="updated")

        # Should reflect new state
        assert client.config.project_name == "updated"


class TestYoloClientConfigUpdate:
    """Tests for configuration update (AC2)."""

    def test_update_config_partial_update(self, tmp_path: Path) -> None:
        """Test update_config with partial settings."""
        config = YoloConfig(project_name="test-project")
        client = YoloClient(config=config, project_path=tmp_path)

        result = client.update_config(quality={"test_coverage_threshold": 0.85})

        assert result.success
        assert client.config.quality.test_coverage_threshold == 0.85
        # Other settings unchanged
        assert client.config.quality.confidence_threshold == 0.90

    def test_update_config_multiple_sections(self, tmp_path: Path) -> None:
        """Test update_config with multiple sections."""
        config = YoloConfig(project_name="test-project")
        client = YoloClient(config=config, project_path=tmp_path)

        result = client.update_config(
            llm={"cheap_model": "gpt-4o"}, quality={"test_coverage_threshold": 0.90}
        )

        assert result.success
        assert client.config.llm.cheap_model == "gpt-4o"
        assert client.config.quality.test_coverage_threshold == 0.90

    def test_update_config_project_name(self, tmp_path: Path) -> None:
        """Test update_config can change project name."""
        config = YoloConfig(project_name="old-name")
        client = YoloClient(config=config, project_path=tmp_path)

        result = client.update_config(project_name="new-name")

        assert result.success
        assert client.config.project_name == "new-name"
        assert result.previous_values["project_name"] == "old-name"
        assert result.new_values["project_name"] == "new-name"

    def test_update_config_tracks_changes(self, tmp_path: Path) -> None:
        """Test update_config tracks previous and new values."""
        config = YoloConfig(project_name="test-project")
        client = YoloClient(config=config, project_path=tmp_path)

        result = client.update_config(quality={"test_coverage_threshold": 0.85})

        assert "quality" in result.previous_values
        assert "quality" in result.new_values
        assert result.previous_values["quality"]["test_coverage_threshold"] == 0.80
        assert result.new_values["quality"]["test_coverage_threshold"] == 0.85

    def test_update_config_validation_error(self, tmp_path: Path) -> None:
        """Test update_config raises error on invalid values."""
        from yolo_developer.sdk.exceptions import ConfigurationAPIError

        config = YoloConfig(project_name="test-project")
        client = YoloClient(config=config, project_path=tmp_path)

        with pytest.raises(ConfigurationAPIError):
            # test_coverage_threshold must be 0.0-1.0
            client.update_config(quality={"test_coverage_threshold": 2.0})

    @pytest.mark.asyncio
    async def test_update_config_async(self, tmp_path: Path) -> None:
        """Test update_config_async works correctly."""
        config = YoloConfig(project_name="test-project")
        client = YoloClient(config=config, project_path=tmp_path)

        result = await client.update_config_async(quality={"confidence_threshold": 0.95})

        assert result.success
        assert client.config.quality.confidence_threshold == 0.95

    def test_update_config_empty_update(self, tmp_path: Path) -> None:
        """Test update_config with no parameters returns success."""
        config = YoloConfig(project_name="test-project")
        client = YoloClient(config=config, project_path=tmp_path)

        result = client.update_config()

        assert result.success
        assert result.previous_values == {}
        assert result.new_values == {}
        # Config should remain unchanged
        assert client.config.project_name == "test-project"


class TestYoloClientConfigValidation:
    """Tests for configuration validation (AC3)."""

    def test_validate_config_returns_result(self, tmp_path: Path) -> None:
        """Test validate_config returns ConfigValidationResult."""
        from yolo_developer.sdk.types import ConfigValidationResult

        config = YoloConfig(project_name="test-project")
        client = YoloClient(config=config, project_path=tmp_path)

        result = client.validate_config()

        assert isinstance(result, ConfigValidationResult)
        assert result.is_valid  # Default config should be valid

    def test_validate_config_reports_warnings(self, tmp_path: Path) -> None:
        """Test validate_config reports warnings."""
        config = YoloConfig(project_name="test-project")
        client = YoloClient(config=config, project_path=tmp_path)

        result = client.validate_config()

        # Should have warnings about missing API keys
        assert len(result.warnings) > 0
        api_key_warning = any("api_key" in issue.field.lower() for issue in result.warnings)
        assert api_key_warning

    def test_validate_config_separates_errors_warnings(self, tmp_path: Path) -> None:
        """Test validation result separates errors and warnings."""
        config = YoloConfig(project_name="test-project")
        client = YoloClient(config=config, project_path=tmp_path)

        result = client.validate_config()

        # Check errors property
        assert isinstance(result.errors, list)
        # Check warnings property
        assert isinstance(result.warnings, list)
        # is_valid should be True if no errors
        assert result.is_valid == (len(result.errors) == 0)

    @pytest.mark.asyncio
    async def test_validate_config_async(self, tmp_path: Path) -> None:
        """Test validate_config_async works correctly."""
        config = YoloConfig(project_name="test-project")
        client = YoloClient(config=config, project_path=tmp_path)

        result = await client.validate_config_async()

        assert result.is_valid


class TestYoloClientConfigPersistence:
    """Tests for configuration persistence (AC4)."""

    def test_save_config_creates_file(self, tmp_path: Path) -> None:
        """Test save_config creates yolo.yaml file."""
        config = YoloConfig(project_name="save-test")
        client = YoloClient(config=config, project_path=tmp_path)

        result = client.save_config()

        assert result.success
        config_file = tmp_path / "yolo.yaml"
        assert config_file.exists()

    def test_save_config_excludes_secrets(self, tmp_path: Path) -> None:
        """Test save_config excludes API keys."""
        config = YoloConfig(project_name="secret-test")
        client = YoloClient(config=config, project_path=tmp_path)

        result = client.save_config()

        assert result.success
        assert "openai_api_key" in result.secrets_excluded
        assert "anthropic_api_key" in result.secrets_excluded

        # Verify file doesn't contain API key fields
        config_file = tmp_path / "yolo.yaml"
        content = config_file.read_text()
        assert "openai_api_key" not in content
        assert "anthropic_api_key" not in content

    def test_save_config_preserves_values(self, tmp_path: Path) -> None:
        """Test saved config can be reloaded correctly."""
        config = YoloConfig(project_name="persist-test")
        client = YoloClient(config=config, project_path=tmp_path)

        # Update and save
        client.update_config(quality={"test_coverage_threshold": 0.85})
        client.save_config()

        # Create new client from same path
        new_client = YoloClient(project_path=tmp_path)

        assert new_client.config.project_name == "persist-test"
        assert new_client.config.quality.test_coverage_threshold == 0.85

    def test_update_config_with_persist(self, tmp_path: Path) -> None:
        """Test update_config with persist=True saves to file."""
        config = YoloConfig(project_name="auto-persist")
        client = YoloClient(config=config, project_path=tmp_path)

        result = client.update_config(quality={"confidence_threshold": 0.95}, persist=True)

        assert result.success
        assert result.persisted

        # Verify file was created
        config_file = tmp_path / "yolo.yaml"
        assert config_file.exists()

    @pytest.mark.asyncio
    async def test_save_config_async(self, tmp_path: Path) -> None:
        """Test save_config_async works correctly."""
        config = YoloConfig(project_name="async-save")
        client = YoloClient(config=config, project_path=tmp_path)

        result = await client.save_config_async()

        assert result.success
        config_file = tmp_path / "yolo.yaml"
        assert config_file.exists()


class TestYoloClientConfigAsyncSync:
    """Tests for async/sync parity (AC5)."""

    def test_sync_methods_exist(self, tmp_path: Path) -> None:
        """Test sync config methods exist."""
        config = YoloConfig(project_name="test-project")
        client = YoloClient(config=config, project_path=tmp_path)

        assert hasattr(client, "update_config")
        assert hasattr(client, "validate_config")
        assert hasattr(client, "save_config")
        assert callable(client.update_config)
        assert callable(client.validate_config)
        assert callable(client.save_config)

    def test_async_methods_exist(self, tmp_path: Path) -> None:
        """Test async config methods exist."""
        config = YoloConfig(project_name="test-project")
        client = YoloClient(config=config, project_path=tmp_path)

        assert hasattr(client, "update_config_async")
        assert hasattr(client, "validate_config_async")
        assert hasattr(client, "save_config_async")

    def test_sync_wraps_async_update(self, tmp_path: Path) -> None:
        """Test sync update_config produces same result as async."""
        config = YoloConfig(project_name="test-project")
        client = YoloClient(config=config, project_path=tmp_path)

        sync_result = client.update_config(quality={"test_coverage_threshold": 0.85})

        assert sync_result.success
        assert client.config.quality.test_coverage_threshold == 0.85

    def test_sync_wraps_async_validate(self, tmp_path: Path) -> None:
        """Test sync validate_config produces same result as async."""
        config = YoloConfig(project_name="test-project")
        client = YoloClient(config=config, project_path=tmp_path)

        sync_result = client.validate_config()

        assert sync_result.is_valid

    def test_sync_wraps_async_save(self, tmp_path: Path) -> None:
        """Test sync save_config produces same result as async."""
        config = YoloConfig(project_name="test-project")
        client = YoloClient(config=config, project_path=tmp_path)

        sync_result = client.save_config()

        assert sync_result.success


# ============================================================================
# Agent Hooks Tests (Story 13.5)
# ============================================================================


class TestYoloClientHookRegistration:
    """Tests for hook registration (AC1)."""

    def test_register_hook_returns_hook_id(self, tmp_path: Path) -> None:
        """Test register_hook returns a unique hook ID."""
        client = YoloClient(project_path=tmp_path)

        def my_hook(agent: str, state: dict) -> dict | None:
            return None

        hook_id = client.register_hook(agent="analyst", phase="pre", callback=my_hook)

        assert hook_id.startswith("hook-")
        assert len(hook_id) > 5

    def test_register_hook_stores_registration(self, tmp_path: Path) -> None:
        """Test register_hook stores the registration."""
        client = YoloClient(project_path=tmp_path)

        def my_hook(agent: str, state: dict) -> dict | None:
            return None

        hook_id = client.register_hook(agent="pm", phase="pre", callback=my_hook)

        hooks = client.list_hooks()
        assert len(hooks) == 1
        assert hooks[0].hook_id == hook_id
        assert hooks[0].agent == "pm"
        assert hooks[0].phase == "pre"

    def test_register_hook_wildcard_agent(self, tmp_path: Path) -> None:
        """Test register_hook with wildcard agent '*'."""
        client = YoloClient(project_path=tmp_path)

        def my_hook(agent: str, state: dict) -> dict | None:
            return None

        client.register_hook(agent="*", phase="pre", callback=my_hook)

        hooks = client.list_hooks()
        assert hooks[0].agent == "*"

    def test_register_multiple_hooks_same_agent(self, tmp_path: Path) -> None:
        """Test registering multiple hooks for the same agent/phase."""
        client = YoloClient(project_path=tmp_path)

        def hook1(agent: str, state: dict) -> dict | None:
            return {"from": "hook1"}

        def hook2(agent: str, state: dict) -> dict | None:
            return {"from": "hook2"}

        id1 = client.register_hook(agent="analyst", phase="pre", callback=hook1)
        id2 = client.register_hook(agent="analyst", phase="pre", callback=hook2)

        hooks = client.list_hooks()
        assert len(hooks) == 2
        assert id1 != id2

    def test_register_hook_post_phase(self, tmp_path: Path) -> None:
        """Test register_hook with post phase."""
        client = YoloClient(project_path=tmp_path)

        def my_hook(agent: str, state: dict, output: dict) -> dict | None:
            return None

        client.register_hook(agent="dev", phase="post", callback=my_hook)

        hooks = client.list_hooks()
        assert hooks[0].phase == "post"


class TestYoloClientHookUnregistration:
    """Tests for hook unregistration (AC6)."""

    def test_unregister_hook_removes_hook(self, tmp_path: Path) -> None:
        """Test unregister_hook removes the hook."""
        client = YoloClient(project_path=tmp_path)

        def my_hook(agent: str, state: dict) -> dict | None:
            return None

        hook_id = client.register_hook(agent="analyst", phase="pre", callback=my_hook)
        assert len(client.list_hooks()) == 1

        result = client.unregister_hook(hook_id)

        assert result is True
        assert len(client.list_hooks()) == 0

    def test_unregister_hook_not_found(self, tmp_path: Path) -> None:
        """Test unregister_hook returns False for non-existent hook."""
        client = YoloClient(project_path=tmp_path)

        result = client.unregister_hook("hook-nonexistent")

        assert result is False

    def test_list_hooks_reflects_removal(self, tmp_path: Path) -> None:
        """Test list_hooks reflects hook removal."""
        client = YoloClient(project_path=tmp_path)

        def hook1(agent: str, state: dict) -> dict | None:
            return None

        def hook2(agent: str, state: dict) -> dict | None:
            return None

        id1 = client.register_hook(agent="analyst", phase="pre", callback=hook1)
        id2 = client.register_hook(agent="pm", phase="pre", callback=hook2)

        assert len(client.list_hooks()) == 2

        client.unregister_hook(id1)

        hooks = client.list_hooks()
        assert len(hooks) == 1
        assert hooks[0].hook_id == id2


class TestYoloClientPreHookExecution:
    """Tests for pre-execution hooks (AC2)."""

    @pytest.mark.asyncio
    async def test_pre_hook_fires_before_agent(self, tmp_path: Path) -> None:
        """Test pre-hooks fire before agent execution."""
        client = YoloClient(project_path=tmp_path)
        called_with = []

        def my_hook(agent: str, state: dict) -> dict | None:
            called_with.append((agent, dict(state)))
            return None

        client.register_hook(agent="analyst", phase="pre", callback=my_hook)

        _modifications, _results = await client._execute_pre_hooks("analyst", {"existing": "data"})

        assert len(called_with) == 1
        assert called_with[0][0] == "analyst"
        assert called_with[0][1] == {"existing": "data"}

    @pytest.mark.asyncio
    async def test_pre_hook_returns_modifications(self, tmp_path: Path) -> None:
        """Test pre-hooks can return state modifications."""
        client = YoloClient(project_path=tmp_path)

        def inject_context(agent: str, state: dict) -> dict | None:
            return {"custom_context": "injected"}

        client.register_hook(agent="analyst", phase="pre", callback=inject_context)

        modifications, results = await client._execute_pre_hooks("analyst", {})

        assert modifications == {"custom_context": "injected"}
        assert results[0].success is True
        assert results[0].modifications == {"custom_context": "injected"}

    @pytest.mark.asyncio
    async def test_pre_hook_none_means_no_modification(self, tmp_path: Path) -> None:
        """Test pre-hooks returning None means no modifications."""
        client = YoloClient(project_path=tmp_path)

        def no_change(agent: str, state: dict) -> dict | None:
            return None

        client.register_hook(agent="analyst", phase="pre", callback=no_change)

        modifications, results = await client._execute_pre_hooks("analyst", {})

        assert modifications is None
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_pre_hooks_fire_in_registration_order(self, tmp_path: Path) -> None:
        """Test pre-hooks fire in registration order."""
        client = YoloClient(project_path=tmp_path)
        order = []

        def hook1(agent: str, state: dict) -> dict | None:
            order.append("hook1")
            return {"first": True}

        def hook2(agent: str, state: dict) -> dict | None:
            order.append("hook2")
            return {"second": True}

        client.register_hook(agent="analyst", phase="pre", callback=hook1)
        client.register_hook(agent="analyst", phase="pre", callback=hook2)

        modifications, _results = await client._execute_pre_hooks("analyst", {})

        assert order == ["hook1", "hook2"]
        # Modifications should be merged
        assert modifications == {"first": True, "second": True}

    @pytest.mark.asyncio
    async def test_pre_hook_wildcard_matches_all(self, tmp_path: Path) -> None:
        """Test wildcard '*' hooks match all agents."""
        client = YoloClient(project_path=tmp_path)
        called_agents = []

        def global_hook(agent: str, state: dict) -> dict | None:
            called_agents.append(agent)
            return None

        client.register_hook(agent="*", phase="pre", callback=global_hook)

        await client._execute_pre_hooks("analyst", {})
        await client._execute_pre_hooks("pm", {})
        await client._execute_pre_hooks("dev", {})

        assert called_agents == ["analyst", "pm", "dev"]


class TestYoloClientPostHookExecution:
    """Tests for post-execution hooks (AC3)."""

    @pytest.mark.asyncio
    async def test_post_hook_fires_after_agent(self, tmp_path: Path) -> None:
        """Test post-hooks fire after agent execution."""
        client = YoloClient(project_path=tmp_path)
        called_with = []

        def my_hook(agent: str, state: dict, output: dict) -> dict | None:
            called_with.append((agent, dict(state), dict(output)))
            return None

        client.register_hook(agent="analyst", phase="post", callback=my_hook)

        _modifications, _results = await client._execute_post_hooks(
            "analyst", {"input": "state"}, {"agent": "output"}
        )

        assert len(called_with) == 1
        assert called_with[0][0] == "analyst"
        assert called_with[0][1] == {"input": "state"}
        assert called_with[0][2] == {"agent": "output"}

    @pytest.mark.asyncio
    async def test_post_hook_can_modify_output(self, tmp_path: Path) -> None:
        """Test post-hooks can modify agent output."""
        client = YoloClient(project_path=tmp_path)

        def modify_output(agent: str, state: dict, output: dict) -> dict | None:
            return {**output, "modified": True}

        client.register_hook(agent="analyst", phase="post", callback=modify_output)

        modifications, _results = await client._execute_post_hooks(
            "analyst", {}, {"original": "data"}
        )

        assert modifications == {"original": "data", "modified": True}

    @pytest.mark.asyncio
    async def test_post_hook_none_uses_original(self, tmp_path: Path) -> None:
        """Test post-hooks returning None uses original output."""
        client = YoloClient(project_path=tmp_path)

        def no_change(agent: str, state: dict, output: dict) -> dict | None:
            return None

        client.register_hook(agent="analyst", phase="post", callback=no_change)

        modifications, results = await client._execute_post_hooks(
            "analyst", {}, {"original": "data"}
        )

        assert modifications is None
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_post_hooks_chain_modifications(self, tmp_path: Path) -> None:
        """Test post-hooks chain their modifications."""
        client = YoloClient(project_path=tmp_path)

        def hook1(agent: str, state: dict, output: dict) -> dict | None:
            return {**output, "from_hook1": True}

        def hook2(agent: str, state: dict, output: dict) -> dict | None:
            return {**output, "from_hook2": True}

        client.register_hook(agent="analyst", phase="post", callback=hook1)
        client.register_hook(agent="analyst", phase="post", callback=hook2)

        modifications, _results = await client._execute_post_hooks(
            "analyst", {}, {"original": True}
        )

        # Hook2 should see hook1's modifications
        assert modifications == {"original": True, "from_hook1": True, "from_hook2": True}


class TestYoloClientHookTypeSafety:
    """Tests for hook type safety (AC4)."""

    def test_pre_hook_protocol_type(self, tmp_path: Path) -> None:
        """Test PreHook protocol is properly defined."""
        from yolo_developer.sdk.types import PreHook

        # Functions matching the protocol should work
        def my_pre_hook(agent: str, state: dict) -> dict | None:
            return None

        # Should be an instance of PreHook protocol
        assert isinstance(my_pre_hook, PreHook)

    def test_post_hook_protocol_type(self, tmp_path: Path) -> None:
        """Test PostHook protocol is properly defined."""
        from yolo_developer.sdk.types import PostHook

        # Functions matching the protocol should work
        def my_post_hook(agent: str, state: dict, output: dict) -> dict | None:
            return None

        # Should be an instance of PostHook protocol
        assert isinstance(my_post_hook, PostHook)

    def test_hook_registration_dataclass(self, tmp_path: Path) -> None:
        """Test HookRegistration dataclass has all fields."""
        from yolo_developer.sdk.types import HookRegistration

        client = YoloClient(project_path=tmp_path)

        def my_hook(agent: str, state: dict) -> dict | None:
            return None

        client.register_hook(agent="analyst", phase="pre", callback=my_hook)
        hooks = client.list_hooks()

        assert len(hooks) == 1
        hook = hooks[0]

        assert isinstance(hook, HookRegistration)
        assert hasattr(hook, "hook_id")
        assert hasattr(hook, "agent")
        assert hasattr(hook, "phase")
        assert hasattr(hook, "callback")
        assert hasattr(hook, "timestamp")


class TestYoloClientHookErrorHandling:
    """Tests for graceful error handling (AC5)."""

    @pytest.mark.asyncio
    async def test_pre_hook_error_continues_execution(self, tmp_path: Path) -> None:
        """Test pre-hook errors don't block workflow."""
        client = YoloClient(project_path=tmp_path)
        hook2_called = []

        def failing_hook(agent: str, state: dict) -> dict | None:
            raise ValueError("Hook error!")

        def succeeding_hook(agent: str, state: dict) -> dict | None:
            hook2_called.append(True)
            return {"success": True}

        client.register_hook(agent="analyst", phase="pre", callback=failing_hook)
        client.register_hook(agent="analyst", phase="pre", callback=succeeding_hook)

        modifications, _results = await client._execute_pre_hooks("analyst", {})

        # Second hook should still be called
        assert len(hook2_called) == 1
        # Modifications from second hook should still apply
        assert modifications == {"success": True}

    @pytest.mark.asyncio
    async def test_pre_hook_error_recorded_in_result(self, tmp_path: Path) -> None:
        """Test pre-hook errors are recorded in HookResult."""
        client = YoloClient(project_path=tmp_path)

        def failing_hook(agent: str, state: dict) -> dict | None:
            raise ValueError("Test error")

        client.register_hook(agent="analyst", phase="pre", callback=failing_hook)

        _modifications, results = await client._execute_pre_hooks("analyst", {})

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error == "Test error"

    @pytest.mark.asyncio
    async def test_post_hook_error_continues_execution(self, tmp_path: Path) -> None:
        """Test post-hook errors don't block workflow."""
        client = YoloClient(project_path=tmp_path)
        hook2_called = []

        def failing_hook(agent: str, state: dict, output: dict) -> dict | None:
            raise RuntimeError("Post hook failed")

        def succeeding_hook(agent: str, state: dict, output: dict) -> dict | None:
            hook2_called.append(True)
            return {**output, "from_hook2": True}

        client.register_hook(agent="analyst", phase="post", callback=failing_hook)
        client.register_hook(agent="analyst", phase="post", callback=succeeding_hook)

        _modifications, results = await client._execute_post_hooks(
            "analyst", {}, {"original": True}
        )

        # Second hook should still be called
        assert len(hook2_called) == 1
        # First result should show failure
        assert results[0].success is False
        # Second result should show success
        assert results[1].success is True

    @pytest.mark.asyncio
    async def test_hook_execution_error_type_available(self, tmp_path: Path) -> None:
        """Test HookExecutionError is available for inspection."""
        from yolo_developer.sdk.exceptions import HookExecutionError

        # Verify the error type exists and has expected attributes
        error = HookExecutionError(
            "Test error",
            hook_id="hook-123",
            agent="analyst",
            phase="pre",
        )

        assert error.hook_id == "hook-123"
        assert error.agent == "analyst"
        assert error.phase == "pre"
        assert str(error) == "Test error"


class TestYoloClientListHooks:
    """Tests for list_hooks method."""

    def test_list_hooks_returns_sorted_by_timestamp(self, tmp_path: Path) -> None:
        """Test list_hooks returns hooks sorted by registration time."""
        import time

        client = YoloClient(project_path=tmp_path)

        def hook1(agent: str, state: dict) -> dict | None:
            return None

        def hook2(agent: str, state: dict) -> dict | None:
            return None

        def hook3(agent: str, state: dict) -> dict | None:
            return None

        id1 = client.register_hook(agent="analyst", phase="pre", callback=hook1)
        time.sleep(0.01)  # Small delay to ensure different timestamps
        id2 = client.register_hook(agent="pm", phase="pre", callback=hook2)
        time.sleep(0.01)
        id3 = client.register_hook(agent="dev", phase="pre", callback=hook3)

        hooks = client.list_hooks()

        assert len(hooks) == 3
        assert hooks[0].hook_id == id1
        assert hooks[1].hook_id == id2
        assert hooks[2].hook_id == id3

    def test_list_hooks_empty_initially(self, tmp_path: Path) -> None:
        """Test list_hooks returns empty list initially."""
        client = YoloClient(project_path=tmp_path)

        hooks = client.list_hooks()

        assert hooks == []


class TestYoloClientHookResult:
    """Tests for HookResult dataclass."""

    def test_hook_result_structure(self, tmp_path: Path) -> None:
        """Test HookResult has all expected fields."""
        from yolo_developer.sdk.types import HookResult

        result = HookResult(
            hook_id="hook-123",
            agent="analyst",
            phase="pre",
            success=True,
            modifications={"key": "value"},
        )

        assert result.hook_id == "hook-123"
        assert result.agent == "analyst"
        assert result.phase == "pre"
        assert result.success is True
        assert result.modifications == {"key": "value"}
        assert result.error is None
        assert result.timestamp is not None

    def test_hook_result_with_error(self, tmp_path: Path) -> None:
        """Test HookResult with error."""
        from yolo_developer.sdk.types import HookResult

        result = HookResult(
            hook_id="hook-123",
            agent="analyst",
            phase="pre",
            success=False,
            error="Something went wrong",
        )

        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.modifications is None


class TestYoloClientHookIntegration:
    """Tests for hook integration with run_async (Story 13.5 fix)."""

    @pytest.mark.asyncio
    async def test_pre_hook_fires_during_run_async(self, tmp_path: Path) -> None:
        """Test pre-hooks fire during workflow execution."""
        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)
        hook_calls: list[tuple[str, dict]] = []

        def capture_pre_hook(agent: str, state: dict) -> dict | None:
            hook_calls.append((agent, dict(state)))
            return {"injected": "data"}

        client.register_hook(agent="analyst", phase="pre", callback=capture_pre_hook)

        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.WorkflowConfig = MagicMock()
        mock_orchestrator.WorkflowConfig.return_value.entry_point = "analyst"
        mock_orchestrator.create_initial_state = MagicMock(
            return_value={"messages": [], "decisions": []}
        )
        mock_orchestrator.run_workflow = AsyncMock(
            return_value={
                "decisions": [],
                "messages": [],
                "current_agent": "analyst",
            }
        )

        with (
            patch("yolo_developer.seed.parse_seed") as mock_parse,
            patch.dict(
                "sys.modules",
                {"yolo_developer.orchestrator": mock_orchestrator},
            ),
        ):
            mock_seed_result = MagicMock()
            mock_seed_result.goal_count = 1
            mock_seed_result.feature_count = 1
            mock_seed_result.constraint_count = 0
            mock_seed_result.has_ambiguities = False
            mock_seed_result.ambiguities = []
            mock_parse.return_value = mock_seed_result

            await client.run_async(seed_content="Build something")

        # Verify pre-hook was called
        assert len(hook_calls) == 1
        assert hook_calls[0][0] == "analyst"

    @pytest.mark.asyncio
    async def test_post_hook_fires_during_run_async(self, tmp_path: Path) -> None:
        """Test post-hooks fire during workflow execution."""
        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)
        hook_calls: list[tuple[str, dict, dict]] = []

        def capture_post_hook(agent: str, state: dict, output: dict) -> dict | None:
            hook_calls.append((agent, dict(state), dict(output)))
            return None

        client.register_hook(agent="analyst", phase="post", callback=capture_post_hook)

        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.WorkflowConfig = MagicMock()
        mock_orchestrator.WorkflowConfig.return_value.entry_point = "analyst"
        mock_orchestrator.create_initial_state = MagicMock(
            return_value={"messages": [], "decisions": []}
        )
        mock_orchestrator.run_workflow = AsyncMock(
            return_value={
                "decisions": [],
                "messages": [],
                "current_agent": "analyst",
            }
        )

        with (
            patch("yolo_developer.seed.parse_seed") as mock_parse,
            patch.dict(
                "sys.modules",
                {"yolo_developer.orchestrator": mock_orchestrator},
            ),
        ):
            mock_seed_result = MagicMock()
            mock_seed_result.goal_count = 1
            mock_seed_result.feature_count = 1
            mock_seed_result.constraint_count = 0
            mock_seed_result.has_ambiguities = False
            mock_seed_result.ambiguities = []
            mock_parse.return_value = mock_seed_result

            await client.run_async(seed_content="Build something")

        # Verify post-hook was called
        assert len(hook_calls) == 1
        assert hook_calls[0][0] == "analyst"
        # Output should contain workflow results
        assert "decisions" in hook_calls[0][2]

    @pytest.mark.asyncio
    async def test_wildcard_hooks_fire_during_run_async(self, tmp_path: Path) -> None:
        """Test wildcard '*' hooks fire during workflow execution."""
        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)
        pre_calls: list[str] = []
        post_calls: list[str] = []

        def wildcard_pre(agent: str, state: dict) -> dict | None:
            pre_calls.append(agent)
            return None

        def wildcard_post(agent: str, state: dict, output: dict) -> dict | None:
            post_calls.append(agent)
            return None

        client.register_hook(agent="*", phase="pre", callback=wildcard_pre)
        client.register_hook(agent="*", phase="post", callback=wildcard_post)

        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.WorkflowConfig = MagicMock()
        mock_orchestrator.WorkflowConfig.return_value.entry_point = "analyst"
        mock_orchestrator.create_initial_state = MagicMock(
            return_value={"messages": [], "decisions": []}
        )
        mock_orchestrator.run_workflow = AsyncMock(
            return_value={
                "decisions": [],
                "messages": [],
                "current_agent": "pm",
            }
        )

        with (
            patch("yolo_developer.seed.parse_seed") as mock_parse,
            patch.dict(
                "sys.modules",
                {"yolo_developer.orchestrator": mock_orchestrator},
            ),
        ):
            mock_seed_result = MagicMock()
            mock_seed_result.goal_count = 1
            mock_seed_result.feature_count = 1
            mock_seed_result.constraint_count = 0
            mock_seed_result.has_ambiguities = False
            mock_seed_result.ambiguities = []
            mock_parse.return_value = mock_seed_result

            await client.run_async(seed_content="Build something")

        # Wildcard hooks should fire for any agent
        assert len(pre_calls) == 1
        assert pre_calls[0] == "analyst"  # Entry agent
        assert len(post_calls) == 1
        assert post_calls[0] == "pm"  # Last agent from workflow

    @pytest.mark.asyncio
    async def test_pre_hook_modifications_injected(self, tmp_path: Path) -> None:
        """Test pre-hook modifications are injected into initial state."""
        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)
        captured_state: list[dict] = []

        def inject_hook(agent: str, state: dict) -> dict | None:
            return {"custom_context": "injected_value"}

        client.register_hook(agent="analyst", phase="pre", callback=inject_hook)

        # Mock orchestrator to capture the state passed to run_workflow
        mock_orchestrator = MagicMock()
        mock_orchestrator.WorkflowConfig = MagicMock()
        mock_orchestrator.WorkflowConfig.return_value.entry_point = "analyst"
        mock_orchestrator.create_initial_state = MagicMock(
            return_value={"messages": [], "decisions": []}
        )

        async def capture_run_workflow(initial_state, config):
            captured_state.append(dict(initial_state))
            return {
                "decisions": [],
                "messages": [],
                "current_agent": "analyst",
            }

        mock_orchestrator.run_workflow = capture_run_workflow

        with (
            patch("yolo_developer.seed.parse_seed") as mock_parse,
            patch.dict(
                "sys.modules",
                {"yolo_developer.orchestrator": mock_orchestrator},
            ),
        ):
            mock_seed_result = MagicMock()
            mock_seed_result.goal_count = 1
            mock_seed_result.feature_count = 1
            mock_seed_result.constraint_count = 0
            mock_seed_result.has_ambiguities = False
            mock_seed_result.ambiguities = []
            mock_parse.return_value = mock_seed_result

            await client.run_async(seed_content="Build something")

        # Verify injected data was passed to workflow
        assert len(captured_state) == 1
        assert captured_state[0].get("custom_context") == "injected_value"


# ============================================================================
# Event Emission Tests (Story 13.6)
# ============================================================================


class TestYoloClientEventTypes:
    """Tests for event type definitions (AC1)."""

    def test_event_type_enum_values(self) -> None:
        """Test EventType enum defines all required event categories."""
        from yolo_developer.sdk.types import EventType

        # Verify all required event types exist
        assert hasattr(EventType, "WORKFLOW_START")
        assert hasattr(EventType, "WORKFLOW_END")
        assert hasattr(EventType, "AGENT_START")
        assert hasattr(EventType, "AGENT_END")
        assert hasattr(EventType, "GATE_PASS")
        assert hasattr(EventType, "GATE_FAIL")
        assert hasattr(EventType, "ERROR")

    def test_event_data_structure(self) -> None:
        """Test EventData dataclass has all required fields."""
        from yolo_developer.sdk.types import EventData, EventType

        event = EventData(
            event_type=EventType.AGENT_START,
            agent="analyst",
            data={"context": "test"},
        )

        assert event.event_type == EventType.AGENT_START
        assert event.agent == "analyst"
        assert event.data == {"context": "test"}
        assert event.timestamp is not None
        assert event.timestamp.tzinfo is not None  # Timezone-aware

    def test_event_data_is_frozen(self) -> None:
        """Test EventData is immutable (frozen dataclass)."""
        from yolo_developer.sdk.types import EventData, EventType

        event = EventData(event_type=EventType.WORKFLOW_START)

        with pytest.raises(AttributeError):
            event.agent = "new_agent"  # type: ignore[misc]

    def test_event_data_defaults(self) -> None:
        """Test EventData has sensible defaults."""
        from yolo_developer.sdk.types import EventData, EventType

        event = EventData(event_type=EventType.ERROR)

        assert event.agent is None
        assert event.data == {}
        assert event.timestamp is not None

    def test_event_callback_error_type(self) -> None:
        """Test EventCallbackError is available and has expected attributes."""
        from yolo_developer.sdk.exceptions import EventCallbackError

        error = EventCallbackError(
            "Callback failed",
            subscription_id="sub-123",
            event_type="AGENT_START",
        )

        assert error.subscription_id == "sub-123"
        assert error.event_type == "AGENT_START"
        assert str(error) == "Callback failed"


class TestYoloClientEventSubscription:
    """Tests for event subscription (AC2)."""

    def test_subscribe_returns_subscription_id(self, tmp_path: Path) -> None:
        """Test subscribe returns a unique subscription ID."""
        client = YoloClient(project_path=tmp_path)

        def my_callback(event: Any) -> None:
            pass

        sub_id = client.subscribe(my_callback)

        assert sub_id.startswith("sub-")
        assert len(sub_id) > 4

    def test_subscribe_stores_subscription(self, tmp_path: Path) -> None:
        """Test subscribe stores the subscription."""
        from yolo_developer.sdk.types import EventType

        client = YoloClient(project_path=tmp_path)

        def my_callback(event: Any) -> None:
            pass

        sub_id = client.subscribe(my_callback, event_types=[EventType.AGENT_START])

        subscriptions = client.list_subscriptions()
        assert len(subscriptions) == 1
        assert subscriptions[0].subscription_id == sub_id
        assert subscriptions[0].event_types == [EventType.AGENT_START]

    def test_subscribe_all_events(self, tmp_path: Path) -> None:
        """Test subscribe with event_types=None subscribes to all events."""
        client = YoloClient(project_path=tmp_path)

        def my_callback(event: Any) -> None:
            pass

        client.subscribe(my_callback)

        subscriptions = client.list_subscriptions()
        assert subscriptions[0].event_types is None

    def test_subscribe_multiple_event_types(self, tmp_path: Path) -> None:
        """Test subscribing to multiple event types."""
        from yolo_developer.sdk.types import EventType

        client = YoloClient(project_path=tmp_path)

        def my_callback(event: Any) -> None:
            pass

        client.subscribe(
            my_callback,
            event_types=[EventType.AGENT_START, EventType.AGENT_END],
        )

        subscriptions = client.list_subscriptions()
        assert subscriptions[0].event_types == [
            EventType.AGENT_START,
            EventType.AGENT_END,
        ]

    def test_multiple_callbacks_same_event(self, tmp_path: Path) -> None:
        """Test multiple callbacks can subscribe to the same event type."""
        from yolo_developer.sdk.types import EventType

        client = YoloClient(project_path=tmp_path)

        def callback1(event: Any) -> None:
            pass

        def callback2(event: Any) -> None:
            pass

        sub_id1 = client.subscribe(callback1, event_types=[EventType.AGENT_START])
        sub_id2 = client.subscribe(callback2, event_types=[EventType.AGENT_START])

        subscriptions = client.list_subscriptions()
        assert len(subscriptions) == 2
        assert sub_id1 != sub_id2


class TestYoloClientEventUnsubscription:
    """Tests for event unsubscription (AC4)."""

    def test_unsubscribe_removes_subscription(self, tmp_path: Path) -> None:
        """Test unsubscribe removes the subscription."""
        client = YoloClient(project_path=tmp_path)

        def my_callback(event: Any) -> None:
            pass

        sub_id = client.subscribe(my_callback)
        assert len(client.list_subscriptions()) == 1

        result = client.unsubscribe(sub_id)

        assert result is True
        assert len(client.list_subscriptions()) == 0

    def test_unsubscribe_not_found(self, tmp_path: Path) -> None:
        """Test unsubscribe returns False for non-existent subscription."""
        client = YoloClient(project_path=tmp_path)

        result = client.unsubscribe("sub-nonexistent")

        assert result is False

    def test_list_subscriptions_reflects_removal(self, tmp_path: Path) -> None:
        """Test list_subscriptions reflects subscription removal."""
        from yolo_developer.sdk.types import EventType

        client = YoloClient(project_path=tmp_path)

        def callback1(event: Any) -> None:
            pass

        def callback2(event: Any) -> None:
            pass

        sub_id1 = client.subscribe(callback1, event_types=[EventType.AGENT_START])
        sub_id2 = client.subscribe(callback2, event_types=[EventType.AGENT_END])

        assert len(client.list_subscriptions()) == 2

        client.unsubscribe(sub_id1)

        subscriptions = client.list_subscriptions()
        assert len(subscriptions) == 1
        assert subscriptions[0].subscription_id == sub_id2


class TestYoloClientEventEmission:
    """Tests for event emission during workflow (AC3)."""

    @pytest.mark.asyncio
    async def test_emit_event_calls_matching_callbacks(self, tmp_path: Path) -> None:
        """Test _emit_event calls callbacks for matching event types."""
        from yolo_developer.sdk.types import EventData, EventType

        client = YoloClient(project_path=tmp_path)
        received_events: list[EventData] = []

        def capture_event(event: EventData) -> None:
            received_events.append(event)

        client.subscribe(capture_event, event_types=[EventType.AGENT_START])

        await client._emit_event(EventType.AGENT_START, agent="analyst")

        assert len(received_events) == 1
        assert received_events[0].event_type == EventType.AGENT_START
        assert received_events[0].agent == "analyst"

    @pytest.mark.asyncio
    async def test_emit_event_filters_by_event_type(self, tmp_path: Path) -> None:
        """Test _emit_event only calls callbacks for matching types."""
        from yolo_developer.sdk.types import EventData, EventType

        client = YoloClient(project_path=tmp_path)
        received_events: list[EventData] = []

        def capture_event(event: EventData) -> None:
            received_events.append(event)

        # Subscribe only to AGENT_END
        client.subscribe(capture_event, event_types=[EventType.AGENT_END])

        # Emit AGENT_START - should NOT trigger callback
        await client._emit_event(EventType.AGENT_START, agent="analyst")

        assert len(received_events) == 0

    @pytest.mark.asyncio
    async def test_emit_event_all_subscribers_receive(self, tmp_path: Path) -> None:
        """Test all matching subscribers receive events."""
        from yolo_developer.sdk.types import EventData, EventType

        client = YoloClient(project_path=tmp_path)
        received1: list[EventData] = []
        received2: list[EventData] = []

        def callback1(event: EventData) -> None:
            received1.append(event)

        def callback2(event: EventData) -> None:
            received2.append(event)

        client.subscribe(callback1, event_types=[EventType.WORKFLOW_START])
        client.subscribe(callback2, event_types=[EventType.WORKFLOW_START])

        await client._emit_event(EventType.WORKFLOW_START, data={"workflow_id": "test-123"})

        assert len(received1) == 1
        assert len(received2) == 1
        assert received1[0].data["workflow_id"] == "test-123"
        assert received2[0].data["workflow_id"] == "test-123"

    @pytest.mark.asyncio
    async def test_emit_event_with_event_data(self, tmp_path: Path) -> None:
        """Test _emit_event passes data to EventData correctly."""
        from yolo_developer.sdk.types import EventData, EventType

        client = YoloClient(project_path=tmp_path)
        received_events: list[EventData] = []

        def capture_event(event: EventData) -> None:
            received_events.append(event)

        client.subscribe(capture_event)

        await client._emit_event(
            EventType.WORKFLOW_END,
            agent="pm",
            data={"status": "completed", "duration": 5.2},
        )

        assert len(received_events) == 1
        event = received_events[0]
        assert event.event_type == EventType.WORKFLOW_END
        assert event.agent == "pm"
        assert event.data["status"] == "completed"
        assert event.data["duration"] == 5.2

    @pytest.mark.asyncio
    async def test_events_fire_during_run_async(self, tmp_path: Path) -> None:
        """Test events fire at appropriate points during workflow execution."""
        from yolo_developer.sdk.types import EventData, EventType

        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)
        received_events: list[EventData] = []

        def capture_event(event: EventData) -> None:
            received_events.append(event)

        client.subscribe(capture_event)

        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.WorkflowConfig = MagicMock()
        mock_orchestrator.WorkflowConfig.return_value.entry_point = "analyst"
        mock_orchestrator.create_initial_state = MagicMock(
            return_value={"messages": [], "decisions": []}
        )
        mock_orchestrator.run_workflow = AsyncMock(
            return_value={
                "decisions": [],
                "messages": [],
                "current_agent": "pm",
            }
        )

        with (
            patch("yolo_developer.seed.parse_seed") as mock_parse,
            patch.dict(
                "sys.modules",
                {"yolo_developer.orchestrator": mock_orchestrator},
            ),
        ):
            mock_seed_result = MagicMock()
            mock_seed_result.goal_count = 1
            mock_seed_result.feature_count = 1
            mock_seed_result.constraint_count = 0
            mock_seed_result.has_ambiguities = False
            mock_seed_result.ambiguities = []
            mock_parse.return_value = mock_seed_result

            await client.run_async(seed_content="Build something")

        # Verify expected events fired
        event_types = [e.event_type for e in received_events]
        assert EventType.WORKFLOW_START in event_types
        assert EventType.AGENT_START in event_types
        assert EventType.AGENT_END in event_types
        assert EventType.GATE_PASS in event_types  # Seed quality + workflow completion
        assert EventType.WORKFLOW_END in event_types

    @pytest.mark.asyncio
    async def test_gate_pass_fires_on_seed_acceptance(self, tmp_path: Path) -> None:
        """Test GATE_PASS event fires when seed passes quality gate."""
        from yolo_developer.sdk.types import EventData, EventType

        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)
        received_events: list[EventData] = []

        def capture_event(event: EventData) -> None:
            received_events.append(event)

        client.subscribe(capture_event, event_types=[EventType.GATE_PASS])

        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.WorkflowConfig = MagicMock()
        mock_orchestrator.WorkflowConfig.return_value.entry_point = "analyst"
        mock_orchestrator.create_initial_state = MagicMock(
            return_value={"messages": [], "decisions": []}
        )
        mock_orchestrator.run_workflow = AsyncMock(
            return_value={
                "decisions": [],
                "messages": [],
                "current_agent": "pm",
            }
        )

        with (
            patch("yolo_developer.seed.parse_seed") as mock_parse,
            patch.dict(
                "sys.modules",
                {"yolo_developer.orchestrator": mock_orchestrator},
            ),
        ):
            mock_seed_result = MagicMock()
            mock_seed_result.goal_count = 1
            mock_seed_result.feature_count = 1
            mock_seed_result.constraint_count = 0
            mock_seed_result.has_ambiguities = False
            mock_seed_result.ambiguities = []
            mock_parse.return_value = mock_seed_result

            await client.run_async(seed_content="Build something")

        # Verify GATE_PASS events fired (seed quality + workflow completion)
        assert len(received_events) >= 2
        gate_names = [e.data.get("gate") for e in received_events]
        assert "seed_quality" in gate_names
        assert "workflow_completion" in gate_names

    @pytest.mark.asyncio
    async def test_gate_fail_fires_on_seed_rejection(self, tmp_path: Path) -> None:
        """Test GATE_FAIL event fires when seed fails quality gate."""
        from yolo_developer.sdk.types import EventData, EventType

        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)
        received_events: list[EventData] = []

        def capture_event(event: EventData) -> None:
            received_events.append(event)

        client.subscribe(capture_event, event_types=[EventType.GATE_FAIL])

        with patch("yolo_developer.seed.parse_seed") as mock_parse:
            # Mock seed result with low quality (no goals, no features, many ambiguities)
            mock_seed_result = MagicMock()
            mock_seed_result.goal_count = 0
            mock_seed_result.feature_count = 0
            mock_seed_result.constraint_count = 0
            mock_seed_result.has_ambiguities = True
            mock_seed_result.ambiguities = [
                MagicMock(description="Ambiguity 1"),
                MagicMock(description="Ambiguity 2"),
                MagicMock(description="Ambiguity 3"),
                MagicMock(description="Ambiguity 4"),
                MagicMock(description="Ambiguity 5"),
                MagicMock(description="Ambiguity 6"),
                MagicMock(description="Ambiguity 7"),
            ]
            mock_parse.return_value = mock_seed_result

            with pytest.raises(WorkflowExecutionError, match="Seed was rejected"):
                await client.run_async(seed_content="vague request")

        # Verify GATE_FAIL event fired
        assert len(received_events) == 1
        event = received_events[0]
        assert event.event_type == EventType.GATE_FAIL
        assert event.data["gate"] == "seed_quality"
        assert "reason" in event.data

    @pytest.mark.asyncio
    async def test_error_event_fires_on_exception(self, tmp_path: Path) -> None:
        """Test ERROR event fires when workflow encounters an exception."""
        from yolo_developer.sdk.types import EventData, EventType

        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)
        received_events: list[EventData] = []

        def capture_event(event: EventData) -> None:
            received_events.append(event)

        client.subscribe(capture_event, event_types=[EventType.ERROR])

        # Mock orchestrator to raise an exception
        mock_orchestrator = MagicMock()
        mock_orchestrator.WorkflowConfig = MagicMock()
        mock_orchestrator.WorkflowConfig.return_value.entry_point = "analyst"
        mock_orchestrator.create_initial_state = MagicMock(
            return_value={"messages": [], "decisions": []}
        )
        mock_orchestrator.run_workflow = AsyncMock(side_effect=RuntimeError("Workflow failed!"))

        with (
            patch("yolo_developer.seed.parse_seed") as mock_parse,
            patch.dict(
                "sys.modules",
                {"yolo_developer.orchestrator": mock_orchestrator},
            ),
        ):
            mock_seed_result = MagicMock()
            mock_seed_result.goal_count = 1
            mock_seed_result.feature_count = 1
            mock_seed_result.constraint_count = 0
            mock_seed_result.has_ambiguities = False
            mock_seed_result.ambiguities = []
            mock_parse.return_value = mock_seed_result

            with pytest.raises(WorkflowExecutionError):
                await client.run_async(seed_content="Build something")

        # Verify ERROR event was emitted
        assert len(received_events) == 1
        assert received_events[0].event_type == EventType.ERROR
        assert "error" in received_events[0].data


class TestYoloClientAsyncCallbackSupport:
    """Tests for async callback support (AC5)."""

    @pytest.mark.asyncio
    async def test_async_callback_awaited(self, tmp_path: Path) -> None:
        """Test async callbacks are awaited properly."""
        from yolo_developer.sdk.types import EventData, EventType

        client = YoloClient(project_path=tmp_path)
        async_called = []

        async def async_callback(event: EventData) -> None:
            await asyncio.sleep(0.001)  # Simulate async work
            async_called.append(event)

        client.subscribe(async_callback, event_types=[EventType.AGENT_START])

        await client._emit_event(EventType.AGENT_START, agent="analyst")

        assert len(async_called) == 1
        assert async_called[0].agent == "analyst"

    @pytest.mark.asyncio
    async def test_sync_callback_executed(self, tmp_path: Path) -> None:
        """Test sync callbacks are executed synchronously."""
        from yolo_developer.sdk.types import EventData, EventType

        client = YoloClient(project_path=tmp_path)
        sync_called = []

        def sync_callback(event: EventData) -> None:
            sync_called.append(event)

        client.subscribe(sync_callback, event_types=[EventType.WORKFLOW_END])

        await client._emit_event(EventType.WORKFLOW_END, data={"status": "done"})

        assert len(sync_called) == 1
        assert sync_called[0].data["status"] == "done"

    @pytest.mark.asyncio
    async def test_mixed_sync_async_callbacks(self, tmp_path: Path) -> None:
        """Test mixed sync and async callbacks both execute."""
        from yolo_developer.sdk.types import EventData, EventType

        client = YoloClient(project_path=tmp_path)
        sync_results: list[str] = []
        async_results: list[str] = []

        def sync_callback(event: EventData) -> None:
            sync_results.append("sync")

        async def async_callback(event: EventData) -> None:
            await asyncio.sleep(0.001)
            async_results.append("async")

        client.subscribe(sync_callback)
        client.subscribe(async_callback)

        await client._emit_event(EventType.WORKFLOW_START)

        assert sync_results == ["sync"]
        assert async_results == ["async"]

    def test_callback_protocol_accepts_sync(self, tmp_path: Path) -> None:
        """Test EventCallback protocol accepts sync functions."""
        from yolo_developer.sdk.types import EventCallback, EventData

        def sync_callback(event: EventData) -> None:
            pass

        # Should match the protocol
        assert isinstance(sync_callback, EventCallback)

    def test_callback_protocol_accepts_async(self, tmp_path: Path) -> None:
        """Test EventCallback protocol accepts async functions."""
        from yolo_developer.sdk.types import EventCallback, EventData

        async def async_callback(event: EventData) -> None:
            pass

        # Should match the protocol
        assert isinstance(async_callback, EventCallback)


class TestYoloClientEventCallbackErrorHandling:
    """Tests for graceful error handling in callbacks (AC6)."""

    @pytest.mark.asyncio
    async def test_callback_error_does_not_block_others(self, tmp_path: Path) -> None:
        """Test callback errors don't block other callbacks."""
        from yolo_developer.sdk.types import EventData, EventType

        client = YoloClient(project_path=tmp_path)
        callback2_called = []

        def failing_callback(event: EventData) -> None:
            raise ValueError("Callback error!")

        def succeeding_callback(event: EventData) -> None:
            callback2_called.append(event)

        client.subscribe(failing_callback)
        client.subscribe(succeeding_callback)

        await client._emit_event(EventType.AGENT_START, agent="analyst")

        # Second callback should still be called
        assert len(callback2_called) == 1

    @pytest.mark.asyncio
    async def test_callback_error_logged(self, tmp_path: Path) -> None:
        """Test callback errors are logged with context."""
        from yolo_developer.sdk.types import EventData, EventType

        client = YoloClient(project_path=tmp_path)

        def failing_callback(event: EventData) -> None:
            raise RuntimeError("Test error")

        client.subscribe(failing_callback)

        # Should not raise, but logs the error
        await client._emit_event(EventType.WORKFLOW_START)

        # The error was handled gracefully (no exception raised)
        # Logging would be verified with structlog capture if needed

    @pytest.mark.asyncio
    async def test_workflow_continues_despite_callback_error(self, tmp_path: Path) -> None:
        """Test workflow execution continues despite callback errors."""
        from yolo_developer.sdk.types import EventData

        (tmp_path / ".yolo").mkdir()
        client = YoloClient(project_path=tmp_path)

        def failing_callback(event: EventData) -> None:
            raise ValueError("Callback failed!")

        client.subscribe(failing_callback)

        # Mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.WorkflowConfig = MagicMock()
        mock_orchestrator.WorkflowConfig.return_value.entry_point = "analyst"
        mock_orchestrator.create_initial_state = MagicMock(
            return_value={"messages": [], "decisions": []}
        )
        mock_orchestrator.run_workflow = AsyncMock(
            return_value={
                "decisions": [],
                "messages": [],
                "current_agent": "analyst",
            }
        )

        with (
            patch("yolo_developer.seed.parse_seed") as mock_parse,
            patch.dict(
                "sys.modules",
                {"yolo_developer.orchestrator": mock_orchestrator},
            ),
        ):
            mock_seed_result = MagicMock()
            mock_seed_result.goal_count = 1
            mock_seed_result.feature_count = 1
            mock_seed_result.constraint_count = 0
            mock_seed_result.has_ambiguities = False
            mock_seed_result.ambiguities = []
            mock_parse.return_value = mock_seed_result

            # Should complete successfully despite callback error
            result = await client.run_async(seed_content="Build something")

        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_async_callback_error_handled(self, tmp_path: Path) -> None:
        """Test async callback errors are handled gracefully."""
        from yolo_developer.sdk.types import EventData, EventType

        client = YoloClient(project_path=tmp_path)
        callback2_called = []

        async def failing_async_callback(event: EventData) -> None:
            await asyncio.sleep(0.001)
            raise RuntimeError("Async callback failed!")

        async def succeeding_async_callback(event: EventData) -> None:
            await asyncio.sleep(0.001)
            callback2_called.append(event)

        client.subscribe(failing_async_callback)
        client.subscribe(succeeding_async_callback)

        await client._emit_event(EventType.AGENT_END, agent="dev")

        # Second callback should still be called
        assert len(callback2_called) == 1


class TestYoloClientEventFiltering:
    """Tests for filtering events by type (AC2, AC3)."""

    @pytest.mark.asyncio
    async def test_filter_single_event_type(self, tmp_path: Path) -> None:
        """Test filtering to receive only specific event type."""
        from yolo_developer.sdk.types import EventData, EventType

        client = YoloClient(project_path=tmp_path)
        received: list[EventData] = []

        def capture(event: EventData) -> None:
            received.append(event)

        client.subscribe(capture, event_types=[EventType.WORKFLOW_END])

        # Emit multiple event types
        await client._emit_event(EventType.WORKFLOW_START)
        await client._emit_event(EventType.AGENT_START, agent="analyst")
        await client._emit_event(EventType.AGENT_END, agent="analyst")
        await client._emit_event(EventType.WORKFLOW_END)

        # Should only receive WORKFLOW_END
        assert len(received) == 1
        assert received[0].event_type == EventType.WORKFLOW_END

    @pytest.mark.asyncio
    async def test_filter_multiple_event_types(self, tmp_path: Path) -> None:
        """Test filtering to receive multiple specific event types."""
        from yolo_developer.sdk.types import EventData, EventType

        client = YoloClient(project_path=tmp_path)
        received: list[EventData] = []

        def capture(event: EventData) -> None:
            received.append(event)

        client.subscribe(capture, event_types=[EventType.AGENT_START, EventType.AGENT_END])

        # Emit multiple event types
        await client._emit_event(EventType.WORKFLOW_START)
        await client._emit_event(EventType.AGENT_START, agent="analyst")
        await client._emit_event(EventType.AGENT_END, agent="analyst")
        await client._emit_event(EventType.WORKFLOW_END)

        # Should only receive AGENT_START and AGENT_END
        assert len(received) == 2
        assert received[0].event_type == EventType.AGENT_START
        assert received[1].event_type == EventType.AGENT_END

    @pytest.mark.asyncio
    async def test_subscribe_all_receives_all(self, tmp_path: Path) -> None:
        """Test subscribing to all events receives all event types."""
        from yolo_developer.sdk.types import EventData, EventType

        client = YoloClient(project_path=tmp_path)
        received: list[EventData] = []

        def capture(event: EventData) -> None:
            received.append(event)

        # Subscribe to all (event_types=None)
        client.subscribe(capture)

        # Emit multiple event types
        await client._emit_event(EventType.WORKFLOW_START)
        await client._emit_event(EventType.AGENT_START, agent="analyst")
        await client._emit_event(EventType.GATE_PASS)
        await client._emit_event(EventType.ERROR, data={"error": "test"})

        # Should receive all
        assert len(received) == 4


class TestYoloClientEventSubscriptionInfo:
    """Tests for EventSubscription dataclass."""

    def test_event_subscription_structure(self, tmp_path: Path) -> None:
        """Test EventSubscription has all expected fields."""
        from yolo_developer.sdk.types import EventType

        client = YoloClient(project_path=tmp_path)

        def my_callback(event: Any) -> None:
            pass

        client.subscribe(my_callback, event_types=[EventType.WORKFLOW_START])

        subscriptions = client.list_subscriptions()
        sub = subscriptions[0]

        assert hasattr(sub, "subscription_id")
        assert hasattr(sub, "event_types")
        assert hasattr(sub, "callback")
        assert hasattr(sub, "timestamp")
        assert sub.timestamp is not None

    def test_list_subscriptions_sorted_by_timestamp(self, tmp_path: Path) -> None:
        """Test list_subscriptions returns subscriptions sorted by creation time."""
        import time

        from yolo_developer.sdk.types import EventType

        client = YoloClient(project_path=tmp_path)

        def cb1(event: Any) -> None:
            pass

        def cb2(event: Any) -> None:
            pass

        def cb3(event: Any) -> None:
            pass

        id1 = client.subscribe(cb1, event_types=[EventType.WORKFLOW_START])
        time.sleep(0.01)
        id2 = client.subscribe(cb2, event_types=[EventType.AGENT_START])
        time.sleep(0.01)
        id3 = client.subscribe(cb3, event_types=[EventType.WORKFLOW_END])

        subscriptions = client.list_subscriptions()

        assert len(subscriptions) == 3
        assert subscriptions[0].subscription_id == id1
        assert subscriptions[1].subscription_id == id2
        assert subscriptions[2].subscription_id == id3

    def test_list_subscriptions_empty_initially(self, tmp_path: Path) -> None:
        """Test list_subscriptions returns empty list initially."""
        client = YoloClient(project_path=tmp_path)

        subscriptions = client.list_subscriptions()

        assert subscriptions == []
