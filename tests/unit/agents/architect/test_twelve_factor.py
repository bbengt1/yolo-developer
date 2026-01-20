"""Tests for Twelve-Factor analysis framework (Story 7.2, Tasks 2-6, 10).

Tests verify the analyze_twelve_factor function and factor-specific analyzers.
"""

from __future__ import annotations

import pytest


class TestAnalyzeTwelveFactor:
    """Test analyze_twelve_factor main function."""

    @pytest.mark.asyncio
    async def test_analyze_twelve_factor_returns_analysis(self) -> None:
        """Test that analyze_twelve_factor returns TwelveFactorAnalysis."""
        from yolo_developer.agents.architect.twelve_factor import analyze_twelve_factor
        from yolo_developer.agents.architect.types import TwelveFactorAnalysis

        story = {
            "id": "story-001",
            "title": "Add user authentication",
            "description": "Implement OAuth2 authentication flow",
        }

        result = await analyze_twelve_factor(story)

        assert isinstance(result, TwelveFactorAnalysis)
        assert isinstance(result.factor_results, dict)
        assert isinstance(result.applicable_factors, tuple)
        assert 0.0 <= result.overall_compliance <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_twelve_factor_with_empty_story(self) -> None:
        """Test analyze_twelve_factor handles empty story gracefully."""
        from yolo_developer.agents.architect.twelve_factor import analyze_twelve_factor

        story: dict[str, str] = {}

        result = await analyze_twelve_factor(story)

        assert result is not None
        assert isinstance(result.factor_results, dict)

    @pytest.mark.asyncio
    async def test_analyze_twelve_factor_returns_factor_results(self) -> None:
        """Test that all 12 factors are analyzed."""
        from yolo_developer.agents.architect.twelve_factor import analyze_twelve_factor
        from yolo_developer.agents.architect.types import TWELVE_FACTORS

        story = {
            "id": "story-001",
            "title": "Setup database connection",
            "description": "Configure PostgreSQL connection",
        }

        result = await analyze_twelve_factor(story)

        # All 12 factors should have results
        assert len(result.factor_results) == 12
        for factor in TWELVE_FACTORS:
            assert factor in result.factor_results

    @pytest.mark.asyncio
    async def test_analyze_twelve_factor_calculates_compliance(self) -> None:
        """Test that overall_compliance is calculated from applicable factors."""
        from yolo_developer.agents.architect.twelve_factor import analyze_twelve_factor

        story = {
            "id": "story-001",
            "title": "Generic story",
            "description": "Some description",
        }

        result = await analyze_twelve_factor(story)

        # Overall compliance should be between 0 and 1
        assert 0.0 <= result.overall_compliance <= 1.0


class TestAnalyzeTwelveFactorImport:
    """Test analyze_twelve_factor is importable from architect module."""

    def test_analyze_twelve_factor_importable(self) -> None:
        """Test function is importable from architect module."""
        from yolo_developer.agents.architect import analyze_twelve_factor

        assert analyze_twelve_factor is not None
        assert callable(analyze_twelve_factor)


class TestAnalyzeFactorHelper:
    """Test _analyze_factor helper function."""

    def test_analyze_factor_returns_factor_result(self) -> None:
        """Test _analyze_factor returns FactorResult."""
        from yolo_developer.agents.architect.twelve_factor import _analyze_factor
        from yolo_developer.agents.architect.types import FactorResult

        story = {
            "id": "story-001",
            "title": "Test story",
            "description": "Test description",
        }

        result = _analyze_factor(story, "config")

        assert isinstance(result, FactorResult)
        assert result.factor_name == "config"

    def test_analyze_factor_handles_unknown_factor(self) -> None:
        """Test _analyze_factor handles unknown factors gracefully."""
        from yolo_developer.agents.architect.twelve_factor import _analyze_factor

        story = {"title": "Test"}

        result = _analyze_factor(story, "unknown_factor")

        assert result.factor_name == "unknown_factor"
        assert result.applies is False


class TestConfigFactorAnalysis:
    """Test Factor III (Config) analysis (AC: 2)."""

    def test_detects_hardcoded_url(self) -> None:
        """Test detection of hardcoded URLs in story."""
        from yolo_developer.agents.architect.twelve_factor import _analyze_factor

        story = {
            "title": "Database setup",
            "description": "Connect to postgresql://localhost:5432/mydb",
        }

        result = _analyze_factor(story, "config")

        assert result.applies is True
        assert result.compliant is False
        assert "hardcoded" in result.finding.lower() or "url" in result.finding.lower()

    def test_detects_hardcoded_api_key(self) -> None:
        """Test detection of hardcoded API keys."""
        from yolo_developer.agents.architect.twelve_factor import _analyze_factor

        story = {
            "title": "API Integration",
            "description": "Use api_key=sk-12345 for authentication",
        }

        result = _analyze_factor(story, "config")

        assert result.applies is True
        assert result.compliant is False

    def test_recommends_environment_variables(self) -> None:
        """Test that config violations recommend environment variables."""
        from yolo_developer.agents.architect.twelve_factor import _analyze_factor

        story = {
            "title": "Config setup",
            "description": "Set database password to secret123",
        }

        result = _analyze_factor(story, "config")

        assert result.applies is True
        if not result.compliant:
            assert (
                "environment" in result.recommendation.lower()
                or "env" in result.recommendation.lower()
            )

    def test_config_compliant_when_using_env_vars(self) -> None:
        """Test config is compliant when story mentions env vars."""
        from yolo_developer.agents.architect.twelve_factor import _analyze_factor

        story = {
            "title": "Config setup",
            "description": "Load database URL from DATABASE_URL environment variable",
        }

        result = _analyze_factor(story, "config")

        assert result.applies is True
        assert result.compliant is True


class TestProcessesFactorAnalysis:
    """Test Factor VI (Processes) analysis (AC: 3)."""

    def test_detects_session_state(self) -> None:
        """Test detection of session state patterns."""
        from yolo_developer.agents.architect.twelve_factor import _analyze_factor

        story = {
            "title": "User sessions",
            "description": "Store user session in local memory",
        }

        result = _analyze_factor(story, "processes")

        assert result.applies is True
        assert result.compliant is False
        assert "session" in result.finding.lower() or "state" in result.finding.lower()

    def test_detects_sticky_sessions(self) -> None:
        """Test detection of sticky session patterns."""
        from yolo_developer.agents.architect.twelve_factor import _analyze_factor

        story = {
            "title": "Load balancing",
            "description": "Configure sticky sessions for user affinity",
        }

        result = _analyze_factor(story, "processes")

        assert result.applies is True
        assert result.compliant is False

    def test_detects_local_file_storage(self) -> None:
        """Test detection of local file storage patterns."""
        from yolo_developer.agents.architect.twelve_factor import _analyze_factor

        story = {
            "title": "File uploads",
            "description": "Store uploaded files in /var/uploads directory",
        }

        result = _analyze_factor(story, "processes")

        assert result.applies is True
        assert result.compliant is False

    def test_stateless_process_compliant(self) -> None:
        """Test compliant stateless process design."""
        from yolo_developer.agents.architect.twelve_factor import _analyze_factor

        story = {
            "title": "Request handling",
            "description": "Process each request independently without shared state",
        }

        result = _analyze_factor(story, "processes")

        # Should be compliant or not applicable
        assert result.compliant is True or result.applies is False


class TestBackingServicesFactorAnalysis:
    """Test Factor IV (Backing Services) analysis (AC: 4)."""

    def test_detects_database_reference(self) -> None:
        """Test detection of database references."""
        from yolo_developer.agents.architect.twelve_factor import _analyze_factor

        story = {
            "title": "Data storage",
            "description": "Store user data in PostgreSQL database",
        }

        result = _analyze_factor(story, "backing_services")

        assert result.applies is True

    def test_detects_cache_reference(self) -> None:
        """Test detection of cache references."""
        from yolo_developer.agents.architect.twelve_factor import _analyze_factor

        story = {
            "title": "Caching",
            "description": "Use Redis for session caching",
        }

        result = _analyze_factor(story, "backing_services")

        assert result.applies is True

    def test_detects_message_queue_reference(self) -> None:
        """Test detection of message queue references."""
        from yolo_developer.agents.architect.twelve_factor import _analyze_factor

        story = {
            "title": "Async processing",
            "description": "Send messages to RabbitMQ queue for processing",
        }

        result = _analyze_factor(story, "backing_services")

        assert result.applies is True

    def test_recommends_connection_string_externalization(self) -> None:
        """Test recommendation for connection string externalization."""
        from yolo_developer.agents.architect.twelve_factor import _analyze_factor

        story = {
            "title": "Database connection",
            "description": "Connect to mysql://user:pass@localhost/db",
        }

        result = _analyze_factor(story, "backing_services")

        assert result.applies is True
        assert result.compliant is False
        assert (
            "connection" in result.recommendation.lower()
            or "external" in result.recommendation.lower()
        )


class TestCodebaseFactorAnalysis:
    """Test Factor I (Codebase) analysis."""

    def test_codebase_factor_exists(self) -> None:
        """Test that codebase factor can be analyzed."""
        from yolo_developer.agents.architect.twelve_factor import _analyze_factor

        story = {
            "title": "Setup repository",
            "description": "Initialize git repository",
        }

        result = _analyze_factor(story, "codebase")

        assert result.factor_name == "codebase"


class TestDependenciesFactorAnalysis:
    """Test Factor II (Dependencies) analysis."""

    def test_dependencies_factor_exists(self) -> None:
        """Test that dependencies factor can be analyzed."""
        from yolo_developer.agents.architect.twelve_factor import _analyze_factor

        story = {
            "title": "Add dependencies",
            "description": "Add required packages to requirements.txt",
        }

        result = _analyze_factor(story, "dependencies")

        assert result.factor_name == "dependencies"


class TestBuildReleaseRunFactorAnalysis:
    """Test Factor V (Build, Release, Run) analysis."""

    def test_build_release_run_factor_exists(self) -> None:
        """Test that build_release_run factor can be analyzed."""
        from yolo_developer.agents.architect.twelve_factor import _analyze_factor

        story = {
            "title": "CI/CD pipeline",
            "description": "Setup build and deployment pipeline",
        }

        result = _analyze_factor(story, "build_release_run")

        assert result.factor_name == "build_release_run"


class TestPortBindingFactorAnalysis:
    """Test Factor VII (Port Binding) analysis."""

    def test_port_binding_factor_exists(self) -> None:
        """Test that port_binding factor can be analyzed."""
        from yolo_developer.agents.architect.twelve_factor import _analyze_factor

        story = {
            "title": "HTTP server",
            "description": "Expose API on port 8080",
        }

        result = _analyze_factor(story, "port_binding")

        assert result.factor_name == "port_binding"


class TestConcurrencyFactorAnalysis:
    """Test Factor VIII (Concurrency) analysis."""

    def test_concurrency_factor_exists(self) -> None:
        """Test that concurrency factor can be analyzed."""
        from yolo_developer.agents.architect.twelve_factor import _analyze_factor

        story = {
            "title": "Scaling",
            "description": "Scale horizontally with multiple instances",
        }

        result = _analyze_factor(story, "concurrency")

        assert result.factor_name == "concurrency"


class TestDisposabilityFactorAnalysis:
    """Test Factor IX (Disposability) analysis."""

    def test_disposability_factor_exists(self) -> None:
        """Test that disposability factor can be analyzed."""
        from yolo_developer.agents.architect.twelve_factor import _analyze_factor

        story = {
            "title": "Graceful shutdown",
            "description": "Handle SIGTERM for graceful shutdown",
        }

        result = _analyze_factor(story, "disposability")

        assert result.factor_name == "disposability"


class TestDevProdParityFactorAnalysis:
    """Test Factor X (Dev/Prod Parity) analysis."""

    def test_dev_prod_parity_factor_exists(self) -> None:
        """Test that dev_prod_parity factor can be analyzed."""
        from yolo_developer.agents.architect.twelve_factor import _analyze_factor

        story = {
            "title": "Environment setup",
            "description": "Use same database in dev and prod",
        }

        result = _analyze_factor(story, "dev_prod_parity")

        assert result.factor_name == "dev_prod_parity"


class TestLogsFactorAnalysis:
    """Test Factor XI (Logs) analysis."""

    def test_logs_factor_exists(self) -> None:
        """Test that logs factor can be analyzed."""
        from yolo_developer.agents.architect.twelve_factor import _analyze_factor

        story = {
            "title": "Logging",
            "description": "Write logs to stdout",
        }

        result = _analyze_factor(story, "logs")

        assert result.factor_name == "logs"


class TestAdminProcessesFactorAnalysis:
    """Test Factor XII (Admin Processes) analysis."""

    def test_admin_processes_factor_exists(self) -> None:
        """Test that admin_processes factor can be analyzed."""
        from yolo_developer.agents.architect.twelve_factor import _analyze_factor

        story = {
            "title": "Database migrations",
            "description": "Run migrations as one-off command",
        }

        result = _analyze_factor(story, "admin_processes")

        assert result.factor_name == "admin_processes"
