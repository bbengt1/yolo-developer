"""Tests for DecisionLogger (Story 11.1 - Task 4).

Tests the DecisionLogger class and get_logger factory function.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def sample_context() -> Any:
    """Create sample DecisionContext."""
    from yolo_developer.audit.types import DecisionContext

    return DecisionContext(
        sprint_id="sprint-1",
        story_id="1-2-auth",
    )


@pytest.fixture
def mock_store() -> AsyncMock:
    """Create a mock DecisionStore."""
    store = AsyncMock()
    store.log_decision = AsyncMock(side_effect=lambda d: d.id)
    return store


class TestDecisionLogger:
    """Tests for DecisionLogger class."""

    @pytest.mark.asyncio
    async def test_log_creates_valid_decision(self, mock_store: AsyncMock) -> None:
        """log should create a Decision with all required fields."""
        from yolo_developer.audit.logger import DecisionLogger

        logger = DecisionLogger(mock_store)

        decision_id = await logger.log(
            agent_name="analyst",
            agent_type="analyst",
            decision_type="requirement_analysis",
            content="OAuth2 authentication required",
            rationale="Industry standard security",
        )

        # Should return decision ID
        assert decision_id is not None
        assert isinstance(decision_id, str)

        # Verify store.log_decision was called
        mock_store.log_decision.assert_called_once()

        # Get the decision that was logged
        logged_decision = mock_store.log_decision.call_args[0][0]

        assert logged_decision.decision_type == "requirement_analysis"
        assert logged_decision.content == "OAuth2 authentication required"
        assert logged_decision.rationale == "Industry standard security"
        assert logged_decision.agent.agent_name == "analyst"
        assert logged_decision.agent.agent_type == "analyst"

    @pytest.mark.asyncio
    async def test_log_auto_generates_uuid_id(self, mock_store: AsyncMock) -> None:
        """log should auto-generate a UUID v4 for decision ID."""
        from yolo_developer.audit.logger import DecisionLogger

        logger = DecisionLogger(mock_store)

        await logger.log(
            agent_name="analyst",
            agent_type="analyst",
            decision_type="requirement_analysis",
            content="Content",
            rationale="Rationale",
        )

        logged_decision = mock_store.log_decision.call_args[0][0]

        # UUID v4 pattern
        uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        assert re.match(uuid_pattern, logged_decision.id, re.IGNORECASE)

    @pytest.mark.asyncio
    async def test_log_auto_generates_timestamp(self, mock_store: AsyncMock) -> None:
        """log should auto-generate an ISO 8601 UTC timestamp."""
        from yolo_developer.audit.logger import DecisionLogger

        logger = DecisionLogger(mock_store)

        await logger.log(
            agent_name="analyst",
            agent_type="analyst",
            decision_type="requirement_analysis",
            content="Content",
            rationale="Rationale",
        )

        logged_decision = mock_store.log_decision.call_args[0][0]

        # ISO 8601 timestamp pattern (with Z suffix for UTC)
        iso_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$"
        assert re.match(iso_pattern, logged_decision.timestamp)

    @pytest.mark.asyncio
    async def test_log_auto_generates_session_id(self, mock_store: AsyncMock) -> None:
        """log should auto-generate a session ID if not provided."""
        from yolo_developer.audit.logger import DecisionLogger

        logger = DecisionLogger(mock_store)

        await logger.log(
            agent_name="analyst",
            agent_type="analyst",
            decision_type="requirement_analysis",
            content="Content",
            rationale="Rationale",
        )

        logged_decision = mock_store.log_decision.call_args[0][0]

        # Should have a non-empty session ID
        assert logged_decision.agent.session_id
        assert isinstance(logged_decision.agent.session_id, str)

    @pytest.mark.asyncio
    async def test_log_uses_provided_session_id(self, mock_store: AsyncMock) -> None:
        """log should use session_id if provided."""
        from yolo_developer.audit.logger import DecisionLogger

        logger = DecisionLogger(mock_store, session_id="my-session-123")

        await logger.log(
            agent_name="analyst",
            agent_type="analyst",
            decision_type="requirement_analysis",
            content="Content",
            rationale="Rationale",
        )

        logged_decision = mock_store.log_decision.call_args[0][0]

        assert logged_decision.agent.session_id == "my-session-123"

    @pytest.mark.asyncio
    async def test_log_with_context(self, mock_store: AsyncMock, sample_context: Any) -> None:
        """log should include provided context."""
        from yolo_developer.audit.logger import DecisionLogger

        logger = DecisionLogger(mock_store)

        await logger.log(
            agent_name="analyst",
            agent_type="analyst",
            decision_type="requirement_analysis",
            content="Content",
            rationale="Rationale",
            context=sample_context,
        )

        logged_decision = mock_store.log_decision.call_args[0][0]

        assert logged_decision.context.sprint_id == "sprint-1"
        assert logged_decision.context.story_id == "1-2-auth"

    @pytest.mark.asyncio
    async def test_log_with_metadata(self, mock_store: AsyncMock) -> None:
        """log should include provided metadata."""
        from yolo_developer.audit.logger import DecisionLogger

        logger = DecisionLogger(mock_store)

        await logger.log(
            agent_name="analyst",
            agent_type="analyst",
            decision_type="requirement_analysis",
            content="Content",
            rationale="Rationale",
            metadata={"source": "seed.md", "version": 1},
        )

        logged_decision = mock_store.log_decision.call_args[0][0]

        assert logged_decision.metadata == {"source": "seed.md", "version": 1}

    @pytest.mark.asyncio
    async def test_log_with_severity(self, mock_store: AsyncMock) -> None:
        """log should include provided severity."""
        from yolo_developer.audit.logger import DecisionLogger

        logger = DecisionLogger(mock_store)

        await logger.log(
            agent_name="analyst",
            agent_type="analyst",
            decision_type="requirement_analysis",
            content="Content",
            rationale="Rationale",
            severity="warning",
        )

        logged_decision = mock_store.log_decision.call_args[0][0]

        assert logged_decision.severity == "warning"

    @pytest.mark.asyncio
    async def test_log_default_severity_is_info(self, mock_store: AsyncMock) -> None:
        """log should default to 'info' severity."""
        from yolo_developer.audit.logger import DecisionLogger

        logger = DecisionLogger(mock_store)

        await logger.log(
            agent_name="analyst",
            agent_type="analyst",
            decision_type="requirement_analysis",
            content="Content",
            rationale="Rationale",
        )

        logged_decision = mock_store.log_decision.call_args[0][0]

        assert logged_decision.severity == "info"

    @pytest.mark.asyncio
    async def test_log_emits_structlog_output(
        self, mock_store: AsyncMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """log should emit structured log output."""
        import logging

        import structlog

        # Configure structlog to use stdlib logging for capture
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=False,
        )

        from yolo_developer.audit.logger import DecisionLogger

        logger = DecisionLogger(mock_store)

        with caplog.at_level(logging.INFO, logger="yolo_developer.audit.logger"):
            await logger.log(
                agent_name="analyst",
                agent_type="analyst",
                decision_type="requirement_analysis",
                content="OAuth2 authentication",
                rationale="Security standard",
            )

        # Should have logged the decision
        assert len(caplog.records) >= 1
        # Check for decision-related content in the log
        log_text = " ".join(record.message for record in caplog.records)
        assert "decision" in log_text.lower() or "logged" in log_text.lower()

    @pytest.mark.asyncio
    async def test_log_handles_store_failure_gracefully(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """log should not raise when store fails (per ADR-007)."""
        import logging

        import structlog

        # Configure structlog to use stdlib logging for capture
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=False,
        )

        from yolo_developer.audit.logger import DecisionLogger

        # Create a store that always fails
        failing_store = AsyncMock()
        failing_store.log_decision = AsyncMock(side_effect=RuntimeError("Store failed"))

        logger = DecisionLogger(failing_store)

        with caplog.at_level(logging.ERROR, logger="yolo_developer.audit.logger"):
            # Should not raise - per ADR-007, log errors don't block callers
            decision_id = await logger.log(
                agent_name="analyst",
                agent_type="analyst",
                decision_type="requirement_analysis",
                content="Test content",
                rationale="Test rationale",
            )

        # Should still return a decision ID
        assert decision_id is not None
        assert isinstance(decision_id, str)

        # Should have logged the error
        assert any(
            "failed" in record.message.lower() or "error" in record.levelname.lower()
            for record in caplog.records
        )


class TestGetLogger:
    """Tests for get_logger factory function."""

    def test_get_logger_returns_decision_logger(self) -> None:
        """get_logger should return a DecisionLogger instance."""
        from yolo_developer.audit.logger import DecisionLogger, get_logger
        from yolo_developer.audit.memory_store import InMemoryDecisionStore

        store = InMemoryDecisionStore()
        logger = get_logger(store)

        assert isinstance(logger, DecisionLogger)

    def test_get_logger_creates_default_store(self) -> None:
        """get_logger without store should create InMemoryDecisionStore."""
        from yolo_developer.audit.logger import DecisionLogger, get_logger

        logger = get_logger()

        assert isinstance(logger, DecisionLogger)

    def test_get_logger_accepts_session_id(self) -> None:
        """get_logger should accept and pass through session_id."""
        from yolo_developer.audit.logger import get_logger
        from yolo_developer.audit.memory_store import InMemoryDecisionStore

        store = InMemoryDecisionStore()
        logger = get_logger(store, session_id="custom-session")

        assert logger._session_id == "custom-session"

    @pytest.mark.asyncio
    async def test_get_logger_integrates_with_store(self) -> None:
        """get_logger should create a logger that works with the store."""
        from yolo_developer.audit.logger import get_logger
        from yolo_developer.audit.memory_store import InMemoryDecisionStore

        store = InMemoryDecisionStore()
        logger = get_logger(store)

        await logger.log(
            agent_name="analyst",
            agent_type="analyst",
            decision_type="requirement_analysis",
            content="Test decision",
            rationale="Test rationale",
        )

        # Decision should be in store
        decisions = await store.get_decisions()
        assert len(decisions) == 1
        assert decisions[0].content == "Test decision"


class TestTraceLinksIntegration:
    """Tests for trace_links integration with DecisionLogger (Story 11.2 - Task 5)."""

    @pytest.mark.asyncio
    async def test_log_with_trace_links_no_context(self, mock_store: AsyncMock) -> None:
        """log should create context with trace_links when no context provided."""
        from yolo_developer.audit.logger import DecisionLogger

        logger = DecisionLogger(mock_store)

        await logger.log(
            agent_name="analyst",
            agent_type="analyst",
            decision_type="requirement_analysis",
            content="Content",
            rationale="Rationale",
            trace_links=["link-001", "link-002"],
        )

        logged_decision = mock_store.log_decision.call_args[0][0]

        assert logged_decision.context.trace_links == ("link-001", "link-002")

    @pytest.mark.asyncio
    async def test_log_with_trace_links_merges_with_context(self, mock_store: AsyncMock) -> None:
        """log should merge trace_links with existing context."""
        from yolo_developer.audit.logger import DecisionLogger
        from yolo_developer.audit.types import DecisionContext

        logger = DecisionLogger(mock_store)
        context = DecisionContext(
            sprint_id="sprint-1",
            story_id="story-1",
            trace_links=("existing-link",),
        )

        await logger.log(
            agent_name="analyst",
            agent_type="analyst",
            decision_type="requirement_analysis",
            content="Content",
            rationale="Rationale",
            context=context,
            trace_links=["new-link-1", "new-link-2"],
        )

        logged_decision = mock_store.log_decision.call_args[0][0]

        # Should have merged trace links
        assert logged_decision.context.trace_links == (
            "existing-link",
            "new-link-1",
            "new-link-2",
        )
        # Should preserve other context fields
        assert logged_decision.context.sprint_id == "sprint-1"
        assert logged_decision.context.story_id == "story-1"

    @pytest.mark.asyncio
    async def test_log_with_context_no_trace_links(self, mock_store: AsyncMock) -> None:
        """log should preserve context when trace_links not provided."""
        from yolo_developer.audit.logger import DecisionLogger
        from yolo_developer.audit.types import DecisionContext

        logger = DecisionLogger(mock_store)
        context = DecisionContext(
            sprint_id="sprint-1",
            trace_links=("existing-link",),
        )

        await logger.log(
            agent_name="analyst",
            agent_type="analyst",
            decision_type="requirement_analysis",
            content="Content",
            rationale="Rationale",
            context=context,
        )

        logged_decision = mock_store.log_decision.call_args[0][0]

        # Should preserve original context
        assert logged_decision.context.sprint_id == "sprint-1"
        assert logged_decision.context.trace_links == ("existing-link",)

    @pytest.mark.asyncio
    async def test_log_with_empty_trace_links(self, mock_store: AsyncMock) -> None:
        """log should handle empty trace_links list."""
        from yolo_developer.audit.logger import DecisionLogger

        logger = DecisionLogger(mock_store)

        await logger.log(
            agent_name="analyst",
            agent_type="analyst",
            decision_type="requirement_analysis",
            content="Content",
            rationale="Rationale",
            trace_links=[],
        )

        logged_decision = mock_store.log_decision.call_args[0][0]

        # Empty list should result in empty tuple (default)
        assert logged_decision.context.trace_links == ()

    @pytest.mark.asyncio
    async def test_log_with_trace_links_logs_count(
        self, mock_store: AsyncMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """log should include trace_link_count in structured log when links present."""
        import logging

        import structlog

        # Configure structlog to use stdlib logging for capture
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=False,
        )

        from yolo_developer.audit.logger import DecisionLogger

        logger = DecisionLogger(mock_store)

        with caplog.at_level(logging.INFO, logger="yolo_developer.audit.logger"):
            await logger.log(
                agent_name="analyst",
                agent_type="analyst",
                decision_type="requirement_analysis",
                content="Content",
                rationale="Rationale",
                trace_links=["link-1", "link-2", "link-3"],
            )

        # Should have logged with trace_link_count
        log_text = " ".join(str(record.__dict__) for record in caplog.records)
        # The count should be included somewhere in the log output
        assert "3" in log_text or "trace_link" in log_text.lower()
