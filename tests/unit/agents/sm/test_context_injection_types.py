"""Tests for context injection types (Story 10.13).

Tests the context injection data types:
- ContextGap: Detected context gap requiring injection
- RetrievedContext: Context retrieved from a source
- InjectionResult: Result of context injection operation
- InjectionConfig: Configuration for context injection
- Constants and literal types

References:
    - FR69: SM Agent can inject context when agents lack information
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

import logging

import pytest

from yolo_developer.agents.sm.context_injection_types import (
    DEFAULT_LOG_INJECTIONS,
    DEFAULT_MAX_CONTEXT_ITEMS,
    DEFAULT_MAX_CONTEXT_SIZE_BYTES,
    DEFAULT_MIN_RELEVANCE_SCORE,
    LONG_CYCLE_TIME_MULTIPLIER,
    MAX_CONFIDENCE,
    MAX_RELEVANCE,
    MIN_CONFIDENCE,
    MIN_RELEVANCE,
    VALID_CONTEXT_SOURCES,
    VALID_GAP_REASONS,
    ContextGap,
    InjectionConfig,
    InjectionResult,
    RetrievedContext,
)


class TestContextGap:
    """Tests for ContextGap dataclass."""

    def test_create_with_required_fields(self) -> None:
        """Test creating ContextGap with required fields."""
        gap = ContextGap(
            gap_id="gap-12345",
            agent="architect",
            reason="clarification_requested",
            context_query="authentication requirements",
            confidence=0.9,
            indicators=("clarification_message_detected",),
        )
        assert gap.gap_id == "gap-12345"
        assert gap.agent == "architect"
        assert gap.reason == "clarification_requested"
        assert gap.context_query == "authentication requirements"
        assert gap.confidence == 0.9
        assert gap.indicators == ("clarification_message_detected",)
        assert gap.detected_at is not None

    def test_immutable(self) -> None:
        """Test that ContextGap is frozen/immutable."""
        gap = ContextGap(
            gap_id="gap-12345",
            agent="architect",
            reason="clarification_requested",
            context_query="auth requirements",
            confidence=0.9,
            indicators=(),
        )
        with pytest.raises(AttributeError):
            gap.confidence = 0.5  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        gap = ContextGap(
            gap_id="gap-54321",
            agent="pm",
            reason="circular_logic",
            context_query="story prioritization",
            confidence=0.8,
            indicators=("cycle_detected", "topic_repeat"),
        )
        d = gap.to_dict()
        assert d["gap_id"] == "gap-54321"
        assert d["agent"] == "pm"
        assert d["reason"] == "circular_logic"
        assert d["context_query"] == "story prioritization"
        assert d["confidence"] == 0.8
        assert d["indicators"] == ["cycle_detected", "topic_repeat"]
        assert "detected_at" in d

    def test_post_init_warning_empty_gap_id(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that empty gap_id logs a warning."""
        with caplog.at_level(logging.WARNING):
            ContextGap(
                gap_id="",
                agent="architect",
                reason="clarification_requested",
                context_query="auth",
                confidence=0.9,
                indicators=(),
            )
        assert "gap_id is empty" in caplog.text

    def test_post_init_warning_empty_agent(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that empty agent logs a warning."""
        with caplog.at_level(logging.WARNING):
            ContextGap(
                gap_id="gap-123",
                agent="",
                reason="clarification_requested",
                context_query="auth",
                confidence=0.9,
                indicators=(),
            )
        assert "agent is empty" in caplog.text

    def test_post_init_warning_invalid_reason(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that invalid reason logs a warning."""
        with caplog.at_level(logging.WARNING):
            ContextGap(
                gap_id="gap-123",
                agent="architect",
                reason="invalid_reason",  # type: ignore[arg-type]
                context_query="auth",
                confidence=0.9,
                indicators=(),
            )
        assert "is not a valid reason value" in caplog.text

    def test_post_init_warning_invalid_confidence(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that invalid confidence logs a warning."""
        with caplog.at_level(logging.WARNING):
            ContextGap(
                gap_id="gap-123",
                agent="architect",
                reason="clarification_requested",
                context_query="auth",
                confidence=1.5,
                indicators=(),
            )
        assert "is outside valid range" in caplog.text

    def test_post_init_warning_empty_context_query(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that empty context_query logs a warning."""
        with caplog.at_level(logging.WARNING):
            ContextGap(
                gap_id="gap-123",
                agent="architect",
                reason="clarification_requested",
                context_query="",
                confidence=0.9,
                indicators=(),
            )
        assert "context_query is empty" in caplog.text


class TestRetrievedContext:
    """Tests for RetrievedContext dataclass."""

    def test_create_with_required_fields(self) -> None:
        """Test creating RetrievedContext with required fields."""
        context = RetrievedContext(
            source="memory",
            content="OAuth2 is required for authentication",
            relevance_score=0.95,
        )
        assert context.source == "memory"
        assert context.content == "OAuth2 is required for authentication"
        assert context.relevance_score == 0.95
        assert context.metadata == {}
        assert context.retrieved_at is not None

    def test_create_with_metadata(self) -> None:
        """Test creating RetrievedContext with metadata."""
        context = RetrievedContext(
            source="state",
            content="Previous decision about auth",
            relevance_score=0.85,
            metadata={"type": "decision", "agent": "architect"},
        )
        assert context.metadata == {"type": "decision", "agent": "architect"}

    def test_immutable(self) -> None:
        """Test that RetrievedContext is frozen/immutable."""
        context = RetrievedContext(
            source="memory",
            content="Some context",
            relevance_score=0.9,
        )
        with pytest.raises(AttributeError):
            context.relevance_score = 0.5  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        context = RetrievedContext(
            source="sprint",
            content="Current sprint goal is X",
            relevance_score=0.75,
            metadata={"sprint_id": "sprint-123"},
        )
        d = context.to_dict()
        assert d["source"] == "sprint"
        assert d["content"] == "Current sprint goal is X"
        assert d["relevance_score"] == 0.75
        assert d["metadata"] == {"sprint_id": "sprint-123"}
        assert "retrieved_at" in d

    def test_post_init_warning_invalid_source(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that invalid source logs a warning."""
        with caplog.at_level(logging.WARNING):
            RetrievedContext(
                source="invalid_source",  # type: ignore[arg-type]
                content="Some content",
                relevance_score=0.9,
            )
        assert "is not a valid source value" in caplog.text

    def test_post_init_warning_empty_content(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that empty content logs a warning."""
        with caplog.at_level(logging.WARNING):
            RetrievedContext(
                source="memory",
                content="",
                relevance_score=0.9,
            )
        assert "content is empty" in caplog.text

    def test_post_init_warning_invalid_relevance(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that invalid relevance_score logs a warning."""
        with caplog.at_level(logging.WARNING):
            RetrievedContext(
                source="memory",
                content="Some content",
                relevance_score=1.5,
            )
        assert "is outside valid range" in caplog.text


class TestInjectionResult:
    """Tests for InjectionResult dataclass."""

    @pytest.fixture
    def sample_gap(self) -> ContextGap:
        """Create a sample ContextGap for testing."""
        return ContextGap(
            gap_id="gap-test",
            agent="architect",
            reason="clarification_requested",
            context_query="auth requirements",
            confidence=0.9,
            indicators=("test_indicator",),
        )

    @pytest.fixture
    def sample_context(self) -> RetrievedContext:
        """Create a sample RetrievedContext for testing."""
        return RetrievedContext(
            source="memory",
            content="OAuth2 required",
            relevance_score=0.95,
        )

    def test_create_successful_injection(
        self, sample_gap: ContextGap, sample_context: RetrievedContext
    ) -> None:
        """Test creating InjectionResult for successful injection."""
        result = InjectionResult(
            gap=sample_gap,
            contexts_retrieved=(sample_context,),
            injected=True,
            injection_target="injected_context",
            total_context_size=1024,
            duration_ms=50.0,
        )
        assert result.gap == sample_gap
        assert result.contexts_retrieved == (sample_context,)
        assert result.injected is True
        assert result.injection_target == "injected_context"
        assert result.total_context_size == 1024
        assert result.duration_ms == 50.0

    def test_create_no_injection(self, sample_gap: ContextGap) -> None:
        """Test creating InjectionResult when no injection happened."""
        result = InjectionResult(
            gap=sample_gap,
            contexts_retrieved=(),
            injected=False,
            injection_target="",
            total_context_size=0,
            duration_ms=10.0,
        )
        assert result.injected is False
        assert result.contexts_retrieved == ()

    def test_immutable(self, sample_gap: ContextGap, sample_context: RetrievedContext) -> None:
        """Test that InjectionResult is frozen/immutable."""
        result = InjectionResult(
            gap=sample_gap,
            contexts_retrieved=(sample_context,),
            injected=True,
            injection_target="injected_context",
            total_context_size=1024,
            duration_ms=50.0,
        )
        with pytest.raises(AttributeError):
            result.injected = False  # type: ignore[misc]

    def test_to_dict(self, sample_gap: ContextGap, sample_context: RetrievedContext) -> None:
        """Test serialization to dictionary."""
        result = InjectionResult(
            gap=sample_gap,
            contexts_retrieved=(sample_context,),
            injected=True,
            injection_target="injected_context",
            total_context_size=1024,
            duration_ms=50.0,
        )
        d = result.to_dict()
        assert d["injected"] is True
        assert d["injection_target"] == "injected_context"
        assert d["total_context_size"] == 1024
        assert d["duration_ms"] == 50.0
        assert "gap" in d
        assert d["gap"]["gap_id"] == "gap-test"
        assert len(d["contexts_retrieved"]) == 1
        assert d["contexts_retrieved"][0]["source"] == "memory"

    def test_post_init_warning_negative_size(
        self, sample_gap: ContextGap, sample_context: RetrievedContext, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that negative total_context_size logs a warning."""
        with caplog.at_level(logging.WARNING):
            InjectionResult(
                gap=sample_gap,
                contexts_retrieved=(sample_context,),
                injected=True,
                injection_target="injected_context",
                total_context_size=-100,
                duration_ms=50.0,
            )
        assert "total_context_size=-100 is negative" in caplog.text

    def test_post_init_warning_negative_duration(
        self, sample_gap: ContextGap, sample_context: RetrievedContext, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that negative duration_ms logs a warning."""
        with caplog.at_level(logging.WARNING):
            InjectionResult(
                gap=sample_gap,
                contexts_retrieved=(sample_context,),
                injected=True,
                injection_target="injected_context",
                total_context_size=1024,
                duration_ms=-10.0,
            )
        assert "duration_ms=-10.00 is negative" in caplog.text

    def test_post_init_warning_injected_empty_target(
        self, sample_gap: ContextGap, sample_context: RetrievedContext, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that injected=True with empty target logs a warning."""
        with caplog.at_level(logging.WARNING):
            InjectionResult(
                gap=sample_gap,
                contexts_retrieved=(sample_context,),
                injected=True,
                injection_target="",
                total_context_size=1024,
                duration_ms=50.0,
            )
        assert "injected=True but injection_target is empty" in caplog.text

    def test_post_init_warning_injected_no_contexts(
        self, sample_gap: ContextGap, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that injected=True with no contexts logs a warning."""
        with caplog.at_level(logging.WARNING):
            InjectionResult(
                gap=sample_gap,
                contexts_retrieved=(),
                injected=True,
                injection_target="injected_context",
                total_context_size=0,
                duration_ms=50.0,
            )
        assert "injected=True but contexts_retrieved is empty" in caplog.text


class TestInjectionConfig:
    """Tests for InjectionConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default values match constants."""
        config = InjectionConfig()
        assert config.max_context_items == DEFAULT_MAX_CONTEXT_ITEMS
        assert config.min_relevance_score == DEFAULT_MIN_RELEVANCE_SCORE
        assert config.max_context_size_bytes == DEFAULT_MAX_CONTEXT_SIZE_BYTES
        assert config.enabled_sources == ("memory", "state")
        assert config.log_injections == DEFAULT_LOG_INJECTIONS

    def test_custom_values(self) -> None:
        """Test creating with custom values."""
        config = InjectionConfig(
            max_context_items=10,
            min_relevance_score=0.8,
            max_context_size_bytes=50_000,
            enabled_sources=("memory", "state", "sprint"),
            log_injections=False,
        )
        assert config.max_context_items == 10
        assert config.min_relevance_score == 0.8
        assert config.max_context_size_bytes == 50_000
        assert config.enabled_sources == ("memory", "state", "sprint")
        assert config.log_injections is False

    def test_immutable(self) -> None:
        """Test that InjectionConfig is frozen/immutable."""
        config = InjectionConfig()
        with pytest.raises(AttributeError):
            config.max_context_items = 20  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        config = InjectionConfig(max_context_items=7, enabled_sources=("memory",))
        d = config.to_dict()
        assert d["max_context_items"] == 7
        assert d["enabled_sources"] == ["memory"]
        assert "min_relevance_score" in d
        assert "max_context_size_bytes" in d
        assert "log_injections" in d

    def test_post_init_warning_invalid_max_items(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that invalid max_context_items logs a warning."""
        with caplog.at_level(logging.WARNING):
            InjectionConfig(max_context_items=0)
        assert "max_context_items=0 should be at least 1" in caplog.text

    def test_post_init_warning_invalid_relevance(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that invalid min_relevance_score logs a warning."""
        with caplog.at_level(logging.WARNING):
            InjectionConfig(min_relevance_score=1.5)
        assert "min_relevance_score=1.500 should be between" in caplog.text

    def test_post_init_warning_negative_relevance(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that negative min_relevance_score logs a warning."""
        with caplog.at_level(logging.WARNING):
            InjectionConfig(min_relevance_score=-0.1)
        assert "min_relevance_score=-0.100 should be between" in caplog.text

    def test_post_init_warning_invalid_max_size(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that invalid max_context_size_bytes logs a warning."""
        with caplog.at_level(logging.WARNING):
            InjectionConfig(max_context_size_bytes=0)
        assert "max_context_size_bytes=0 should be at least 1" in caplog.text

    def test_post_init_warning_invalid_source(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that invalid source in enabled_sources logs a warning."""
        with caplog.at_level(logging.WARNING):
            InjectionConfig(enabled_sources=("memory", "invalid"))  # type: ignore[arg-type]
        assert "contains invalid source='invalid'" in caplog.text


class TestConstants:
    """Tests for module constants."""

    def test_valid_context_sources(self) -> None:
        """Test VALID_CONTEXT_SOURCES contains expected values."""
        assert VALID_CONTEXT_SOURCES == frozenset({"memory", "state", "sprint", "architecture"})
        assert "memory" in VALID_CONTEXT_SOURCES
        assert "state" in VALID_CONTEXT_SOURCES
        assert "sprint" in VALID_CONTEXT_SOURCES
        assert "architecture" in VALID_CONTEXT_SOURCES

    def test_valid_gap_reasons(self) -> None:
        """Test VALID_GAP_REASONS contains expected values."""
        assert VALID_GAP_REASONS == frozenset({
            "clarification_requested",
            "circular_logic",
            "long_cycle_time",
            "gate_failure",
            "explicit_flag",
        })

    def test_default_max_context_items(self) -> None:
        """Test DEFAULT_MAX_CONTEXT_ITEMS is reasonable."""
        assert DEFAULT_MAX_CONTEXT_ITEMS >= 1
        assert DEFAULT_MAX_CONTEXT_ITEMS <= 20

    def test_default_min_relevance_score(self) -> None:
        """Test DEFAULT_MIN_RELEVANCE_SCORE is valid."""
        assert MIN_RELEVANCE <= DEFAULT_MIN_RELEVANCE_SCORE <= MAX_RELEVANCE

    def test_default_max_context_size(self) -> None:
        """Test DEFAULT_MAX_CONTEXT_SIZE_BYTES is reasonable."""
        assert DEFAULT_MAX_CONTEXT_SIZE_BYTES >= 1000  # At least 1KB
        assert DEFAULT_MAX_CONTEXT_SIZE_BYTES <= 10_000_000  # At most 10MB

    def test_confidence_bounds(self) -> None:
        """Test confidence bound constants."""
        assert MIN_CONFIDENCE == 0.0
        assert MAX_CONFIDENCE == 1.0
        assert MIN_CONFIDENCE < MAX_CONFIDENCE

    def test_relevance_bounds(self) -> None:
        """Test relevance bound constants."""
        assert MIN_RELEVANCE == 0.0
        assert MAX_RELEVANCE == 1.0
        assert MIN_RELEVANCE < MAX_RELEVANCE

    def test_long_cycle_time_multiplier(self) -> None:
        """Test LONG_CYCLE_TIME_MULTIPLIER is positive."""
        assert LONG_CYCLE_TIME_MULTIPLIER > 1.0
