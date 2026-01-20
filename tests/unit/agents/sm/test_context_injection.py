"""Tests for context injection functions (Story 10.13).

Tests the context injection functionality:
- detect_context_gap: Gap detection with various indicators
- retrieve_relevant_context: Context retrieval from sources
- inject_context: Context injection into state
- manage_context_injection: End-to-end flow

References:
    - FR69: SM Agent can inject context when agents lack information
    - ADR-001: TypedDict for graph state, frozen dataclasses for internal types
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from yolo_developer.agents.sm.context_injection import (
    CLARIFICATION_KEYWORDS,
    DEFAULT_INJECTION_TARGET,
    _build_context_query,
    _build_injection_payload,
    _calculate_relevance_score,
    _check_circular_logic_gap,
    _check_clarification_requested,
    _check_explicit_flag,
    _check_gate_failure,
    _check_long_cycle_time,
    _retrieve_from_sprint,
    _retrieve_from_state,
    detect_context_gap,
    inject_context,
    manage_context_injection,
    retrieve_relevant_context,
)
from yolo_developer.agents.sm.context_injection_types import (
    ContextGap,
    InjectionConfig,
    RetrievedContext,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def empty_state() -> dict[str, Any]:
    """Create an empty state for testing."""
    return {
        "messages": [],
        "current_agent": "analyst",
        "handoff_context": None,
        "decisions": [],
    }


@pytest.fixture
def state_with_messages() -> dict[str, Any]:
    """Create a state with some messages."""
    messages = [
        AIMessage(
            content="Starting analysis",
            additional_kwargs={"agent": "analyst"},
        ),
        AIMessage(
            content="Requirements look good",
            additional_kwargs={"agent": "pm"},
        ),
    ]
    return {
        "messages": messages,
        "current_agent": "pm",
        "handoff_context": None,
        "decisions": [],
    }


@pytest.fixture
def sample_gap() -> ContextGap:
    """Create a sample context gap for testing."""
    return ContextGap(
        gap_id="gap-test-123",
        agent="architect",
        reason="clarification_requested",
        context_query="authentication requirements OAuth2",
        confidence=0.9,
        indicators=("clarification_message_detected",),
    )


@pytest.fixture
def sample_context() -> RetrievedContext:
    """Create a sample retrieved context for testing."""
    return RetrievedContext(
        source="memory",
        content="OAuth2 is required for authentication per ADR-003",
        relevance_score=0.95,
        metadata={"type": "decision", "agent": "architect"},
    )


@pytest.fixture
def sample_config() -> InjectionConfig:
    """Create a sample injection config for testing."""
    return InjectionConfig(
        max_context_items=5,
        min_relevance_score=0.7,
        max_context_size_bytes=50_000,
        enabled_sources=("memory", "state"),
        log_injections=False,
    )


# =============================================================================
# Tests for Helper Functions
# =============================================================================


class TestCheckClarificationRequested:
    """Tests for _check_clarification_requested helper."""

    def test_empty_messages_returns_false(self, empty_state: dict[str, Any]) -> None:
        """Test returns False when no messages present."""
        result = _check_clarification_requested(empty_state)
        assert result is False

    def test_no_clarification_returns_false(self) -> None:
        """Test returns False when no clarification indicators."""
        state = {
            "messages": [
                AIMessage(content="All good", additional_kwargs={"agent": "analyst"}),
            ],
            "current_agent": "pm",
        }
        result = _check_clarification_requested(state)
        assert result is False

    def test_multiple_question_marks_returns_true(self) -> None:
        """Test returns True when multiple question marks detected."""
        state = {
            "messages": [
                AIMessage(
                    content="What do you mean? Is this correct?",
                    additional_kwargs={"agent": "analyst"},
                ),
            ],
            "current_agent": "pm",
        }
        result = _check_clarification_requested(state)
        assert result is True

    @pytest.mark.parametrize("keyword", list(CLARIFICATION_KEYWORDS)[:5])
    def test_clarification_keywords_return_true(self, keyword: str) -> None:
        """Test returns True when clarification keywords are present."""
        state = {
            "messages": [
                AIMessage(
                    content=f"I {keyword} the requirements",
                    additional_kwargs={"agent": "analyst"},
                ),
            ],
            "current_agent": "pm",
        }
        result = _check_clarification_requested(state)
        assert result is True


class TestCheckCircularLogicGap:
    """Tests for _check_circular_logic_gap helper."""

    def test_no_cycle_analysis_returns_false(self, empty_state: dict[str, Any]) -> None:
        """Test returns False when no cycle_analysis in state."""
        result = _check_circular_logic_gap(empty_state)
        assert result is False

    def test_circular_detected_dict_returns_true(self) -> None:
        """Test returns True when cycle_analysis dict shows circular."""
        state = {
            "cycle_analysis": {"circular_detected": True},
            "current_agent": "pm",
        }
        result = _check_circular_logic_gap(state)
        assert result is True

    def test_no_circular_detected_returns_false(self) -> None:
        """Test returns False when circular_detected is False."""
        state = {
            "cycle_analysis": {"circular_detected": False},
            "current_agent": "pm",
        }
        result = _check_circular_logic_gap(state)
        assert result is False


class TestCheckLongCycleTime:
    """Tests for _check_long_cycle_time helper."""

    def test_no_health_status_returns_false(self, empty_state: dict[str, Any]) -> None:
        """Test returns False when no health_status in state."""
        result = _check_long_cycle_time(empty_state)
        assert result is False

    def test_long_cycle_time_returns_true(self) -> None:
        """Test returns True when cycle time exceeds 2x baseline."""
        state = {
            "current_agent": "architect",
            "health_status": {
                "metrics": {
                    "agent_cycle_times": {"architect": 10000.0},
                    "overall_cycle_time": 3000.0,
                }
            },
        }
        result = _check_long_cycle_time(state)
        assert result is True

    def test_normal_cycle_time_returns_false(self) -> None:
        """Test returns False when cycle time is within threshold."""
        state = {
            "current_agent": "architect",
            "health_status": {
                "metrics": {
                    "agent_cycle_times": {"architect": 4000.0},
                    "overall_cycle_time": 3000.0,
                }
            },
        }
        result = _check_long_cycle_time(state)
        assert result is False


class TestCheckGateFailure:
    """Tests for _check_gate_failure helper."""

    def test_no_gate_blocked_returns_false(self, empty_state: dict[str, Any]) -> None:
        """Test returns False when gate_blocked is False/missing."""
        result = _check_gate_failure(empty_state)
        assert result is False

    def test_gate_blocked_with_context_reason_returns_true(self) -> None:
        """Test returns True when gate failure mentions missing context."""
        state = {
            "gate_blocked": True,
            "gate_failure": "Missing context for requirement validation",
            "current_agent": "analyst",
        }
        result = _check_gate_failure(state)
        assert result is True

    def test_gate_blocked_unrelated_reason_returns_false(self) -> None:
        """Test returns False when gate failure doesn't mention context."""
        state = {
            "gate_blocked": True,
            "gate_failure": "Test coverage below threshold",
            "current_agent": "tea",
        }
        result = _check_gate_failure(state)
        assert result is False


class TestCheckExplicitFlag:
    """Tests for _check_explicit_flag helper."""

    def test_no_flag_returns_false(self, empty_state: dict[str, Any]) -> None:
        """Test returns False when no explicit flag present."""
        result = _check_explicit_flag(empty_state)
        assert result is False

    def test_missing_context_flag_returns_true(self) -> None:
        """Test returns True when missing_context flag is set."""
        state = {
            "missing_context": True,
            "current_agent": "architect",
        }
        result = _check_explicit_flag(state)
        assert result is True

    def test_handoff_with_required_context_returns_true(self) -> None:
        """Test returns True when handoff_context has required_context."""
        state = {
            "handoff_context": {
                "required_context": ["auth_decisions", "api_design"],
            },
            "current_agent": "dev",
        }
        result = _check_explicit_flag(state)
        assert result is True


class TestBuildContextQuery:
    """Tests for _build_context_query helper."""

    def test_builds_query_with_agent_and_reason(self, empty_state: dict[str, Any]) -> None:
        """Test builds query including agent and reason context."""
        query = _build_context_query(empty_state, "clarification_requested")
        assert "analyst" in query.lower()
        assert "clarification" in query.lower()

    def test_includes_recent_message_content(self) -> None:
        """Test includes content from recent messages."""
        state = {
            "messages": [
                AIMessage(
                    content="What is the OAuth2 authentication flow for this API?",
                    additional_kwargs={"agent": "dev"},
                ),
            ],
            "current_agent": "dev",
        }
        query = _build_context_query(state, "clarification_requested")
        assert "OAuth2" in query or "oauth2" in query.lower()


class TestCalculateRelevanceScore:
    """Tests for _calculate_relevance_score helper."""

    def test_empty_inputs_return_zero(self) -> None:
        """Test returns 0 for empty inputs."""
        assert _calculate_relevance_score("", "content") == 0.0
        assert _calculate_relevance_score("query", "") == 0.0

    def test_identical_content_high_score(self) -> None:
        """Test high score for identical/similar content."""
        score = _calculate_relevance_score(
            "OAuth authentication requirements",
            "OAuth authentication requirements specification",
        )
        assert score > 0.5

    def test_unrelated_content_low_score(self) -> None:
        """Test low score for unrelated content."""
        score = _calculate_relevance_score(
            "database schema design",
            "frontend button styling guide",
        )
        assert score < 0.3


# =============================================================================
# Tests for detect_context_gap
# =============================================================================


class TestDetectContextGap:
    """Tests for detect_context_gap function."""

    @pytest.mark.asyncio
    async def test_empty_state_no_gap(self, empty_state: dict[str, Any]) -> None:
        """Test returns None for empty state with no indicators."""
        gap = await detect_context_gap(empty_state)
        assert gap is None

    @pytest.mark.asyncio
    async def test_clarification_requested_gap(self) -> None:
        """Test detects gap when clarification requested."""
        state = {
            "messages": [
                AIMessage(
                    content="I'm unclear about the requirements. Can you clarify?",
                    additional_kwargs={"agent": "analyst"},
                ),
            ],
            "current_agent": "analyst",
        }
        gap = await detect_context_gap(state)
        assert gap is not None
        assert gap.reason == "clarification_requested"
        assert gap.confidence == 0.9

    @pytest.mark.asyncio
    async def test_circular_logic_gap(self) -> None:
        """Test detects gap when circular logic detected."""
        state = {
            "messages": [],
            "current_agent": "pm",
            "cycle_analysis": {"circular_detected": True},
        }
        gap = await detect_context_gap(state)
        assert gap is not None
        assert gap.reason == "circular_logic"
        assert gap.confidence == 0.8

    @pytest.mark.asyncio
    async def test_gate_failure_gap(self) -> None:
        """Test detects gap when gate fails due to missing context."""
        state = {
            "messages": [],
            "current_agent": "analyst",
            "gate_blocked": True,
            "gate_failure": "Missing context information",
        }
        gap = await detect_context_gap(state)
        assert gap is not None
        assert gap.reason == "gate_failure"
        assert gap.confidence == 0.85

    @pytest.mark.asyncio
    async def test_explicit_flag_gap(self) -> None:
        """Test detects gap when explicit flag is set."""
        state = {
            "messages": [],
            "current_agent": "dev",
            "missing_context": True,
        }
        gap = await detect_context_gap(state)
        assert gap is not None
        assert gap.reason == "explicit_flag"
        assert gap.confidence == 0.95

    @pytest.mark.asyncio
    async def test_long_cycle_time_gap(self) -> None:
        """Test detects gap when cycle time exceeds baseline."""
        state = {
            "messages": [],
            "current_agent": "architect",
            "health_status": {
                "metrics": {
                    "agent_cycle_times": {"architect": 15000.0},
                    "overall_cycle_time": 3000.0,
                }
            },
        }
        gap = await detect_context_gap(state)
        assert gap is not None
        assert gap.reason == "long_cycle_time"
        assert gap.confidence == 0.6

    @pytest.mark.asyncio
    async def test_gap_id_is_unique(self, empty_state: dict[str, Any]) -> None:
        """Test that gap IDs are unique."""
        empty_state["missing_context"] = True

        gap1 = await detect_context_gap(empty_state)
        gap2 = await detect_context_gap(empty_state)

        assert gap1 is not None
        assert gap2 is not None
        assert gap1.gap_id != gap2.gap_id


# =============================================================================
# Tests for Context Retrieval
# =============================================================================


class TestRetrieveFromState:
    """Tests for _retrieve_from_state function."""

    def test_empty_state_no_contexts(
        self, empty_state: dict[str, Any], sample_config: InjectionConfig
    ) -> None:
        """Test returns empty list for empty state."""
        contexts = _retrieve_from_state("test query", empty_state, sample_config)
        assert contexts == []

    def test_retrieves_relevant_messages(self, sample_config: InjectionConfig) -> None:
        """Test retrieves messages matching query."""
        state = {
            "messages": [
                AIMessage(
                    content="OAuth2 authentication is required for the API endpoints",
                    additional_kwargs={"agent": "architect"},
                ),
                AIMessage(
                    content="CSS styling for buttons",
                    additional_kwargs={"agent": "dev"},
                ),
            ],
            "current_agent": "dev",
            "decisions": [],
        }
        # Use lower threshold to ensure match
        config = InjectionConfig(min_relevance_score=0.1)
        contexts = _retrieve_from_state("OAuth authentication API", state, config)

        # Should find at least the OAuth message
        assert len(contexts) >= 1
        oauth_contexts = [c for c in contexts if "OAuth" in c.content]
        assert len(oauth_contexts) >= 1


class TestRetrieveFromSprint:
    """Tests for _retrieve_from_sprint function."""

    def test_empty_sprint_data_no_contexts(
        self, empty_state: dict[str, Any], sample_config: InjectionConfig
    ) -> None:
        """Test returns empty list when no sprint data."""
        contexts = _retrieve_from_sprint("test query", empty_state, sample_config)
        assert contexts == []

    def test_retrieves_from_sprint_plan(self, sample_config: InjectionConfig) -> None:
        """Test retrieves context from sprint plan."""
        state = {
            "sprint_plan": {
                "sprint_id": "sprint-1",
                "goal": "Implement authentication module with OAuth2",
                "stories": ["auth-1", "auth-2"],
            },
            "sprint_progress": None,
        }
        # Use lower threshold
        config = InjectionConfig(min_relevance_score=0.1)
        contexts = _retrieve_from_sprint("OAuth authentication", state, config)

        # Should find sprint plan context
        assert len(contexts) >= 1
        assert contexts[0].source == "sprint"


class TestRetrieveRelevantContext:
    """Tests for retrieve_relevant_context function."""

    @pytest.mark.asyncio
    async def test_empty_state_no_contexts(
        self,
        sample_gap: ContextGap,
        empty_state: dict[str, Any],
        sample_config: InjectionConfig,
    ) -> None:
        """Test returns empty tuple for empty state without memory."""
        contexts = await retrieve_relevant_context(sample_gap, None, empty_state, sample_config)
        assert contexts == ()

    @pytest.mark.asyncio
    async def test_retrieves_from_memory(
        self, sample_gap: ContextGap, empty_state: dict[str, Any]
    ) -> None:
        """Test retrieves from memory when available."""
        # Create mock memory store
        mock_memory = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = "OAuth2 authentication decision"
        mock_result.score = 0.95
        mock_result.metadata = {"type": "decision"}
        mock_memory.search_similar = AsyncMock(return_value=[mock_result])

        config = InjectionConfig(enabled_sources=("memory",))

        contexts = await retrieve_relevant_context(sample_gap, mock_memory, empty_state, config)

        assert len(contexts) >= 1
        assert contexts[0].source == "memory"
        mock_memory.search_similar.assert_called_once()

    @pytest.mark.asyncio
    async def test_respects_max_context_items(
        self, sample_gap: ContextGap, empty_state: dict[str, Any]
    ) -> None:
        """Test limits results to max_context_items."""
        # Create mock memory returning many results
        mock_memory = AsyncMock()
        mock_results = []
        for i in range(10):
            mock_result = MagicMock()
            mock_result.content = f"Context item {i}"
            mock_result.score = 0.9 - (i * 0.05)
            mock_result.metadata = {}
            mock_results.append(mock_result)
        mock_memory.search_similar = AsyncMock(return_value=mock_results)

        config = InjectionConfig(
            max_context_items=3,
            enabled_sources=("memory",),
            min_relevance_score=0.5,
        )

        contexts = await retrieve_relevant_context(sample_gap, mock_memory, empty_state, config)

        assert len(contexts) <= 3

    @pytest.mark.asyncio
    async def test_sorts_by_relevance(
        self, sample_gap: ContextGap, empty_state: dict[str, Any]
    ) -> None:
        """Test results are sorted by relevance score."""
        mock_memory = AsyncMock()
        mock_results = []
        for score in [0.7, 0.9, 0.8]:
            mock_result = MagicMock()
            mock_result.content = f"Content with score {score}"
            mock_result.score = score
            mock_result.metadata = {}
            mock_results.append(mock_result)
        mock_memory.search_similar = AsyncMock(return_value=mock_results)

        config = InjectionConfig(enabled_sources=("memory",), min_relevance_score=0.5)

        contexts = await retrieve_relevant_context(sample_gap, mock_memory, empty_state, config)

        # Should be sorted highest to lowest
        if len(contexts) >= 2:
            for i in range(len(contexts) - 1):
                assert contexts[i].relevance_score >= contexts[i + 1].relevance_score


# =============================================================================
# Tests for Context Injection
# =============================================================================


class TestBuildInjectionPayload:
    """Tests for _build_injection_payload function."""

    def test_builds_payload_with_contexts(
        self, sample_context: RetrievedContext, sample_config: InjectionConfig
    ) -> None:
        """Test builds payload from contexts."""
        payload = _build_injection_payload([sample_context], sample_config)

        assert "contexts" in payload
        assert len(payload["contexts"]) == 1
        assert payload["contexts"][0]["source"] == "memory"
        assert payload["total_items"] == 1
        assert "source_summary" in payload
        assert payload["source_summary"]["memory"] == 1

    def test_respects_max_size_limit(self, sample_config: InjectionConfig) -> None:
        """Test truncates when max size exceeded."""
        # Create large contexts
        large_contexts = []
        for _i in range(10):
            large_contexts.append(
                RetrievedContext(
                    source="memory",
                    content="A" * 10000,  # 10KB each
                    relevance_score=0.9,
                )
            )

        config = InjectionConfig(max_context_size_bytes=25000)  # 25KB limit
        payload = _build_injection_payload(large_contexts, config)

        # Should be truncated
        assert len(payload["contexts"]) < 10
        assert payload["total_size_bytes"] <= 25000


class TestInjectContext:
    """Tests for inject_context function."""

    @pytest.mark.asyncio
    async def test_successful_injection(
        self,
        sample_gap: ContextGap,
        sample_context: RetrievedContext,
        sample_config: InjectionConfig,
    ) -> None:
        """Test successful context injection."""
        result = await inject_context(sample_gap, [sample_context], sample_config)

        assert result.injected is True
        assert result.injection_target == DEFAULT_INJECTION_TARGET
        assert result.total_context_size > 0
        assert result.duration_ms >= 0
        assert result.gap == sample_gap
        assert len(result.contexts_retrieved) == 1

    @pytest.mark.asyncio
    async def test_no_injection_when_no_contexts(
        self, sample_gap: ContextGap, sample_config: InjectionConfig
    ) -> None:
        """Test no injection when no contexts provided."""
        result = await inject_context(sample_gap, [], sample_config)

        assert result.injected is False
        assert result.total_context_size == 0


# =============================================================================
# Tests for manage_context_injection
# =============================================================================


class TestManageContextInjection:
    """Tests for manage_context_injection function."""

    @pytest.mark.asyncio
    async def test_no_gap_returns_none(self, empty_state: dict[str, Any]) -> None:
        """Test returns None when no gap detected."""
        result, payload = await manage_context_injection(empty_state)

        assert result is None
        assert payload is None

    @pytest.mark.asyncio
    async def test_gap_with_contexts_returns_result(self) -> None:
        """Test returns result when gap found and contexts retrieved."""
        state = {
            "messages": [
                AIMessage(
                    content="I need clarification on the OAuth requirements",
                    additional_kwargs={"agent": "dev"},
                ),
            ],
            "current_agent": "dev",
            "decisions": [],
        }

        mock_memory = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = "OAuth2 with PKCE flow required"
        mock_result.score = 0.95
        mock_result.metadata = {}
        mock_memory.search_similar = AsyncMock(return_value=[mock_result])

        config = InjectionConfig(enabled_sources=("memory",))

        result, payload = await manage_context_injection(state, mock_memory, config)

        assert result is not None
        assert result.gap.reason == "clarification_requested"
        assert result.injected is True
        assert payload is not None
        assert DEFAULT_INJECTION_TARGET in payload

    @pytest.mark.asyncio
    async def test_gap_without_contexts_returns_empty_result(self) -> None:
        """Test returns result with injected=False when no contexts found."""
        state = {
            "messages": [],
            "current_agent": "analyst",
            "missing_context": True,  # Explicit flag
            "decisions": [],
        }

        # No memory, no relevant state data
        config = InjectionConfig(
            enabled_sources=("memory",),  # Only memory, which is None
            min_relevance_score=0.99,  # Very high threshold
        )

        result, payload = await manage_context_injection(state, None, config)

        assert result is not None
        assert result.gap.reason == "explicit_flag"
        assert result.injected is False
        assert payload is None

    @pytest.mark.asyncio
    async def test_uses_default_config_when_none(self, empty_state: dict[str, Any]) -> None:
        """Test uses default config when None provided."""
        # This should not raise an error
        result, payload = await manage_context_injection(empty_state, None, None)
        # No gap in empty state
        assert result is None
        assert payload is None


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_handles_memory_error_gracefully(
        self, sample_gap: ContextGap, empty_state: dict[str, Any]
    ) -> None:
        """Test handles memory retrieval errors gracefully."""
        mock_memory = AsyncMock()
        mock_memory.search_similar = AsyncMock(side_effect=Exception("Connection error"))

        config = InjectionConfig(enabled_sources=("memory",))

        # Should not raise, just return empty
        contexts = await retrieve_relevant_context(sample_gap, mock_memory, empty_state, config)
        assert contexts == ()

    @pytest.mark.asyncio
    async def test_handles_malformed_state(self) -> None:
        """Test handles malformed state data gracefully."""
        malformed_state: dict[str, Any] = {
            "messages": "not a list",  # Wrong type
            "current_agent": 123,  # Wrong type
        }

        # Should not crash
        gap = await detect_context_gap(malformed_state)
        # May or may not find a gap, but shouldn't crash
        assert gap is None or isinstance(gap, ContextGap)

    def test_relevance_calculation_with_unicode(self) -> None:
        """Test relevance calculation handles unicode."""
        score = _calculate_relevance_score(
            "日本語 authentication",
            "日本語 authentication requirements",
        )
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_injection_with_empty_config_sources(
        self, sample_gap: ContextGap, empty_state: dict[str, Any]
    ) -> None:
        """Test injection with no enabled sources."""
        config = InjectionConfig(enabled_sources=())

        contexts = await retrieve_relevant_context(sample_gap, None, empty_state, config)
        assert contexts == ()
