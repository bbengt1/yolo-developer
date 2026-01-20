"""Tests for ADR generation service (Story 11.8).

Tests for ADRGenerationService including generation from decisions,
content generation helpers, file export, and factory function.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_developer.audit.adr_memory_store import InMemoryADRStore
from yolo_developer.audit.adr_service import (
    ADRGenerationService,
    _generate_adr_markdown,
    _generate_consequences,
    _generate_context,
    _generate_decision_text,
    _generate_title,
    get_adr_generation_service,
)
from yolo_developer.audit.adr_types import AutoADR
from yolo_developer.audit.memory_store import InMemoryDecisionStore
from yolo_developer.audit.types import (
    AgentIdentity,
    Decision,
    DecisionContext,
)


def _make_decision(
    decision_id: str = "dec-001",
    decision_type: str = "architecture_choice",
    content: str = "Use PostgreSQL for data storage",
    rationale: str = "Strong ACID compliance and reliability",
    story_id: str | None = "1-2-database",
    sprint_id: str | None = "sprint-1",
    session_id: str = "session-123",
    severity: str = "info",
) -> Decision:
    """Create a test Decision with default values."""
    return Decision(
        id=decision_id,
        decision_type=decision_type,  # type: ignore[arg-type]
        content=content,
        rationale=rationale,
        agent=AgentIdentity(
            agent_name="architect",
            agent_type="architect",
            session_id=session_id,
        ),
        context=DecisionContext(
            story_id=story_id,
            sprint_id=sprint_id,
        ),
        timestamp="2026-01-15T10:00:00+00:00",
        severity=severity,  # type: ignore[arg-type]
    )


class TestADRGenerationServiceInit:
    """Tests for ADRGenerationService initialization."""

    def test_service_initializes_with_stores(self) -> None:
        """Test that service initializes with decision and ADR stores."""
        decision_store = InMemoryDecisionStore()
        adr_store = InMemoryADRStore()

        service = ADRGenerationService(
            decision_store=decision_store,
            adr_store=adr_store,
        )

        assert service._decision_store is decision_store
        assert service._adr_store is adr_store


class TestGenerateADRFromDecision:
    """Tests for generate_adr_from_decision method."""

    @pytest.mark.asyncio
    async def test_generates_adr_for_architecture_choice(self) -> None:
        """Test that architecture_choice decisions generate ADRs."""
        decision_store = InMemoryDecisionStore()
        adr_store = InMemoryADRStore()
        service = ADRGenerationService(decision_store, adr_store)

        decision = _make_decision(decision_type="architecture_choice")

        result = await service.generate_adr_from_decision(decision)

        assert result is not None
        assert result.id == "ADR-001"
        assert result.status == "proposed"
        assert result.source_decision_id == "dec-001"

    @pytest.mark.asyncio
    async def test_skips_non_architectural_decisions(self) -> None:
        """Test that non-architecture_choice decisions are skipped."""
        decision_store = InMemoryDecisionStore()
        adr_store = InMemoryADRStore()
        service = ADRGenerationService(decision_store, adr_store)

        decision = _make_decision(decision_type="requirement_analysis")

        result = await service.generate_adr_from_decision(decision)

        assert result is None

    @pytest.mark.asyncio
    async def test_skips_implementation_choice(self) -> None:
        """Test that implementation_choice decisions don't generate ADRs."""
        decision_store = InMemoryDecisionStore()
        adr_store = InMemoryADRStore()
        service = ADRGenerationService(decision_store, adr_store)

        decision = _make_decision(decision_type="implementation_choice")

        result = await service.generate_adr_from_decision(decision)

        assert result is None

    @pytest.mark.asyncio
    async def test_links_story_from_context(self) -> None:
        """Test that story_id from context is linked to ADR."""
        decision_store = InMemoryDecisionStore()
        adr_store = InMemoryADRStore()
        service = ADRGenerationService(decision_store, adr_store)

        decision = _make_decision(story_id="1-2-database")

        result = await service.generate_adr_from_decision(decision)

        assert result is not None
        assert result.story_ids == ("1-2-database",)

    @pytest.mark.asyncio
    async def test_handles_no_story_id(self) -> None:
        """Test that ADR is generated even without story_id."""
        decision_store = InMemoryDecisionStore()
        adr_store = InMemoryADRStore()
        service = ADRGenerationService(decision_store, adr_store)

        decision = _make_decision(story_id=None)

        result = await service.generate_adr_from_decision(decision)

        assert result is not None
        assert result.story_ids == ()

    @pytest.mark.asyncio
    async def test_stores_adr_in_store(self) -> None:
        """Test that generated ADR is stored in the ADR store."""
        decision_store = InMemoryDecisionStore()
        adr_store = InMemoryADRStore()
        service = ADRGenerationService(decision_store, adr_store)

        decision = _make_decision()

        result = await service.generate_adr_from_decision(decision)

        # Verify it's in the store
        stored = await adr_store.get_adr(result.id)
        assert stored is not None
        assert stored.id == result.id

    @pytest.mark.asyncio
    async def test_sequential_adr_numbers(self) -> None:
        """Test that ADR numbers are sequential."""
        decision_store = InMemoryDecisionStore()
        adr_store = InMemoryADRStore()
        service = ADRGenerationService(decision_store, adr_store)

        decision1 = _make_decision(decision_id="dec-001")
        decision2 = _make_decision(decision_id="dec-002")
        decision3 = _make_decision(decision_id="dec-003")

        adr1 = await service.generate_adr_from_decision(decision1)
        adr2 = await service.generate_adr_from_decision(decision2)
        adr3 = await service.generate_adr_from_decision(decision3)

        assert adr1.id == "ADR-001"
        assert adr2.id == "ADR-002"
        assert adr3.id == "ADR-003"


class TestGenerateADRsForSession:
    """Tests for generate_adrs_for_session method."""

    @pytest.mark.asyncio
    async def test_generates_adrs_for_session_decisions(self) -> None:
        """Test that ADRs are generated for all session decisions."""
        decision_store = InMemoryDecisionStore()
        adr_store = InMemoryADRStore()
        service = ADRGenerationService(decision_store, adr_store)

        # Store some decisions
        decision1 = _make_decision(
            decision_id="dec-001",
            session_id="session-123",
        )
        decision2 = _make_decision(
            decision_id="dec-002",
            session_id="session-123",
        )
        await decision_store.log_decision(decision1)
        await decision_store.log_decision(decision2)

        result = await service.generate_adrs_for_session("session-123")

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_filters_by_session_id(self) -> None:
        """Test that only decisions from specified session are processed."""
        decision_store = InMemoryDecisionStore()
        adr_store = InMemoryADRStore()
        service = ADRGenerationService(decision_store, adr_store)

        # Store decisions from different sessions
        decision1 = _make_decision(decision_id="dec-001", session_id="session-A")
        decision2 = _make_decision(decision_id="dec-002", session_id="session-B")
        decision3 = _make_decision(decision_id="dec-003", session_id="session-A")
        await decision_store.log_decision(decision1)
        await decision_store.log_decision(decision2)
        await decision_store.log_decision(decision3)

        result = await service.generate_adrs_for_session("session-A")

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_empty_session(self) -> None:
        """Test handling of session with no decisions."""
        decision_store = InMemoryDecisionStore()
        adr_store = InMemoryADRStore()
        service = ADRGenerationService(decision_store, adr_store)

        result = await service.generate_adrs_for_session("empty-session")

        assert result == []

    @pytest.mark.asyncio
    async def test_skips_non_architectural_in_session(self) -> None:
        """Test that non-architectural decisions in session are skipped."""
        decision_store = InMemoryDecisionStore()
        adr_store = InMemoryADRStore()
        service = ADRGenerationService(decision_store, adr_store)

        # Store mixed decisions
        arch_decision = _make_decision(
            decision_id="dec-001",
            decision_type="architecture_choice",
            session_id="session-123",
        )
        impl_decision = _make_decision(
            decision_id="dec-002",
            decision_type="implementation_choice",
            session_id="session-123",
        )
        await decision_store.log_decision(arch_decision)
        await decision_store.log_decision(impl_decision)

        result = await service.generate_adrs_for_session("session-123")

        # Only the architecture_choice should generate an ADR
        assert len(result) == 1
        assert result[0].source_decision_id == "dec-001"


class TestContentGenerationHelpers:
    """Tests for content generation helper functions."""

    def test_generate_title_uses_first_sentence(self) -> None:
        """Test that title uses first sentence of content."""
        decision = _make_decision(content="Use PostgreSQL. It provides strong ACID compliance.")

        result = _generate_title(decision)

        assert result == "Use PostgreSQL"

    def test_generate_title_truncates_long_content(self) -> None:
        """Test that long content without sentences is truncated."""
        long_content = "A" * 100
        decision = _make_decision(content=long_content)

        result = _generate_title(decision)

        assert len(result) == 53  # 50 chars + "..."
        assert result.endswith("...")

    def test_generate_title_short_content(self) -> None:
        """Test that short content is used as-is."""
        decision = _make_decision(content="Use Redis")

        result = _generate_title(decision)

        assert result == "Use Redis"

    def test_generate_context_includes_content(self) -> None:
        """Test that context includes decision content."""
        decision = _make_decision(content="Use PostgreSQL")

        result = _generate_context(decision)

        assert "Use PostgreSQL" in result

    def test_generate_context_includes_story_id(self) -> None:
        """Test that context includes story reference."""
        decision = _make_decision(story_id="1-2-database")

        result = _generate_context(decision)

        assert "1-2-database" in result

    def test_generate_context_includes_sprint_id(self) -> None:
        """Test that context includes sprint reference."""
        decision = _make_decision(sprint_id="sprint-1")

        result = _generate_context(decision)

        assert "sprint-1" in result

    def test_generate_context_no_story_or_sprint(self) -> None:
        """Test context generation without story or sprint."""
        decision = _make_decision(story_id=None, sprint_id=None)

        result = _generate_context(decision)

        # Should still have base context
        assert "architectural decision was needed" in result

    def test_generate_decision_text_includes_content(self) -> None:
        """Test that decision text includes content."""
        decision = _make_decision(content="Use PostgreSQL")

        result = _generate_decision_text(decision)

        assert "Use PostgreSQL" in result

    def test_generate_decision_text_includes_rationale(self) -> None:
        """Test that decision text includes rationale."""
        decision = _make_decision(rationale="Strong ACID compliance")

        result = _generate_decision_text(decision)

        assert "Strong ACID compliance" in result

    def test_generate_consequences_basic(self) -> None:
        """Test basic consequences generation."""
        decision = _make_decision(severity="info")

        result = _generate_consequences(decision)

        assert "Positive" in result
        assert "Trade-offs" in result

    def test_generate_consequences_critical_severity(self) -> None:
        """Test consequences for critical decisions."""
        decision = _make_decision(severity="critical")

        result = _generate_consequences(decision)

        assert "critical decision" in result
        assert "significantly impacts" in result

    def test_generate_consequences_warning_severity(self) -> None:
        """Test consequences for warning decisions."""
        decision = _make_decision(severity="warning")

        result = _generate_consequences(decision)

        assert "may need attention" in result


class TestFactoryFunction:
    """Tests for get_adr_generation_service factory function."""

    def test_factory_creates_service(self) -> None:
        """Test that factory function creates a service."""
        decision_store = InMemoryDecisionStore()
        adr_store = InMemoryADRStore()

        service = get_adr_generation_service(
            decision_store=decision_store,
            adr_store=adr_store,
        )

        assert isinstance(service, ADRGenerationService)
        assert service._decision_store is decision_store
        assert service._adr_store is adr_store


def _make_adr(
    adr_id: str = "ADR-001",
    title: str = "Use PostgreSQL for data storage",
    status: str = "proposed",
    context: str = "An architectural decision was needed for data storage.",
    decision: str = "Decision: Use PostgreSQL. Rationale: Strong ACID compliance.",
    consequences: str = "Positive: Reliable data storage. Trade-offs: Learning curve.",
    source_decision_id: str = "dec-001",
    story_ids: tuple[str, ...] = ("1-2-database",),
    created_at: str = "2026-01-15T10:00:00+00:00",
) -> AutoADR:
    """Create a test AutoADR with default values."""
    return AutoADR(
        id=adr_id,
        title=title,
        status=status,  # type: ignore[arg-type]
        context=context,
        decision=decision,
        consequences=consequences,
        source_decision_id=source_decision_id,
        story_ids=story_ids,
        created_at=created_at,
    )


class TestGenerateADRMarkdown:
    """Tests for _generate_adr_markdown helper function."""

    def test_generates_title_with_id(self) -> None:
        """Test that markdown includes ADR ID and title."""
        adr = _make_adr(adr_id="ADR-001", title="Use PostgreSQL")

        result = _generate_adr_markdown(adr)

        assert "# ADR-001: Use PostgreSQL" in result

    def test_includes_status(self) -> None:
        """Test that markdown includes status."""
        adr = _make_adr(status="proposed")

        result = _generate_adr_markdown(adr)

        assert "**Status:** proposed" in result

    def test_includes_date(self) -> None:
        """Test that markdown includes date extracted from created_at."""
        adr = _make_adr(created_at="2026-01-15T10:00:00+00:00")

        result = _generate_adr_markdown(adr)

        assert "**Date:** 2026-01-15" in result

    def test_includes_story_references(self) -> None:
        """Test that markdown includes story references when present."""
        adr = _make_adr(story_ids=("story-1", "story-2"))

        result = _generate_adr_markdown(adr)

        assert "**Stories:** story-1, story-2" in result

    def test_no_stories_section_when_empty(self) -> None:
        """Test that stories section is omitted when no story_ids."""
        adr = _make_adr(story_ids=())

        result = _generate_adr_markdown(adr)

        assert "**Stories:**" not in result

    def test_includes_context_section(self) -> None:
        """Test that markdown includes Context section."""
        adr = _make_adr(context="Need a reliable database")

        result = _generate_adr_markdown(adr)

        assert "## Context" in result
        assert "Need a reliable database" in result

    def test_includes_decision_section(self) -> None:
        """Test that markdown includes Decision section."""
        adr = _make_adr(decision="Use PostgreSQL for ACID compliance")

        result = _generate_adr_markdown(adr)

        assert "## Decision" in result
        assert "Use PostgreSQL for ACID compliance" in result

    def test_includes_consequences_section(self) -> None:
        """Test that markdown includes Consequences section."""
        adr = _make_adr(consequences="Positive: Strong reliability")

        result = _generate_adr_markdown(adr)

        assert "## Consequences" in result
        assert "Strong reliability" in result

    def test_includes_source_reference(self) -> None:
        """Test that markdown includes source decision ID reference."""
        adr = _make_adr(source_decision_id="dec-123")

        result = _generate_adr_markdown(adr)

        assert "*Source Decision ID: dec-123*" in result


class TestExportADRToFile:
    """Tests for export_adr_to_file method."""

    @pytest.mark.asyncio
    async def test_creates_markdown_file(self, tmp_path: Path) -> None:
        """Test that export creates a markdown file."""
        decision_store = InMemoryDecisionStore()
        adr_store = InMemoryADRStore()
        service = ADRGenerationService(decision_store, adr_store)
        adr = _make_adr(adr_id="ADR-001")

        result = await service.export_adr_to_file(adr, tmp_path)

        assert result.exists()
        assert result.name == "ADR-001.md"

    @pytest.mark.asyncio
    async def test_file_contains_markdown_content(self, tmp_path: Path) -> None:
        """Test that exported file contains proper markdown."""
        decision_store = InMemoryDecisionStore()
        adr_store = InMemoryADRStore()
        service = ADRGenerationService(decision_store, adr_store)
        adr = _make_adr(adr_id="ADR-001", title="Use PostgreSQL")

        result = await service.export_adr_to_file(adr, tmp_path)
        content = result.read_text()

        assert "# ADR-001: Use PostgreSQL" in content
        assert "## Context" in content
        assert "## Decision" in content
        assert "## Consequences" in content

    @pytest.mark.asyncio
    async def test_creates_output_directory(self, tmp_path: Path) -> None:
        """Test that export creates output directory if it doesn't exist."""
        decision_store = InMemoryDecisionStore()
        adr_store = InMemoryADRStore()
        service = ADRGenerationService(decision_store, adr_store)
        adr = _make_adr()

        new_dir = tmp_path / "nested" / "adrs"
        result = await service.export_adr_to_file(adr, new_dir)

        assert new_dir.exists()
        assert result.exists()

    @pytest.mark.asyncio
    async def test_accepts_string_path(self, tmp_path: Path) -> None:
        """Test that export accepts string path as well as Path."""
        decision_store = InMemoryDecisionStore()
        adr_store = InMemoryADRStore()
        service = ADRGenerationService(decision_store, adr_store)
        adr = _make_adr()

        result = await service.export_adr_to_file(adr, str(tmp_path))

        assert result.exists()


class TestExportAllADRsToDirectory:
    """Tests for export_all_adrs_to_directory method."""

    @pytest.mark.asyncio
    async def test_exports_all_adrs(self, tmp_path: Path) -> None:
        """Test that all ADRs in store are exported."""
        decision_store = InMemoryDecisionStore()
        adr_store = InMemoryADRStore()
        service = ADRGenerationService(decision_store, adr_store)

        # Store some ADRs
        await adr_store.store_adr(_make_adr(adr_id="ADR-001"))
        await adr_store.store_adr(_make_adr(adr_id="ADR-002"))
        await adr_store.store_adr(_make_adr(adr_id="ADR-003"))

        result = await service.export_all_adrs_to_directory(tmp_path)

        assert len(result) == 3
        assert (tmp_path / "ADR-001.md").exists()
        assert (tmp_path / "ADR-002.md").exists()
        assert (tmp_path / "ADR-003.md").exists()

    @pytest.mark.asyncio
    async def test_returns_empty_list_for_empty_store(self, tmp_path: Path) -> None:
        """Test that empty store returns empty list."""
        decision_store = InMemoryDecisionStore()
        adr_store = InMemoryADRStore()
        service = ADRGenerationService(decision_store, adr_store)

        result = await service.export_all_adrs_to_directory(tmp_path)

        assert result == []

    @pytest.mark.asyncio
    async def test_creates_output_directory(self, tmp_path: Path) -> None:
        """Test that export creates output directory if needed."""
        decision_store = InMemoryDecisionStore()
        adr_store = InMemoryADRStore()
        service = ADRGenerationService(decision_store, adr_store)
        await adr_store.store_adr(_make_adr())

        new_dir = tmp_path / "new_adrs"
        await service.export_all_adrs_to_directory(new_dir)

        assert new_dir.exists()
