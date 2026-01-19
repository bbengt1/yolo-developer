"""ADR generation service for auto-generating ADRs from audit decisions (Story 11.8).

This module provides the ADRGenerationService that creates Architecture Decision
Records from Decision records in the audit trail.

The service monitors for `architecture_choice` decision types and generates
corresponding ADRs with context, decision, and consequences sections.

Example:
    >>> from yolo_developer.audit.adr_service import (
    ...     ADRGenerationService,
    ...     get_adr_generation_service,
    ... )
    >>>
    >>> service = get_adr_generation_service(
    ...     decision_store=decision_store,
    ...     adr_store=adr_store,
    ... )
    >>> adr = await service.generate_adr_from_decision(decision)
    >>> adrs = await service.generate_adrs_for_session("session-123")
    >>>
    >>> # Export ADR to file system (AC #4)
    >>> await service.export_adr_to_file(adr, "/path/to/adrs")

References:
    - FR88: System can generate Architecture Decision Records automatically
    - Story 11.1: Decision types and DecisionStore
    - Story 7.3: ADR generation pattern (Architect Agent)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from yolo_developer.audit.adr_types import AutoADR

if TYPE_CHECKING:
    from yolo_developer.audit.adr_store import ADRStore
    from yolo_developer.audit.store import DecisionStore
    from yolo_developer.audit.types import Decision

_logger = structlog.get_logger(__name__)


class ADRGenerationService:
    """Service for auto-generating ADRs from audit decisions.

    Processes Decision records from the audit trail and generates
    corresponding ADRs for architecture_choice decision types.

    Attributes:
        _decision_store: Store for retrieving Decision records.
        _adr_store: Store for persisting generated ADRs.

    Example:
        >>> service = ADRGenerationService(
        ...     decision_store=decision_store,
        ...     adr_store=adr_store,
        ... )
        >>> adr = await service.generate_adr_from_decision(decision)
    """

    def __init__(
        self,
        decision_store: DecisionStore,
        adr_store: ADRStore,
    ) -> None:
        """Initialize the ADR generation service.

        Args:
            decision_store: Store for retrieving Decision records.
            adr_store: Store for persisting generated ADRs.
        """
        self._decision_store = decision_store
        self._adr_store = adr_store
        _logger.debug(
            "adr_generation_service_initialized",
        )

    async def generate_adr_from_decision(self, decision: Decision) -> AutoADR | None:
        """Generate an ADR from a single decision.

        Only generates ADRs for architecture_choice decisions.
        Non-architectural decisions are skipped with a debug log.

        Args:
            decision: The Decision to potentially convert to ADR.

        Returns:
            The generated AutoADR, or None if decision type doesn't warrant ADR.

        Example:
            >>> adr = await service.generate_adr_from_decision(decision)
            >>> if adr:
            ...     print(f"Generated {adr.id}")
        """
        if decision.decision_type != "architecture_choice":
            _logger.debug(
                "skipping_non_architectural_decision",
                decision_id=decision.id,
                decision_type=decision.decision_type,
            )
            return None

        adr_number = await self._adr_store.get_next_adr_number()

        # Build story_ids tuple from context
        story_ids: tuple[str, ...] = ()
        if decision.context.story_id:
            story_ids = (decision.context.story_id,)

        adr = AutoADR(
            id=f"ADR-{adr_number:03d}",
            title=_generate_title(decision),
            status="proposed",
            context=_generate_context(decision),
            decision=_generate_decision_text(decision),
            consequences=_generate_consequences(decision),
            source_decision_id=decision.id,
            story_ids=story_ids,
        )

        await self._adr_store.store_adr(adr)

        _logger.info(
            "adr_generated",
            adr_id=adr.id,
            source_decision_id=decision.id,
            story_ids=story_ids,
        )

        return adr

    async def generate_adrs_for_session(self, session_id: str) -> list[AutoADR]:
        """Generate ADRs for all architectural decisions in a session.

        Queries the decision store for all architecture_choice decisions
        made in the specified session and generates ADRs for each.

        Args:
            session_id: The session ID to process.

        Returns:
            List of generated ADRs.

        Example:
            >>> adrs = await service.generate_adrs_for_session("session-123")
            >>> print(f"Generated {len(adrs)} ADRs")
        """
        from yolo_developer.audit.store import DecisionFilters

        _logger.debug(
            "generating_adrs_for_session",
            session_id=session_id,
        )

        filters = DecisionFilters(decision_type="architecture_choice")
        decisions = await self._decision_store.get_decisions(filters)

        # Filter by session (check agent.session_id)
        session_decisions = [
            d for d in decisions if d.agent.session_id == session_id
        ]

        adrs: list[AutoADR] = []
        for decision in session_decisions:
            adr = await self.generate_adr_from_decision(decision)
            if adr is not None:
                adrs.append(adr)

        _logger.info(
            "session_adrs_generated",
            session_id=session_id,
            adr_count=len(adrs),
        )

        return adrs

    async def export_adr_to_file(
        self,
        adr: AutoADR,
        output_dir: str | Path,
    ) -> Path:
        """Export an ADR to a markdown file in the specified directory.

        Creates a markdown file with the ADR content following standard
        ADR format conventions.

        Args:
            adr: The ADR to export.
            output_dir: Directory to write the ADR file to.

        Returns:
            Path to the created markdown file.

        Example:
            >>> path = await service.export_adr_to_file(adr, "/path/to/adrs")
            >>> print(f"ADR written to {path}")
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename from ADR ID (e.g., "ADR-001.md")
        filename = f"{adr.id}.md"
        file_path = output_path / filename

        # Generate markdown content
        markdown = _generate_adr_markdown(adr)

        # Write to file
        file_path.write_text(markdown, encoding="utf-8")

        _logger.info(
            "adr_exported_to_file",
            adr_id=adr.id,
            file_path=str(file_path),
        )

        return file_path

    async def export_all_adrs_to_directory(
        self,
        output_dir: str | Path,
    ) -> list[Path]:
        """Export all ADRs from the store to markdown files.

        Creates a markdown file for each ADR in the specified directory.

        Args:
            output_dir: Directory to write the ADR files to.

        Returns:
            List of paths to created markdown files.

        Example:
            >>> paths = await service.export_all_adrs_to_directory("/path/to/adrs")
            >>> print(f"Exported {len(paths)} ADRs")
        """
        adrs = await self._adr_store.get_all_adrs()

        paths: list[Path] = []
        for adr in adrs:
            path = await self.export_adr_to_file(adr, output_dir)
            paths.append(path)

        _logger.info(
            "all_adrs_exported",
            output_dir=str(output_dir),
            adr_count=len(paths),
        )

        return paths


# =============================================================================
# Content Generation Helpers
# =============================================================================


def _generate_title(decision: Decision) -> str:
    """Generate ADR title from decision content.

    Extracts a concise title from the decision content, using the
    first sentence or truncating to 50 characters.

    Args:
        decision: The Decision to generate a title for.

    Returns:
        A concise title string.
    """
    content = decision.content

    # Use first sentence if available
    if ". " in content:
        title = content.split(". ")[0]
    else:
        # Truncate long content
        title = content[:50] + "..." if len(content) > 50 else content

    return title


def _generate_context(decision: Decision) -> str:
    """Generate ADR context section.

    Creates a context explaining why the decision was needed,
    including story and sprint references when available.

    Args:
        decision: The Decision to generate context for.

    Returns:
        A context string explaining the decision need.
    """
    parts = []

    # Core context from decision
    parts.append(f"An architectural decision was needed: {decision.content}")

    # Add story reference
    if decision.context.story_id:
        parts.append(f"This decision was made for story {decision.context.story_id}.")

    # Add sprint reference
    if decision.context.sprint_id:
        parts.append(f"Sprint: {decision.context.sprint_id}.")

    return " ".join(parts)


def _generate_decision_text(decision: Decision) -> str:
    """Generate ADR decision section.

    Creates the decision section stating what was decided
    and the rationale.

    Args:
        decision: The Decision to document.

    Returns:
        A decision string with the chosen approach and rationale.
    """
    parts = []

    # What was decided
    parts.append(f"Decision: {decision.content}")

    # Rationale
    parts.append(f"Rationale: {decision.rationale}")

    return " ".join(parts)


def _generate_consequences(decision: Decision) -> str:
    """Generate ADR consequences section.

    Creates the consequences section documenting positive effects,
    trade-offs, and any severity notes.

    Args:
        decision: The Decision to document consequences for.

    Returns:
        A consequences string with effects and trade-offs.
    """
    parts = []

    # Positive effects
    parts.append("Positive: Addresses architectural need with documented rationale.")

    # Severity-based notes
    if decision.severity == "critical":
        parts.append(
            "Note: This is a critical decision that significantly impacts system behavior."
        )
    elif decision.severity == "warning":
        parts.append(
            "Note: This decision may need attention and should be reviewed."
        )

    # Trade-offs
    parts.append("Trade-offs: Requires careful implementation and monitoring.")

    return " ".join(parts)


def _generate_adr_markdown(adr: AutoADR) -> str:
    """Generate markdown content for an ADR.

    Creates a standard ADR format markdown document with all
    required sections.

    Args:
        adr: The AutoADR to convert to markdown.

    Returns:
        Markdown string representing the ADR.
    """
    lines = [
        f"# {adr.id}: {adr.title}",
        "",
        f"**Status:** {adr.status}",
        f"**Date:** {adr.created_at[:10]}",  # Extract date portion
        "",
    ]

    # Add story references if available
    if adr.story_ids:
        stories = ", ".join(adr.story_ids)
        lines.extend([f"**Stories:** {stories}", ""])

    # Context section
    lines.extend([
        "## Context",
        "",
        adr.context,
        "",
    ])

    # Decision section
    lines.extend([
        "## Decision",
        "",
        adr.decision,
        "",
    ])

    # Consequences section
    lines.extend([
        "## Consequences",
        "",
        adr.consequences,
        "",
    ])

    # Source reference
    lines.extend([
        "---",
        "",
        f"*Source Decision ID: {adr.source_decision_id}*",
    ])

    return "\n".join(lines)


# =============================================================================
# Factory Function
# =============================================================================


def get_adr_generation_service(
    decision_store: DecisionStore,
    adr_store: ADRStore,
) -> ADRGenerationService:
    """Factory function to create ADRGenerationService.

    Creates an ADRGenerationService with the provided stores.

    Args:
        decision_store: Store for retrieving Decision records.
        adr_store: Store for persisting generated ADRs.

    Returns:
        Configured ADRGenerationService instance.

    Example:
        >>> service = get_adr_generation_service(
        ...     decision_store=decision_store,
        ...     adr_store=adr_store,
        ... )
    """
    return ADRGenerationService(
        decision_store=decision_store,
        adr_store=adr_store,
    )
