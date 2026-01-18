"""PDF exporter for audit data (Story 11.4).

This module provides PDF export functionality for audit trails.

The PdfAuditExporter produces human-readable PDF reports with:
- Title page with export metadata
- Decision section with formatted entries
- Traceability section with artifact/link tables
- Color-coded severity levels
- Support for redaction of sensitive data

Example:
    >>> from yolo_developer.audit.pdf_exporter import PdfAuditExporter
    >>> from yolo_developer.audit.export_types import ExportOptions, RedactionConfig
    >>>
    >>> exporter = PdfAuditExporter()
    >>> decisions = [...]  # list of Decision objects
    >>> pdf_bytes = exporter.export_decisions(decisions)
    >>>
    >>> # With redaction
    >>> redaction = RedactionConfig(redact_metadata=True)
    >>> options = ExportOptions(redaction_config=redaction)
    >>> pdf_bytes = exporter.export_decisions(decisions, options=options)

References:
    - FR84: System can export audit trail for compliance reporting
    - AC #1: Data exported in requested format (PDF)
    - AC #2: All relevant fields included
    - AC #3: Export is complete and accurate
    - AC #4: Sensitive data can be redacted
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from yolo_developer.audit.export_types import (
    DEFAULT_EXPORT_OPTIONS,
    ExportOptions,
    RedactionConfig,
)
from yolo_developer.audit.traceability_types import TraceableArtifact, TraceLink
from yolo_developer.audit.types import Decision

REDACTED_PLACEHOLDER = "[REDACTED]"

# Severity colors
SEVERITY_COLORS = {
    "critical": colors.red,
    "warning": colors.orange,
    "info": colors.green,
}


class PdfAuditExporter:
    """PDF exporter for audit data.

    Exports audit trail data to PDF format with:
    - Title page with export metadata
    - Formatted decision entries
    - Traceability tables
    - Color-coded severity levels
    - Configurable redaction support

    Example:
        >>> exporter = PdfAuditExporter()
        >>> decisions = [decision1, decision2]
        >>> pdf_bytes = exporter.export_decisions(decisions)
    """

    def __init__(self) -> None:
        """Initialize the PDF exporter with styles."""
        self._styles = getSampleStyleSheet()
        self._title_style = ParagraphStyle(
            "Title",
            parent=self._styles["Heading1"],
            fontSize=24,
            spaceAfter=30,
        )
        self._heading_style = ParagraphStyle(
            "CustomHeading",
            parent=self._styles["Heading2"],
            fontSize=16,
            spaceAfter=12,
        )
        self._body_style = self._styles["Normal"]

    def export_decisions(
        self, decisions: list[Decision], options: ExportOptions | None = None
    ) -> bytes:
        """Export decisions to PDF format.

        Args:
            decisions: List of decisions to export.
            options: Optional export options (uses defaults if None).

        Returns:
            PDF bytes.
        """
        options = options or DEFAULT_EXPORT_OPTIONS
        redaction = options.redaction_config

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        story = []

        # Title
        story.append(Paragraph("Audit Trail Export - Decisions", self._title_style))
        story.append(Spacer(1, 0.25 * inch))

        # Metadata
        export_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        story.append(Paragraph(f"Export Date: {export_time}", self._body_style))
        story.append(Paragraph(f"Total Decisions: {len(decisions)}", self._body_style))
        story.append(Spacer(1, 0.5 * inch))

        # Decisions
        if decisions:
            story.append(Paragraph("Decisions", self._heading_style))
            for decision in decisions:
                self._add_decision_to_story(story, decision, redaction)
        else:
            story.append(Paragraph("No decisions to export.", self._body_style))

        doc.build(story)
        return buffer.getvalue()

    def export_traces(
        self,
        artifacts: list[TraceableArtifact],
        links: list[TraceLink],
        options: ExportOptions | None = None,
    ) -> bytes:
        """Export traceability data to PDF format.

        Args:
            artifacts: List of traceable artifacts to export.
            links: List of trace links between artifacts.
            options: Optional export options (uses defaults if None).

        Returns:
            PDF bytes.
        """
        options = options or DEFAULT_EXPORT_OPTIONS
        redaction = options.redaction_config

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        story = []

        # Title
        story.append(Paragraph("Audit Trail Export - Traceability", self._title_style))
        story.append(Spacer(1, 0.25 * inch))

        # Metadata
        export_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        story.append(Paragraph(f"Export Date: {export_time}", self._body_style))
        story.append(Paragraph(f"Total Artifacts: {len(artifacts)}", self._body_style))
        story.append(Paragraph(f"Total Links: {len(links)}", self._body_style))
        story.append(Spacer(1, 0.5 * inch))

        # Artifacts table
        if artifacts:
            story.append(Paragraph("Artifacts", self._heading_style))
            story.append(self._create_artifacts_table(artifacts, redaction))
            story.append(Spacer(1, 0.25 * inch))

        # Links table
        if links:
            story.append(Paragraph("Trace Links", self._heading_style))
            story.append(self._create_links_table(links, redaction))

        if not artifacts and not links:
            story.append(Paragraph("No traceability data to export.", self._body_style))

        doc.build(story)
        return buffer.getvalue()

    def export_full_audit(
        self,
        decisions: list[Decision],
        artifacts: list[TraceableArtifact],
        links: list[TraceLink],
        options: ExportOptions | None = None,
    ) -> bytes:
        """Export complete audit trail to PDF format.

        Args:
            decisions: List of decisions to export.
            artifacts: List of traceable artifacts to export.
            links: List of trace links between artifacts.
            options: Optional export options (uses defaults if None).

        Returns:
            PDF bytes.
        """
        options = options or DEFAULT_EXPORT_OPTIONS
        redaction = options.redaction_config

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        story = []

        # Title
        story.append(Paragraph("Complete Audit Trail Export", self._title_style))
        story.append(Spacer(1, 0.25 * inch))

        # Metadata
        export_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        story.append(Paragraph(f"Export Date: {export_time}", self._body_style))
        story.append(Paragraph(f"Total Decisions: {len(decisions)}", self._body_style))
        story.append(Paragraph(f"Total Artifacts: {len(artifacts)}", self._body_style))
        story.append(Paragraph(f"Total Links: {len(links)}", self._body_style))
        story.append(Spacer(1, 0.5 * inch))

        # Decisions Section
        story.append(Paragraph("Decisions", self._heading_style))
        if decisions:
            for decision in decisions:
                self._add_decision_to_story(story, decision, redaction)
        else:
            story.append(Paragraph("No decisions.", self._body_style))
        story.append(Spacer(1, 0.25 * inch))

        # Artifacts Section
        story.append(Paragraph("Artifacts", self._heading_style))
        if artifacts:
            story.append(self._create_artifacts_table(artifacts, redaction))
        else:
            story.append(Paragraph("No artifacts.", self._body_style))
        story.append(Spacer(1, 0.25 * inch))

        # Links Section
        story.append(Paragraph("Trace Links", self._heading_style))
        if links:
            story.append(self._create_links_table(links, redaction))
        else:
            story.append(Paragraph("No links.", self._body_style))

        doc.build(story)
        return buffer.getvalue()

    def get_file_extension(self) -> str:
        """Return the file extension for PDF format.

        Returns:
            File extension ".pdf".
        """
        return ".pdf"

    def get_content_type(self) -> str:
        """Return the MIME content type for PDF.

        Returns:
            MIME type "application/pdf".
        """
        return "application/pdf"

    def _add_decision_to_story(
        self,
        story: list[Any],
        decision: Decision,
        redaction: RedactionConfig,
    ) -> None:
        """Add a decision entry to the PDF story.

        Args:
            story: List of PDF elements.
            decision: Decision to add.
            redaction: Redaction configuration.
        """
        d = decision.to_dict()

        # Get severity color
        severity = d.get("severity", "info")
        severity_color = SEVERITY_COLORS.get(severity, colors.black)

        # Apply redactions
        session_id = d.get("agent", {}).get("session_id", "")
        if redaction.redact_session_ids:
            session_id = REDACTED_PLACEHOLDER

        metadata = d.get("metadata", {})
        if redaction.redact_metadata:
            metadata_str = REDACTED_PLACEHOLDER
        else:
            metadata_str = str(metadata) if metadata else ""

        # Create severity style
        severity_style = ParagraphStyle(
            "SeverityStyle",
            parent=self._body_style,
            textColor=severity_color,
            fontName="Helvetica-Bold",
        )

        # Decision header
        decision_id = d.get("id", "unknown")
        decision_type = d.get("decision_type", "unknown")

        story.append(
            Paragraph(
                f"<b>Decision {decision_id}</b> ({decision_type})",
                self._body_style,
            )
        )
        story.append(
            Paragraph(f"Severity: <b>{severity}</b>", severity_style)
        )
        story.append(
            Paragraph(f"Content: {d.get('content', '')}", self._body_style)
        )
        story.append(
            Paragraph(f"Rationale: {d.get('rationale', '')}", self._body_style)
        )
        story.append(
            Paragraph(
                f"Agent: {d.get('agent', {}).get('agent_name', '')} "
                f"({d.get('agent', {}).get('agent_type', '')})",
                self._body_style,
            )
        )
        story.append(Paragraph(f"Session ID: {session_id}", self._body_style))
        story.append(
            Paragraph(f"Timestamp: {d.get('timestamp', '')}", self._body_style)
        )
        if metadata_str:
            story.append(Paragraph(f"Metadata: {metadata_str}", self._body_style))

        story.append(Spacer(1, 0.2 * inch))

    def _create_artifacts_table(
        self,
        artifacts: list[TraceableArtifact],
        redaction: RedactionConfig,
    ) -> Table:
        """Create a table for artifacts.

        Args:
            artifacts: List of artifacts.
            redaction: Redaction configuration.

        Returns:
            ReportLab Table element.
        """
        data = [["ID", "Type", "Name", "Description", "Metadata"]]

        for artifact in artifacts:
            d = artifact.to_dict()
            description = d.get("description", "")
            desc_display = (
                description[:40] + "..."
                if len(description) > 40
                else description
            )

            # Apply redaction to metadata
            metadata = d.get("metadata", {})
            if redaction.redact_metadata:
                metadata_str = REDACTED_PLACEHOLDER
            else:
                metadata_str = str(metadata)[:30] + "..." if len(str(metadata)) > 30 else str(metadata)

            data.append([
                d.get("id", ""),
                d.get("artifact_type", ""),
                d.get("name", ""),
                desc_display,
                metadata_str,
            ])

        table = Table(data, colWidths=[1 * inch, 0.8 * inch, 1.5 * inch, 2 * inch, 1.4 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))

        return table

    def _create_links_table(
        self,
        links: list[TraceLink],
        redaction: RedactionConfig,
    ) -> Table:
        """Create a table for trace links.

        Args:
            links: List of links.
            redaction: Redaction configuration.

        Returns:
            ReportLab Table element.
        """
        data = [["ID", "Source", "Target", "Type", "Metadata"]]

        for link in links:
            d = link.to_dict()

            # Apply redaction to metadata
            metadata = d.get("metadata", {})
            if redaction.redact_metadata:
                metadata_str = REDACTED_PLACEHOLDER
            else:
                metadata_str = str(metadata)[:25] + "..." if len(str(metadata)) > 25 else str(metadata)

            data.append([
                d.get("id", ""),
                f"{d.get('source_id', '')} ({d.get('source_type', '')})",
                f"{d.get('target_id', '')} ({d.get('target_type', '')})",
                d.get("link_type", ""),
                metadata_str,
            ])

        table = Table(data, colWidths=[1 * inch, 1.6 * inch, 1.6 * inch, 1.2 * inch, 1.3 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))

        return table
