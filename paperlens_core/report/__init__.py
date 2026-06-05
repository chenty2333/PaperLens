from __future__ import annotations

from paperlens_core.report.graph_export import (
    core_graph_report_filename,
    write_core_graph_report_view,
)
from paperlens_core.report.graph_view import (
    GraphReportDraft,
    ReportParagraph,
    ReportSection,
    audit_report_draft_against_graph,
    build_report_draft_from_graph,
    render_graph_report_markdown,
)
from paperlens_core.report.memory_context import (
    build_report_memory_context,
    compact_core_memory_view_for_report,
    compact_paper_memory_for_report,
    core_memory_pages,
    core_memory_view_dict,
    report_focus_pages,
    report_focus_queries,
)

__all__ = [
    "GraphReportDraft",
    "ReportParagraph",
    "ReportSection",
    "audit_report_draft_against_graph",
    "build_report_memory_context",
    "build_report_draft_from_graph",
    "compact_core_memory_view_for_report",
    "compact_paper_memory_for_report",
    "core_graph_report_filename",
    "core_memory_pages",
    "core_memory_view_dict",
    "report_focus_pages",
    "report_focus_queries",
    "render_graph_report_markdown",
    "write_core_graph_report_view",
]
