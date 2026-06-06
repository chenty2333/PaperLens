from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from paperlens_core.audit import AuditFinding, AuditSeverity
from paperlens_core.dom import PaperDOM
from paperlens_core.grounding import text_overlaps_any_reference
from paperlens_core.graph import ClaimGraph


class ReportParagraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paragraph_id: str
    markdown: str
    used_node_ids: list[str] = Field(default_factory=list)
    used_evidence_ids: list[str] = Field(default_factory=list)


class ReportSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str
    title: str
    paragraphs: list[ReportParagraph] = Field(default_factory=list)


class GraphReportDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "graph_report_draft.v2"
    paper_id: str
    sections: list[ReportSection] = Field(default_factory=list)


SECTION_TITLES = {
    "problem": {"zh": "问题与动机", "en": "Problem"},
    "claim": {"zh": "核心主张", "en": "Claims"},
    "mechanism": {"zh": "方法机制", "en": "Mechanism"},
    "implementation": {"zh": "实现细节", "en": "Implementation"},
    "evaluation": {"zh": "实验设置", "en": "Evaluation Setup"},
    "result": {"zh": "结果与结论", "en": "Results"},
    "limitation": {"zh": "限制与边界", "en": "Limitations"},
    "concept": {"zh": "相关概念", "en": "Concept Bridge"},
}

NUMBER_TEXT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:\s?%|[A-Za-z]{1,6})?"
)


def build_report_draft_from_graph(
    graph: ClaimGraph, *, max_nodes_per_kind: int = 8, output_language: str = "zh"
) -> GraphReportDraft:
    sections: list[ReportSection] = []
    for kind, titles in SECTION_TITLES.items():
        nodes = [node for node in graph.nodes.values() if node.kind == kind][:max_nodes_per_kind]
        if not nodes:
            continue
        paragraphs = []
        for index, node in enumerate(nodes, start=1):
            evidence_ids = graph.evidence_ids_for(node.node_id)
            paragraphs.append(
                ReportParagraph(
                    paragraph_id=f"{kind}_{index:02d}",
                    markdown=node.label,
                    used_node_ids=[node.node_id],
                    used_evidence_ids=evidence_ids,
                )
            )
        sections.append(
            ReportSection(
                section_id=kind,
                title=localized_section_title(kind, titles, output_language=output_language),
                paragraphs=paragraphs,
            )
        )
    return GraphReportDraft(paper_id=graph.paper_id, sections=sections)


def audit_report_draft_against_graph(
    draft: GraphReportDraft,
    graph: ClaimGraph,
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    if draft.paper_id != graph.paper_id:
        findings.append(
            AuditFinding(
                finding_id=f"report_draft_paper_id_mismatch:{draft.paper_id}:{graph.paper_id}",
                severity=AuditSeverity.ERROR,
                code="report_draft_paper_id_mismatch",
                message=f"Report draft paper_id does not match ClaimGraph: {draft.paper_id} != {graph.paper_id}",
            )
        )
    node_ids = set(graph.nodes)
    evidence_node_ids = {node.node_id for node in graph.nodes.values() if node.kind == "evidence"}
    support_edges = {
        (edge.source_id, edge.target_id) for edge in graph.edges if edge.kind == "supported_by"
    }
    for section in draft.sections:
        for paragraph in section.paragraphs:
            if not paragraph.used_node_ids:
                findings.append(
                    AuditFinding(
                        finding_id=f"paragraph_missing_nodes:{paragraph.paragraph_id}",
                        severity=AuditSeverity.ERROR,
                        code="report_paragraph_missing_node_ids",
                        message="Report paragraph does not declare graph node IDs",
                    )
                )
            if not paragraph.used_evidence_ids:
                findings.append(
                    AuditFinding(
                        finding_id=f"paragraph_missing_evidence:{paragraph.paragraph_id}",
                        severity=AuditSeverity.ERROR,
                        code="report_paragraph_missing_evidence_ids",
                        message="Report paragraph does not declare evidence IDs",
                    )
                )
            for node_id in paragraph.used_node_ids:
                if node_id not in node_ids:
                    findings.append(
                        AuditFinding(
                            finding_id=f"paragraph_unknown_node:{paragraph.paragraph_id}:{node_id}",
                            severity=AuditSeverity.ERROR,
                            code="report_paragraph_unknown_node_id",
                            message=f"Report paragraph references unknown graph node: {node_id}",
                            node_id=node_id,
                        )
                    )
            for evidence_id in paragraph.used_evidence_ids:
                if evidence_id not in evidence_node_ids:
                    findings.append(
                        AuditFinding(
                            finding_id=(
                                f"paragraph_unknown_evidence:{paragraph.paragraph_id}:{evidence_id}"
                            ),
                            severity=AuditSeverity.ERROR,
                            code="report_paragraph_unknown_evidence_id",
                            message=f"Report paragraph references unknown evidence node: {evidence_id}",
                            node_id=evidence_id,
                        )
                    )
            known_node_ids = [node_id for node_id in paragraph.used_node_ids if node_id in node_ids]
            known_evidence_ids = [
                evidence_id
                for evidence_id in paragraph.used_evidence_ids
                if evidence_id in evidence_node_ids
            ]
            known_source_ids = graph_report_source_ids(
                graph=graph,
                evidence_ids=known_evidence_ids,
            )
            if known_evidence_ids and not known_source_ids:
                findings.append(
                    AuditFinding(
                        finding_id=f"paragraph_missing_sources:{paragraph.paragraph_id}",
                        severity=AuditSeverity.ERROR,
                        code="report_paragraph_missing_source_ids",
                        message=(
                            "Report paragraph declares evidence IDs, but those evidence nodes do "
                            "not expose PaperDOM source IDs"
                        ),
                    )
                )
            for node_id in known_node_ids:
                if not any(
                    (node_id, evidence_id) in support_edges for evidence_id in known_evidence_ids
                ):
                    findings.append(
                        AuditFinding(
                            finding_id=(
                                "paragraph_node_without_declared_support:"
                                f"{paragraph.paragraph_id}:{node_id}"
                            ),
                            severity=AuditSeverity.ERROR,
                            code="report_paragraph_node_missing_declared_evidence",
                            message=(
                                "Report paragraph declares a graph node but none of its declared "
                                "evidence IDs support that node"
                            ),
                            node_id=node_id,
                        )
                    )
            paragraph_overlaps_any_node = bool(known_node_ids) and text_overlaps_any_reference(
                paragraph.markdown,
                [graph.nodes[node_id].label for node_id in known_node_ids],
            )
            if known_node_ids and not paragraph_overlaps_any_node:
                findings.append(
                    AuditFinding(
                        finding_id=f"paragraph_text_not_grounded:{paragraph.paragraph_id}",
                        severity=AuditSeverity.ERROR,
                        code="report_paragraph_text_not_grounded_in_declared_nodes",
                        message=(
                            "Report paragraph text does not overlap declared ClaimGraph node labels"
                        ),
                        node_id=known_node_ids[0],
                        source_ids=known_source_ids,
                    )
                )
            for node_id in known_node_ids:
                if paragraph_overlaps_any_node and not text_overlaps_any_reference(
                    paragraph.markdown, [graph.nodes[node_id].label]
                ):
                    findings.append(
                        AuditFinding(
                            finding_id=(
                                "paragraph_declared_node_not_used_in_text:"
                                f"{paragraph.paragraph_id}:{node_id}"
                            ),
                            severity=AuditSeverity.ERROR,
                            code="report_paragraph_declared_node_not_used_in_text",
                            message=(
                                "Report paragraph declares a ClaimGraph node whose label is not "
                                "reflected in the paragraph text"
                            ),
                            node_id=node_id,
                        )
                    )
            declared_node_texts = [graph.nodes[node_id].label for node_id in known_node_ids]
            if paragraph_overlaps_any_node:
                for number_text in number_texts(paragraph.markdown):
                    if declared_node_texts and not any(
                        number_text_is_located(number_text, node_text)
                        for node_text in declared_node_texts
                    ):
                        findings.append(
                            AuditFinding(
                                finding_id=(
                                    "paragraph_number_not_grounded:"
                                    f"{paragraph.paragraph_id}:"
                                    f"{normalized_number_text(number_text)}"
                                ),
                                severity=AuditSeverity.ERROR,
                                code="report_paragraph_number_not_grounded_in_declared_nodes",
                                message=(
                                    "Report paragraph includes a numeric value that is not present "
                                    "in its declared ClaimGraph node labels"
                                ),
                                node_id=known_node_ids[0] if known_node_ids else None,
                                source_ids=known_source_ids,
                            )
                        )
            for evidence_id in known_evidence_ids:
                if not any((node_id, evidence_id) in support_edges for node_id in known_node_ids):
                    findings.append(
                        AuditFinding(
                            finding_id=(
                                "paragraph_evidence_without_declared_node:"
                                f"{paragraph.paragraph_id}:{evidence_id}"
                            ),
                            severity=AuditSeverity.ERROR,
                            code="report_paragraph_evidence_not_linked_to_declared_node",
                            message=(
                                "Report paragraph declares an evidence ID that does not support "
                                "any declared graph node"
                            ),
                            node_id=evidence_id,
                            source_ids=evidence_source_ids(graph, evidence_id),
                        )
                    )
    return findings


def evidence_source_ids(graph: ClaimGraph, evidence_id: str) -> list[str]:
    node = graph.nodes.get(evidence_id)
    if node is None:
        return []
    source_id = str(node.payload.get("source_id") or "")
    return [source_id] if source_id else []


def render_graph_report_markdown(
    *,
    title: str,
    draft: GraphReportDraft,
    graph: ClaimGraph,
    dom: PaperDOM,
    quality: dict | None = None,
    output_language: str = "zh",
) -> str:
    quality = quality or {}
    output_language = "en" if output_language == "en" else "zh"
    display_title = reader_display_title(title=title, draft=draft, dom=dom)
    lines = [
        f"# {display_title}",
        "",
        report_review_line(quality, output_language=output_language),
    ]

    anchor = report_anchor(draft, output_language=output_language)
    if anchor:
        label = "First hold this abstraction:" if output_language == "en" else "先抓住这个抽象："
        lines.extend(["", f"**{label}** {anchor}"])
    lines.append("")

    for section in draft.sections:
        section_title = reader_section_title(section, output_language=output_language)
        section_texts = [
            text
            for text in (
                reader_markdown(paragraph.markdown, output_language=output_language)
                for paragraph in section.paragraphs
            )
            if text
        ]
        if not section_texts:
            continue
        lines.extend([f"## {section_title}", ""])
        for text in section_texts:
            lines.extend([text, ""])

    boundary = report_trust_boundary(quality, output_language=output_language)
    if boundary:
        label = "Trust boundary" if output_language == "en" else "可信边界"
        lines.extend([f"{label}：{boundary}", ""])
    return "\n".join(lines).rstrip() + "\n"


def localized_section_title(
    kind: str,
    titles: dict[str, str],
    *,
    output_language: str,
) -> str:
    return titles.get("en" if output_language == "en" else "zh") or titles.get("zh") or kind


def reader_display_title(*, title: str, draft: GraphReportDraft, dom: PaperDOM) -> str:
    candidate = (title or "").strip()
    dom_title = (dom.title or "").strip()
    document_title = first_document_title(dom)
    if document_title and (
        not candidate
        or title_looks_like_filename(candidate)
        or title_looks_like_filename(dom_title)
    ):
        return document_title
    if dom_title and (not candidate or title_looks_like_filename(candidate)):
        return dom_title
    return candidate or dom_title or draft.paper_id


def first_document_title(dom: PaperDOM) -> str:
    for span in dom.spans[:80]:
        text = " ".join(str(span.text or "").split()).strip()
        if not text or len(text) < 24:
            continue
        if span.page_no and span.page_no > 2:
            break
        lower = text.lower()
        if lower.startswith(("abstract", "index terms", "keywords")):
            continue
        if "," in text and len(text.split(",")) >= 3:
            continue
        if re.fullmatch(r"[\d\s.]+", text):
            continue
        return text
    return ""


def title_looks_like_filename(title: str) -> bool:
    text = title.strip()
    return bool(
        re.search(r"(^\d+[_-])|[_]{2,}|[_].*[_]|\.pdf$", text, flags=re.IGNORECASE)
        or text.startswith("p_")
    )


def report_review_line(quality: dict, *, output_language: str) -> str:
    status = str(quality.get("publish_status") or quality.get("artifact_publish_status") or "")
    if output_language == "en":
        return f"Review: {review_status_label(status, output_language=output_language)}"
    return f"复核：{review_status_label(status, output_language=output_language)}"


def review_status_label(status: str, *, output_language: str) -> str:
    normalized = status.strip().upper()
    if output_language == "en":
        return {
            "REVIEWED": "passed",
            "REVIEWED_WITH_LIMITS": "passed with limits",
            "DRAFT_WEAK": "weak draft",
            "BLOCKED": "blocked",
        }.get(normalized, "completed")
    return {
        "REVIEWED": "已通过",
        "REVIEWED_WITH_LIMITS": "通过但有边界",
        "DRAFT_WEAK": "弱草稿",
        "BLOCKED": "未通过",
    }.get(normalized, "已完成")


def report_anchor(draft: GraphReportDraft, *, output_language: str) -> str:
    priority = ["problem", "claim", "mechanism"]
    selected: list[str] = []
    for section_id in priority:
        for section in draft.sections:
            if section.section_id != section_id:
                continue
            for paragraph in section.paragraphs:
                text = reader_markdown(paragraph.markdown, output_language=output_language)
                if text:
                    selected.append(strip_markdown_heading(text))
                    break
            break
    if not selected:
        return ""
    if output_language == "en":
        return compact_source_text(" ".join(selected), limit=360)
    return compact_source_text(" ".join(selected), limit=220)


def strip_markdown_heading(text: str) -> str:
    return re.sub(r"^\s{0,3}#{1,6}\s+", "", text.strip())


def reader_section_title(section: ReportSection, *, output_language: str) -> str:
    titles = SECTION_TITLES.get(section.section_id)
    if titles:
        return localized_section_title(section.section_id, titles, output_language=output_language)
    return clean_internal_markers(section.title).strip() or section.section_id


def reader_markdown(markdown: str, *, output_language: str = "zh") -> str:
    cleaned_lines: list[str] = []
    for line in str(markdown or "").splitlines():
        if is_internal_report_line(line):
            continue
        cleaned = clean_internal_markers(line)
        cleaned = strip_observation_prefix(cleaned)
        if output_language == "zh" and is_english_mechanical_evidence(cleaned):
            continue
        cleaned_lines.append(cleaned)
    return "\n".join(line for line in cleaned_lines if line.strip()).strip()


def strip_observation_prefix(text: str) -> str:
    return re.sub(
        r"^(问题定位|核心主张|方法机制|实现路径|评估设置|实验结果|限制边界|概念桥接|相关工作定位|可复现性)证据：\s*",
        "",
        text.strip(),
    )


def is_english_mechanical_evidence(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.lower().startswith(("abstract-", "abstract—", "to solve ", "where c ")):
        return True
    cjk = len(re.findall(r"[\u4e00-\u9fff]", stripped))
    latin = len(re.findall(r"[A-Za-z]", stripped))
    return cjk < 8 and latin > 40


def is_internal_report_line(line: str) -> bool:
    stripped = line.strip()
    if re.search(r"\b(?:problem|claim|mechanism|implementation|evaluation|result|limitation|concept):obs_", stripped):
        return True
    if "evidence:" in stripped or "span:" in stripped:
        return True
    return stripped.startswith(
        (
            "- ClaimGraph nodes:",
            "- Evidence nodes:",
            "- PaperDOM sources:",
            "ClaimGraph nodes:",
            "Evidence nodes:",
            "PaperDOM sources:",
        )
    )


def clean_internal_markers(text: str) -> str:
    cleaned = str(text or "")
    for marker in ["ClaimGraph", "PaperDOM", "source_id", "evidence_id", "observation_id"]:
        cleaned = cleaned.replace(marker, "")
    cleaned = re.sub(
        r"`?(?:problem|claim|mechanism|implementation|evaluation|result|limitation|concept):obs_[A-Za-z0-9_:-]+`?",
        "",
        cleaned,
    )
    cleaned = re.sub(r"`?evidence:[A-Za-z0-9_:/.-]+`?", "", cleaned)
    cleaned = re.sub(r"`?span:[A-Za-z0-9_:/.-]+`?", "", cleaned)
    return cleaned.strip()


def report_trust_boundary(quality: dict, *, output_language: str) -> str:
    error_count = int(quality.get("current_audit_error_count") or 0)
    warning_count = int(quality.get("current_audit_warning_count") or 0)
    status = str(quality.get("publish_status") or "").upper()
    if not error_count and not warning_count and status == "REVIEWED":
        return ""
    if output_language == "en":
        if error_count:
            return "Automatic review found unresolved evidence issues; use this as a reading lead."
        return "Automatic review passed with some evidence boundaries."
    if error_count:
        return "自动复核发现未解决的证据问题，这份报告只能作为阅读线索。"
    return "自动复核已通过，但仍建议在引用关键数值、实现细节或强结论前回到原文核对。"


def graph_report_source_ids(*, graph: ClaimGraph, evidence_ids: list[str]) -> list[str]:
    source_ids = []
    for evidence_id in evidence_ids:
        for source_id in evidence_source_ids(graph, evidence_id):
            if source_id not in source_ids:
                source_ids.append(source_id)
    return source_ids


def inline_code_list(values: list[str]) -> str:
    return ", ".join(f"`{value}`" for value in values) if values else "`none`"


def describe_dom_source(dom: PaperDOM, source_id: str) -> str:
    for span in dom.spans:
        if span.source_id == source_id:
            return (
                f"`{source_id}` ({span.kind}, page {span.page_no}): "
                f"{compact_source_text(span.text)}"
            )
    for figure in dom.figures:
        if figure.source_id == source_id:
            return (
                f"`{source_id}` ({figure.kind}, page {figure.page_no}): "
                f"{compact_source_text(figure.caption or '')}"
            )
    for table in dom.tables:
        if table.source_id == source_id:
            return (
                f"`{source_id}` ({table.kind}, page {table.page_no}): "
                f"{compact_source_text(table.caption or '')}"
            )
    for equation in dom.equations:
        if equation.source_id == source_id:
            return (
                f"`{source_id}` ({equation.kind}, page {equation.page_no}): "
                f"{compact_source_text(equation.latex_or_text)}"
            )
    for section in dom.sections:
        if section.source_id == source_id:
            return (
                f"`{source_id}` ({section.kind}, page {section.page_no}): "
                f"{compact_source_text(section.title)}"
            )
    return ""


def compact_source_text(text: str, *, limit: int = 260) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def number_texts(text: str) -> list[str]:
    result: list[str] = []
    for match in NUMBER_TEXT_PATTERN.finditer(str(text or "")):
        value = match.group(0).strip()
        if value and value not in result:
            result.append(value)
    return result


def number_text_is_located(number_text: str, source_text: str) -> bool:
    needle = normalized_number_text(number_text)
    haystack = normalized_number_text(source_text)
    return bool(needle and needle in haystack)


def normalized_number_text(text: str) -> str:
    return re.sub(r"[\s,]+", "", str(text or "").casefold())
