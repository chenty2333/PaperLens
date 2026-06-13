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
    "principle_method": {"zh": "原理与方法", "en": "Principle and Method"},
    "evaluation_setup": {"zh": "实验设置", "en": "Evaluation Setup"},
    "results": {"zh": "结果与消融", "en": "Results and Ablations"},
    "limitations": {"zh": "局限与边界", "en": "Limitations"},
    # Legacy section ids are accepted so older drafts can still be rendered safely.
    "claim": {"zh": "核心主张与贡献", "en": "Claims and Contributions"},
    "mechanism": {"zh": "方法机制", "en": "Mechanism"},
    "implementation": {"zh": "实现细节与训练流程", "en": "Implementation and Training"},
    "evaluation": {"zh": "实验设置", "en": "Evaluation Setup"},
    "result": {"zh": "结果与结论", "en": "Results"},
    "limitation": {"zh": "局限与边界", "en": "Limitations"},
    "concept": {"zh": "相关概念", "en": "Concept Bridge"},
}
READER_SECTION_ORDER = (
    "problem",
    "principle_method",
    "evaluation_setup",
    "results",
    "limitations",
)
PRINCIPLE_METHOD_KINDS = {"claim", "mechanism", "implementation", "concept"}

NUMBER_TEXT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:\s?%|[A-Za-z]{1,6})?"
)
READER_HOSTILE_PHRASES = (
    "你给到",
    "你提供",
    "供给的片段",
    "供给片段",
    "供给的图示",
    "证据包",
    "提供的页面",
    "提供的材料",
    "提供的证据",
    "supplied excerpts",
    "provided excerpts",
    "provided excerpt",
    "the user provided",
    "evidence pack",
)
RESULT_OR_ABLATION_MARKERS = (
    "ablation",
    "outperform",
    "outperforms",
    "outperformed",
    "improvement",
    "improves",
    "improved",
    "better than",
    "higher than",
    "lower than",
    "reduce",
    "reduces",
    "reduced",
    "increase",
    "increases",
    "increased",
    "achieve",
    "achieves",
    "achieved",
    "performance gain",
    "state-of-the-art",
    "sota",
    "best result",
    "性能",
    "结果",
    "消融",
    "提升",
    "改进",
    "优于",
    "超过",
    "降低",
    "下降",
    "达到",
)
QUANTITATIVE_RESULT_CONTEXT = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "bleu",
    "rouge",
    "mae",
    "mse",
    "error",
    "latency",
    "throughput",
    "score",
    "metric",
    "measure",
    "benchmark",
    "result",
    "performance",
    "准确率",
    "召回",
    "误差",
    "延迟",
    "吞吐",
    "分数",
    "指标",
    "度量",
    "基准",
    "表现",
)


def fallback_node_belongs_to_results(section_id: str, text: str) -> bool:
    return section_id == "evaluation_setup" and looks_like_result_or_ablation(text)


def looks_like_result_or_ablation(text: str) -> bool:
    lowered = str(text or "").lower()
    if any(marker in lowered for marker in RESULT_OR_ABLATION_MARKERS):
        return True
    return bool(NUMBER_TEXT_PATTERN.search(text or "")) and any(
        marker in lowered for marker in QUANTITATIVE_RESULT_CONTEXT
    )


def build_report_draft_from_graph(
    graph: ClaimGraph, *, max_nodes_per_kind: int | None = None, output_language: str = "zh"
) -> GraphReportDraft:
    sections: list[ReportSection] = []
    for section_id, kinds in {
        "problem": {"problem"},
        "principle_method": PRINCIPLE_METHOD_KINDS,
        "evaluation_setup": {"evaluation"},
        "results": {"result"},
        "limitations": {"limitation"},
    }.items():
        titles = SECTION_TITLES[section_id]
        nodes = [
            node
            for node in graph.nodes.values()
            if node.kind in kinds and not fallback_node_belongs_to_results(section_id, node.label)
        ]
        if max_nodes_per_kind is not None:
            nodes = nodes[: max(0, max_nodes_per_kind)]
        if section_id == "results":
            nodes.extend(
                node
                for node in graph.nodes.values()
                if node.kind == "evaluation" and looks_like_result_or_ablation(node.label)
            )
            if max_nodes_per_kind is not None:
                nodes = nodes[: max(0, max_nodes_per_kind)]
        if not nodes:
            continue
        paragraphs = []
        for index, node in enumerate(nodes, start=1):
            evidence_ids = graph.evidence_ids_for(node.node_id)
            paragraphs.append(
                ReportParagraph(
                    paragraph_id=f"{section_id}_{index:02d}",
                    markdown=node.label,
                    used_node_ids=[node.node_id],
                    used_evidence_ids=evidence_ids,
                )
            )
        sections.append(
            ReportSection(
                section_id=section_id,
                title=localized_section_title(section_id, titles, output_language=output_language),
                paragraphs=paragraphs,
            )
        )
    return GraphReportDraft(paper_id=graph.paper_id, sections=sections)


def audit_report_draft_against_graph(
    draft: GraphReportDraft,
    graph: ClaimGraph,
    dom: PaperDOM | None = None,
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
            declared_source_texts = (
                report_source_texts(dom, known_source_ids) if dom is not None else []
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
                [
                    *[graph.nodes[node_id].label for node_id in known_node_ids],
                    *declared_source_texts,
                ],
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
            declared_node_texts = [
                *[graph.nodes[node_id].label for node_id in known_node_ids],
                *declared_source_texts,
            ]
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
                                code="report_paragraph_number_not_grounded_in_declared_evidence",
                                message=(
                                    "Report paragraph includes a numeric value that is not present "
                                    "in its declared ClaimGraph node labels or evidence sources"
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


def report_source_texts(dom: PaperDOM, source_ids: list[str]) -> list[str]:
    texts: list[str] = []
    source_id_set = set(source_ids)
    for span in dom.spans:
        if span.source_id in source_id_set and span.text:
            texts.append(span.text)
    for figure in dom.figures:
        if figure.source_id in source_id_set and figure.caption:
            texts.append(figure.caption)
    for table in dom.tables:
        if table.source_id in source_id_set and table.caption:
            texts.append(table.caption)
    for equation in dom.equations:
        if equation.source_id in source_id_set and equation.latex_or_text:
            texts.append(equation.latex_or_text)
    return texts


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
    draft = coerce_reader_report_draft(draft, output_language=output_language)
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

    seen_text_keys: set[str] = set()
    for section in draft.sections:
        section_title = reader_section_title(section, output_language=output_language)
        section_texts = dedupe_reader_texts(
            [
                text
                for text in (
                    reader_markdown(paragraph.markdown, output_language=output_language)
                    for paragraph in section.paragraphs
                )
                if text
            ],
            seen_text_keys=seen_text_keys,
        )
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


def coerce_reader_report_draft(
    draft: GraphReportDraft,
    *,
    output_language: str,
) -> GraphReportDraft:
    buckets: dict[str, list[ReportParagraph]] = {section_id: [] for section_id in READER_SECTION_ORDER}
    for section_index, section in enumerate(draft.sections):
        default_bucket = reader_section_bucket(section) or positional_reader_section_bucket(
            section, section_index
        )
        for paragraph in section.paragraphs:
            bucket = default_bucket
            text = paragraph.markdown
            if bucket == "principle_method" and is_detached_concept_note(text):
                continue
            if bucket in buckets:
                buckets[bucket].append(paragraph)

    sections: list[ReportSection] = []
    for section_id in READER_SECTION_ORDER:
        paragraphs = buckets.get(section_id) or []
        if not paragraphs:
            continue
        paragraphs = [
            paragraph.model_copy(update={"paragraph_id": f"{section_id}_{index:02d}"})
            for index, paragraph in enumerate(paragraphs, start=1)
        ]
        sections.append(
            ReportSection(
                section_id=section_id,
                title=localized_section_title(
                    section_id,
                    SECTION_TITLES[section_id],
                    output_language=output_language,
                ),
                paragraphs=paragraphs,
            )
        )
    return draft.model_copy(update={"sections": sections or draft.sections})


def positional_reader_section_bucket(section: ReportSection, section_index: int) -> str | None:
    text = f"{section.section_id} {section.title}".strip().lower()
    if re.fullmatch(r"(?:section[_ -]?)?\d+", section.section_id.strip().lower()) or re.fullmatch(
        r"(?:section[_ -]?)?\d+", section.title.strip().lower()
    ):
        if 0 <= section_index < len(READER_SECTION_ORDER):
            return READER_SECTION_ORDER[section_index]
    if text in {"section_01 section_01", "section 01 section 01"}:
        return "problem"
    return None


def reader_section_bucket(section: ReportSection) -> str | None:
    section_id = section.section_id.lower().strip()
    title = section.title.lower().strip()
    if section_id in {"problem", "motivation"} or "问题" in title or "动机" in title:
        return "problem"
    if section_id in {
        "claim",
        "claims",
        "mechanism",
        "method",
        "methodology",
        "implementation",
        "concept",
        "concept_bridge",
        "principle_method",
    }:
        return "principle_method"
    if any(marker in title for marker in ("主张", "贡献", "机制", "方法", "实现", "概念", "原理")):
        return "principle_method"
    if section_id in {"evaluation", "evaluation_setup", "experiment", "experiments"}:
        return "evaluation_setup"
    if any(marker in title for marker in ("实验设置", "评估设置")):
        return "evaluation_setup"
    if section_id in {"result", "results"}:
        return "results"
    if any(marker in title for marker in ("结果", "消融", "结论")):
        return "results"
    if section_id in {"limitation", "limitations"}:
        return "limitations"
    if any(marker in title for marker in ("限制", "局限", "边界")):
        return "limitations"
    return None


def is_detached_concept_note(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return True
    concept_only_markers = ("其核心概念是", "相关概念", "读懂论文必须掌握")
    if any(marker in stripped for marker in concept_only_markers) and not any(
        marker in stripped for marker in ("具体实现", "该模块", "该方法", "通过", "因此")
    ):
        return True
    return False


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
            "DRAFT_WEAK": "draft with evidence limits",
            "BLOCKED": "needs evidence check",
        }.get(normalized, "completed")
    return {
        "REVIEWED": "已通过",
        "REVIEWED_WITH_LIMITS": "通过但有边界",
        "DRAFT_WEAK": "可读草稿",
        "BLOCKED": "需要核对证据",
    }.get(normalized, "已完成")


def report_anchor(draft: GraphReportDraft, *, output_language: str) -> str:
    priority = ["problem", "principle_method", "claim", "mechanism"]
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
        cleaned = normalize_reader_loss_symbols(cleaned)
        cleaned = strip_observation_prefix(cleaned)
        if output_language == "zh" and is_english_mechanical_evidence(cleaned):
            continue
        cleaned_lines.append(cleaned)
    return "\n".join(line for line in cleaned_lines if line.strip()).strip()


def normalize_reader_loss_symbols(text: str) -> str:
    return re.sub(
        r"(?<![A-Za-z0-9_])L\s*[_\-\s]+([A-Za-z][A-Za-z0-9]{1,16})(?![A-Za-z0-9_])",
        lambda match: f"L_{match.group(1)}",
        text,
        flags=re.IGNORECASE,
    )


def dedupe_reader_texts(texts: list[str], *, seen_text_keys: set[str]) -> list[str]:
    result = []
    for text in texts:
        key = reader_text_key(text)
        if not key:
            continue
        if key in seen_text_keys:
            continue
        if any(reader_text_similarity(key, existing) > 0.9 for existing in seen_text_keys):
            continue
        seen_text_keys.add(key)
        result.append(text)
    return result


def reader_text_key(text: str) -> str:
    cleaned = clean_internal_markers(strip_observation_prefix(text))
    return " ".join(sorted(reader_text_tokens(cleaned))[:120])


def reader_text_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for chunk in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9][a-z0-9_+./%-]{2,}", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            if len(chunk) <= 12:
                tokens.add(chunk)
            for size in (2, 3, 4):
                if len(chunk) >= size:
                    tokens.update(
                        chunk[index : index + size] for index in range(len(chunk) - size + 1)
                    )
            continue
        tokens.add(chunk.strip("._-/"))
    return {token for token in tokens if token}


def reader_text_similarity(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))


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
    if stripped.lower().startswith(
        (
            "abstract-",
            "abstract—",
            "to solve ",
            "where c ",
            "the proposed ",
            "we present ",
            "we propose ",
        )
    ):
        return True
    cjk = len(re.findall(r"[\u4e00-\u9fff]", stripped))
    latin = len(re.findall(r"[A-Za-z]", stripped))
    return cjk < 8 and latin > 24


def is_internal_report_line(line: str) -> bool:
    stripped = line.strip()
    lowered = stripped.lower()
    if any(phrase in stripped or phrase in lowered for phrase in READER_HOSTILE_PHRASES):
        return True
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
