from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from paperlens_core.report.overview import display_paper_title
from paperlens_core.report.rows import dedupe_evidence_refs
from paperlens_core.report.text import (
    clean_model_inline_text,
    clean_model_markdown,
    compact_compare_text,
    compact_reason,
    readable_model_body,
    recommendation_for_grade,
    sanitize_reader_hostile_text,
    user_facing_uncertainty_note,
)
from paperlens_core.schemas import ClassificationDecision, PaperCard, PaperRecord, SkimCard


def render_paper_report(
    *,
    paper: PaperRecord,
    skim: SkimCard | None,
    decision: ClassificationDecision | None,
    card: PaperCard | None,
    layout: dict[str, Any],
    topic: str | None,
    idea: str | None,
    formal_run: bool,
    model_report: dict[str, Any] | None,
    report_audit: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    output_language: str = "zh",
) -> str:
    _ = (topic, idea)
    if not formal_run:
        return render_debug_paper_diagnostic(
            paper=paper,
            skim=skim,
            decision=decision,
            card=card,
            layout=layout,
        )
    if not model_report:
        raise RuntimeError(f"Missing model-generated final report for {paper.paper_id}")
    return render_freeform_paper_report(
        paper=paper,
        decision=decision,
        card=card,
        layout=layout,
        model_report=model_report,
        report_audit=report_audit,
        output_dir=output_dir,
        output_language=output_language,
    )


def render_freeform_paper_report(
    *,
    paper: PaperRecord,
    decision: ClassificationDecision | None,
    card: PaperCard | None = None,
    layout: dict[str, Any] | None = None,
    model_report: dict[str, Any],
    report_audit: dict[str, Any] | None,
    output_dir: Path | None = None,
    output_language: str = "zh",
) -> str:
    grade = _string_or_none(model_report.get("grade")) or (
        decision.class_label if decision else "HOLD"
    )
    recommendation = localized_recommendation(
        _string_or_none(model_report.get("read_recommendation"))
        or recommendation_for_grade(grade),
        output_language=output_language,
    )
    review_status = display_review_status(
        model_report, report_audit, output_language=output_language
    )
    reason = compact_reason(
        sanitize_reader_hostile_text(clean_model_inline_text(model_report.get("one_line_reason")))
        or "模型没有给出一句话理由。"
    )
    core_takeaway = sanitize_reader_hostile_text(
        clean_model_markdown(model_report.get("core_takeaway"))
    )
    body = (
        sanitize_reader_hostile_text(readable_model_body(model_report.get("explanation_markdown")))
        or "模型没有给出可用讲解。"
    )
    body = trim_redundant_body_opening(body, core_takeaway)
    uncertainty = sanitize_reader_hostile_text(
        user_facing_uncertainty_note(model_report.get("uncertainty_note"))
    )
    visual_markdown = render_key_visual_crops(
        paper=paper,
        model_report=model_report,
        card=card,
        layout=layout or {},
        output_dir=output_dir,
        output_language=output_language,
    )
    trust_boundary = report_trust_boundary(report_audit, output_language=output_language)
    labels = report_display_labels(output_language)
    lines = [
        f"# {display_paper_title(paper)}",
        "",
        labels["meta"].format(
            grade=grade, review_status=review_status, recommendation=recommendation
        ),
    ]
    if reason and not body_starts_with_reason(body, reason):
        lines.extend(["", f"> {reason}"])
    if core_takeaway:
        lines.extend(["", f"**{labels['core_anchor']}** {core_takeaway}"])
    lines.extend(["", body.strip()])
    if visual_markdown:
        lines.extend(["", visual_markdown])
    if uncertainty:
        lines.extend(["", f"{labels['uncertainty']}：{uncertainty}"])
    if trust_boundary:
        lines.extend(["", f"{labels['trust_boundary']}：{trust_boundary}"])
    return "\n".join(lines).rstrip() + "\n"


def report_display_labels(output_language: str) -> dict[str, str]:
    if output_language == "en":
        return {
            "meta": "Grade: {grade} · Review: {review_status} · Recommendation: {recommendation}",
            "core_anchor": "First hold this abstraction:",
            "uncertainty": "Uncertainty",
            "trust_boundary": "Trust boundary",
        }
    return {
        "meta": "等级：{grade} · 复核：{review_status} · 建议：{recommendation}",
        "core_anchor": "先抓住这个抽象：",
        "uncertainty": "不确定",
        "trust_boundary": "可信边界",
    }


def localized_recommendation(recommendation: str, *, output_language: str) -> str:
    if output_language != "en":
        return recommendation
    return {
        "重点关注": "high priority",
        "标准读": "standard read",
        "低优先级": "lower priority",
        "需确认": "needs confirmation",
    }.get(recommendation, recommendation)


def report_trust_boundary(
    report_audit: dict[str, Any] | None, *, output_language: str = "zh"
) -> str:
    if not report_audit or report_audit.get("verdict") == "PASS":
        return ""
    if report_audit.get("verdict") == "PASS_WITH_WEAKNESSES":
        if output_language == "en":
            return (
                "This capsule passed review with evidence boundaries; ask follow-up questions or "
                "check the source before citing exact numbers, broad extrapolations, or implementation details."
            )
        return "这份胶囊已经过复核，但仍存在证据边界；具体数值、外推结论和实现细节建议按需追问或回到原文核对。"
    if report_audit.get("verdict") == "NEED_HUMAN_REVIEW":
        if output_language == "en":
            return "This capsule did not pass automatic review and should only be used as a reading lead."
        return "这份胶囊未通过自动复核，只能作为阅读线索，不能直接当作可靠结论。"
    return ""


def trim_redundant_body_opening(body: str, core_takeaway: str) -> str:
    if not body or not core_takeaway:
        return body
    paragraphs = body.split("\n\n")
    if not paragraphs:
        return body
    first = paragraphs[0]
    sentences = split_report_sentences(first)
    if len(sentences) < 2:
        return body
    removed = 0
    while sentences and removed < 2 and sentence_overlaps_anchor(sentences[0], core_takeaway):
        sentences.pop(0)
        removed += 1
    if (
        removed
        and sentences
        and clean_model_inline_text(sentences[0]).startswith(
            ("理解了这个", "有了这个", "在这个基础上")
        )
    ):
        sentences.pop(0)
    if not removed or not sentences:
        return body
    paragraphs[0] = "".join(sentences).strip()
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph.strip())


def split_report_sentences(paragraph_text: str) -> list[str]:
    parts = re.findall(r"[^。！？.!?]+[。！？.!?]?", paragraph_text.strip())
    return [part for part in parts if part]


def sentence_overlaps_anchor(sentence: str, anchor: str) -> bool:
    sentence_chars = {char for char in sentence if "\u4e00" <= char <= "\u9fff"}
    anchor_chars = {char for char in anchor if "\u4e00" <= char <= "\u9fff"}
    if sentence_chars and anchor_chars:
        overlap = len(sentence_chars & anchor_chars) / max(1, len(sentence_chars))
        if overlap >= 0.45:
            return True
    sentence_terms = set(re.findall(r"[A-Za-z0-9_+-]{3,}", sentence.lower()))
    anchor_terms = set(re.findall(r"[A-Za-z0-9_+-]{3,}", anchor.lower()))
    if sentence_terms and anchor_terms:
        return len(sentence_terms & anchor_terms) / max(1, len(sentence_terms)) >= 0.5
    return False


def render_key_visual_crops(
    *,
    paper: PaperRecord,
    model_report: dict[str, Any],
    card: PaperCard | None,
    layout: dict[str, Any],
    output_dir: Path | None,
    output_language: str = "zh",
) -> str:
    if output_dir is None:
        return ""
    pages = select_key_visual_pages(
        paper=paper,
        model_report=model_report,
        card=card,
        layout=layout,
        limit=3,
        output_language=output_language,
    )
    if not pages:
        return ""
    pages_by_no = layout_pages_by_no(layout)
    visuals: list[dict[str, Any]] = []
    for page in pages:
        page_no = page["page_no"]
        layout_page = pages_by_no.get(page_no)
        if not layout_page:
            continue
        bbox = visual_crop_bbox_for_page(layout_page)
        if not bbox:
            continue
        image_path = render_visual_crop(
            output_dir=output_dir,
            paper=paper,
            page_no=page_no,
            bbox=bbox,
            visual_index=len(visuals) + 1,
        )
        if not image_path:
            continue
        fallback_reason = (
            f"Page {page_no} contains key visual evidence."
            if output_language == "en"
            else f"第 {page_no} 页包含关键视觉证据。"
        )
        reason = clean_model_inline_text(page.get("reason")) or fallback_reason
        visuals.append({"page_no": page_no, "reason": reason, "image_path": image_path})
    if not visuals:
        return ""
    lines = ["## Key Figures" if output_language == "en" else "## 关键图表"]
    for visual in visuals:
        page_no = visual["page_no"]
        reason = visual_reader_reason(visual["reason"], output_language=output_language)
        image_path = visual["image_path"]
        caption_prefix = (
            f"Page {page_no} crop" if output_language == "en" else f"第 {page_no} 页裁剪"
        )
        lines.extend(
            [
                "",
                "<figure>",
                f'  <img src="{image_path}" alt="{display_paper_title(paper)} visual crop from page {page_no}" width="720">',
                f"  <figcaption>{caption_prefix}: {reason}</figcaption>",
                "</figure>",
            ]
        )
    return "\n".join(lines)


def layout_pages_by_no(layout: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        page.get("page_no"): page
        for page in layout.get("pages", [])
        if isinstance(page, dict) and isinstance(page.get("page_no"), int)
    }


def visual_crop_bbox_for_page(page: dict[str, Any]) -> list[float] | None:
    page_width = positive_float(page.get("page_width"))
    page_height = positive_float(page.get("page_height"))
    if page_width is None or page_height is None:
        return None
    page_area = page_width * page_height
    if page_area <= 0:
        return None
    base_bboxes: list[list[float]] = []
    captions = [item for item in page.get("captions") or [] if isinstance(item, dict)]
    for kind in ["figures", "tables", "images"]:
        for item in page.get(kind) or []:
            if not isinstance(item, dict):
                continue
            bbox = valid_visual_bbox(item.get("bbox"), page_width, page_height)
            if bbox is None:
                continue
            base_bboxes.append(bbox)
    candidates: list[tuple[float, list[float]]] = []
    for bbox in merge_visual_bbox_groups(base_bboxes, page_width, page_height):
        merged = merge_nearby_caption_bboxes(bbox, captions, page_width, page_height)
        crop = expand_visual_bbox(merged, page_width, page_height)
        if not crop or not visual_bbox_is_reportable(crop, page_area):
            continue
        candidates.append((visual_bbox_area(crop), crop))
    if not candidates:
        for bbox in base_bboxes:
            merged = merge_nearby_caption_bboxes(bbox, captions, page_width, page_height)
            crop = expand_visual_bbox(merged, page_width, page_height)
            if not crop or not visual_bbox_is_reportable(crop, page_area):
                continue
            candidates.append((visual_bbox_area(crop), crop))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def render_visual_crop(
    *,
    output_dir: Path,
    paper: PaperRecord,
    page_no: int,
    bbox: list[float],
    visual_index: int,
) -> str | None:
    pdf_path = Path(paper.file_path)
    if not pdf_path.exists():
        return None
    try:
        from paperlens_core.pdf.pymupdf_parser import require_pymupdf

        fitz = require_pymupdf()
        with fitz.open(pdf_path) as doc:
            if page_no < 1 or page_no > len(doc):
                return None
            page = doc[page_no - 1]
            clipped = [
                max(0.0, min(float(bbox[0]), float(page.rect.width))),
                max(0.0, min(float(bbox[1]), float(page.rect.height))),
                max(0.0, min(float(bbox[2]), float(page.rect.width))),
                max(0.0, min(float(bbox[3]), float(page.rect.height))),
            ]
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                return None
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(2, 2), clip=fitz.Rect(*clipped), alpha=False
            )
            if pixmap.width < 120 or pixmap.height < 80:
                return None
            figures_dir = output_dir / ".paperlens" / "figures" / paper.paper_id
            figures_dir.mkdir(parents=True, exist_ok=True)
            filename = f"page_{page_no:04d}_visual_{visual_index:02d}.png"
            crop_path = figures_dir / filename
            pixmap.save(crop_path)
            return f"../.paperlens/figures/{paper.paper_id}/{filename}"
    except Exception:
        return None


def positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def valid_visual_bbox(value: Any, page_width: float, page_height: float) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    bbox = [
        max(0.0, min(bbox[0], page_width)),
        max(0.0, min(bbox[1], page_height)),
        max(0.0, min(bbox[2], page_width)),
        max(0.0, min(bbox[3], page_height)),
    ]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    if (bbox[2] - bbox[0]) < 24 or (bbox[3] - bbox[1]) < 24:
        return None
    page_area = page_width * page_height
    if page_area <= 0 or visual_bbox_area(bbox) / page_area > 0.65:
        return None
    return bbox


def valid_caption_bbox(value: Any, page_width: float, page_height: float) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    bbox = [
        max(0.0, min(bbox[0], page_width)),
        max(0.0, min(bbox[1], page_height)),
        max(0.0, min(bbox[2], page_width)),
        max(0.0, min(bbox[3], page_height)),
    ]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    if (bbox[2] - bbox[0]) < 12 or (bbox[3] - bbox[1]) < 6:
        return None
    page_area = page_width * page_height
    if page_area <= 0 or visual_bbox_area(bbox) / page_area > 0.35:
        return None
    return bbox


def merge_visual_bbox_groups(
    bboxes: list[list[float]],
    page_width: float,
    page_height: float,
) -> list[list[float]]:
    groups: list[list[float]] = []
    gap = max(24.0, min(page_width, page_height) * 0.06)
    for bbox in sorted(bboxes, key=lambda item: (item[1], item[0])):
        target_index = None
        for index, group in enumerate(groups):
            if visual_bboxes_near(group, bbox, gap=gap):
                target_index = index
                break
        if target_index is None:
            groups.append(bbox)
        else:
            groups[target_index] = union_visual_bbox(groups[target_index], bbox)
    return groups


def merge_nearby_caption_bboxes(
    bbox: list[float],
    captions: list[dict[str, Any]],
    page_width: float,
    page_height: float,
) -> list[float]:
    caption_bboxes = [
        valid_caption_bbox(caption.get("bbox"), page_width, page_height)
        for caption in captions
        if isinstance(caption, dict)
    ]
    caption_bboxes = [caption for caption in caption_bboxes if caption is not None]
    if not caption_bboxes:
        return bbox
    merged = bbox
    for caption in sorted(caption_bboxes, key=lambda item: visual_bbox_distance(bbox, item)):
        if not caption_is_near_visual(bbox, caption, page_width, page_height):
            continue
        candidate = union_visual_bbox(merged, caption)
        page_area = page_width * page_height
        if page_area <= 0 or visual_bbox_area(candidate) / page_area > 0.65:
            continue
        merged = candidate
    return merged


def caption_is_near_visual(
    bbox: list[float],
    caption: list[float],
    page_width: float,
    page_height: float,
) -> bool:
    if visual_bbox_distance(bbox, caption) <= max(140.0, page_height * 0.22):
        return True
    vertical_gap = 0.0
    if caption[1] >= bbox[3]:
        vertical_gap = caption[1] - bbox[3]
    elif bbox[1] >= caption[3]:
        vertical_gap = bbox[1] - caption[3]
    horizontal_overlap = max(0.0, min(bbox[2], caption[2]) - max(bbox[0], caption[0]))
    horizontal_span = max(1.0, min(bbox[2] - bbox[0], caption[2] - caption[0]))
    return (
        vertical_gap <= max(40.0, min(page_width, page_height) * 0.06)
        and horizontal_overlap / horizontal_span >= 0.25
    )


def visual_bboxes_near(left: list[float], right: list[float], *, gap: float) -> bool:
    merged = union_visual_bbox(left, right)
    max_area = max(visual_bbox_area(left), visual_bbox_area(right))
    if max_area and visual_bbox_area(merged) > max_area * 4.5:
        return False
    return not (
        left[2] + gap < right[0]
        or right[2] + gap < left[0]
        or left[3] + gap < right[1]
        or right[3] + gap < left[1]
    )


def expand_visual_bbox(
    bbox: list[float], page_width: float, page_height: float
) -> list[float] | None:
    margin = 10.0
    expanded = [
        max(0.0, bbox[0] - margin),
        max(0.0, bbox[1] - margin),
        min(page_width, bbox[2] + margin),
        min(page_height, bbox[3] + margin),
    ]
    return expanded if expanded[2] > expanded[0] and expanded[3] > expanded[1] else None


def visual_bbox_is_reportable(bbox: list[float], page_area: float) -> bool:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width < 36 or height < 36:
        return False
    ratio = visual_bbox_area(bbox) / page_area if page_area else 1.0
    return 0.002 <= ratio <= 0.65


def visual_bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def union_visual_bbox(left: list[float], right: list[float]) -> list[float]:
    return [
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    ]


def visual_bbox_distance(left: list[float], right: list[float]) -> float:
    left_center_x = (left[0] + left[2]) / 2
    right_center_x = (right[0] + right[2]) / 2
    horizontal = abs(left_center_x - right_center_x)
    if right[1] >= left[3]:
        vertical = right[1] - left[3]
    elif left[1] >= right[3]:
        vertical = left[1] - right[3]
    else:
        vertical = 0.0
    return vertical * 4 + horizontal


def visual_reader_reason(reason: str, *, output_language: str = "zh") -> str:
    cleaned = clean_model_inline_text(reason)
    if not cleaned:
        if output_language == "en":
            return "This visual helps ground the paper's key mechanism or evidence."
        return "看这页可以辅助理解论文的关键机制或证据。"
    if re.match(r"^(Figure|Fig\.?|Table)\s+\w+", cleaned, flags=re.IGNORECASE):
        return visual_caption_to_reader_reason(cleaned, output_language=output_language)
    return cleaned


def visual_caption_to_reader_reason(caption: str, *, output_language: str = "zh") -> str:
    cleaned = clean_model_inline_text(caption)
    lowered = cleaned.lower()
    if "architecture" in lowered or "overview" in lowered or "system" in lowered:
        if output_language == "en":
            return "This visual is useful for understanding the system structure and component relationships."
        return "这张图适合用来建立系统结构和组件关系的直觉。"
    if (
        "evaluation" in lowered
        or "result" in lowered
        or "throughput" in lowered
        or "latency" in lowered
    ):
        if output_language == "en":
            return "This visual is useful for checking metrics, baselines, and the boundary of the results."
        return "这张图适合用来核对实验指标、基线对比和结论边界。"
    if "algorithm" in lowered or "example" in lowered or "illustration" in lowered:
        if output_language == "en":
            return "This visual is useful for seeing how the mechanism works step by step."
        return "这张图适合用来理解论文机制如何一步步工作。"
    if output_language == "en":
        return "This visual helps ground the paper's key mechanism or evidence."
    return "这张图适合辅助理解论文的关键机制或证据。"


def select_key_visual_pages(
    *,
    paper: PaperRecord,
    model_report: dict[str, Any],
    card: PaperCard | None,
    layout: dict[str, Any],
    limit: int,
    output_language: str = "zh",
) -> list[dict[str, Any]]:
    _ = paper
    pages_by_no = {
        page.get("page_no"): page
        for page in layout.get("pages", [])
        if isinstance(page, dict) and isinstance(page.get("page_no"), int)
    }
    selected: list[dict[str, Any]] = []

    def visual_reason(page_no: int, fallback: str) -> str:
        page = pages_by_no.get(page_no) or {}
        captions = page.get("captions") if isinstance(page.get("captions"), list) else []
        figures = page.get("figures") if isinstance(page.get("figures"), list) else []
        tables = page.get("tables") if isinstance(page.get("tables"), list) else []
        visual_notes = (
            page.get("visual_notes") if isinstance(page.get("visual_notes"), list) else []
        )
        for item in captions + figures + tables + visual_notes:
            if not isinstance(item, dict):
                continue
            text = clean_model_inline_text(
                item.get("text")
                or item.get("caption")
                or item.get("visual_summary")
                or item.get("summary")
            )
            if text:
                return visual_caption_to_reader_reason(text, output_language=output_language)
        return fallback

    def add(page_no: Any, reason: str, *, prefer_reason: bool = False) -> None:
        if not isinstance(page_no, int) or page_no <= 0:
            return
        if any(item["page_no"] == page_no for item in selected):
            return
        selected_reason = (
            compact_reason(clean_model_inline_text(reason), max_chars=180) if prefer_reason else ""
        )
        selected.append(
            {"page_no": page_no, "reason": selected_reason or visual_reason(page_no, reason)}
        )

    for item in model_report.get("key_visual_pages") or []:
        if not isinstance(item, dict):
            continue
        add(
            item.get("page_no"),
            clean_model_inline_text(item.get("reason"))
            or (
                "PaperLens selected this page as useful visual evidence."
                if output_language == "en"
                else "模型认为这页有助于理解论文。"
            ),
            prefer_reason=True,
        )
        if len(selected) >= limit:
            return selected[:limit]

    if card:
        for ref in card.evidence_refs:
            if ref.figure_id or ref.table_id or ref.bbox:
                add(
                    ref.page_no,
                    "This page contains figure/table evidence that supports the capsule."
                    if output_language == "en"
                    else "这页包含支撑正文理解的图表或版面证据。",
                )
            if len(selected) >= limit:
                return selected[:limit]

    scored: list[tuple[int, int]] = []
    for page_no, page in pages_by_no.items():
        score = 0
        if page.get("figures"):
            score += 3
        if page.get("tables"):
            score += 3
        if page.get("captions"):
            score += 2
        if page.get("visual_notes"):
            score += 2
        text = _normalize_for_search(str(page.get("text") or ""))
        if any(
            term in text
            for term in [
                "figure",
                "fig.",
                "table",
                "overview",
                "architecture",
                "evaluation",
                "result",
            ]
        ):
            score += 1
        if score:
            scored.append((score, page_no))
    scored.sort(key=lambda item: (-item[0], item[1]))
    for _score, page_no in scored:
        add(
            page_no,
            "This page contains visual or tabular evidence worth viewing with the capsule."
            if output_language == "en"
            else "这页包含图表、表格或关键版面信息，适合和正文一起查看。",
        )
        if len(selected) >= limit:
            break
    return selected[:limit]


def body_starts_with_reason(body: str, reason: str) -> bool:
    body_key = compact_compare_text(body)
    reason_key = compact_compare_text(reason)
    if not body_key or not reason_key:
        return False
    prefix_len = min(max(40, len(reason_key) // 2), len(reason_key))
    return body_key.startswith(reason_key[:prefix_len])


def display_review_status(
    model_report: dict[str, Any],
    report_audit: dict[str, Any] | None,
    output_language: str = "zh",
) -> str:
    verdict = _string_or_none(report_audit.get("verdict")) if report_audit else None
    if output_language == "en":
        if verdict == "PASS":
            return "reviewed"
        if verdict == "PASS_WITH_WEAKNESSES":
            return "reviewed with evidence boundaries"
        if verdict == "NEED_HUMAN_REVIEW":
            return "needs review"
        raw = _string_or_none(model_report.get("review_status"))
        if raw == "格式归一化":
            return "normalized"
        return raw or "not reviewed"
    if verdict == "PASS":
        return "已复核"
    if verdict == "PASS_WITH_WEAKNESSES":
        return "已复核（有证据边界）"
    if verdict == "NEED_HUMAN_REVIEW":
        return "需复查"
    raw = _string_or_none(model_report.get("review_status"))
    if raw == "格式归一化":
        return "已归一化"
    return raw or "未复核"


def render_debug_paper_diagnostic(
    *,
    paper: PaperRecord,
    skim: SkimCard | None,
    decision: ClassificationDecision | None,
    card: PaperCard | None,
    layout: dict[str, Any],
) -> str:
    pages = layout.get("pages") if isinstance(layout.get("pages"), list) else []
    metrics = layout.get("metrics") if isinstance(layout.get("metrics"), dict) else {}
    refs = dedupe_evidence_refs(
        (skim.evidence_refs if skim else []) + (card.evidence_refs if card else [])
    )
    metric_bits = [f"{key}={value}" for key, value in sorted(metrics.items())[:6]]
    metrics_summary = "；".join(metric_bits) if metric_bits else "未记录解析指标"
    lines = [
        f"# {paper.canonical_title or paper.paper_id}",
        "",
        "离线调试模式只检查 PDF 解析、页面渲染和证据链路，不会生成论文价值判断或正式总结。",
        "",
        f"Paper ID：`{paper.paper_id}`；解析质量：`{paper.parse_quality or 'unknown'}`；页数：{len(pages) or paper.page_count}；初始等级：`{decision.class_label if decision else 'missing'}`。",
        "",
        f"解析信号：{metrics_summary}。",
        "",
        f"已连接的证据引用数：{len(refs)}。如果要生成真正的论文讲解，请用模型 provider 运行正式流程。",
    ]
    return "\n".join(lines) + "\n"


def _string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"\s+", " ", value).strip()
    return cleaned or None


def _normalize_for_search(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()
