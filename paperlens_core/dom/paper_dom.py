from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


SourceKind = Literal["section", "span", "figure", "table", "equation"]


class PaperDOMNode(BaseModel):
    source_id: str
    paper_id: str
    page_no: int | None = None
    kind: SourceKind

    @field_validator("source_id", "paper_id")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("PaperDOM identifiers cannot be blank")
        return value


class PaperSpan(PaperDOMNode):
    kind: Literal["span"] = "span"
    section_id: str | None = None
    text: str
    char_start: int | None = None
    char_end: int | None = None


class PaperSection(PaperDOMNode):
    kind: Literal["section"] = "section"
    title: str
    level: int = 1
    span_ids: list[str] = Field(default_factory=list)


class PaperFigure(PaperDOMNode):
    kind: Literal["figure"] = "figure"
    caption: str | None = None
    bbox: list[float] | None = None


class PaperTable(PaperDOMNode):
    kind: Literal["table"] = "table"
    caption: str | None = None
    bbox: list[float] | None = None


class PaperEquation(PaperDOMNode):
    kind: Literal["equation"] = "equation"
    latex_or_text: str
    section_id: str | None = None


class PaperDOM(BaseModel):
    schema_version: str = "paper_dom.v1"
    paper_id: str
    title: str | None = None
    sections: list[PaperSection] = Field(default_factory=list)
    spans: list[PaperSpan] = Field(default_factory=list)
    figures: list[PaperFigure] = Field(default_factory=list)
    tables: list[PaperTable] = Field(default_factory=list)
    equations: list[PaperEquation] = Field(default_factory=list)
    parse_warnings: list[str] = Field(default_factory=list)

    def source_ids(self) -> set[str]:
        return {
            node.source_id
            for group in [self.sections, self.spans, self.figures, self.tables, self.equations]
            for node in group
        }

    def source_exists(self, source_id: str) -> bool:
        return source_id in self.source_ids()


def stable_source_id(paper_id: str, kind: SourceKind, *parts: Any) -> str:
    safe_paper = slug_part(paper_id)
    safe_parts = [slug_part(str(part)) for part in parts if str(part).strip()]
    suffix = ":".join(safe_parts)
    return f"{kind}:{safe_paper}:{suffix}" if suffix else f"{kind}:{safe_paper}"


def build_paper_dom_from_layout(
    *,
    paper_id: str,
    title: str | None,
    layout: dict[str, Any],
) -> PaperDOM:
    pages = layout.get("pages") if isinstance(layout.get("pages"), list) else []
    sections: list[PaperSection] = []
    spans: list[PaperSpan] = []
    figures: list[PaperFigure] = []
    tables: list[PaperTable] = []
    equations: list[PaperEquation] = []
    warnings: list[str] = []

    current_section_id: str | None = None
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_no = safe_int(page.get("page_no"))
        section_candidates = [
            item for item in list_value(page.get("section_candidates")) if isinstance(item, dict)
        ]
        if section_candidates:
            candidate = section_candidates[0]
            title_text = str(candidate.get("title") or candidate.get("text") or "").strip()
            if title_text:
                current_section_id = stable_source_id(
                    paper_id, "section", f"p{page_no or 0}", len(sections) + 1
                )
                sections.append(
                    PaperSection(
                        source_id=current_section_id,
                        paper_id=paper_id,
                        page_no=page_no,
                        title=title_text,
                        level=safe_int(candidate.get("level")) or 1,
                    )
                )
        if current_section_id is None:
            current_section_id = stable_source_id(paper_id, "section", "front")
            sections.append(
                PaperSection(
                    source_id=current_section_id,
                    paper_id=paper_id,
                    page_no=page_no,
                    title="Front Matter",
                    level=1,
                )
            )
        text = str(page.get("text") or "")
        for index, paragraph in enumerate(split_paragraphs(text), start=1):
            source_id = stable_source_id(paper_id, "span", f"p{page_no or 0}", index)
            span = PaperSpan(
                source_id=source_id,
                paper_id=paper_id,
                page_no=page_no,
                section_id=current_section_id,
                text=paragraph,
            )
            spans.append(span)
            for section in reversed(sections):
                if section.source_id == current_section_id:
                    section.span_ids.append(source_id)
                    break
            equations.extend(extract_equations(paper_id, page_no, current_section_id, paragraph))
        figures.extend(extract_visual_nodes(paper_id, page_no, page, key="figures"))
        tables.extend(extract_visual_nodes(paper_id, page_no, page, key="tables"))
        if page_no is None:
            warnings.append("page_missing_number")

    if not spans:
        warnings.append("paper_dom_empty_spans")
    return PaperDOM(
        paper_id=paper_id,
        title=title,
        sections=sections,
        spans=spans,
        figures=figures,
        tables=tables,
        equations=equations,
        parse_warnings=warnings,
    )


def extract_visual_nodes(
    paper_id: str,
    page_no: int | None,
    page: dict[str, Any],
    *,
    key: Literal["figures", "tables"],
) -> list[PaperFigure] | list[PaperTable]:
    result = []
    kind: Literal["figure", "table"] = "figure" if key == "figures" else "table"
    for index, item in enumerate(list_value(page.get(key)), start=1):
        if not isinstance(item, dict):
            continue
        caption = str(item.get("caption") or item.get("text") or "").strip() or None
        bbox = item.get("bbox") if isinstance(item.get("bbox"), list) else None
        payload = {
            "source_id": stable_source_id(paper_id, kind, f"p{page_no or 0}", index),
            "paper_id": paper_id,
            "page_no": page_no,
            "caption": caption,
            "bbox": bbox,
        }
        result.append(PaperFigure(**payload) if kind == "figure" else PaperTable(**payload))
    return result


def extract_equations(
    paper_id: str,
    page_no: int | None,
    section_id: str | None,
    paragraph: str,
) -> list[PaperEquation]:
    equations = []
    for index, match in enumerate(
        re.finditer(r"(\$[^$]{2,}\$|\\\[[^\]]{2,}\\\])", paragraph), start=1
    ):
        equations.append(
            PaperEquation(
                source_id=stable_source_id(paper_id, "equation", f"p{page_no or 0}", index),
                paper_id=paper_id,
                page_no=page_no,
                section_id=section_id,
                latex_or_text=match.group(0),
            )
        )
    return equations


def split_paragraphs(text: str) -> list[str]:
    chunks = [re.sub(r"\s+", " ", item).strip() for item in re.split(r"\n\s*\n", text)]
    if len(chunks) <= 1:
        chunks = [re.sub(r"\s+", " ", item).strip() for item in text.splitlines()]
    return [item for item in chunks if item]


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def safe_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def slug_part(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_\-.]+", "_", value)
    return value.strip("_") or "unknown"
