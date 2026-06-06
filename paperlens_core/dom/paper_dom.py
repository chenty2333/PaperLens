from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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
    source_span_id: str | None = None


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

    @field_validator("paper_id")
    @classmethod
    def nonempty_paper_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("PaperDOM paper_id cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_source_address_space(self) -> "PaperDOM":
        nodes = self.all_nodes()
        seen: set[str] = set()
        duplicates: list[str] = []
        safe_paper = slug_part(self.paper_id)
        for node in nodes:
            if node.source_id in seen and node.source_id not in duplicates:
                duplicates.append(node.source_id)
            seen.add(node.source_id)
            if node.paper_id != self.paper_id:
                raise ValueError(
                    f"PaperDOM node paper_id mismatch: {node.source_id} has "
                    f"{node.paper_id} != {self.paper_id}"
                )
            expected_prefix = f"{node.kind}:{safe_paper}"
            if node.source_id != expected_prefix and not node.source_id.startswith(
                f"{expected_prefix}:"
            ):
                raise ValueError(
                    f"PaperDOM node source_id does not match kind/paper_id: {node.source_id}"
                )
        if duplicates:
            raise ValueError("PaperDOM source_id values must be unique: " + ", ".join(duplicates))

        span_ids = {span.source_id for span in self.spans}
        section_ids = {section.source_id for section in self.sections}
        missing_span_refs = sorted(
            {
                span_id
                for section in self.sections
                for span_id in section.span_ids
                if span_id not in span_ids
            }
        )
        if missing_span_refs:
            raise ValueError(
                "PaperDOM section span_ids reference missing spans: "
                + ", ".join(missing_span_refs[:8])
            )
        missing_section_refs = sorted(
            {
                section_id
                for section_id in [
                    *(span.section_id for span in self.spans),
                    *(equation.section_id for equation in self.equations),
                ]
                if section_id and section_id not in section_ids
            }
        )
        if missing_section_refs:
            raise ValueError(
                "PaperDOM nodes reference missing sections: "
                + ", ".join(missing_section_refs[:8])
            )
        missing_equation_span_refs = sorted(
            {
                equation.source_span_id
                for equation in self.equations
                if equation.source_span_id and equation.source_span_id not in span_ids
            }
        )
        if missing_equation_span_refs:
            raise ValueError(
                "PaperDOM equations reference missing spans: "
                + ", ".join(missing_equation_span_refs[:8])
            )
        return self

    def source_ids(self) -> set[str]:
        return {
            node.source_id
            for node in self.all_nodes()
        }

    def source_exists(self, source_id: str) -> bool:
        return source_id in self.source_ids()

    def all_nodes(self) -> list[PaperDOMNode]:
        return [
            node
            for group in [self.sections, self.spans, self.figures, self.tables, self.equations]
            for node in group
        ]


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
    for page_index, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            continue
        page_no = safe_int(page.get("page_no"))
        page_key = f"p{page_no}" if page_no is not None else f"page{page_index}"
        section_candidates = [
            item for item in list_value(page.get("section_candidates")) if isinstance(item, dict)
        ]
        if section_candidates:
            candidate = section_candidates[0]
            title_text = str(candidate.get("title") or candidate.get("text") or "").strip()
            if title_text:
                current_section_id = stable_source_id(
                    paper_id, "section", page_key, len(sections) + 1
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
        for index, paragraph_item in enumerate(page_text_units(page, page_key, paper_id), start=1):
            paragraph = paragraph_item["text"]
            source_id = paragraph_item["source_id"] or stable_source_id(
                paper_id, "span", page_key, index
            )
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
            equations.extend(
                extract_equations(
                    paper_id,
                    page_key,
                    page_no,
                    current_section_id,
                    paragraph_index=index,
                    paragraph=paragraph,
                    span_source_id=source_id,
                )
            )
        figures.extend(extract_visual_nodes(paper_id, page_key, page_no, page, key="figures"))
        tables.extend(extract_visual_nodes(paper_id, page_key, page_no, page, key="tables"))
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
    page_key: str,
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
            "source_id": str(item.get("source_id") or "").strip()
            or stable_source_id(paper_id, kind, page_key, index),
            "paper_id": paper_id,
            "page_no": page_no,
            "caption": caption,
            "bbox": bbox,
        }
        result.append(PaperFigure(**payload) if kind == "figure" else PaperTable(**payload))
    return result


def page_text_units(page: dict[str, Any], page_key: str, paper_id: str) -> list[dict[str, str]]:
    units = []
    for index, block in enumerate(list_value(page.get("blocks")), start=1):
        if not isinstance(block, dict):
            continue
        text = clean_text(str(block.get("text") or ""))
        if not text:
            continue
        source_id = str(block.get("source_id") or "").strip()
        units.append(
            {
                "source_id": source_id or stable_source_id(paper_id, "span", page_key, index),
                "text": text,
            }
        )
    if units:
        return units
    return [
        {
            "source_id": stable_source_id(paper_id, "span", page_key, index),
            "text": paragraph,
        }
        for index, paragraph in enumerate(split_paragraphs(str(page.get("text") or "")), start=1)
    ]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_equations(
    paper_id: str,
    page_key: str,
    page_no: int | None,
    section_id: str | None,
    *,
    paragraph_index: int,
    paragraph: str,
    span_source_id: str,
) -> list[PaperEquation]:
    equations = []
    for index, equation_text in enumerate(equation_candidates(paragraph), start=1):
        equations.append(
            PaperEquation(
                source_id=stable_source_id(
                    paper_id, "equation", page_key, f"s{paragraph_index}", index
                ),
                paper_id=paper_id,
                page_no=page_no,
                section_id=section_id,
                source_span_id=span_source_id,
                latex_or_text=equation_text,
            )
        )
    return equations


def equation_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(r"(\$\$?[^$]{2,}\$\$?|\\\[[^\]]{2,}\\\]|\\\([^\)]{2,}\\\))", text):
        add_unique_text(candidates, match.group(0))
    for line in split_equation_lines(text):
        if looks_like_equation(line):
            add_unique_text(candidates, line)
    return candidates


def split_equation_lines(text: str) -> list[str]:
    return [
        clean_text(line)
        for line in re.split(r"(?:\n|;\s+)", text)
        if clean_text(line)
    ]


def looks_like_equation(text: str) -> bool:
    normalized = clean_text(text)
    if len(normalized) < 5 or len(normalized) > 260:
        return False
    lower = normalized.lower()
    if lower.startswith(("figure ", "fig. ", "table ", "algorithm ")):
        return False
    math_markers = [
        "=",
        "\u2264",
        "\u2265",
        "\\sum",
        "\\prod",
        "\\frac",
        "\\arg",
        "\\min",
        "\\max",
        "\u2211",
        "\u220f",
        "\u221a",
        "\u2202",
        "\u2207",
        "\u2248",
        "\u2243",
        "\u2208",
        "\u2200",
    ]
    if not any(marker in normalized for marker in math_markers):
        return False
    symbol_count = len(
        re.findall(
            "[=+\\-*/^_{}()[\\]<>"
            "\\u2264\\u2265\\u2248\\u2211\\u220f\\u221a\\u2202\\u2207\\u2208\\u2200]",
            normalized,
        )
    )
    digit_or_latin = len(re.findall(r"[A-Za-z0-9]", normalized))
    word_count = len(re.findall(r"[A-Za-z]{3,}", normalized))
    if symbol_count < 2:
        return False
    if digit_or_latin == 0:
        return False
    return word_count <= 18 or symbol_count >= 4


def add_unique_text(values: list[str], text: str) -> None:
    cleaned = clean_text(text)
    if cleaned and cleaned not in values:
        values.append(cleaned)


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
