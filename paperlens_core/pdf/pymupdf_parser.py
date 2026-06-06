from __future__ import annotations

import re
import contextlib
import io
from pathlib import Path
from typing import Any

from paperlens_core.dom import stable_source_id
from paperlens_core.schemas import PageArtifact, PaperRecord

try:
    import pymupdf
except ImportError:
    pymupdf = None  # type: ignore[assignment]


def require_pymupdf() -> Any:
    if pymupdf is None:
        raise RuntimeError("PyMuPDF is not installed. Install PaperLens Core dependencies first.")
    return pymupdf


def normalize_title(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    value = re.sub(r"\s+", " ", value).strip()
    return value or fallback


def block_tuple_to_dict(block: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "bbox": [float(block[0]), float(block[1]), float(block[2]), float(block[3])],
        "text": block[4] if len(block) > 4 else "",
        "block_no": block[5] if len(block) > 5 else None,
        "block_type": block[6] if len(block) > 6 else None,
    }


def assign_block_source_ids(paper_id: str, page_no: int, blocks: list[dict[str, Any]]) -> None:
    page_key = f"p{page_no}"
    text_index = 0
    for block in blocks:
        if not str(block.get("text") or "").strip():
            continue
        text_index += 1
        block["source_id"] = stable_source_id(paper_id, "span", page_key, text_index)


def word_tuple_to_dict(word: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "bbox": [float(word[0]), float(word[1]), float(word[2]), float(word[3])],
        "text": word[4] if len(word) > 4 else "",
        "block_no": word[5] if len(word) > 5 else None,
        "line_no": word[6] if len(word) > 6 else None,
        "word_no": word[7] if len(word) > 7 else None,
    }


def valid_bbox(bbox: Any) -> list[float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def union_bbox(left: list[float], right: list[float]) -> list[float]:
    return [
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    ]


def bboxes_near(left: list[float], right: list[float], *, gap: float) -> bool:
    return not (
        left[2] + gap < right[0]
        or right[2] + gap < left[0]
        or left[3] + gap < right[1]
        or right[3] + gap < left[1]
    )


def merge_bbox_fragments(
    items: list[dict[str, Any]],
    *,
    id_key: str,
    prefix: str,
    gap: float,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    sorted_items = sorted(
        (item for item in items if valid_bbox(item.get("bbox"))),
        key=lambda item: (valid_bbox(item.get("bbox")) or [0.0, 0.0, 0.0, 0.0])[:2],
    )
    for item in sorted_items:
        bbox = valid_bbox(item.get("bbox"))
        if bbox is None:
            continue
        target: dict[str, Any] | None = None
        for group in groups:
            group_bbox = valid_bbox(group.get("bbox"))
            if group_bbox and bboxes_near(group_bbox, bbox, gap=gap):
                target = group
                break
        if target is None:
            payload = dict(item)
            payload[id_key] = f"{prefix}_{len(groups) + 1}"
            payload["bbox"] = bbox
            payload["source_count"] = 1
            payload["source_fragments"] = [item]
            groups.append(payload)
            continue
        target["bbox"] = union_bbox(valid_bbox(target["bbox"]) or bbox, bbox)
        target["source_count"] = int(target.get("source_count") or 1) + 1
        target.setdefault("source_fragments", []).append(item)
        if item.get("row_count") is not None:
            target["row_count"] = max(int(target.get("row_count") or 0), int(item["row_count"]))
        if item.get("col_count") is not None:
            target["col_count"] = max(int(target.get("col_count") or 0), int(item["col_count"]))
    return groups


def extract_images(page_dict: dict[str, Any]) -> list[dict[str, Any]]:
    fragments: list[dict[str, Any]] = []
    for index, block in enumerate(page_dict.get("blocks", []), start=1):
        if block.get("type") != 1:
            continue
        bbox = valid_bbox(block.get("bbox"))
        if bbox is None or bbox_area(bbox) < 16:
            continue
        fragments.append(
            {
                "image_id": f"img_{index}",
                "bbox": bbox,
                "width": block.get("width"),
                "height": block.get("height"),
                "ext": block.get("ext"),
            }
        )
    return merge_bbox_fragments(fragments, id_key="image_id", prefix="img", gap=18)


def extract_tables(page: Any) -> list[dict[str, Any]]:
    if not hasattr(page, "find_tables"):
        return []
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            found = page.find_tables()
    except Exception:
        return []
    tables: list[dict[str, Any]] = []
    for index, table in enumerate(getattr(found, "tables", []), start=1):
        bbox = valid_bbox(getattr(table, "bbox", None))
        if bbox is None or bbox_area(bbox) < 64:
            continue
        tables.append(
            {
                "table_id": f"tbl_{index}",
                "source_id": None,
                "bbox": bbox,
                "row_count": getattr(table, "row_count", None),
                "col_count": getattr(table, "col_count", None),
            }
        )
    return merge_bbox_fragments(tables, id_key="table_id", prefix="tbl", gap=10)


CAPTION_RE = re.compile(
    r"^(?P<kind>fig\.?|figure|table)\s*(?P<number>[A-Za-z0-9][A-Za-z0-9.\-:]*)",
    flags=re.IGNORECASE,
)


def infer_captions(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    captions = []
    for index, block in enumerate(blocks, start=1):
        text = str(block.get("text") or "").strip()
        match = CAPTION_RE.match(text)
        if match:
            kind = match.group("kind").lower()
            caption_type = "figure" if kind.startswith("fig") else "table"
            number = match.group("number").rstrip(".:")
            captions.append(
                {
                    "caption_id": f"cap_{index}",
                    "caption_type": caption_type,
                    "label": f"{caption_type} {number}".lower(),
                    "text": text,
                    "bbox": block.get("bbox"),
                    "source_id": block.get("source_id"),
                }
            )
    return captions


SECTION_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\s+)?"
    r"(abstract|introduction|background|design|implementation|evaluation|"
    r"method|methods|results|discussion|limitations|related work|conclusion|references)\b",
    flags=re.IGNORECASE,
)


def infer_section_candidates(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sections = []
    for index, block in enumerate(blocks, start=1):
        text = re.sub(r"\s+", " ", str(block.get("text") or "")).strip()
        if not text or len(text) > 120:
            continue
        if SECTION_RE.match(text) or re.match(r"^\d+(?:\.\d+)*\s+[A-Z][A-Za-z ]{2,80}$", text):
            sections.append({"section_id": f"sec_{index}", "title": text, "bbox": block.get("bbox")})
    return sections


def bbox_distance(anchor: list[float] | None, candidate: list[float] | None) -> float:
    if not anchor or not candidate:
        return float("inf")
    anchor_center_x = (anchor[0] + anchor[2]) / 2
    candidate_center_x = (candidate[0] + candidate[2]) / 2
    horizontal = abs(anchor_center_x - candidate_center_x)
    if candidate[1] >= anchor[3]:
        vertical = candidate[1] - anchor[3]
    elif anchor[1] >= candidate[3]:
        vertical = anchor[1] - candidate[3]
    else:
        vertical = 0
    return vertical * 4 + horizontal


def nearest_caption(
    item: dict[str, Any],
    captions: list[dict[str, Any]],
    caption_type: str,
) -> dict[str, Any] | None:
    bbox = valid_bbox(item.get("bbox"))
    typed = [caption for caption in captions if caption.get("caption_type") == caption_type]
    if not bbox or not typed:
        return None
    scored = [
        (bbox_distance(bbox, valid_bbox(caption.get("bbox"))), caption)
        for caption in typed
    ]
    scored.sort(key=lambda row: row[0])
    if scored and scored[0][0] < 500:
        return scored[0][1]
    return None


def infer_figures(images: list[dict[str, Any]], captions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    figures = []
    for index, image in enumerate(images, start=1):
        caption = nearest_caption(image, captions, "figure")
        figures.append(
            {
                "figure_id": f"fig_{index}",
                "source_id": None,
                "bbox": image.get("bbox"),
                "caption_id": caption.get("caption_id") if caption else None,
                "caption": caption.get("text") if caption else None,
                "label": caption.get("label") if caption else None,
                "source_image_id": image.get("image_id"),
                "source_count": image.get("source_count", 1),
            }
        )
    return figures


def attach_table_captions(
    tables: list[dict[str, Any]],
    captions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched = []
    for table in tables:
        payload = dict(table)
        caption = nearest_caption(payload, captions, "table")
        if caption:
            payload["caption_id"] = caption.get("caption_id")
            payload["caption"] = caption.get("text")
            payload["label"] = caption.get("label")
        enriched.append(payload)
    return enriched


def assign_visual_source_ids(
    paper_id: str,
    page_no: int,
    figures: list[dict[str, Any]],
    tables: list[dict[str, Any]],
) -> None:
    page_key = f"p{page_no}"
    for index, figure in enumerate(figures, start=1):
        figure["source_id"] = stable_source_id(paper_id, "figure", page_key, index)
    for index, table in enumerate(tables, start=1):
        table["source_id"] = stable_source_id(paper_id, "table", page_key, index)


def metadata_authors(value: str | None) -> list[str]:
    if not value:
        return []
    parts = re.split(r";|,|\band\b", value)
    return [part.strip() for part in parts if part.strip()]


def metadata_year(metadata: dict[str, Any]) -> int | None:
    for key in ("creationDate", "modDate", "date"):
        value = str(metadata.get(key) or "")
        match = re.search(r"(19|20)\d{2}", value)
        if match:
            return int(match.group(0))
    return None


def infer_doi(text: str) -> str | None:
    match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", text, flags=re.IGNORECASE)
    return match.group(0).rstrip(".,;") if match else None


def infer_arxiv_id(text: str) -> str | None:
    match = re.search(r"\barXiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)\b", text, flags=re.IGNORECASE)
    return match.group(1) if match else None


def infer_year_from_text(text: str) -> int | None:
    matches = [int(value) for value in re.findall(r"\b(19\d{2}|20\d{2})\b", text)]
    plausible = [value for value in matches if 1950 <= value <= 2100]
    return plausible[0] if plausible else None


def make_bibtex_key(authors: list[str], title: str | None, year: int | None, fallback: str) -> str:
    author = authors[0].split()[-1] if authors else "paper"
    title_word = "paper"
    if title:
        words = re.findall(r"[A-Za-z0-9]+", title)
        if words:
            title_word = words[0]
    key = f"{author}{year or ''}{title_word}"
    key = re.sub(r"[^A-Za-z0-9]+", "", key)
    return key[:64] or fallback


def parse_pdf(
    paper: PaperRecord,
    evidence_root: Path,
    *,
    render_zoom: float,
) -> tuple[PaperRecord, list[PageArtifact]]:
    fitz = require_pymupdf()
    pdf_path = Path(paper.file_path)
    render_dir = evidence_root / "pages" / paper.paper_id
    render_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(pdf_path) as doc:
        metadata = doc.metadata or {}
        paper.page_count = len(doc)
        paper.canonical_title = normalize_title(metadata.get("title"), pdf_path.stem)
        paper.authors = metadata_authors(metadata.get("author"))
        paper.year = metadata_year(metadata)
        paper.venue = metadata.get("subject") or paper.venue
        artifacts: list[PageArtifact] = []

        first_page_text = ""
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_no = page_index + 1
            text = page.get_text("text", sort=True)
            if page_no == 1:
                first_page_text = text
            blocks = [block_tuple_to_dict(block) for block in page.get_text("blocks", sort=True)]
            assign_block_source_ids(paper.paper_id, page_no, blocks)
            words = [word_tuple_to_dict(word) for word in page.get_text("words", sort=True)]
            page_dict = page.get_text("dict")
            images = extract_images(page_dict)
            captions = infer_captions(blocks)
            tables = attach_table_captions(extract_tables(page), captions)
            sections = infer_section_candidates(blocks)
            figures = infer_figures(images, captions)
            assign_visual_source_ids(paper.paper_id, page_no, figures, tables)

            render_path = render_dir / f"page_{page_no:04d}.png"
            matrix = fitz.Matrix(render_zoom, render_zoom)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            pixmap.save(render_path)

            visual_required = bool(images or tables) or len(text.strip()) < 300
            flags = []
            if len(text.strip()) < 100:
                flags.append("low_text")
            if images:
                flags.append("has_images")
            if tables:
                flags.append("has_tables")

            artifact = PageArtifact(
                paper_id=paper.paper_id,
                page_no=page_no,
                text=text,
                page_width=float(page.rect.width),
                page_height=float(page.rect.height),
                render_width=int(pixmap.width),
                render_height=int(pixmap.height),
                blocks=blocks,
                words=words,
                images=images,
                tables=tables,
                figures=figures,
                captions=captions,
                render_path=str(render_path),
                low_confidence_flags=flags,
                visual_required=visual_required,
                section_candidates=sections,
                crop_paths=[],
            )
            artifacts.append(artifact)

        metadata_text = "\n".join(
            value for value in [metadata.get("title"), metadata.get("author"), first_page_text[:5000]] if value
        )
        paper.doi = paper.doi or infer_doi(metadata_text)
        paper.arxiv_id = paper.arxiv_id or infer_arxiv_id(metadata_text)
        paper.year = paper.year or infer_year_from_text(metadata_text)
        paper.bibtex_key = paper.bibtex_key or make_bibtex_key(
            paper.authors,
            paper.canonical_title,
            paper.year,
            paper.paper_id,
        )

    paper.status = "PARSED"
    return paper, artifacts
