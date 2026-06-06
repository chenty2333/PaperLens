from __future__ import annotations

from pathlib import Path
from typing import Any

from paperlens_core.library import write_paperlens_library
from paperlens_core.library_graph import read_core_v2_graph_summary
from paperlens_core.report import (
    markdown_title,
    paper_report_filename,
    render_core_graph_report_view,
    render_paperlens_report,
)
from paperlens_core.schemas import ClassificationDecision, PaperRecord, ReviewItem, SkimCard


def write_final_report_bundle(
    *,
    output_dir: Path,
    data_dir: Path,
    papers: list[PaperRecord],
    skim_cards: list[SkimCard],
    decisions: list[ClassificationDecision],
    review_items: list[ReviewItem],
    budget: dict[str, Any],
    budget_provider: Any | None = None,
    config: dict[str, Any],
    topic: str | None,
    idea: str | None,
) -> list[Path]:
    formal_run = not bool(config.get("offline_debug"))
    output_language = str(config.get("output_language") or "zh")
    if output_language not in {"zh", "en"}:
        output_language = "zh"
    read_mode = str(config.get("read_mode") or "standard")
    if read_mode != "standard":
        raise ValueError("PaperLens Core currently supports only read_mode='standard'")
    skim_by_id = {card.paper_id: card for card in skim_cards}
    decision_by_id = {decision.paper_id: decision for decision in decisions}
    paper_report_rows: list[dict[str, Any]] = []
    written: list[Path] = []

    for paper in papers:
        report_name = paper_report_filename(paper)
        report_path = output_dir / "papers" / report_name
        report_markdown = render_core_graph_report_view(
            data_dir=data_dir,
            paper_id=paper.paper_id,
            title=paper.canonical_title or paper.paper_id,
        )
        if report_markdown is None:
            raise RuntimeError(
                f"Core v2 report is required for {paper.paper_id}; export only publishes "
                "reviewed ClaimGraph reports"
            )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_markdown, encoding="utf-8")
        written.append(report_path)
        graph_summary = read_core_v2_graph_summary(output_dir, paper.paper_id)
        paper_report_rows.append(
            {
                "paper": paper,
                "skim": skim_by_id.get(paper.paper_id),
                "decision": decision_by_id.get(paper.paper_id),
                "report_name": report_name,
                "core_graph_report_name": report_name,
                "report_title": markdown_title(report_markdown) or paper.canonical_title,
                "core_v2_graph_summary": graph_summary,
            }
        )

    final_budget = budget_provider() if budget_provider else budget
    paperlens_report = render_paperlens_report(
        rows=paper_report_rows,
        review_items=review_items,
        budget=final_budget,
        topic=topic,
        idea=idea,
        formal_run=formal_run,
        output_language=output_language,
    )
    main_path = output_dir / "PaperLens.md"
    main_path.write_text(paperlens_report, encoding="utf-8")
    written.append(main_path)
    written.extend(
        write_paperlens_library(
            output_dir=output_dir,
            rows=paper_report_rows,
            topic=topic,
            idea=idea,
        )
    )
    return written
