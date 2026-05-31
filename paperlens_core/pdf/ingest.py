from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from paperlens_core.pdf.fingerprint import file_sha256, paper_id_from_hash
from paperlens_core.schemas import PaperRecord


def scan_pdfs(input_dir: Path) -> list[PaperRecord]:
    paths = sorted(p for p in input_dir.rglob("*.pdf") if p.is_file())
    papers: list[PaperRecord] = []
    for path in paths:
        file_hash = file_sha256(path)
        papers.append(
            PaperRecord(
                paper_id=paper_id_from_hash(file_hash),
                file_path=str(path),
                file_hash=file_hash,
                canonical_title=path.stem,
                status="INGESTED",
            )
        )

    by_hash: dict[str, list[PaperRecord]] = defaultdict(list)
    for paper in papers:
        by_hash[paper.file_hash].append(paper)
    for group_index, group in enumerate(by_hash.values(), start=1):
        if len(group) <= 1:
            continue
        duplicate_group = f"dup_{group_index:04d}"
        for paper in group:
            paper.duplicate_group = duplicate_group
    return papers
