from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="score-quality-benchmark")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--benchmark",
        default="tests/quality_benchmark/sosp_systems.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    benchmark_path = Path(args.benchmark).resolve()
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    min_score = float(benchmark.get("min_score") or 8.0)
    reports = list((output_dir / "papers").glob("*.md"))
    memory_records = read_memory_records(output_dir)
    results = []
    failed = False
    for paper in benchmark.get("papers", []):
        result = score_one(paper, reports, memory_records, min_score=min_score)
        results.append(result)
        if result["status"] != "PASS":
            failed = True
    payload = {
        "benchmark": benchmark.get("name", benchmark_path.name),
        "output_dir": str(output_dir),
        "min_score": min_score,
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 1 if failed else 0


def score_one(
    paper: dict[str, Any],
    reports: list[Path],
    memory_records: list[dict[str, Any]],
    *,
    min_score: float,
) -> dict[str, Any]:
    title = str(paper.get("title_contains") or "").lower()
    expected_terms = [str(term) for term in paper.get("expected_terms") or []]
    forbidden_terms = [str(term) for term in paper.get("forbidden_terms") or []]
    candidates = []
    for report in reports:
        text = report.read_text(encoding="utf-8", errors="replace")
        haystack = f"{report.name}\n{text}".lower()
        if title and title in haystack:
            candidates.append((report, text))
    if not candidates:
        return {
            "title_contains": paper.get("title_contains"),
            "status": "MISSING_REPORT",
            "score": 0.0,
            "issues": ["missing_report"],
        }
    report, text = candidates[0]
    from paperlens_core.library import validate_memory_record
    from paperlens_core.quality import evaluate_capsule_quality

    quality = evaluate_capsule_quality(text, expected_terms=expected_terms)
    issues = list(quality["issues"])
    forbidden_hits = [
        term for term in forbidden_terms if term and term.lower() in text.lower()
    ]
    if forbidden_hits:
        issues.append("forbidden_terms:" + ",".join(forbidden_hits))
    record = find_memory_record(memory_records, title=title)
    memory_issues: list[str] = []
    qa_results = []
    if paper.get("require_memory", True):
        if not record:
            memory_issues.append("missing_memory_record")
        else:
            memory_issues.extend(validate_memory_record(record))
    for qa_check in paper.get("qa_checks") or []:
        if not isinstance(qa_check, dict):
            continue
        qa_result = score_qa_check(qa_check, text=text, record=record)
        qa_results.append(qa_result)
        if qa_result["status"] != "PASS":
            issues.append("qa_check_failed:" + qa_result["question"])
    if memory_issues:
        issues.extend("memory:" + issue for issue in memory_issues)
    status = "PASS" if quality["score"] >= min_score and not forbidden_hits and not memory_issues and all(
        item["status"] == "PASS" for item in qa_results
    ) else "LOW_QUALITY"
    return {
        "title_contains": paper.get("title_contains"),
        "report": str(report),
        "status": status,
        **quality,
        "issues": issues,
        "forbidden_hits": forbidden_hits,
        "memory_record": record.get("paper_id") if record else None,
        "memory_issues": memory_issues,
        "qa_results": qa_results,
    }


def read_memory_records(output_dir: Path) -> list[dict[str, Any]]:
    try:
        from paperlens_core.library import read_library_records
    except Exception:
        return []
    return read_library_records(output_dir)


def find_memory_record(records: list[dict[str, Any]], *, title: str) -> dict[str, Any] | None:
    if not title:
        return None
    for record in records:
        haystack = f"{record.get('title', '')}\n{record.get('outputs', {}).get('briefing_md', '')}".lower()
        if title in haystack:
            return record
    return None


def score_qa_check(qa_check: dict[str, Any], *, text: str, record: dict[str, Any] | None) -> dict[str, Any]:
    question = str(qa_check.get("question") or "")
    expected_terms = [str(term) for term in qa_check.get("expected_terms") or []]
    forbidden_terms = [str(term) for term in qa_check.get("forbidden_terms") or []]
    haystack = text + "\n" + json.dumps(record or {}, ensure_ascii=False)
    normalized = haystack.lower()
    matched = [term for term in expected_terms if term.lower() in normalized]
    forbidden = [term for term in forbidden_terms if term.lower() in normalized]
    min_hits = int(qa_check.get("min_hits") or max(1, min(2, len(expected_terms))))
    status = "PASS" if len(matched) >= min_hits and not forbidden else "FAIL"
    return {
        "question": question,
        "status": status,
        "expected_terms": expected_terms,
        "matched_terms": matched,
        "forbidden_hits": forbidden,
    }


if __name__ == "__main__":
    raise SystemExit(main())
