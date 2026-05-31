from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a real arXiv PDF corpus.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--query",
        default="cat:cs.OS OR cat:cs.CR OR cat:cs.PL",
        help="arXiv API search_query.",
    )
    parser.add_argument("--delay-seconds", type=float, default=0.35)
    return parser.parse_args()


def request_bytes(url: str, *, timeout: int = 90) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "PaperLens/0.1 batch validation (local research tool)",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_entries(query: str, start: int, max_results: int) -> list[dict[str, str]]:
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    data = request_bytes(f"https://export.arxiv.org/api/query?{params}", timeout=60)
    root = ET.fromstring(data)
    entries: list[dict[str, str]] = []
    for entry in root.findall(f"{ATOM}entry"):
        arxiv_id = (entry.findtext(f"{ATOM}id") or "").rsplit("/", 1)[-1]
        title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
        published = entry.findtext(f"{ATOM}published") or ""
        primary = entry.find(f"{ARXIV}primary_category")
        category = primary.attrib.get("term", "") if primary is not None else ""
        pdf_url = ""
        for link in entry.findall(f"{ATOM}link"):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")
                break
        if arxiv_id and pdf_url:
            entries.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "published": published,
                    "category": category,
                    "pdf_url": pdf_url.replace("http://", "https://"),
                }
            )
    return entries


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned[:120] or "paper"


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"
    entries = fetch_entries(args.query, args.start, args.max_results)
    rows = []
    for index, entry in enumerate(entries, start=1):
        filename = f"{index:03d}_{safe_name(entry['arxiv_id'])}.pdf"
        path = output_dir / filename
        status = "exists" if path.exists() and path.stat().st_size > 1024 else "downloaded"
        if status == "downloaded":
            data = request_bytes(entry["pdf_url"])
            if not data.startswith(b"%PDF"):
                status = "non_pdf_response"
                path.with_suffix(".error.txt").write_bytes(data[:4000])
            else:
                path.write_bytes(data)
                time.sleep(args.delay_seconds)
        rows.append({**entry, "file": filename, "status": status, "bytes": path.stat().st_size if path.exists() else 0})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file", "arxiv_id", "title", "published", "category", "pdf_url", "status", "bytes"],
        )
        writer.writeheader()
        writer.writerows(rows)
    ok = sum(1 for row in rows if row["status"] in {"downloaded", "exists"} and int(row["bytes"]) > 1024)
    print(json.dumps({"output_dir": str(output_dir), "pdf_count": ok}, ensure_ascii=False))
    return 0 if ok >= min(args.max_results, len(entries)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
