from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from urllib.parse import quote
from pathlib import Path

from paperlens_core.service import (
    EventStream,
    PaperLensHttpServer,
    PaperLensRequestHandler,
    PaperLensServiceState,
    load_report,
    load_public_workspace,
)
from paperlens_core.library import LIBRARY_RECORD_FILENAME, LIBRARY_RECORD_SCHEMA_VERSION


def write_sample_library(output_dir: Path) -> None:
    library_dir = output_dir / ".paperlens" / "library"
    library_dir.mkdir(parents=True)
    record = {
        "schema_version": LIBRARY_RECORD_SCHEMA_VERSION,
        "paper_id": "p_test",
        "title": "A Useful Paper",
        "grade": "A",
        "recommendation": "重点关注",
        "tags": ["systems", "memory"],
        "memory": {
            "brief": "这篇论文把复杂系统问题讲清楚。",
            "core_idea": "核心抽象",
            "concepts": [{"term": "KV cache", "explanation": "cached key/value states"}],
            "claims": [{"claim": "claim"}],
            "evidence_items": [{"id": "E1"}],
        },
        "outputs": {"briefing_md": "papers/p_test.md"},
        "source": {"year": 2026, "pages": 12},
        "quality": {"report_audit_verdict": "PASS"},
        "search_text": "internal search text should not be returned to the UI",
    }
    (library_dir / LIBRARY_RECORD_FILENAME).write_text(
        json.dumps(record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_dir = output_dir / "papers"
    report_dir.mkdir()
    (report_dir / "p_test.md").write_text("# A Useful Paper\n\n正文。", encoding="utf-8")


def test_public_workspace_hides_internal_library_record_fields(tmp_path: Path) -> None:
    write_sample_library(tmp_path)

    workspace = load_public_workspace(tmp_path)

    assert workspace["paper_count"] == 1
    paper = workspace["papers"][0]
    assert paper["paper_id"] == "p_test"
    assert paper["title"] == "A Useful Paper"
    assert paper["brief"] == "这篇论文把复杂系统问题讲清楚。"
    assert paper["concepts"] == ["KV cache"]
    assert paper["memory"] == {
        "claim_count": 1,
        "evidence_count": 1,
        "memory_v3_path": None,
    }
    assert "search_text" not in paper
    assert LIBRARY_RECORD_FILENAME not in json.dumps(workspace, ensure_ascii=False)


def test_service_requires_auth_for_workspace_routes(tmp_path: Path) -> None:
    write_sample_library(tmp_path)
    state = PaperLensServiceState(token="test-token")
    server = PaperLensHttpServer(("127.0.0.1", 0), PaperLensRequestHandler, state=state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
            assert response.status == 200

        request = urllib.request.Request(
            f"{base_url}/workspaces/open",
            data=json.dumps({"output_dir": str(tmp_path)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=5)
            raise AssertionError("expected unauthorized response")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401

        request.add_header("Authorization", "Bearer test-token")
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["papers"][0]["paper_id"] == "p_test"
    finally:
        server.shutdown()
        server.server_close()


def test_service_serves_only_workspace_assets(tmp_path: Path) -> None:
    write_sample_library(tmp_path)
    asset_path = tmp_path / ".paperlens" / "figures" / "p_test" / "figure.png"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    outside = tmp_path.parent / "outside.png"
    outside.write_bytes(b"outside")
    state = PaperLensServiceState(token="test-token")
    server = PaperLensHttpServer(("127.0.0.1", 0), PaperLensRequestHandler, state=state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        path_query = quote(".paperlens/figures/p_test/figure.png", safe="")
        output_query = quote(str(tmp_path), safe="")
        with urllib.request.urlopen(
            f"{base_url}/assets?output_dir={output_query}&path={path_query}&token=test-token",
            timeout=5,
        ) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "image/png"
            assert response.read() == b"\x89PNG\r\n\x1a\n"

        escaped_path_query = quote("../outside.png", safe="")
        try:
            urllib.request.urlopen(
                f"{base_url}/assets?output_dir={output_query}&path={escaped_path_query}&token=test-token",
                timeout=5,
            )
            raise AssertionError("expected bad request for escaped asset path")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_load_report_inlines_local_images_for_ui(tmp_path: Path) -> None:
    write_sample_library(tmp_path)
    image_path = tmp_path / ".paperlens" / "figures" / "p_test" / "figure.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "papers" / "p_test.md").write_text(
        '# A Useful Paper\n\n<figure>\n  <img src="../.paperlens/figures/p_test/figure.png" alt="figure">\n</figure>',
        encoding="utf-8",
    )

    report = load_report(tmp_path, "p_test")

    assert 'src="data:image/png;base64,' in report["markdown"]
    assert "../.paperlens/figures/p_test/figure.png" not in report["markdown"]


def test_event_stream_assigns_sequence_numbers() -> None:
    stream = EventStream()
    stream.append({"type": "answer_started", "data": {}})
    stream.append({"type": "answer_completed", "data": {"answer": {"answer_markdown": "ok"}}})

    events = stream.snapshot()

    assert [event["seq"] for event in events] == [0, 1]
    assert events[1]["data"]["answer"]["answer_markdown"] == "ok"
