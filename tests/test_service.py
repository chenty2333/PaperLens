from __future__ import annotations

import json
import io
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import quote
from pathlib import Path

import pytest

from paperlens_core.protocol import RunResult
from paperlens_core.service import (
    EventStream,
    PaperLensHttpServer,
    PaperLensRequestHandler,
    PaperLensServiceState,
    load_report,
    load_public_workspace,
    read_request_json,
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


def test_load_report_preserves_local_images_for_frontend_asset_resolver(tmp_path: Path) -> None:
    write_sample_library(tmp_path)
    image_path = tmp_path / ".paperlens" / "figures" / "p_test" / "figure.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    image_src = "../.paperlens/figures/p_test/figure.png"
    (tmp_path / "papers" / "p_test.md").write_text(
        f'# A Useful Paper\n\n<figure>\n  <img src="{image_src}" alt="figure">\n</figure>',
        encoding="utf-8",
    )

    report = load_report(tmp_path, "p_test")

    assert f'src="{image_src}"' in report["markdown"]
    assert 'src="data:' not in report["markdown"]


def test_workspace_rejects_escaped_report_path(tmp_path: Path) -> None:
    write_sample_library(tmp_path)
    outside = tmp_path.parent / "outside.md"
    outside.write_text("# outside", encoding="utf-8")
    records_path = tmp_path / ".paperlens" / "library" / LIBRARY_RECORD_FILENAME
    record = json.loads(records_path.read_text(encoding="utf-8").splitlines()[0])
    record["outputs"]["briefing_md"] = "../outside.md"
    records_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    workspace = load_public_workspace(tmp_path)

    assert workspace["papers"][0]["report_path"] == ""
    with pytest.raises(FileNotFoundError, match="No report"):
        load_report(tmp_path, "p_test")


def test_read_request_json_rejects_oversized_body() -> None:
    class Handler:
        headers = {"Content-Length": "2000001"}
        rfile = io.BytesIO(b"{}")

    with pytest.raises(ValueError, match="too large"):
        read_request_json(Handler())  # type: ignore[arg-type]


def test_event_stream_assigns_sequence_numbers() -> None:
    stream = EventStream()
    stream.append({"type": "answer_started", "data": {}})
    stream.append({"type": "answer_completed", "data": {"answer": {"answer_markdown": "ok"}}})

    events = stream.snapshot()

    assert [event["seq"] for event in events] == [0, 1]
    assert events[1]["data"]["answer"]["answer_markdown"] == "ok"


def test_service_rejects_concurrent_read_jobs_for_same_output(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    state = PaperLensServiceState(token="test-token")
    started = threading.Event()
    release = threading.Event()

    def blocked_run_job(*_args, **_kwargs):  # noqa: ANN001
        started.set()
        release.wait(timeout=5)
        return RunResult(status="ok", data={})

    state.engine.run_job = blocked_run_job  # type: ignore[method-assign]
    first = state.start_read_job({"input_dir": str(input_dir), "output_dir": str(output_dir)})
    assert started.wait(timeout=2)

    try:
        with pytest.raises(ValueError, match="already being processed"):
            state.start_read_job({"input_dir": str(input_dir), "output_dir": str(output_dir)})
    finally:
        release.set()
        job = state.jobs[first["job_id"]]
        if job.thread:
            job.thread.join(timeout=2)


def test_retry_job_ignores_non_workflow_current_stage(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    state = PaperLensServiceState(token="test-token")
    captured: list[object] = []

    def run_job(request, **_kwargs):  # noqa: ANN001
        captured.append(request)
        return RunResult(status="ok", data={})

    state.engine.run_job = run_job  # type: ignore[method-assign]
    first = state.start_read_job({"input_dir": str(input_dir), "output_dir": str(output_dir)})
    deadline = time.time() + 2
    while state.jobs[first["job_id"]].status not in {"completed", "failed"} and time.time() < deadline:
        time.sleep(0.01)
    state.jobs[first["job_id"]].current_stage = "startup"

    retry = state.retry_job(first["job_id"], {})
    deadline = time.time() + 2
    while state.jobs[retry["job_id"]].status not in {"completed", "failed"} and time.time() < deadline:
        time.sleep(0.01)

    assert len(captured) >= 2
    assert captured[-1].from_stage is None
