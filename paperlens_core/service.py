from __future__ import annotations

import json
import mimetypes
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from paperlens_core.control import ControlState
from paperlens_core.core_manifest import inspect_core_v2_artifact_set
from paperlens_core.engine import PaperLensEngine
from paperlens_core.library import read_library_records, search_library
from paperlens_core.protocol import LibraryQuestionRequest, PaperQuestionRequest, RunRequest
from paperlens_core.runtime import read_typed_artifact
from paperlens_core.version import display_version
from paperlens_core.workflow.stages import normalize_workflow_stage


SERVER_VERSION = "paperlens-core-service.v1"
INTERNAL_DIR = ".paperlens"
MAX_JSON_REQUEST_BYTES = 2_000_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json_file(path: Path, fallback: Any = None) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def read_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    except OSError:
        return []
    return rows[-limit:] if limit else rows


def write_json_response(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(status)
    write_common_headers(handler)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def write_file_response(handler: BaseHTTPRequestHandler, path: Path) -> None:
    body = path.read_bytes()
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    handler.send_response(HTTPStatus.OK)
    write_common_headers(handler)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def write_common_headers(handler: BaseHTTPRequestHandler) -> None:
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "authorization, content-type, x-paperlens-token")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Cache-Control", "no-store")


def read_request_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    try:
        length = int(handler.headers.get("Content-Length") or "0")
    except ValueError as exc:
        raise ValueError("Invalid Content-Length") from exc
    if length <= 0:
        return {}
    if length > MAX_JSON_REQUEST_BYTES:
        raise ValueError(
            f"JSON request body is too large: {length} bytes exceeds {MAX_JSON_REQUEST_BYTES}"
        )
    raw = handler.rfile.read(length)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON request body: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON request body must be an object")
    return value


def string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def chat_history_from_payload(payload: dict[str, Any]) -> list[dict[str, str]]:
    raw = payload.get("chat_history")
    if not isinstance(raw, list):
        return []
    history: list[dict[str, str]] = []
    for item in raw[-12:]:
        if not isinstance(item, dict):
            continue
        role = string_or_none(item.get("role"))
        content = string_or_none(item.get("content"))
        if role not in {"user", "assistant"} or not content:
            continue
        history.append({"role": role, "content": content[:2000]})
    return history[-8:]


def int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def path_from_payload(payload: dict[str, Any], key: str) -> Path:
    value = string_or_none(payload.get(key))
    if not value:
        raise ValueError(f"Missing required path: {key}")
    return Path(value).expanduser().resolve()


def safe_workflow_stage(value: str | None) -> str | None:
    try:
        return normalize_workflow_stage(value)
    except ValueError:
        return None


def output_dir_from_query(query: dict[str, list[str]]) -> Path:
    value = query.get("output_dir", [""])[0]
    if value:
        return Path(value).expanduser().resolve()
    raise ValueError("Missing output_dir")


def config_overrides_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    provider = payload.get("provider") if isinstance(payload.get("provider"), dict) else {}
    provider_kind = string_or_none(payload.get("provider_kind")) or string_or_none(
        provider.get("kind")
    )
    base_url = string_or_none(payload.get("base_url")) or string_or_none(provider.get("base_url"))
    model = string_or_none(payload.get("model")) or string_or_none(provider.get("model"))
    reasoning_model = string_or_none(payload.get("reasoning_model")) or string_or_none(
        provider.get("reasoning_model")
    )
    timeout_seconds = payload.get("timeout_seconds") or provider.get("timeout_seconds")
    api_key = string_or_none(payload.get("api_key")) or string_or_none(provider.get("api_key"))
    return {
        "provider": {
            "kind": provider_kind,
            "base_url": base_url,
            "model": model,
            "reasoning_model": reasoning_model,
            "timeout_seconds": timeout_seconds,
            "api_key": api_key,
            "api_key_env": string_or_none(payload.get("api_key_env"))
            or string_or_none(provider.get("api_key_env"))
            or "PAPERLENS_API_KEY",
        },
        "concurrency": int_or_default(payload.get("concurrency"), 1)
        if payload.get("concurrency") is not None
        else None,
        "offline_debug": bool(payload.get("offline_debug", False)),
        "topic": string_or_none(payload.get("topic")),
        "idea": string_or_none(payload.get("idea")),
        "output_language": string_or_none(payload.get("output_language")),
        "read_mode": string_or_none(payload.get("read_mode")),
        "visual_verification_mode": string_or_none(payload.get("visual_verification_mode")),
        "visual_verification_max_pages": payload.get("visual_verification_max_pages"),
    }


def public_report_path(output_dir: Path, record: dict[str, Any]) -> str:
    outputs = record.get("outputs") if isinstance(record.get("outputs"), dict) else {}
    report = string_or_none(outputs.get("briefing_md"))
    if report:
        safe_report = safe_public_report_path(output_dir, report)
        if safe_report:
            return safe_report
    paper_id = string_or_none(record.get("paper_id")) or ""
    for path in sorted((output_dir / "papers").glob(f"{paper_id}_*.md")):
        return str(path.relative_to(output_dir)).replace("\\", "/")
    return ""


def resolve_output_relative_path(output_dir: Path, relative_path: str) -> Path:
    if not relative_path:
        raise ValueError("Missing output-relative path")
    if "\x00" in relative_path:
        raise ValueError("Invalid output-relative path")
    output_dir = output_dir.expanduser().resolve()
    requested = Path(relative_path)
    if requested.is_absolute():
        raise ValueError("Path must be relative to output_dir")
    target = (output_dir / requested).resolve()
    try:
        target.relative_to(output_dir)
    except ValueError as exc:
        raise ValueError("Path escapes output_dir") from exc
    return target


def safe_public_report_path(output_dir: Path, report_path: str) -> str:
    try:
        target = resolve_output_relative_path(output_dir, report_path)
    except ValueError:
        return ""
    if target.suffix.lower() != ".md":
        return ""
    output_dir = output_dir.expanduser().resolve()
    try:
        relative = target.relative_to(output_dir)
    except ValueError:
        return ""
    if not relative.parts or relative.parts[0] != "papers":
        return ""
    return str(relative).replace("\\", "/")


def latest_qa_by_paper(output_dir: Path) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(output_dir / INTERNAL_DIR / "data" / "qa_trace.jsonl"):
        paper_id = string_or_none(row.get("paper_id"))
        if not paper_id:
            continue
        current = grouped.setdefault(paper_id, {"count": 0, "last_question": "", "last_time": ""})
        current["count"] = int(current.get("count") or 0) + 1
        current["last_question"] = string_or_none(row.get("question")) or ""
        current["last_time"] = string_or_none(row.get("time")) or ""
    return grouped


def public_paper_record(output_dir: Path, record: dict[str, Any], qa_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    memory = record.get("memory") if isinstance(record.get("memory"), dict) else {}
    provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    paper_id = string_or_none(record.get("paper_id")) or ""
    concepts = memory.get("concepts") if isinstance(memory.get("concepts"), list) else []
    concept_terms = [
        string_or_none(item.get("term")) if isinstance(item, dict) else string_or_none(item)
        for item in concepts
    ]
    report_path = public_report_path(output_dir, record)
    return {
        "paper_id": paper_id,
        "title": string_or_none(record.get("title")) or paper_id,
        "grade": string_or_none(record.get("grade")) or "HOLD",
        "recommendation": string_or_none(record.get("recommendation")) or "",
        "brief": string_or_none(memory.get("brief"))
        or string_or_none(memory.get("core_idea"))
        or "",
        "core_idea": string_or_none(memory.get("core_idea")) or "",
        "concepts": [term for term in concept_terms if term][:12],
        "tags": record.get("tags") if isinstance(record.get("tags"), list) else [],
        "report_path": report_path,
        "report_file": str((output_dir / report_path).resolve()) if report_path else "",
        "source": {
            "year": source.get("year"),
            "venue": source.get("venue"),
            "pages": source.get("pages"),
            "doi": source.get("doi"),
            "original_path": source.get("original_path"),
        },
        "quality": record.get("quality") if isinstance(record.get("quality"), dict) else {},
        "memory": {
            "claim_count": len(memory.get("claims") or []),
            "evidence_count": len(memory.get("evidence_items") or []),
            "core_v2": provenance.get("core_v2") if isinstance(provenance.get("core_v2"), dict) else {},
        },
        "qa": qa_map.get(paper_id, {"count": 0, "last_question": "", "last_time": ""}),
    }


def load_public_workspace(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    run = read_json_file(output_dir / INTERNAL_DIR / "data" / "run.json", {}) or {}
    try:
        records = read_library_records(output_dir)
    except Exception:
        records = []
    qa_map = latest_qa_by_paper(output_dir)
    papers = [public_paper_record(output_dir, record, qa_map) for record in records]
    return {
        "output_dir": str(output_dir),
        "status": (run.get("status") if isinstance(run, dict) else None) or "unknown",
        "manifest": run.get("manifest") if isinstance(run, dict) else {},
        "papers": papers,
        "paper_count": len(papers),
    }


def load_report(output_dir: Path, paper_id: str) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    workspace = load_public_workspace(output_dir)
    paper = next((item for item in workspace["papers"] if item.get("paper_id") == paper_id), None)
    if not paper:
        raise FileNotFoundError(f"Unknown paper_id: {paper_id}")
    report_path = string_or_none(paper.get("report_path"))
    if not report_path:
        raise FileNotFoundError(f"No report for paper_id: {paper_id}")
    path = resolve_output_relative_path(output_dir, report_path)
    if path.suffix.lower() != ".md":
        raise ValueError("Report path must point to a Markdown file")
    if not path.exists():
        raise FileNotFoundError(f"Report file missing: {path}")
    markdown = path.read_text(encoding="utf-8")
    return {
        "paper": paper,
        "path": str(path),
        "base_dir": str(path.parent),
        "markdown": markdown,
    }


def load_evidence(output_dir: Path, paper_id: str) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    data_dir = output_dir / INTERNAL_DIR / "data"
    root = data_dir / "core" / "v2" / paper_id
    manifest = inspect_core_v2_artifact_set(data_dir, paper_id)
    issues = list(manifest.get("issues") if isinstance(manifest.get("issues"), list) else [])
    try:
        dom = read_typed_artifact(root / "paper_dom.v1.json", expected_type="paper_dom").data
        graph = read_typed_artifact(root / "claim_graph.v1.json", expected_type="claim_graph").data
        quality = read_typed_artifact(
            root / "quality_metrics.v1.json", expected_type="core_quality_metrics"
        ).data
        audit = read_typed_artifact(
            root / "audit_findings.v1.json", expected_type="audit_findings"
        ).data
    except (FileNotFoundError, ValueError) as exc:
        return {
            "paper_id": paper_id,
            "status": "INCOMPLETE",
            "publish_status": None,
            "consumable": False,
            "claims": [],
            "evidence": [],
            "quality": {},
            "audit": [],
            "issues": [*issues, str(exc)],
        }
    dom_sources = core_v2_dom_source_index(dom if isinstance(dom, dict) else {})
    nodes = (graph.get("nodes") if isinstance(graph, dict) else {}) or {}
    edges = (
        graph.get("edges")
        if isinstance(graph, dict) and isinstance(graph.get("edges"), list)
        else []
    )
    if not isinstance(nodes, dict):
        nodes = {}
    claims: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for node_id, raw_node in nodes.items():
        if not isinstance(raw_node, dict):
            continue
        kind = string_or_none(raw_node.get("kind")) or ""
        payload = raw_node.get("payload") if isinstance(raw_node.get("payload"), dict) else {}
        if kind == "evidence":
            source_id = string_or_none(payload.get("source_id")) or ""
            source = dom_sources.get(source_id, {})
            evidence.append(
                {
                    "id": str(node_id),
                    "source_id": source_id,
                    "source_kind": source.get("kind"),
                    "page_no": source.get("page_no"),
                    "interpretation": string_or_none(raw_node.get("label")) or "",
                    "text": source.get("text") or "",
                }
            )
            continue
        evidence_ids = graph_evidence_ids_for(edges, str(node_id), nodes)
        source_ids = [
            string_or_none(nodes[evidence_id].get("payload", {}).get("source_id")) or ""
            for evidence_id in evidence_ids
            if isinstance(nodes.get(evidence_id), dict)
            and isinstance(nodes[evidence_id].get("payload"), dict)
        ]
        claims.append(
            {
                "id": str(node_id),
                "kind": kind,
                "label": string_or_none(raw_node.get("label")) or "",
                "confidence": payload.get("confidence"),
                "uncertainty": string_or_none(payload.get("uncertainty")) or "",
                "evidence_ids": evidence_ids,
                "source_ids": [source_id for source_id in source_ids if source_id],
                "covered_outputs": payload.get("covered_outputs")
                if isinstance(payload.get("covered_outputs"), list)
                else [],
                "extracted_numbers": payload.get("extracted_numbers")
                if isinstance(payload.get("extracted_numbers"), list)
                else [],
            }
        )
    return {
        "paper_id": paper_id,
        "status": manifest.get("status"),
        "publish_status": manifest.get("publish_status"),
        "consumable": manifest.get("consumable"),
        "claims": claims,
        "evidence": evidence,
        "quality": quality if isinstance(quality, dict) else {},
        "audit": audit if isinstance(audit, list) else [],
        "issues": issues,
    }


def core_v2_dom_source_index(dom: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for group_name in ["sections", "spans", "figures", "tables", "equations"]:
        group = dom.get(group_name) if isinstance(dom.get(group_name), list) else []
        for item in group:
            if not isinstance(item, dict):
                continue
            source_id = string_or_none(item.get("source_id"))
            if not source_id:
                continue
            text = (
                string_or_none(item.get("text"))
                or string_or_none(item.get("caption"))
                or string_or_none(item.get("latex_or_text"))
                or string_or_none(item.get("title"))
                or ""
            )
            result[source_id] = {
                "source_id": source_id,
                "kind": string_or_none(item.get("kind")) or group_name.rstrip("s"),
                "page_no": item.get("page_no"),
                "text": text,
            }
    return result


def graph_evidence_ids_for(
    edges: list[Any], node_id: str, nodes: dict[str, dict[str, Any]]
) -> list[str]:
    evidence_ids = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        target_id = string_or_none(edge.get("target_id")) or ""
        target = nodes.get(target_id)
        if (
            string_or_none(edge.get("source_id")) == node_id
            and string_or_none(edge.get("kind")) == "supported_by"
            and isinstance(target, dict)
            and string_or_none(target.get("kind")) == "evidence"
            and target_id not in evidence_ids
        ):
            evidence_ids.append(target_id)
    return evidence_ids


def resolve_output_asset(output_dir: Path, asset_path: str) -> Path:
    if not asset_path:
        raise ValueError("Missing asset path")
    output_dir = output_dir.expanduser().resolve()
    target = resolve_output_relative_path(output_dir, asset_path)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"Asset file missing: {asset_path}")
    return target


class EventStream:
    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._condition = threading.Condition()

    def append(self, event: dict[str, Any]) -> None:
        with self._condition:
            payload = dict(event)
            payload.setdefault("time", utc_now())
            payload["seq"] = len(self._events)
            self._events.append(payload)
            self._condition.notify_all()

    def snapshot(self) -> list[dict[str, Any]]:
        with self._condition:
            return list(self._events)

    def wait_after(self, cursor: int, timeout: float = 15.0) -> list[dict[str, Any]]:
        with self._condition:
            if len(self._events) <= cursor:
                self._condition.wait(timeout)
            return list(self._events[cursor:])


@dataclass
class ManagedJob:
    job_id: str
    request_payload: dict[str, Any]
    input_dir: Path
    output_dir: Path
    control: ControlState = field(default_factory=ControlState)
    events: EventStream = field(default_factory=EventStream)
    status: str = "queued"
    current_stage: str = "queued"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    thread: threading.Thread | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "current_stage": self.current_stage,
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "latest_event": self.events.snapshot()[-1] if self.events.snapshot() else None,
            "result": self.result,
        }


@dataclass
class ManagedAnswer:
    answer_id: str
    request_payload: dict[str, Any]
    scope: str
    output_dir: Path
    paper_id: str | None
    question: str
    events: EventStream = field(default_factory=EventStream)
    status: str = "queued"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    completed_at: str | None = None
    answer: dict[str, Any] | None = None
    error: str | None = None
    thread: threading.Thread | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "answer_id": self.answer_id,
            "status": self.status,
            "scope": self.scope,
            "output_dir": str(self.output_dir),
            "paper_id": self.paper_id,
            "question": self.question,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "answer": self.answer,
            "error": self.error,
            "latest_event": self.events.snapshot()[-1] if self.events.snapshot() else None,
        }


class PaperLensServiceState:
    def __init__(self, *, token: str, config_path: Path | None = None) -> None:
        self.token = token
        self.config_path = config_path
        self.engine = PaperLensEngine()
        self.jobs: dict[str, ManagedJob] = {}
        self.answers: dict[str, ManagedAnswer] = {}
        self._lock = threading.Lock()

    def start_read_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        input_dir = path_from_payload(payload, "input_dir")
        output_dir = path_from_payload(payload, "output_dir")
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = ManagedJob(
            job_id=job_id,
            request_payload=dict(payload),
            input_dir=input_dir,
            output_dir=output_dir,
        )
        with self._lock:
            active_job = self.active_job_for_output_dir(output_dir)
            if active_job:
                raise ValueError(
                    f"Output directory is already being processed by job {active_job.job_id}"
                )
            self.jobs[job_id] = job
        job.events.append(
            {
                "type": "job_queued",
                "level": "info",
                "stage": "queued",
                "message": "PaperLens read job queued",
                "data": {"job_id": job_id},
            }
        )
        thread = threading.Thread(target=self._run_job_thread, args=(job,), daemon=True)
        job.thread = thread
        thread.start()
        return job.summary()

    def _run_job_thread(self, job: ManagedJob) -> None:
        job.status = "running"
        job.updated_at = utc_now()
        job.events.append(
            {
                "type": "job_started",
                "level": "info",
                "stage": "startup",
                "message": "PaperLens read job started",
                "data": {"job_id": job.job_id},
            }
        )

        def on_event(event: dict[str, Any]) -> None:
            job.current_stage = str(event.get("stage") or job.current_stage)
            job.updated_at = utc_now()
            if event.get("level") in {"error", "critical"}:
                job.status = "failed"
                job.error = str(event.get("message") or "Job failed")
            job.events.append({**event, "job_id": job.job_id})

        payload = job.request_payload
        overrides = config_overrides_from_payload(payload)
        try:
            result = self.engine.run_job(
                RunRequest(
                    input_dir=job.input_dir,
                    output_dir=job.output_dir,
                    config_path=Path(payload["config_path"]).resolve()
                    if string_or_none(payload.get("config_path"))
                    else self.config_path,
                    config_overrides=overrides,
                    from_stage=string_or_none(payload.get("from_stage")),
                    only_stage=string_or_none(payload.get("only_stage")),
                    use_stdin_control=False,
                ),
                control=job.control,
                event_callback=on_event,
            )
            job.result = result.model_dump()
            job.status = "completed" if result.status == "ok" else "failed"
            if result.status != "ok":
                job.error = str(result.data.get("reason") or "Job failed")
        except Exception as exc:  # pragma: no cover - defensive service boundary
            job.status = "failed"
            job.error = str(exc)
            job.events.append(
                {
                    "type": "job_failed",
                    "level": "error",
                    "stage": job.current_stage,
                    "message": str(exc),
                    "data": {"job_id": job.job_id},
                }
            )
        finally:
            job.completed_at = utc_now()
            job.updated_at = job.completed_at
            job.events.append(
                {
                    "type": "job_completed" if job.status == "completed" else "job_failed",
                    "level": "info" if job.status == "completed" else "error",
                    "stage": job.current_stage,
                    "message": "PaperLens read job completed"
                    if job.status == "completed"
                    else job.error or "PaperLens read job failed",
                    "data": {"job_id": job.job_id, "status": job.status},
                }
            )

    def retry_job(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        original = self.jobs.get(job_id)
        if not original:
            raise KeyError(f"Unknown job: {job_id}")
        retry_payload = dict(original.request_payload)
        from_stage = safe_workflow_stage(
            string_or_none(payload.get("from_stage")) or original.current_stage
        )
        if from_stage:
            retry_payload["from_stage"] = from_stage
        else:
            retry_payload.pop("from_stage", None)
        return self.start_read_job(retry_payload)

    def active_job_for_output_dir(self, output_dir: Path) -> ManagedJob | None:
        resolved = output_dir.expanduser().resolve()
        for job in self.jobs.values():
            if job.output_dir == resolved and job.status in {
                "queued",
                "running",
                "paused",
                "cancelling",
            }:
                return job
        return None

    def control_job(self, job_id: str, command: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(f"Unknown job: {job_id}")
        if command == "cancel":
            job.control.cancel()
            job.status = "cancelling"
        elif command == "pause":
            job.control.pause()
            job.status = "paused"
        elif command == "resume":
            job.control.resume()
            job.status = "running"
        else:
            raise ValueError(f"Unknown job control command: {command}")
        job.updated_at = utc_now()
        job.events.append(
            {
                "type": f"job_{command}",
                "level": "info",
                "stage": job.current_stage,
                "message": f"Job {command} requested",
                "data": {"job_id": job_id},
            }
        )
        return job.summary()

    def start_answer(self, payload: dict[str, Any]) -> dict[str, Any]:
        output_dir = path_from_payload(payload, "output_dir")
        scope = string_or_none(payload.get("scope")) or "paper"
        question = string_or_none(payload.get("question"))
        if not question:
            raise ValueError("Missing question")
        answer_id = f"ask_{uuid.uuid4().hex[:12]}"
        answer = ManagedAnswer(
            answer_id=answer_id,
            request_payload=dict(payload),
            scope=scope,
            output_dir=output_dir,
            paper_id=string_or_none(payload.get("paper_id")),
            question=question,
        )
        with self._lock:
            self.answers[answer_id] = answer
        answer.events.append(
            {
                "type": "answer_queued",
                "level": "info",
                "message": "Question queued",
                "data": {"answer_id": answer_id, "scope": scope},
            }
        )
        thread = threading.Thread(target=self._answer_thread, args=(answer,), daemon=True)
        answer.thread = thread
        thread.start()
        return answer.summary()

    def _answer_thread(self, answer: ManagedAnswer) -> None:
        answer.status = "running"
        answer.updated_at = utc_now()
        answer.events.append(
            {
                "type": "answer_started",
                "level": "info",
                "message": "Checking PaperLens ClaimGraph and evidence",
                "data": {"answer_id": answer.answer_id, "scope": answer.scope},
            }
        )
        payload = answer.request_payload
        overrides = config_overrides_from_payload(payload)
        try:
            if answer.scope == "library":
                result = self.engine.answer_library_question(
                    LibraryQuestionRequest(
                        output_dir=answer.output_dir,
                        config_path=Path(payload["config_path"]).resolve()
                        if string_or_none(payload.get("config_path"))
                        else self.config_path,
                        config_overrides=overrides,
                        question=answer.question,
                        limit=int_or_default(payload.get("limit"), 8),
                        chat_history=chat_history_from_payload(payload),
                    )
                )
            else:
                result = self.engine.answer_paper_question(
                    PaperQuestionRequest(
                        output_dir=answer.output_dir,
                        config_path=Path(payload["config_path"]).resolve()
                        if string_or_none(payload.get("config_path"))
                        else self.config_path,
                        config_overrides=overrides,
                        paper_id=answer.paper_id,
                        question=answer.question,
                        chat_history=chat_history_from_payload(payload),
                    )
                )
            answer.answer = result
            answer.status = "completed"
        except Exception as exc:  # pragma: no cover - defensive service boundary
            answer.status = "failed"
            answer.error = str(exc)
            answer.events.append(
                {
                    "type": "answer_failed",
                    "level": "error",
                    "message": str(exc),
                    "data": {"answer_id": answer.answer_id},
                }
            )
        finally:
            answer.completed_at = utc_now()
            answer.updated_at = answer.completed_at
            answer.events.append(
                {
                    "type": "answer_completed" if answer.status == "completed" else "answer_failed",
                    "level": "info" if answer.status == "completed" else "error",
                    "message": "Answer ready" if answer.status == "completed" else answer.error,
                    "data": {
                        "answer_id": answer.answer_id,
                        "status": answer.status,
                        "answer": answer.answer if answer.status == "completed" else None,
                    },
                }
            )


class PaperLensHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass: type[BaseHTTPRequestHandler],
        *,
        state: PaperLensServiceState,
    ) -> None:
        super().__init__(server_address, RequestHandlerClass)
        self.state = state


class PaperLensRequestHandler(BaseHTTPRequestHandler):
    server: PaperLensHttpServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    @property
    def state(self) -> PaperLensServiceState:
        return self.server.state

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        write_common_headers(self)
        self.end_headers()

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def _handle(self, method: str) -> None:
        parsed = urlparse(self.path)
        parts = [unquote(part) for part in parsed.path.strip("/").split("/") if part]
        query = parse_qs(parsed.query)
        try:
            if not self._authorized(parts, query):
                write_json_response(self, HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            if method == "GET":
                result = self._route_get(parts, query)
            elif method == "POST":
                result = self._route_post(parts, query)
            else:
                raise RouteError(HTTPStatus.METHOD_NOT_ALLOWED, "method not allowed")
            if result is not None:
                if isinstance(result, FileResult):
                    write_file_response(self, result.path)
                else:
                    write_json_response(self, HTTPStatus.OK, result)
        except RouteError as exc:
            write_json_response(self, int(exc.status), {"error": exc.message})
        except FileNotFoundError as exc:
            write_json_response(self, HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except KeyError as exc:
            write_json_response(self, HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except ValueError as exc:
            write_json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - service boundary
            write_json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _authorized(self, parts: list[str], query: dict[str, list[str]]) -> bool:
        if parts in ([], ["health"], ["version"]):
            return True
        token = self.headers.get("X-PaperLens-Token")
        auth = self.headers.get("Authorization") or ""
        if auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1]
        if not token:
            token = query.get("token", [""])[0]
        return bool(token) and secrets.compare_digest(token, self.state.token)

    def _route_get(self, parts: list[str], query: dict[str, list[str]]) -> Any:
        if parts == [] or parts == ["health"]:
            return {"status": "ok", "service": SERVER_VERSION}
        if parts == ["version"]:
            return {"version": display_version(), "service": SERVER_VERSION}
        if parts == ["workspaces", "current"]:
            output_dir = output_dir_from_query(query)
            return load_public_workspace(output_dir)
        if parts == ["library"]:
            output_dir = output_dir_from_query(query)
            return {"workspace": load_public_workspace(output_dir)}
        if parts == ["papers"]:
            output_dir = output_dir_from_query(query)
            return {"papers": load_public_workspace(output_dir)["papers"]}
        if len(parts) == 2 and parts[0] == "papers":
            output_dir = output_dir_from_query(query)
            paper_id = parts[1]
            workspace = load_public_workspace(output_dir)
            paper = next((item for item in workspace["papers"] if item.get("paper_id") == paper_id), None)
            if not paper:
                raise FileNotFoundError(f"Unknown paper_id: {paper_id}")
            return {"paper": paper}
        if len(parts) == 3 and parts[0] == "papers" and parts[2] == "report":
            output_dir = output_dir_from_query(query)
            return load_report(output_dir, parts[1])
        if len(parts) == 3 and parts[0] == "papers" and parts[2] == "evidence":
            output_dir = output_dir_from_query(query)
            return load_evidence(output_dir, parts[1])
        if parts == ["assets"]:
            output_dir = output_dir_from_query(query)
            asset_path = query.get("path", [""])[0]
            return FileResult(resolve_output_asset(output_dir, asset_path))
        if parts == ["jobs"]:
            return {"jobs": [job.summary() for job in self.state.jobs.values()]}
        if len(parts) == 2 and parts[0] == "jobs":
            job = self.state.jobs.get(parts[1])
            if not job:
                raise KeyError(f"Unknown job: {parts[1]}")
            return job.summary()
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "events":
            job = self.state.jobs.get(parts[1])
            if not job:
                raise KeyError(f"Unknown job: {parts[1]}")
            self._stream_events(job.events, terminal_status=lambda: job.status)
            return None
        if len(parts) == 2 and parts[0] == "ask":
            answer = self.state.answers.get(parts[1])
            if not answer:
                raise KeyError(f"Unknown answer: {parts[1]}")
            return answer.summary()
        if len(parts) == 3 and parts[0] == "ask" and parts[2] == "events":
            answer = self.state.answers.get(parts[1])
            if not answer:
                raise KeyError(f"Unknown answer: {parts[1]}")
            self._stream_events(answer.events, terminal_status=lambda: answer.status)
            return None
        raise RouteError(HTTPStatus.NOT_FOUND, f"Unknown route: /{'/'.join(parts)}")

    def _route_post(self, parts: list[str], query: dict[str, list[str]]) -> Any:
        payload = read_request_json(self)
        if parts == ["workspaces", "open"]:
            output_dir = path_from_payload(payload, "output_dir")
            return load_public_workspace(output_dir)
        if parts == ["jobs", "read"]:
            return self.state.start_read_job(payload)
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] in {"cancel", "pause", "resume"}:
            return self.state.control_job(parts[1], parts[2])
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "retry":
            return self.state.retry_job(parts[1], payload)
        if parts == ["ask"]:
            return self.state.start_answer(payload)
        if parts == ["shutdown"]:
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return {"status": "shutting_down"}
        if parts == ["library", "search"]:
            output_dir = path_from_payload(payload, "output_dir")
            query_text = string_or_none(payload.get("query")) or ""
            return search_library(output_dir=output_dir, query=query_text, limit=int_or_default(payload.get("limit"), 8))
        raise RouteError(HTTPStatus.NOT_FOUND, f"Unknown route: /{'/'.join(parts)}")

    def _stream_events(self, stream: EventStream, *, terminal_status: Any) -> None:
        self.send_response(HTTPStatus.OK)
        write_common_headers(self)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Connection", "close")
        self.end_headers()
        cursor = 0
        while True:
            events = stream.wait_after(cursor, timeout=10.0)
            for event in events:
                cursor = max(cursor, int(event.get("seq", cursor)) + 1)
                payload = json.dumps(event, ensure_ascii=False, default=str)
                self.wfile.write(f"id: {event.get('seq', cursor)}\n".encode("utf-8"))
                self.wfile.write(b"event: paperlens\n")
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
            if terminal_status() in {"completed", "failed", "cancelled"} and cursor >= len(stream.snapshot()):
                break
            if not events:
                self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
        self.close_connection = True


class RouteError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class FileResult:
    path: Path


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    token: str | None = None,
    config_path: Path | None = None,
) -> int:
    token = token or secrets.token_urlsafe(32)
    state = PaperLensServiceState(token=token, config_path=config_path)
    server = PaperLensHttpServer((host, port), PaperLensRequestHandler, state=state)
    actual_host, actual_port = server.server_address
    print(
        json.dumps(
            {
                "type": "server_started",
                "service": SERVER_VERSION,
                "host": actual_host,
                "port": actual_port,
                "base_url": f"http://{actual_host}:{actual_port}",
                "token": token,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        time.sleep(0.05)
    return 0
