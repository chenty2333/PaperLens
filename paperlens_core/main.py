from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from paperlens_core.engine import PaperLensEngine
from paperlens_core.events import emit_fatal
from paperlens_core.protocol import (
    LibraryBuildRequest,
    LibraryDoctorRequest,
    LibraryQuestionRequest,
    LibraryRebuildIndexRequest,
    LibrarySearchRequest,
    PaperQuestionRequest,
    RunRequest,
)
from paperlens_core.service import serve


def configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paperlens-core")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the PaperLens pipeline")
    run.add_argument("--input-dir", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--config")
    run.add_argument("--provider-kind")
    run.add_argument("--base-url")
    run.add_argument("--model")
    run.add_argument("--reasoning-model")
    run.add_argument("--timeout-seconds", type=int)
    run.add_argument("--api-key-env", default="PAPERLENS_API_KEY")
    run.add_argument("--budget", type=float)
    run.add_argument("--concurrency", type=int)
    run.add_argument("--offline-debug", action="store_true")
    run.add_argument("--topic")
    run.add_argument("--idea")
    run.add_argument("--output-language", choices=["en", "zh"])
    run.add_argument("--read-mode", choices=["standard"], default=None)
    run.add_argument("--from-stage", help="Resume from a completed run's later stage")
    run.add_argument("--only-stage", help="Run one stage using previously saved state")
    run.add_argument(
        "--visual-verification-mode", choices=["parse_issues", "all_marked_pages", "off"]
    )
    run.add_argument("--visual-verification-max-pages", type=int)
    run.add_argument("--enable-llm-stages", action="store_true", help=argparse.SUPPRESS)

    ask = subparsers.add_parser("ask", help="Ask a question about a generated paper report")
    ask.add_argument("--output-dir", required=True)
    ask.add_argument("--paper-id")
    ask.add_argument("--question", required=True)
    ask.add_argument("--config")
    ask.add_argument("--provider-kind")
    ask.add_argument("--base-url")
    ask.add_argument("--model")
    ask.add_argument("--timeout-seconds", type=int)
    ask.add_argument("--api-key-env", default="PAPERLENS_API_KEY")
    ask.add_argument("--offline-debug", action="store_true")

    library = subparsers.add_parser(
        "library", help="Build, search, or ask the local PaperLens library"
    )
    library_subparsers = library.add_subparsers(dest="library_command", required=True)
    library_build = library_subparsers.add_parser(
        "build", help="Rebuild PaperLens local memory library"
    )
    library_build.add_argument("--output-dir", required=True)

    library_rebuild_index = library_subparsers.add_parser(
        "rebuild-index", help="Rebuild the derived PaperLens search index"
    )
    library_rebuild_index.add_argument("--output-dir", required=True)

    library_doctor = library_subparsers.add_parser(
        "doctor", help="Check local PaperLens library health"
    )
    library_doctor.add_argument("--output-dir", required=True)

    library_search = library_subparsers.add_parser("search", help="Search PaperLens local memory")
    library_search.add_argument("--output-dir", required=True)
    library_search.add_argument("--query", required=True)
    library_search.add_argument("--limit", type=int, default=8)

    library_ask = library_subparsers.add_parser("ask", help="Ask across PaperLens local memory")
    library_ask.add_argument("--output-dir", required=True)
    library_ask.add_argument("--question", required=True)
    library_ask.add_argument("--limit", type=int, default=8)
    library_ask.add_argument("--config")
    library_ask.add_argument("--provider-kind")
    library_ask.add_argument("--base-url")
    library_ask.add_argument("--model")
    library_ask.add_argument("--timeout-seconds", type=int)
    library_ask.add_argument("--api-key-env", default="PAPERLENS_API_KEY")
    library_ask.add_argument("--offline-debug", action="store_true")

    subparsers.add_parser("version", help="Print PaperLens Core version")

    serve_parser = subparsers.add_parser("serve", help="Run the local PaperLens Core service")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=0)
    serve_parser.add_argument("--token")
    serve_parser.add_argument("--config")

    return parser


def run_command(args: argparse.Namespace) -> int:
    overrides: dict[str, Any] = {
        "provider": {
            "kind": args.provider_kind,
            "base_url": args.base_url,
            "model": args.model,
            "reasoning_model": args.reasoning_model,
            "timeout_seconds": args.timeout_seconds,
            "api_key_env": args.api_key_env,
        },
        "budget": {"max_usd": args.budget} if args.budget is not None else None,
        "concurrency": args.concurrency,
        "offline_debug": args.offline_debug,
        "topic": args.topic,
        "idea": args.idea,
        "output_language": args.output_language,
        "read_mode": args.read_mode,
        "visual_verification_mode": args.visual_verification_mode,
        "visual_verification_max_pages": args.visual_verification_max_pages,
    }
    result = PaperLensEngine().run_job(
        RunRequest(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            config_path=Path(args.config).resolve() if args.config else None,
            config_overrides=overrides,
            from_stage=args.from_stage,
            only_stage=args.only_stage,
            use_stdin_control=True,
        )
    )
    return 0 if result.status == "ok" else 2


def ask_command(args: argparse.Namespace) -> int:
    overrides: dict[str, Any] = {
        "provider": {
            "kind": args.provider_kind,
            "base_url": args.base_url,
            "model": args.model,
            "timeout_seconds": args.timeout_seconds,
            "api_key_env": args.api_key_env,
        },
        "offline_debug": args.offline_debug,
    }
    answer = PaperLensEngine().answer_paper_question(
        PaperQuestionRequest(
            output_dir=Path(args.output_dir),
            config_path=Path(args.config).resolve() if args.config else None,
            config_overrides=overrides,
            paper_id=args.paper_id,
            question=args.question,
        )
    )
    print(json.dumps(answer, ensure_ascii=False, default=str), flush=True)
    return 0


def library_command(args: argparse.Namespace) -> int:
    engine = PaperLensEngine()
    if args.library_command == "build":
        result = engine.build_library(LibraryBuildRequest(output_dir=Path(args.output_dir)))
        print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
        return 0
    if args.library_command == "search":
        result = engine.search_library(
            LibrarySearchRequest(output_dir=Path(args.output_dir), query=args.query, limit=args.limit)
        )
        print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
        return 0
    if args.library_command == "rebuild-index":
        result = engine.rebuild_library_index(
            LibraryRebuildIndexRequest(output_dir=Path(args.output_dir))
        )
        print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
        return 0
    if args.library_command == "doctor":
        result = engine.doctor_library(LibraryDoctorRequest(output_dir=Path(args.output_dir)))
        print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
        return 0
    if args.library_command == "ask":
        overrides: dict[str, Any] = {
            "provider": {
                "kind": args.provider_kind,
                "base_url": args.base_url,
                "model": args.model,
                "timeout_seconds": args.timeout_seconds,
                "api_key_env": args.api_key_env,
            },
            "offline_debug": args.offline_debug,
        }
        answer = engine.answer_library_question(
            LibraryQuestionRequest(
                output_dir=Path(args.output_dir),
                config_path=Path(args.config).resolve() if args.config else None,
                config_overrides=overrides,
                question=args.question,
                limit=args.limit,
            )
        )
        print(json.dumps(answer, ensure_ascii=False, default=str), flush=True)
        return 0
    raise ValueError(f"Unknown library command: {args.library_command}")


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "version":
        print("paperlens-core 0.1.1")
        return 0
    try:
        if args.command == "library":
            return library_command(args)
        if args.command == "serve":
            return serve(
                host=args.host,
                port=args.port,
                token=args.token,
                config_path=Path(args.config).resolve() if args.config else None,
            )
        if args.command == "ask":
            return ask_command(args)
        return run_command(args)
    except Exception as exc:
        emit_fatal(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
