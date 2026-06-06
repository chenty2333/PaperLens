from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from paperlens_core.schemas import (
    ArtifactVersion,
    ClassificationDecision,
    PageArtifact,
    PaperRecord,
    PaperState,
    ReviewItem,
    SkimCard,
)


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS paper_records (
  paper_id TEXT PRIMARY KEY,
  file_path TEXT NOT NULL,
  file_hash TEXT NOT NULL,
  canonical_title TEXT,
  authors_json TEXT NOT NULL,
  year INTEGER,
  page_count INTEGER NOT NULL,
  duplicate_group TEXT,
  status TEXT NOT NULL,
  parse_quality TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  venue TEXT,
  doi TEXT,
  arxiv_id TEXT,
  bibtex_key TEXT,
  side_statuses_json TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_paper_hash ON paper_records(file_hash);

CREATE TABLE IF NOT EXISTS page_artifacts (
  paper_id TEXT NOT NULL,
  page_no INTEGER NOT NULL,
  artifact_json TEXT NOT NULL,
  PRIMARY KEY (paper_id, page_no)
);

CREATE TABLE IF NOT EXISTS skim_cards (
  paper_id TEXT PRIMARY KEY,
  card_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS classifications (
  paper_id TEXT PRIMARY KEY,
  decision_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_state (
  key TEXT PRIMARY KEY,
  value_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_states (
  paper_id TEXT PRIMARY KEY,
  state_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact_versions (
  artifact_id TEXT PRIMARY KEY,
  artifact_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_items (
  item_id TEXT PRIMARY KEY,
  item_json TEXT NOT NULL
);
"""


class ArtifactDb:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._ensure_paper_columns()
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _ensure_paper_columns(self) -> None:
        existing = {
            row["name"] for row in self.conn.execute("PRAGMA table_info(paper_records)").fetchall()
        }
        migrations = {
            "venue": "ALTER TABLE paper_records ADD COLUMN venue TEXT",
            "doi": "ALTER TABLE paper_records ADD COLUMN doi TEXT",
            "arxiv_id": "ALTER TABLE paper_records ADD COLUMN arxiv_id TEXT",
            "bibtex_key": "ALTER TABLE paper_records ADD COLUMN bibtex_key TEXT",
            "side_statuses_json": "ALTER TABLE paper_records ADD COLUMN side_statuses_json TEXT NOT NULL DEFAULT '[]'",
        }
        for column, statement in migrations.items():
            if column not in existing:
                self.conn.execute(statement)

    def upsert_paper(self, paper: PaperRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO paper_records (
              paper_id, file_path, file_hash, canonical_title, authors_json, year,
              page_count, duplicate_group, status, parse_quality, created_at, updated_at,
              venue, doi, arxiv_id, bibtex_key, side_statuses_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
              file_path=excluded.file_path,
              canonical_title=excluded.canonical_title,
              authors_json=excluded.authors_json,
              year=excluded.year,
              page_count=excluded.page_count,
              duplicate_group=excluded.duplicate_group,
              status=excluded.status,
              parse_quality=excluded.parse_quality,
              updated_at=excluded.updated_at,
              venue=excluded.venue,
              doi=excluded.doi,
              arxiv_id=excluded.arxiv_id,
              bibtex_key=excluded.bibtex_key,
              side_statuses_json=excluded.side_statuses_json
            """,
            (
                paper.paper_id,
                paper.file_path,
                paper.file_hash,
                paper.canonical_title,
                json.dumps(paper.authors, ensure_ascii=False),
                paper.year,
                paper.page_count,
                paper.duplicate_group,
                paper.status,
                paper.parse_quality,
                paper.created_at,
                paper.updated_at,
                paper.venue,
                paper.doi,
                paper.arxiv_id,
                paper.bibtex_key,
                json.dumps(paper.side_statuses, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def update_paper_status(
        self, paper_id: str, status: str, parse_quality: str | None = None
    ) -> None:
        self.conn.execute(
            "UPDATE paper_records SET status=?, parse_quality=COALESCE(?, parse_quality) WHERE paper_id=?",
            (status, parse_quality, paper_id),
        )
        self.conn.commit()

    def insert_page_artifacts(self, artifacts: Iterable[PageArtifact]) -> None:
        self.conn.executemany(
            """
            INSERT INTO page_artifacts (paper_id, page_no, artifact_json)
            VALUES (?, ?, ?)
            ON CONFLICT(paper_id, page_no) DO UPDATE SET artifact_json=excluded.artifact_json
            """,
            [
                (artifact.paper_id, artifact.page_no, artifact.model_dump_json())
                for artifact in artifacts
            ],
        )
        self.conn.commit()

    def upsert_skim(self, card: SkimCard) -> None:
        self.conn.execute(
            """
            INSERT INTO skim_cards (paper_id, card_json)
            VALUES (?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET card_json=excluded.card_json
            """,
            (card.paper_id, card.model_dump_json()),
        )
        self.conn.commit()

    def upsert_classification(self, decision: ClassificationDecision) -> None:
        self.conn.execute(
            """
            INSERT INTO classifications (paper_id, decision_json)
            VALUES (?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET decision_json=excluded.decision_json
            """,
            (decision.paper_id, decision.model_dump_json()),
        )
        self.conn.commit()

    def upsert_paper_state(self, state: PaperState) -> None:
        self.conn.execute(
            """
            INSERT INTO paper_states (paper_id, state_json)
            VALUES (?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET state_json=excluded.state_json
            """,
            (state.paper_id, state.model_dump_json()),
        )
        self.conn.commit()

    def get_paper_state(self, paper_id: str) -> PaperState | None:
        row = self.conn.execute(
            "SELECT state_json FROM paper_states WHERE paper_id=?",
            (paper_id,),
        ).fetchone()
        return PaperState.model_validate_json(row["state_json"]) if row else None

    def list_papers(self) -> list[PaperRecord]:
        rows = self.conn.execute("SELECT * FROM paper_records ORDER BY paper_id").fetchall()
        return [
            PaperRecord(
                paper_id=row["paper_id"],
                file_path=row["file_path"],
                file_hash=row["file_hash"],
                canonical_title=row["canonical_title"],
                authors=json.loads(row["authors_json"]),
                year=row["year"],
                page_count=row["page_count"],
                duplicate_group=row["duplicate_group"],
                status=row["status"],
                parse_quality=row["parse_quality"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                venue=row["venue"],
                doi=row["doi"],
                arxiv_id=row["arxiv_id"],
                bibtex_key=row["bibtex_key"],
                side_statuses=json.loads(row["side_statuses_json"] or "[]"),
            )
            for row in rows
        ]

    def list_skim_cards(self) -> list[SkimCard]:
        rows = self.conn.execute("SELECT card_json FROM skim_cards ORDER BY paper_id").fetchall()
        return [SkimCard.model_validate_json(row["card_json"]) for row in rows]

    def list_classifications(self) -> list[ClassificationDecision]:
        rows = self.conn.execute(
            "SELECT decision_json FROM classifications ORDER BY paper_id"
        ).fetchall()
        return [ClassificationDecision.model_validate_json(row["decision_json"]) for row in rows]

    def upsert_artifact_version(self, artifact: ArtifactVersion) -> None:
        self.conn.execute(
            """
            INSERT INTO artifact_versions (artifact_id, artifact_json)
            VALUES (?, ?)
            ON CONFLICT(artifact_id) DO UPDATE SET artifact_json=excluded.artifact_json
            """,
            (artifact.artifact_id, artifact.model_dump_json()),
        )
        self.conn.commit()

    def list_artifact_versions(self) -> list[ArtifactVersion]:
        rows = self.conn.execute(
            "SELECT artifact_json FROM artifact_versions ORDER BY artifact_id"
        ).fetchall()
        return [ArtifactVersion.model_validate_json(row["artifact_json"]) for row in rows]

    def upsert_review_item(self, item: ReviewItem) -> None:
        self.conn.execute(
            """
            INSERT INTO review_items (item_id, item_json)
            VALUES (?, ?)
            ON CONFLICT(item_id) DO UPDATE SET item_json=excluded.item_json
            """,
            (item.item_id, item.model_dump_json()),
        )
        self.conn.commit()

    def list_review_items(self) -> list[ReviewItem]:
        rows = self.conn.execute("SELECT item_json FROM review_items ORDER BY item_id").fetchall()
        return [ReviewItem.model_validate_json(row["item_json"]) for row in rows]

    def get_page_artifacts(self, paper_id: str) -> list[PageArtifact]:
        rows = self.conn.execute(
            "SELECT artifact_json FROM page_artifacts WHERE paper_id=? ORDER BY page_no",
            (paper_id,),
        ).fetchall()
        return [PageArtifact.model_validate_json(row["artifact_json"]) for row in rows]

    def set_state(self, key: str, value: Any) -> None:
        self.conn.execute(
            """
            INSERT INTO run_state (key, value_json) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json
            """,
            (key, json.dumps(value, ensure_ascii=False, default=str)),
        )
        self.conn.commit()

    def get_state(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute(
            "SELECT value_json FROM run_state WHERE key=?",
            (key,),
        ).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value_json"])
        except json.JSONDecodeError:
            return default
