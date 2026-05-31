from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def paper_id_from_hash(file_hash: str) -> str:
    return f"p_{file_hash[:16]}"
