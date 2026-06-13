from __future__ import annotations

import hashlib
from pathlib import Path


def hash_file(path: Path, *, length: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    hexdigest = digest.hexdigest()
    return hexdigest[:length] if length is not None else hexdigest
