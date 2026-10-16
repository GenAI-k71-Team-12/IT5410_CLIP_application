"""SHA-256 of files and of canonical JSON (sorted keys, compact separators, UTF-8, no newline)."""

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()
