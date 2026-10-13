"""Validate JSONL, identity uniqueness and optionally actual files/dimensions."""

import argparse
import json
from pathlib import Path

from clip_retrieval.contracts import Metadata

parser = argparse.ArgumentParser()
parser.add_argument("file", type=Path)
parser.add_argument("--data-root", type=Path)
args = parser.parse_args()
seen = set()
count = 0
for line_no, line in enumerate(args.file.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
        continue
    try:
        record = Metadata.model_validate(json.loads(line))
        if record.id in seen:
            raise ValueError(f"Duplicate image id: {record.id}")
        seen.add(record.id)
        if args.data_root:
            root = args.data_root.resolve()
            for relative in (record.image_path, record.thumbnail_path):
                if relative is not None:
                    path = (root / relative).resolve()
                    if not path.is_relative_to(root) or not path.is_file():
                        raise ValueError(f"Missing or unsafe file: {relative}")
        count += 1
    except Exception as exc:
        raise SystemExit(f"Line {line_no}: {exc}") from exc
if count == 0:
    raise SystemExit("Metadata cannot be empty")
print(f"Valid metadata: {count} unique images")
