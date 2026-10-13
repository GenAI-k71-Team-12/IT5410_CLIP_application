"""Run from repo root. --check rejects contract drift in CI."""

import argparse
import json
from pathlib import Path

from clip_retrieval.backend.app import create_app
from clip_retrieval.contracts import Metadata

parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
outputs = {
    "contracts/openapi.json": create_app("mock").openapi(),
    "contracts/metadata.schema.json": Metadata.model_json_schema(),
}
for name, payload in outputs.items():
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path = Path(name)
    if args.check:
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            raise SystemExit(f"Contract drift: regenerate {name}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
print("Contracts are up to date")
