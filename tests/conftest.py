import importlib.util
import sys
from pathlib import Path

# Cho phép `import scripts.*` khi chạy `pytest` trực tiếp, không chỉ `python -m pytest`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# tests/ml needs requirements-ml.txt; the contract CI job installs requirements-lock.txt only.
collect_ignore = [] if all(map(importlib.util.find_spec, ("numpy", "faiss", "PIL"))) else ["ml"]
