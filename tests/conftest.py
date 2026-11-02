import importlib.util

# tests/ml needs requirements-ml.txt; the contract CI job installs requirements-lock.txt only.
collect_ignore = [] if all(map(importlib.util.find_spec, ("numpy", "faiss", "PIL"))) else ["ml"]
