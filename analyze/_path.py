"""
_path.py — Repo-root bootstrap for analyze/ scripts.
Import as the first local import:  import _path  # noqa: F401
"""
from __future__ import annotations
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
