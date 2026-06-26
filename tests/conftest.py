"""Shared pytest configuration ensuring the project root is importable."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Generator
from pathlib import Path
from uuid import uuid4

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def tmp_path() -> Generator[Path, None, None]:
    """Provide repo-local temp dirs when the Windows temp root is inaccessible."""
    base = _ROOT / ".pytest_tmp"
    base.mkdir(exist_ok=True)
    path = base / uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
