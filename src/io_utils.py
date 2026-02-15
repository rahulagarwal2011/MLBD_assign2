"""I/O helpers for loading and normalizing documents."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_document(text: str) -> str:
    """Normalize document text to lowercase and single spaces.

    The assignment states documents contain only lowercase letters and spaces.
    We defensively normalize whitespace to a single space and strip edges.
    """
    normalized = text.strip().lower()
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized


def read_document(path: Path) -> str:
    """Read a document from disk and normalize it."""
    if not path.exists():
        raise FileNotFoundError(f"Missing document: {path}")
    raw = path.read_text(encoding="utf-8")
    return normalize_document(raw)


def load_documents(data_dir: Path, doc_names: Sequence[str]) -> dict[str, str]:
    """Load all documents from a directory by filename.

    Returns a mapping from document id (e.g., "D1") to normalized text.
    """
    documents: dict[str, str] = {}
    for name in doc_names:
        doc_id = Path(name).stem
        documents[doc_id] = read_document(data_dir / name)
    return documents
