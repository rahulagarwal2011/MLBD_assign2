"""K-gram construction utilities."""

from __future__ import annotations

from typing import Iterable


def char_kgrams(text: str, k: int) -> set[str]:
    """Create character k-grams (including spaces)."""
    if k <= 0:
        raise ValueError("k must be positive")
    if len(text) < k:
        return set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def word_kgrams(text: str, k: int) -> set[str]:
    """Create word-based k-grams (space-delimited)."""
    if k <= 0:
        raise ValueError("k must be positive")
    words = text.split(" ") if text else []
    if len(words) < k:
        return set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def build_kgrams_for_documents(
    documents: dict[str, str],
    mode: str,
    k: int,
) -> dict[str, set[str]]:
    """Build k-grams for each document.

    mode: "char" or "word".
    """
    grams_by_doc: dict[str, set[str]] = {}
    for doc_id, text in documents.items():
        if mode == "char":
            grams_by_doc[doc_id] = char_kgrams(text, k)
        elif mode == "word":
            grams_by_doc[doc_id] = word_kgrams(text, k)
        else:
            raise ValueError(f"Unsupported k-gram mode: {mode}")
    return grams_by_doc


def iter_kgram_types() -> Iterable[tuple[str, int]]:
    """Yield supported k-gram configurations."""
    yield ("char", 2)
    yield ("char", 3)
    yield ("word", 2)
