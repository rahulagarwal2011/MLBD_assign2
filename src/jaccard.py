"""Jaccard similarity utilities."""

from __future__ import annotations

from typing import AbstractSet, TypeVar

T = TypeVar("T")


def jaccard_similarity(a: AbstractSet[T], b: AbstractSet[T]) -> float:
    """Compute Jaccard similarity between two sets.

    If both sets are empty, returns 1.0 by convention.
    """
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)
