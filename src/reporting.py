"""Reporting helpers for Parts 1–3 outputs."""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

from . import jaccard as jaccard_lib
from . import kgrams as kgrams_lib
from . import lsh as lsh_lib
from . import minhash as minhash_lib


Pair = tuple[str, str]


def pairwise_ids(doc_ids: Sequence[str]) -> list[Pair]:
    """Return all unordered document pairs."""
    return [(a, b) for a, b in combinations(doc_ids, 2)]


def compute_pairwise_jaccard(grams_by_doc: dict[str, set[str]]) -> dict[Pair, float]:
    """Compute Jaccard similarities for all doc pairs."""
    doc_ids = sorted(grams_by_doc.keys())
    results: dict[Pair, float] = {}
    for a, b in pairwise_ids(doc_ids):
        results[(a, b)] = jaccard_lib.jaccard_similarity(grams_by_doc[a], grams_by_doc[b])
    return results


def compute_kgram_jaccards(
    documents: dict[str, str],
    specs: Sequence[tuple[str, int]],
) -> dict[str, dict[Pair, float]]:
    """Compute pairwise Jaccard values for each k-gram spec."""
    output: dict[str, dict[Pair, float]] = {}
    for mode, k in specs:
        key = f"{mode}_{k}"
        grams_by_doc = kgrams_lib.build_kgrams_for_documents(documents, mode, k)
        output[key] = compute_pairwise_jaccard(grams_by_doc)
    return output


def compute_minhash_estimates(
    grams_d1: set[str],
    grams_d2: set[str],
    t_values: Sequence[int],
    m: int,
    seed: int,
) -> list[dict[str, float]]:
    """Compute MinHash estimates for each t and include errors."""
    exact = jaccard_lib.jaccard_similarity(grams_d1, grams_d2)
    rows: list[dict[str, float]] = []
    for t in t_values:
        estimate = minhash_lib.approximate_jaccard(grams_d1, grams_d2, t=t, m=m, seed=seed)
        rows.append(
            {
                "t": float(t),
                "estimate": estimate,
                "exact": exact,
                "abs_error": abs(estimate - exact),
            }
        )
    return rows


def recommend_t(rows: Sequence[dict[str, float]]) -> int:
    """Recommend a good t based on diminishing error reduction.

    Heuristic: pick the smallest t such that additional gains are < 0.01.
    """
    if not rows:
        raise ValueError("No rows to analyze")

    sorted_rows = sorted(rows, key=lambda r: r["t"])
    errors = [row["abs_error"] for row in sorted_rows]
    ts = [int(row["t"]) for row in sorted_rows]

    for i in range(1, len(errors)):
        improvement = errors[i - 1] - errors[i]
        if improvement < 0.01:
            return ts[i]
    return ts[-1]


def compute_lsh_probabilities(
    jaccard_by_pair: dict[Pair, float],
    params: lsh_lib.LshParams,
) -> dict[Pair, float]:
    """Compute LSH candidate probabilities for each pair."""
    results: dict[Pair, float] = {}
    for pair, sim in jaccard_by_pair.items():
        results[pair] = lsh_lib.lsh_probability(sim, r=params.r, b=params.b)
    return results
