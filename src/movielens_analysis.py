"""Analysis utilities for MovieLens parts (4–5)."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np

from . import jaccard as jaccard_lib
from . import lsh as lsh_lib
from . import minhash as minhash_lib

Pair = tuple[int, int]


def iter_pairs(user_ids: list[int]) -> Iterable[Pair]:
    """Yield all unordered user id pairs."""
    return combinations(user_ids, 2)


def compute_exact_jaccard(
    user_ids: list[int],
    user_sets: list[set[int]],
) -> dict[Pair, float]:
    """Compute exact Jaccard for all user pairs."""
    results: dict[Pair, float] = {}
    for i, j in combinations(range(len(user_ids)), 2):
        pair = (user_ids[i], user_ids[j])
        results[pair] = jaccard_lib.jaccard_similarity(user_sets[i], user_sets[j])
    return results


def pairs_above_threshold(
    similarities: dict[Pair, float],
    threshold: float,
) -> set[Pair]:
    """Return all pairs with similarity >= threshold."""
    return {pair for pair, sim in similarities.items() if sim >= threshold}


def minhash_signatures_matrix(
    user_sets: list[set[int]],
    t: int,
    m: int,
    seed: int,
) -> np.ndarray:
    """Compute a MinHash signature matrix of shape (num_users, t)."""
    hash_functions = minhash_lib.generate_hash_functions(t, seed)
    signatures = minhash_lib.minhash_signatures_for_sets(user_sets, hash_functions, m)
    return np.array(signatures, dtype=np.uint32)


def estimated_pairs_from_signatures(
    user_ids: list[int],
    signatures: np.ndarray,
    threshold: float,
) -> dict[Pair, float]:
    """Estimate Jaccard for all pairs and return those >= threshold."""
    num_users, t = signatures.shape
    results: dict[Pair, float] = {}

    for i in range(num_users):
        sig_i = signatures[i]
        for j in range(i + 1, num_users):
            sim = float(np.mean(sig_i == signatures[j]))
            if sim >= threshold:
                results[(user_ids[i], user_ids[j])] = sim
    return results


def lsh_candidate_pairs(
    user_ids: list[int],
    signatures: np.ndarray,
    params: lsh_lib.LshParams,
) -> set[Pair]:
    """Compute LSH candidate pairs for given signatures and parameters."""
    num_users, t = signatures.shape
    if params.r * params.b != t:
        raise ValueError("LSH params r*b must equal signature length")

    candidates: set[Pair] = set()
    for band_index in range(params.r):
        start = band_index * params.b
        end = start + params.b
        buckets: dict[tuple[int, ...], list[int]] = {}
        for i in range(num_users):
            key = tuple(signatures[i, start:end].tolist())
            buckets.setdefault(key, []).append(i)

        for bucket_indices in buckets.values():
            if len(bucket_indices) < 2:
                continue
            for a, b in combinations(bucket_indices, 2):
                pair = (user_ids[a], user_ids[b])
                candidates.add(pair)

    return candidates


def compute_fp_fn(
    predicted_pairs: set[Pair],
    true_pairs: set[Pair],
) -> tuple[int, int]:
    """Compute false positives and false negatives given predicted/true pairs."""
    false_positives = len(predicted_pairs - true_pairs)
    false_negatives = len(true_pairs - predicted_pairs)
    return false_positives, false_negatives
