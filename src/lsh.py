"""LSH probability and parameter selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class LshParams:
    """LSH parameterization using the assignment's notation.

    r: number of bands
    b: hash functions (rows) per band
    """

    r: int
    b: int


def lsh_probability(similarity: float, r: int, b: int) -> float:
    """Compute LSH candidate probability f(s) = 1 - (1 - s^b)^r."""
    if similarity < 0.0 or similarity > 1.0:
        raise ValueError("similarity must be in [0, 1]")
    return 1.0 - (1.0 - similarity**b) ** r


def lsh_slope(similarity: float, r: int, b: int) -> float:
    """Derivative of f(s) at a given similarity value."""
    if similarity <= 0.0:
        return 0.0
    if similarity >= 1.0:
        return 0.0
    return r * b * (similarity ** (b - 1)) * ((1.0 - similarity**b) ** (r - 1))


def factor_pairs(n: int) -> Iterable[tuple[int, int]]:
    """Yield (r, b) pairs such that r * b = n."""
    for r in range(1, n + 1):
        if n % r == 0:
            yield (r, n // r)


def choose_lsh_params(t: int, tau: float) -> LshParams:
    """Choose (r, b) with good separation at tau.

    Heuristic: pick (r, b) that makes f(tau) closest to 0.5,
    and break ties using the steepest slope at tau.
    """
    best_params: LshParams | None = None
    best_distance: float | None = None
    best_slope: float | None = None

    for r, b in factor_pairs(t):
        f_tau = lsh_probability(tau, r, b)
        distance = abs(f_tau - 0.5)
        slope = lsh_slope(tau, r, b)

        if best_params is None:
            best_params = LshParams(r=r, b=b)
            best_distance = distance
            best_slope = slope
            continue

        if distance < (best_distance or 0.0):
            best_params = LshParams(r=r, b=b)
            best_distance = distance
            best_slope = slope
        elif distance == best_distance and slope > (best_slope or 0.0):
            best_params = LshParams(r=r, b=b)
            best_distance = distance
            best_slope = slope

    if best_params is None:
        raise ValueError("No valid LSH parameters found")

    return best_params
