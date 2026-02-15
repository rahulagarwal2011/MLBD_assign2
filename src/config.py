"""Project configuration constants."""

from __future__ import annotations

DOC_NAMES: tuple[str, ...] = ("D1.txt", "D2.txt", "D3.txt", "D4.txt")

# K-gram specs: (mode, k)
KGRAM_SPECS: tuple[tuple[str, int], ...] = (
    ("char", 2),
    ("char", 3),
    ("word", 2),
)

# MinHash settings for Part 2
MINHASH_T_VALUES: tuple[int, ...] = (20, 60, 150, 300, 600)
MINHASH_M: int = 10007  # must be > 10,000

# LSH settings for Part 3
LSH_T: int = 160
LSH_TAU: float = 0.7

# Reproducibility
DEFAULT_SEED: int = 42

# MovieLens settings for Parts 4–5
MOVIELENS_RATINGS_PATH: str = "data/movielens/ml-100k/u.data"
MOVIELENS_T_VALUES: tuple[int, ...] = (50, 100, 200)
MOVIELENS_SIM_THRESHOLD: float = 0.5
MOVIELENS_LSH_THRESHOLDS: tuple[float, ...] = (0.6, 0.8)
MOVIELENS_RUNS: int = 5
MOVIELENS_LSH_CONFIGS: tuple[tuple[int, int, int], ...] = (
    (50, 5, 10),
    (100, 5, 20),
    (200, 5, 40),
    (200, 10, 20),
)
