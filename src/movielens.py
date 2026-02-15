"""MovieLens 100k dataset loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _resolve_ratings_path(path: Path) -> Path:
    """Resolve to the ratings file (u.data) given a file or directory path."""
    if path.is_dir():
        candidate = path / "u.data"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"u.data not found in directory: {path}")
    if path.exists():
        return path
    raise FileNotFoundError(f"Ratings file not found: {path}")


def load_user_movie_sets(ratings_path: Path) -> dict[int, set[int]]:
    """Load MovieLens ratings and return user -> set of movie IDs.

    Expected format: tab-separated with columns user_id, movie_id, rating, timestamp.
    Only user_id and movie_id are used.
    """
    resolved = _resolve_ratings_path(ratings_path)
    user_movies: dict[int, set[int]] = {}
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            user_id = int(parts[0])
            movie_id = int(parts[1])
            user_movies.setdefault(user_id, set()).add(movie_id)
    return user_movies


def sorted_user_ids(user_movies: dict[int, set[int]]) -> list[int]:
    """Return user IDs in sorted order."""
    return sorted(user_movies.keys())


def iter_user_sets(user_movies: dict[int, set[int]]) -> Iterable[set[int]]:
    """Yield user movie sets in sorted user id order."""
    for user_id in sorted_user_ids(user_movies):
        yield user_movies[user_id]
