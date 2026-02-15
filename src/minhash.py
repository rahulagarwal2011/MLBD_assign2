"""MinHash implementation utilities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Iterable, Sequence

PRIME: int = 4_294_967_311  # largest 32-bit prime


@dataclass(frozen=True)
class HashFunction:
    """Linear hash function h(x) = (a*x + b) mod prime."""

    a: int
    b: int
    prime: int = PRIME

    def apply(self, x: int, m: int) -> int:
        """Apply the hash function and map into [0, m)."""
        return ((self.a * x + self.b) % self.prime) % m


def generate_hash_functions(count: int, seed: int) -> list[HashFunction]:
    """Generate a list of hash functions with a fixed seed."""
    rng = random.Random(seed)
    functions: list[HashFunction] = []
    for _ in range(count):
        a = rng.randrange(1, PRIME)
        b = rng.randrange(0, PRIME)
        functions.append(HashFunction(a=a, b=b))
    return functions


def hash_token(token: str) -> int:
    """Hash a token deterministically to an integer."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def minhash_signature(
    tokens: Iterable[str],
    hash_functions: Sequence[HashFunction],
    m: int,
) -> list[int]:
    """Compute the MinHash signature for a set of tokens."""
    token_hashes = [hash_token(token) for token in tokens]
    if not token_hashes:
        return [m] * len(hash_functions)

    signature: list[int] = []
    for func in hash_functions:
        min_value = min(func.apply(h, m) for h in token_hashes)
        signature.append(min_value)
    return signature


def minhash_signature_from_ints(
    items: Iterable[int],
    hash_functions: Sequence[HashFunction],
    m: int,
) -> list[int]:
    """Compute the MinHash signature for a set of integer items."""
    item_list = list(items)
    if not item_list:
        return [m] * len(hash_functions)

    signature: list[int] = []
    for func in hash_functions:
        min_value = min(func.apply(item, m) for item in item_list)
        signature.append(min_value)
    return signature


def minhash_signatures_for_sets(
    item_sets: Sequence[set[int]],
    hash_functions: Sequence[HashFunction],
    m: int,
) -> list[list[int]]:
    """Compute MinHash signatures for multiple item sets."""
    signatures: list[list[int]] = []
    for items in item_sets:
        signatures.append(minhash_signature_from_ints(items, hash_functions, m))
    return signatures


def estimate_jaccard_from_signatures(sig_a: Sequence[int], sig_b: Sequence[int]) -> float:
    """Estimate Jaccard similarity from two signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("Signatures must have the same length")
    if not sig_a:
        return 1.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def approximate_jaccard(
    tokens_a: Iterable[str],
    tokens_b: Iterable[str],
    t: int,
    m: int,
    seed: int,
) -> float:
    """Compute MinHash-based Jaccard estimate for two token sets."""
    hash_functions = generate_hash_functions(t, seed)
    sig_a = minhash_signature(tokens_a, hash_functions, m)
    sig_b = minhash_signature(tokens_b, hash_functions, m)
    return estimate_jaccard_from_signatures(sig_a, sig_b)
