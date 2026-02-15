"""CLI entrypoint for Parts 1–3 of the MinHash/LSH assignment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from . import config
from . import io_utils
from . import kgrams
from . import lsh
from . import reporting


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="MinHash and LSH assignment runner")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/minhash"),
        help="Directory containing D1.txt–D4.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to write result files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.DEFAULT_SEED,
        help="Random seed for MinHash",
    )
    parser.add_argument(
        "--minhash-m",
        type=int,
        default=config.MINHASH_M,
        help="Range size m for hash functions (must be > 10,000)",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Disable printing results to the terminal",
    )
    return parser.parse_args(argv)


def write_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    """Write a simple CSV file."""
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Format rows as a fixed-width table for terminal output."""
    if not rows:
        return "No rows to display."

    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * width for width in widths)
    body_lines = [
        " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows
    ]
    return "\n".join([header_line, separator, *body_lines])


def run() -> None:
    """Run Parts 1–3 and write outputs."""
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load documents
    documents = io_utils.load_documents(args.data_dir, config.DOC_NAMES)

    # Part 1: k-grams and pairwise Jaccard
    kgram_jaccards = reporting.compute_kgram_jaccards(documents, config.KGRAM_SPECS)
    part1_rows: list[list[str]] = []
    for kgram_type, pairs in kgram_jaccards.items():
        for (doc_a, doc_b), value in pairs.items():
            part1_rows.append([kgram_type, doc_a, doc_b, f"{value:.6f}"])

    write_csv(
        output_dir / "part1_kgrams_jaccard.csv",
        headers=["kgram_type", "doc_a", "doc_b", "jaccard"],
        rows=part1_rows,
    )

    # Part 2: MinHash for D1 vs D2 with 3-grams (characters)
    grams_3 = kgrams.build_kgrams_for_documents(documents, "char", 3)
    d1_grams = grams_3["D1"]
    d2_grams = grams_3["D2"]

    minhash_rows = reporting.compute_minhash_estimates(
        d1_grams,
        d2_grams,
        t_values=config.MINHASH_T_VALUES,
        m=args.minhash_m,
        seed=args.seed,
    )

    part2_rows: list[list[str]] = []
    for row in minhash_rows:
        part2_rows.append(
            [
                f"{int(row['t'])}",
                f"{row['estimate']:.6f}",
                f"{row['exact']:.6f}",
                f"{row['abs_error']:.6f}",
            ]
        )

    write_csv(
        output_dir / "part2_minhash_d1_d2.csv",
        headers=["t", "estimate", "exact", "abs_error"],
        rows=part2_rows,
    )

    recommended_t = reporting.recommend_t(minhash_rows)
    recommendation_text = (
        "Recommended t based on diminishing error improvement:\n"
        f"t = {recommended_t}\n\n"
        "Justification: smallest t where additional error reduction < 0.01."
    )
    (output_dir / "part2_t_recommendation.txt").write_text(
        recommendation_text, encoding="utf-8"
    )

    # Part 3: LSH parameter selection and probabilities
    params = lsh.choose_lsh_params(config.LSH_T, config.LSH_TAU)
    tau_probability = lsh.lsh_probability(config.LSH_TAU, params.r, params.b)
    tau_slope = lsh.lsh_slope(config.LSH_TAU, params.r, params.b)

    lsh_params_text = (
        f"LSH parameters (t={config.LSH_T}, tau={config.LSH_TAU}):\n"
        f"r (bands) = {params.r}\n"
        f"b (rows per band) = {params.b}\n"
        f"f(tau) = {tau_probability:.6f}\n"
        f"slope(tau) = {tau_slope:.6f}\n"
    )
    (output_dir / "part3_lsh_params.txt").write_text(lsh_params_text, encoding="utf-8")

    # Probabilities for each pair using 3-grams
    pairwise_jaccard = kgram_jaccards["char_3"]
    lsh_probabilities = reporting.compute_lsh_probabilities(pairwise_jaccard, params)

    part3_rows: list[list[str]] = []
    for (doc_a, doc_b), prob in lsh_probabilities.items():
        part3_rows.append(
            [
                doc_a,
                doc_b,
                f"{pairwise_jaccard[(doc_a, doc_b)]:.6f}",
                f"{prob:.6f}",
            ]
        )

    write_csv(
        output_dir / "part3_lsh_probabilities.csv",
        headers=["doc_a", "doc_b", "jaccard_3gram", "probability"],
        rows=part3_rows,
    )

    if not args.no_print:
        print("\n=== Part 1: K-grams Jaccard ===")
        print(format_table(["kgram_type", "doc_a", "doc_b", "jaccard"], part1_rows))

        print("\n=== Part 2: MinHash Estimates (D1 vs D2) ===")
        print(format_table(["t", "estimate", "exact", "abs_error"], part2_rows))
        print("\n" + recommendation_text.strip())

        print("\n=== Part 3: LSH Parameters ===")
        print(lsh_params_text.strip())

        print("\n=== Part 3: LSH Probabilities (3-grams) ===")
        print(
            format_table(
                ["doc_a", "doc_b", "jaccard_3gram", "probability"], part3_rows
            )
        )

    print("\nOutputs written to:", output_dir)


if __name__ == "__main__":
    run()
