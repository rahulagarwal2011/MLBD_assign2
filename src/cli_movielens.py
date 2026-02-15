"""CLI entrypoint for MovieLens parts (4–5)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from . import config
from . import lsh
from . import movielens
from . import movielens_analysis


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="MovieLens MinHash/LSH runner")
    parser.add_argument(
        "--ratings-path",
        type=Path,
        default=Path(config.MOVIELENS_RATINGS_PATH),
        help="Path to MovieLens u.data (or directory containing it)",
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
        help="Base random seed for MinHash runs",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=config.MOVIELENS_RUNS,
        help="Number of random runs to average",
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
    parser.add_argument(
        "--print-limit",
        type=int,
        default=10,
        help="Max rows to print for large tables (0 for all)",
    )
    return parser.parse_args(argv)


def write_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    """Write a simple CSV file."""
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_table(
    headers: Sequence[str], rows: Sequence[Sequence[str]], limit: int | None
) -> str:
    """Format rows as a fixed-width table for terminal output."""
    if limit is not None:
        rows = rows[:limit]
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
    """Run Parts 4–5 and write outputs."""
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    user_movies = movielens.load_user_movie_sets(args.ratings_path)
    user_ids = movielens.sorted_user_ids(user_movies)
    user_sets = [user_movies[user_id] for user_id in user_ids]
    print_limit = None if args.print_limit <= 0 else args.print_limit

    # Exact Jaccard for all pairs
    exact_jaccard = movielens_analysis.compute_exact_jaccard(user_ids, user_sets)

    # True pairs for thresholds
    true_pairs_by_threshold: dict[float, set[movielens_analysis.Pair]] = {}
    for threshold in (config.MOVIELENS_SIM_THRESHOLD, *config.MOVIELENS_LSH_THRESHOLDS):
        true_pairs_by_threshold[threshold] = movielens_analysis.pairs_above_threshold(
            exact_jaccard, threshold
        )

    # Part 4: output pairs with exact similarity >= 0.5
    exact_pairs = true_pairs_by_threshold[config.MOVIELENS_SIM_THRESHOLD]
    exact_rows: list[list[str]] = []
    for (u, v), sim in exact_jaccard.items():
        if (u, v) in exact_pairs:
            exact_rows.append([str(u), str(v), f"{sim:.6f}"])

    write_csv(
        output_dir / "part4_exact_pairs_ge_0_5.csv",
        headers=["user_a", "user_b", "jaccard"],
        rows=exact_rows,
    )

    # Part 4: MinHash estimates (t=50,100,200), 5 runs
    part4_summary_rows: list[list[str]] = []
    run1_pairs_by_t: dict[int, dict[movielens_analysis.Pair, float]] = {}

    # Precompute signatures for each t and run
    signatures_by_t: dict[int, list[object]] = {}
    for t in config.MOVIELENS_T_VALUES:
        signatures_by_t[t] = []
        for run_index in range(args.runs):
            seed = args.seed + run_index
            signatures = movielens_analysis.minhash_signatures_matrix(
                user_sets=user_sets,
                t=t,
                m=args.minhash_m,
                seed=seed,
            )
            signatures_by_t[t].append(signatures)

        fp_total = 0
        fn_total = 0
        run1_pairs: dict[movielens_analysis.Pair, float] | None = None
        for run_index, signatures in enumerate(signatures_by_t[t]):
            estimated_pairs = movielens_analysis.estimated_pairs_from_signatures(
                user_ids=user_ids,
                signatures=signatures,
                threshold=config.MOVIELENS_SIM_THRESHOLD,
            )
            if run_index == 0:
                run1_pairs = estimated_pairs
            predicted_set = set(estimated_pairs.keys())
            fp, fn = movielens_analysis.compute_fp_fn(
                predicted_set, true_pairs_by_threshold[config.MOVIELENS_SIM_THRESHOLD]
            )
            fp_total += fp
            fn_total += fn

        avg_fp = fp_total / args.runs
        avg_fn = fn_total / args.runs
        part4_summary_rows.append([str(t), f"{avg_fp:.2f}", f"{avg_fn:.2f}"])

        if run1_pairs is not None:
            run1_pairs_by_t[t] = run1_pairs
            run1_rows = [
                [str(u), str(v), f"{sim:.6f}"] for (u, v), sim in run1_pairs.items()
            ]
            write_csv(
                output_dir / f"part4_minhash_pairs_t{t}_run1.csv",
                headers=["user_a", "user_b", "estimate"],
                rows=run1_rows,
            )

    write_csv(
        output_dir / "part4_minhash_summary.csv",
        headers=["t", "avg_false_positives", "avg_false_negatives"],
        rows=part4_summary_rows,
    )

    # Part 5: LSH with fixed parameters for thresholds 0.6 and 0.8
    part5_candidate_counts: dict[float, list[tuple[int, int, int, int]]] = {}
    part5_candidate_pairs: dict[float, dict[tuple[int, int, int], set[movielens_analysis.Pair]]] = {}
    for tau in config.MOVIELENS_LSH_THRESHOLDS:
        part5_rows: list[list[str]] = []
        true_pairs = true_pairs_by_threshold[tau]
        part5_candidate_counts[tau] = []
        part5_candidate_pairs[tau] = {}

        for t, r, b in config.MOVIELENS_LSH_CONFIGS:
            params = lsh.LshParams(r=r, b=b)
            fp_total = 0
            fn_total = 0
            run1_candidates: set[movielens_analysis.Pair] | None = None

            for run_index, signatures in enumerate(signatures_by_t[t]):
                candidates = movielens_analysis.lsh_candidate_pairs(
                    user_ids=user_ids,
                    signatures=signatures,
                    params=params,
                )
                if run_index == 0:
                    run1_candidates = candidates
                fp, fn = movielens_analysis.compute_fp_fn(candidates, true_pairs)
                fp_total += fp
                fn_total += fn

            avg_fp = fp_total / args.runs
            avg_fn = fn_total / args.runs
            part5_rows.append(
                [
                    str(t),
                    str(params.r),
                    str(params.b),
                    f"{avg_fp:.2f}",
                    f"{avg_fn:.2f}",
                ]
            )

            if run1_candidates is not None:
                part5_candidate_counts[tau].append((t, r, b, len(run1_candidates)))
                part5_candidate_pairs[tau][(t, r, b)] = run1_candidates
                candidate_rows = [
                    [str(u), str(v)] for (u, v) in sorted(run1_candidates)
                ]
                write_csv(
                    output_dir
                    / f"part5_lsh_candidates_tau_{tau:.1f}_t{t}_r{r}_b{b}_run1.csv",
                    headers=["user_a", "user_b"],
                    rows=candidate_rows,
                )

        write_csv(
            output_dir / f"part5_lsh_summary_tau_{tau:.1f}.csv",
            headers=["t", "r", "b", "avg_false_positives", "avg_false_negatives"],
            rows=part5_rows,
        )

    if not args.no_print:
        print("\n=== MovieLens Summary ===")
        print(f"Users: {len(user_ids)}")
        print(f"Total pairs: {len(user_ids) * (len(user_ids) - 1) // 2}")
        print(
            f"Pairs with Jaccard >= {config.MOVIELENS_SIM_THRESHOLD}: {len(exact_rows)}"
        )

        exact_display = sorted(
            ((int(u), int(v), float(sim)) for u, v, sim in exact_rows),
            key=lambda x: x[2],
            reverse=True,
        )
        exact_table_rows = [
            [str(u), str(v), f"{sim:.6f}"] for u, v, sim in exact_display
        ]
        print("\n--- Part 4: Exact pairs (top rows) ---")
        print(format_table(["user_a", "user_b", "jaccard"], exact_table_rows, print_limit))

        print("\n--- Part 4: MinHash summary ---")
        print(
            format_table(
                ["t", "avg_false_positives", "avg_false_negatives"],
                part4_summary_rows,
                None,
            )
        )

        for t, pairs in run1_pairs_by_t.items():
            pair_list = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
            rows = [[str(u), str(v), f"{sim:.6f}"] for (u, v), sim in pair_list]
            print(f"\n--- Part 4: MinHash pairs run1 (t={t}) ---")
            print(format_table(["user_a", "user_b", "estimate"], rows, print_limit))

        for tau in config.MOVIELENS_LSH_THRESHOLDS:
            summary_path = output_dir / f"part5_lsh_summary_tau_{tau:.1f}.csv"
            print(f"\n--- Part 5: LSH summary (tau={tau:.1f}) ---")
            print(summary_path.read_text(encoding="utf-8").strip())

            print(f"\n--- Part 5: LSH run1 candidate counts (tau={tau:.1f}) ---")
            candidate_rows = [
                [str(t), str(r), str(b), str(count)]
                for t, r, b, count in part5_candidate_counts[tau]
            ]
            print(
                format_table(
                    ["t", "r", "b", "run1_candidate_count"],
                    candidate_rows,
                    None,
                )
            )

            print(f"\n--- Part 5: LSH run1 candidate pairs (tau={tau:.1f}) ---")
            for (t, r, b), pairs in part5_candidate_pairs[tau].items():
                pair_rows = [[str(u), str(v)] for (u, v) in sorted(pairs)]
                print(f"\nConfig t={t}, r={r}, b={b}")
                print(format_table(["user_a", "user_b"], pair_rows, None))

    print("\nMovieLens outputs written to:", output_dir)


if __name__ == "__main__":
    run()
