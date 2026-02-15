"""Format CSV outputs into Markdown tables for PDF-ready reporting."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Sequence

from . import config

_TITLE_MAP: dict[str, str] = {
    "part1_kgrams_jaccard.md": "Part 1: K-grams Jaccard",
    "part2_minhash_d1_d2.md": "Part 2: MinHash Estimates (D1 vs D2)",
    "part3_lsh_probabilities.md": "Part 3: LSH Probabilities",
    "part4_exact_pairs_ge_0_5.md": "Part 4: Exact Pairs (Jaccard ≥ 0.5)",
    "part4_minhash_summary.md": "Part 4: MinHash Summary (Avg FP/FN)",
    "part5_lsh_summary_tau_0.6.md": "Part 5: LSH Summary (τ = 0.6)",
    "part5_lsh_summary_tau_0.8.md": "Part 5: LSH Summary (τ = 0.8)",
}


def csv_to_markdown(
    input_path: Path,
    output_path: Path,
    max_rows: int | None = None,
) -> None:
    """Convert a CSV file into a Markdown table."""
    with input_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        headers = next(reader, None)
        if headers is None:
            raise ValueError(f"Empty CSV file: {input_path}")
        rows = list(reader)

    if max_rows is not None:
        rows = rows[:max_rows]

    lines: list[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_csv_files(output_dir: Path, include_candidates: bool) -> Iterable[Path]:
    """Enumerate relevant Part 1–5 CSV outputs if they exist."""
    files: list[Path] = []
    # Parts 1–3
    files.append(output_dir / "part1_kgrams_jaccard.csv")
    files.append(output_dir / "part2_minhash_d1_d2.csv")
    files.append(output_dir / "part3_lsh_probabilities.csv")

    # Parts 4–5
    files.append(output_dir / "part4_exact_pairs_ge_0_5.csv")
    files.append(output_dir / "part4_minhash_summary.csv")

    for t in config.MOVIELENS_T_VALUES:
        files.append(output_dir / f"part4_minhash_pairs_t{t}_run1.csv")

    for tau in config.MOVIELENS_LSH_THRESHOLDS:
        files.append(output_dir / f"part5_lsh_summary_tau_{tau:.1f}.csv")
        if include_candidates:
            files.extend(
                sorted(
                    output_dir.glob(
                        f"part5_lsh_candidates_tau_{tau:.1f}_t*_r*_b*_run1.csv"
                    )
                )
            )

    return [path for path in files if path.exists()]


def find_text_files(output_dir: Path) -> list[Path]:
    """Locate supplemental text outputs to include in the bundle."""
    candidates = [
        output_dir / "part2_t_recommendation.txt",
        output_dir / "part3_lsh_params.txt",
    ]
    return [path for path in candidates if path.exists()]


def bundle_markdown(
    output_dir: Path,
    include_candidates: bool,
    max_rows: int | None,
    bundle_name: str,
) -> None:
    """Bundle Markdown tables and text blocks into a single report."""
    csv_files = find_csv_files(output_dir, include_candidates=include_candidates)
    for csv_path in csv_files:
        md_path = csv_path.with_suffix(".md")
        if not md_path.exists():
            csv_to_markdown(csv_path, md_path, max_rows=max_rows)

    md_files = [path.with_suffix(".md") for path in csv_files]
    text_files = find_text_files(output_dir)

    sections: list[str] = []
    sections.append("# Assignment Results")

    for md_path in md_files:
        title = _TITLE_MAP.get(md_path.name, md_path.stem.replace("_", " ").title())
        sections.append(f"\n## {title}\n")
        sections.append(md_path.read_text(encoding="utf-8").strip())

    if text_files:
        sections.append("\n## Notes\n")
        for text_path in text_files:
            title = text_path.stem.replace("_", " ").title()
            sections.append(f"### {title}\n")
            sections.append(text_path.read_text(encoding="utf-8").strip())

    bundle_path = output_dir / bundle_name
    bundle_path.write_text("\n".join(sections).strip() + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Format MovieLens outputs to Markdown")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing output CSVs",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for very large tables",
    )
    parser.add_argument(
        "--include-candidates",
        action="store_true",
        help="Include LSH candidate pair tables (can be large)",
    )
    parser.add_argument(
        "--bundle",
        action="store_true",
        help="Bundle all Markdown tables into a single report.md",
    )
    parser.add_argument(
        "--bundle-name",
        type=str,
        default="report.md",
        help="Name of the bundled Markdown report",
    )
    return parser.parse_args(argv)


def run() -> None:
    """Format Part 4–5 outputs into Markdown tables."""
    args = parse_args()
    output_dir: Path = args.output_dir

    csv_files = find_csv_files(output_dir, include_candidates=args.include_candidates)
    if not csv_files:
        print("No Part 4–5 CSV outputs found in:", output_dir)
        return

    for csv_path in csv_files:
        md_path = csv_path.with_suffix(".md")
        csv_to_markdown(csv_path, md_path, max_rows=args.max_rows)

    if args.bundle:
        bundle_markdown(
            output_dir,
            include_candidates=args.include_candidates,
            max_rows=args.max_rows,
            bundle_name=args.bundle_name,
        )

    print("Markdown tables written to:", output_dir)


if __name__ == "__main__":
    run()
