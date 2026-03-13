#!/usr/bin/env python3
"""Compare Python and Rust benchmark results.

Reads ``bench/results_python.json`` and ``bench/results_rust.json`` produced
by pytest-benchmark, then prints a comparison table and optionally writes
``bench/report.md``.

Usage::

    # First, generate result files:
    pytest bench/bench_python.py --benchmark-json=bench/results_python.json
    pytest bench/bench_rust.py   --benchmark-json=bench/results_rust.json

    # Then run the reporter:
    python bench/report.py
    python bench/report.py --output bench/report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _parse_name(name: str) -> tuple[str, str]:
    """Extract (operation, batch_size) from a benchmark name like
    ``test_batch_compose[n=1000]``.
    """
    # Strip 'test_' prefix.
    if name.startswith("test_"):
        name = name[5:]
    # Split off parametrize suffix if present.
    if "[" in name:
        op, rest = name.split("[", 1)
        size = rest.rstrip("]")
        # Handle 'n=1000' or bare '1000' formats.
        if "=" in size:
            size = size.split("=", 1)[1]
    else:
        op = name
        size = "1"
    return op, size


def build_table(
    py_data: dict, rs_data: dict
) -> list[tuple[str, str, float, float, float]]:
    """Return rows of (operation, batch_size, python_mean_ms, rust_mean_ms, speedup)."""
    # Index Rust benchmarks by (op, size).
    rs_index: dict[tuple[str, str], float] = {}
    for bench in rs_data.get("benchmarks", []):
        op, size = _parse_name(bench["name"])
        mean_ms = bench["stats"]["mean"] * 1000.0
        rs_index[(op, size)] = mean_ms

    rows = []
    for bench in py_data.get("benchmarks", []):
        op, size = _parse_name(bench["name"])
        py_mean = bench["stats"]["mean"] * 1000.0
        rs_mean = rs_index.get((op, size))
        if rs_mean is None:
            continue
        speedup = py_mean / rs_mean if rs_mean > 0 else float("inf")
        rows.append((op, size, py_mean, rs_mean, speedup))

    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def format_table(rows: list[tuple]) -> str:
    header = (
        f"{'operation':<25} {'batch_size':>12} {'python_mean_ms':>16} "
        f"{'rust_mean_ms':>14} {'speedup':>10}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for op, size, py_ms, rs_ms, speedup in rows:
        lines.append(
            f"{op:<25} {size:>12} {py_ms:>16.4f} {rs_ms:>14.4f} {speedup:>9.2f}x"
        )
    return "\n".join(lines)


def format_markdown(rows: list[tuple]) -> str:
    lines = [
        "| operation | batch_size | python_mean_ms | rust_mean_ms | speedup |",
        "|-----------|------------|---------------|-------------|---------|",
    ]
    for op, size, py_ms, rs_ms, speedup in rows:
        lines.append(
            f"| {op} | {size} | {py_ms:.4f} | {rs_ms:.4f} | {speedup:.2f}x |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Python and Rust benchmarks.")
    parser.add_argument(
        "--python-json",
        default="bench/results_python.json",
        help="Path to Python benchmark JSON (default: bench/results_python.json)",
    )
    parser.add_argument(
        "--rust-json",
        default="bench/results_rust.json",
        help="Path to Rust benchmark JSON (default: bench/results_rust.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="If given, write a Markdown report to this path.",
    )
    args = parser.parse_args()

    for path in (args.python_json, args.rust_json):
        if not os.path.exists(path):
            print(f"ERROR: result file not found: {path}", file=sys.stderr)
            print(
                "Run benchmarks first:\n"
                "  pytest bench/bench_python.py --benchmark-json=bench/results_python.json\n"
                "  pytest bench/bench_rust.py   --benchmark-json=bench/results_rust.json",
                file=sys.stderr,
            )
            sys.exit(1)

    py_data = _load(args.python_json)
    rs_data = _load(args.rust_json)
    rows = build_table(py_data, rs_data)

    print(format_table(rows))

    if args.output:
        md = format_markdown(rows)
        Path(args.output).write_text(md + "\n")
        print(f"\nMarkdown report written to {args.output}")


if __name__ == "__main__":
    main()
