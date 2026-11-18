"""Evaluate transparent expected-term coverage on pre-recorded RAG answers.

This is a lightweight smoke/evidence check, not a substitute for retrieval recall,
groundedness, citation verification, answer relevance or human review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def score(answer: str, expected_terms: list[str]) -> float:
    if not isinstance(answer, str):
        raise ValueError("answer must be a string")
    if not isinstance(expected_terms, list) or not expected_terms:
        raise ValueError("expected_terms must be a non-empty list")
    if any(not isinstance(term, str) or not term.strip() for term in expected_terms):
        raise ValueError("expected_terms must contain non-empty strings")

    normalized_answer = answer.casefold()
    return sum(term.casefold() in normalized_answer for term in expected_terms) / len(expected_terms)


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Prediction file not found: {source}")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}") from exc
        if not isinstance(row, dict) or "answer" not in row or "expected_terms" not in row:
            raise ValueError(
                f"Line {line_number} must contain answer and expected_terms fields"
            )
        rows.append(row)

    if not rows:
        raise ValueError("Prediction file contains no evaluation records")
    return rows


def evaluate(path: str | Path) -> dict[str, float | int]:
    rows = load_predictions(path)
    scores = [score(row["answer"], row["expected_terms"]) for row in rows]
    return {
        "records": len(scores),
        "mean_term_coverage": round(mean(scores), 4),
        "min_term_coverage": round(min(scores), 4),
        "max_term_coverage": round(max(scores), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="?", default="evaluation/predictions.example.jsonl")
    parser.add_argument(
        "--min-term-coverage",
        type=float,
        default=None,
        help="Optional smoke-test threshold; not a production quality gate.",
    )
    args = parser.parse_args()

    if args.min_term_coverage is not None and not 0.0 <= args.min_term_coverage <= 1.0:
        parser.error("--min-term-coverage must be between 0 and 1")

    report = evaluate(args.predictions)
    print(json.dumps(report, indent=2))
    if args.min_term_coverage is not None:
        return 0 if report["mean_term_coverage"] >= args.min_term_coverage else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
