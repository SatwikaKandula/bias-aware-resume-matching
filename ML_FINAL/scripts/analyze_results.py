#!/usr/bin/env python3
"""Create counterfactual fairness summaries for resume-matching scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize(scores: pd.DataFrame, score_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    originals = (
        scores[scores["version"] == "original"][["resume_id", "job_id", score_col]]
        .rename(columns={score_col: "original_score"})
    )
    deltas = scores.merge(originals, on=["resume_id", "job_id"], how="left")
    deltas["delta"] = deltas[score_col] - deltas["original_score"]
    changed = deltas[deltas["changed_signal"] != "none"].copy()

    signal_summary = (
        changed.groupby("changed_signal")["delta"]
        .agg(mean_delta="mean", median_delta="median", std_delta="std", min_delta="min", max_delta="max")
        .reset_index()
    )
    signal_summary["mean_abs_delta"] = (
        changed.groupby("changed_signal")["delta"].apply(lambda s: s.abs().mean()).values
    )

    domain_summary = (
        changed[changed["resume_domain"] == changed["job_domain"]]
        .groupby("changed_signal")["delta"]
        .agg(correct_domain_mean_delta="mean", correct_domain_median_delta="median")
        .reset_index()
    )
    domain_summary["correct_domain_mean_abs_delta"] = (
        changed[changed["resume_domain"] == changed["job_domain"]]
        .groupby("changed_signal")["delta"]
        .apply(lambda s: s.abs().mean())
        .values
    )

    return signal_summary, domain_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="ML_FINAL/results/sbert_scores.csv")
    parser.add_argument("--score-col", default="sbert_score")
    parser.add_argument("--out-dir", default="ML_FINAL/results")
    args = parser.parse_args()

    scores = pd.read_csv(args.scores)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    signal_summary, domain_summary = summarize(scores, args.score_col)
    signal_path = out_dir / f"{args.score_col}_bias_summary.csv"
    domain_path = out_dir / f"{args.score_col}_correct_domain_summary.csv"
    signal_summary.to_csv(signal_path, index=False)
    domain_summary.to_csv(domain_path, index=False)

    print(f"Wrote {signal_path}")
    print(signal_summary.to_string(index=False))
    print(f"\nWrote {domain_path}")
    print(domain_summary.to_string(index=False))


if __name__ == "__main__":
    main()
