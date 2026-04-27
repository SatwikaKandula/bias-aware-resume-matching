#!/usr/bin/env python3
"""Score every resume/job pair with Sentence-BERT cosine similarity."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", default="ML_FINAL/data/jobs.csv")
    parser.add_argument("--resumes", default="ML_FINAL/data/resume_variants.csv")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--output", default="ML_FINAL/results/sbert_scores.csv")
    args = parser.parse_args()

    jobs = pd.read_csv(args.jobs)
    resumes = pd.read_csv(args.resumes)

    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer(args.model)

    job_embeddings = model.encode(jobs["job_description"].astype(str).tolist(), show_progress_bar=True)
    resume_embeddings = model.encode(resumes["resume_text"].astype(str).tolist(), show_progress_bar=True)

    rows = []
    for resume_idx, resume_row in resumes.iterrows():
        for job_idx, job_row in jobs.iterrows():
            score = cosine_similarity([resume_embeddings[resume_idx]], [job_embeddings[job_idx]])[0][0]
            rows.append(
                {
                    "resume_id": resume_row["resume_id"],
                    "resume_domain": resume_row["domain"],
                    "version": resume_row["version"],
                    "changed_signal": resume_row["changed_signal"],
                    "name": resume_row["name"],
                    "pronouns": resume_row["pronouns"],
                    "university": resume_row["university"],
                    "job_id": job_row["job_id"],
                    "job_domain": job_row["domain"],
                    "job_title": job_row["title"],
                    "company_name": job_row["company_name"],
                    "sbert_score": float(score),
                }
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(rows)} rows")


if __name__ == "__main__":
    main()
