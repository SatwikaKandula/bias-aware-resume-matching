# Bias-Aware Job-Resume Matching

This project audits demographic bias in automated resume-job matching systems.

We compare Sentence-BERT and an LLM-based scoring method to examine whether resume matching scores change when only one demographic signal is modified.

## Dataset

The dataset contains:

- 5 job descriptions across five domains:
  - Software Engineering
  - Finance
  - Marketing
  - Healthcare
  - Education

- 10 synthetic base resumes
- 40 controlled resume variants

Each resume has four versions:

1. Original
2. Name changed
3. Pronoun changed
4. University changed

This gives:

40 resume variants × 5 job descriptions = 200 resume-job comparisons

## Files

- `data/jobs.csv`  
  Contains job descriptions.

- `data/base_resumes.csv`  
  Contains the original synthetic resumes.

- `data/resume_variants.csv`  
  Contains original and counterfactual resume versions.

## Goal

The goal is to measure whether SBERT or an LLM assigns different matching scores to resumes that are identical except for one demographic signal.
