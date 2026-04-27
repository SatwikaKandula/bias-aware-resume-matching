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

- `ML_FINAL/scripts/run_sbert.py`  
  Regenerates Sentence-BERT resume-job similarity scores.

- `ML_FINAL/scripts/analyze_results.py`  
  Computes counterfactual score-shift summaries.

- `ML_FINAL/scripts/run_llm_openai.py`  
  Adds a current LLM scoring path using OpenAI's Responses API. It defaults to
  `gpt-5-mini` and requires `OPENAI_API_KEY`.

- `ML_FINAL/report/final_report.pdf`  
  Final ICML-style project report.

- `ML_FINAL/demo/index.html`  
  Static browser demo for trying a local resume/job match and presenting the project results.

## Goal

The goal is to measure whether SBERT or an LLM assigns different matching scores to resumes that are identical except for one demographic signal.

## Reproducing the Results

Install dependencies:

```bash
pip install -r requirements.txt
```

Run SBERT scoring:

```bash
python ML_FINAL/scripts/run_sbert.py
```

Create bias summaries:

```bash
python ML_FINAL/scripts/analyze_results.py
```

Run a small LLM smoke test:

```bash
export OPENAI_API_KEY=your_key_here
python ML_FINAL/scripts/run_llm_openai.py --limit 5
```

Run the full LLM audit:

```bash
python ML_FINAL/scripts/run_llm_openai.py
python ML_FINAL/scripts/analyze_results.py --scores ML_FINAL/results/llm_scores.csv --score-col llm_score
```

## Demo

Open `ML_FINAL/demo/index.html` in a browser to paste or upload a resume PDF or
plain-text resume, paste a job description, get a local demo match score, and
show the dataset design, SBERT bias summary, and optional LLM scoring workflow.
PDF parsing is handled locally through the vendored PDF.js files in
`ML_FINAL/demo/vendor/pdfjs/`.

For live LLM scoring, run the optional local server instead of opening the HTML
file directly:

```bash
export OPENAI_API_KEY=your_key_here
python ML_FINAL/demo/server.py
```

Then open `http://127.0.0.1:8000/` and use **Analyze with LLM**.
Using the bundled Python also enables the server-side PDF extraction fallback.
