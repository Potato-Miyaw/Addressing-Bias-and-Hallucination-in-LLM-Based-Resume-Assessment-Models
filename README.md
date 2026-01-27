# Addressing Bias and Hallucination in LLM-Based Resume Assessment Models

## Overview

This repository contains an end-to-end prototype for resume assessment that combines information extraction, hallucination checks, job matching, and fairness-aware ranking. It includes a FastAPI backend and a Streamlit UI to demonstrate the workflow on resume and job description data.

The goal is to study how bias and hallucination affect automated screening and to test mitigation techniques while keeping the system modular and auditable.

## System Pipeline (What this project does)

1. Job description extraction: Pulls required skills, experience, education, and certifications.
2. Resume parsing: Uses BERT NER (generic and resume-specific variants) to extract structured fields.
3. Hallucination detection: Verifies extracted claims with token overlap + BERTScore + logistic regression.
4. Job-resume matching: Computes skill/experience/education fit and identifies gaps.
5. Fairness-aware ranking: Uses XGBoost for ranking with optional Fairlearn Demographic Parity constraints.

## Features & Implementations

**Job Description Extraction**  
Implementation: `backend/services/feature1_jd_extractor.py` + `backend/routers/jd_router.py`
- Extracts required skills, experience, education, and certifications from JD text or files.

**Resume Parsing (BERT NER)**  
Implementation: `backend/services/feature2_bert_ner.py`, `feature2_resume_ner_v2.py` + `backend/routers/resume_router.py`
- Extracts structured fields (skills, education, experience, contact info).

**Hallucination Detection**  
Implementation: `backend/services/feature3_claim_verifier.py` + `backend/routers/verification_router.py`
- Uses token overlap + BERTScore + logistic regression to flag hallucinated claims.

**Job–Resume Matching**  
Implementation: `backend/services/feature4_matcher.py` + `backend/routers/matching_router.py`
- Computes match scores and identifies skill gaps.

**Fairness-Aware Screening + Ranking Engine (XGBoost)**  
Implementation: `backend/services/feature5_xgb_ranker.py` + `backend/routers/ranking_router.py`
- Engineers features from resumes + JD: `skills_count`, `education_level`, `certification_count`,
  `experience_years`, `skills_match_score`, `resume_keywords_count`, `jd_keywords_count`,
  `tfidf_similarity`, and `ai_score`.
- Produces `hire_probability`, ranked shortlist, and `decision` (HIRE/REJECT).
- Fairness Mode toggle (`use_fairness`) with Demographic Parity mitigation (Fairlearn).
- Reports fairness metrics: Impact Ratio, Demographic Parity diff, Equal Opportunity diff.

**End‑to‑End Pipeline**  
Implementation: `backend/routers/pipeline_router.py`
- Runs JD extraction → resume parsing → verification → matching → ranking in one call.

**Downloadable Audit Report (PDF/CSV)**  
Implementation: `backend/utils/reporting.py` + `backend/routers/audit_router.py`
- Exports shortlist CSV (rank, hire probability, decision, verification status).
- Generates fairness + hallucination audit PDF (impact ratio, DP/EO, flagged claims summary).
- Downloaded as a zip bundle via `POST /api/audit/export`.

**Streamlit UI (Demo)**  
Implementation: `frontend/streamlit_app.py`
- Frontend to run the full workflow, view metrics, and export reports.

## What the Code Does (Mapping to Files)

- **FastAPI backend** (`backend/app.py`):
  - Registers all routers and serves the API at `http://localhost:8000`.
- **JD extraction** (`backend/services/feature1_jd_extractor.py` + `backend/routers/jd_router.py`):
  - Regex + NER based extraction of skills, education, and experience from JD text.
- **Resume parsing** (`backend/services/feature2_bert_ner.py`, `feature2_resume_ner_v2.py` + `backend/routers/resume_router.py`):
  - Extracts name, contact info, skills, education, and experience from resumes.
- **Hallucination detection** (`backend/services/feature3_claim_verifier.py` + `backend/routers/verification_router.py`):
  - Verifies extracted claims using token overlap + BERTScore + logistic regression.
- **Matching** (`backend/services/feature4_matcher.py` + `backend/routers/matching_router.py`):
  - Computes skill/experience/education fit and highlights gaps.
- **Ranking + fairness** (`backend/services/feature5_xgb_ranker.py` + `backend/routers/ranking_router.py`):
  - Engineers features (skills match, TF‑IDF similarity, experience, etc.)
  - Produces `hire_probability`, ranked shortlist, and fairness metrics.
- **Audit exports** (`backend/utils/reporting.py` + `backend/routers/audit_router.py`):
  - Generates PDF + CSV compliance report and downloads as zip.
- **Streamlit UI** (`frontend/streamlit_app.py`):
  - Frontend demo for JD parsing, resume upload, matching, ranking, fairness metrics, and audit export.

## Project Structure

- backend/: FastAPI service with modular routers and feature services
  - services/feature1_jd_extractor.py: JD extraction (skills, experience, education, certifications)
  - services/feature2_bert_ner.py: Resume parsing with generic BERT NER
  - services/feature2_resume_ner_v2.py: Resume parsing with resume-specific BERT NER
  - services/feature3_claim_verifier.py: Hallucination detection (token overlap + BERTScore + LR)
  - services/feature4_matcher.py: Job-resume matching and gap analysis
  - services/feature5_xgb_ranker.py: XGBoost ranking with optional fairness constraints
  - routers/: API endpoints for JD parsing, resume parsing, matching, verification, ranking, and pipeline
- frontend/streamlit_app.py: Streamlit UI to demo the full workflow
- data/raw/: Sample data (e.g., AI_Resume_Screening.csv, candidate*.txt)
- requirements.txt: Python dependencies for backend, frontend, and notebooks

## Quick Start (API + UI)

1. Install dependencies
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Start the backend
   ```bash
   python backend/app.py
   ```
   This starts the FastAPI server at http://localhost:8000.

3. Start the Streamlit UI
   ```bash
   streamlit run frontend/streamlit_app.py
   ```


## Run Instructions (Step-by-step)

1) **Backend API**  
```bash
python backend/app.py
```
Open API docs at `http://localhost:8000/docs`.

2) **Frontend UI (Streamlit)**  
```bash
streamlit run frontend/streamlit_app.py
```
Use the tabs to upload JDs/resumes, run ranking, view fairness metrics, and export audit reports.

3) **Audit Export**  
From the UI, click **Generate Audit Report (PDF + CSV)** to download a zip.
Or call the endpoint directly:
```bash
POST /api/audit/export
```

## API Endpoints (Backend)

- Job description
  - POST /api/jd/extract: Extract structured requirements from text
  - POST /api/jd/upload: Upload and parse JD files (PDF, DOCX, DOC, TXT)
- Resume parsing
  - POST /api/resume/parse: Parse resume text with generic BERT NER
  - POST /api/resume/parse-v2: Parse resume text with resume-specific BERT NER
  - POST /api/resume/upload: Upload and parse resume files
  - POST /api/resume/batch: Batch upload resumes
- Matching and ranking
  - POST /api/match/: Match a resume to a JD
  - POST /api/rank/: Rank candidates using XGBoost (fairness optional)
- Hallucination detection
  - POST /api/verify/claim: Verify a single claim
  - POST /api/verify/resume: Verify claims in a resume extraction
- End-to-end pipeline
  - POST /api/pipeline/complete: Run the full pipeline on a JD + multiple resumes
- Audit exports
  - POST /api/audit/export: Downloadable zip with shortlist CSV and audit PDF

Note: The backend currently runs in-memory only. Database persistence has been removed.

## Example Workflow (Quick Demo)

1. **Job Description**: paste JD text in the Streamlit UI (Tab 1).
2. **Resumes**: upload resumes or paste multiple (Tab 2 or Tab 5).
3. **Match & Rank**: run ranking and view shortlist (Tab 3).
4. **Fairness Metrics**: view Impact Ratio / DP / EO (Tab 3 or Tab 5).
5. **Audit Export**: generate and download PDF+CSV report.

## Data Notes

- Sample data lives in data/raw/.
- Some notebooks also download datasets with KaggleHub. See notebook cells for dataset IDs and setup.

## Known Limitations

- Hallucination verification is strongest when ground-truth data is provided; otherwise it reports NO_CLAIMS.
- DOC file parsing on Windows requires Microsoft Word installed (via pywin32).
- Model checkpoints are pulled from Hugging Face at runtime (requires internet access).

---

For detailed results, use the API and UI to generate reports.
