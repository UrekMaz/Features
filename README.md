
# ReadabilityLens (Coh-Metrix + FS5)

ReadabilityLens is a readability classification app with:
- A FastAPI backend (`api.py`) for feature extraction + grade prediction
- A Streamlit frontend (`app.py`) for analysis, feature interpretation, nudges, and rewrite workflow

The model predicts grades 2–8 and returns confidence, grade probabilities, top feature contributions, and full extracted features.

## Current Project Layout

- `api.py` — FastAPI backend (`/health`, `/classify`, `/grade-means`, `/nudge`)
- `app.py` — Streamlit frontend
- `fs5_extractor.py` — FS5 feature extraction pipeline
- `artifacts/` — model + scaler + feature list + label encoder artifacts
- `CohMetrixCore/` — bundled Coh-Metrix runtime files used by extraction components
- `requirements.txt` — Python dependencies

## Prerequisites

- Windows (project currently developed/tested on Windows)
- Python 3.10 or 3.11
- `pip`
- Internet on first run (for model/resource downloads if missing)

## 1) Setup Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install streamlit plotly requests uvicorn
```

## 2) Run Backend (FastAPI)

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Backend URL: `http://127.0.0.1:8000`

### API Endpoints

- `GET /health` — service/model status
- `POST /classify` — classify one text
- `GET /grade-means` — per-grade mean values for model features
- `POST /nudge` — feature-level suggestions toward a target grade

### Quick classify example

```powershell
curl -X POST http://127.0.0.1:8000/classify -H "Content-Type: application/json" -d "{\"text\": \"This is a sufficiently long sample text that should pass minimum word requirements for testing purposes.\"}"
```

## 3) Run Frontend (Streamlit)

Open a second terminal:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Frontend URL: usually `http://localhost:8501`

## Rewrite + Nudge Behavior (Important)

- Rewrite uses Sarvam (`SARVAM_API_KEY`) with prompt instructions based on:
  - original text
  - current predicted grade
  - selected target grade
- After rewrite, the app re-classifies rewritten text and shows the actual predicted grade/confidence.
- Nudges are computed from current feature values vs target-grade mean values.
- Nudges are shown to the user, but are **not currently injected into the rewrite prompt**.

## Environment Variables

- `API_URL` (default: `http://localhost:8000`)
- `API_CONNECT_TIMEOUT_SEC` (default: `10`)
- `API_CLASSIFY_TIMEOUT_SEC` (default: `120`)
- `API_NUDGE_TIMEOUT_SEC` (default: `120`)
- `API_GRADE_MEANS_TIMEOUT_SEC` (default: `60`)
- `SARVAM_API_URL` (default: `https://api.sarvam.ai/v1/chat/completions`)
- `SARVAM_MODEL` (default: `sarvam-m`)
- `SARVAM_API_KEY` (required for rewrite)

You can set `SARVAM_API_KEY` either in environment or `.streamlit/secrets.toml`.

## Notes on Performance

- `POST /classify` is CPU-heavy (feature extraction + model inference) and can take tens of seconds on CPU.
- Backend classifies in a threadpool to avoid blocking other API endpoints while one request is running.
- If Streamlit appears to wait too long, reduce `API_CLASSIFY_TIMEOUT_SEC`.

## Troubleshooting

- If Streamlit cannot connect, confirm backend is running at `127.0.0.1:8000` and check `GET /health`.
- If rewrite fails, check `SARVAM_API_KEY` setup.
- If dependencies mismatch after environment changes, reinstall pinned packages from `requirements.txt` and restart both backend + frontend.











