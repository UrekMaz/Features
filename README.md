
Extract 80+ core discourse features (and 200+ with optional modules) from text using a FastAPI backend and a Streamlit frontend.

The pipeline combines:
- Coreference + entity grid features
- Lexical chain and constituency features
- POS and readability features
- Optional LFTK feature set
- Optional Coh-Metrix CLI indices

## Project Layout

- `app.py` — FastAPI API server
- `discourse_pipeline.py` — full feature extraction pipeline
- `frontend.py` — Streamlit UI
- `CohMetrixCore/` — Coh-Metrix CLI binaries and runtime files
- `cohmetrix_output/` — temporary/output files for Coh-Metrix runs
- `requirements.txt` — backend and NLP dependencies

## Prerequisites

- Windows (recommended for bundled `CohMetrixCoreCLI.exe`)
- Python 3.10 or 3.11
- `pip`
- Internet on first run (downloads NLTK, Stanza, spaCy and sentence-transformer resources)

Optional (for Coh-Metrix module):
- Ensure `CohMetrixCore/CohMetrixCoreCLI.exe` is present and runnable

## 1) Initialize Environment

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install UI-only packages (not pinned in `requirements.txt`):

```powershell
pip install streamlit plotly requests
```

## 2) Start the Backend (FastAPI)

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Backend URL: `http://127.0.0.1:8000`

Useful endpoints:
- `GET /` — API info
- `GET /health` — model + Coh-Metrix availability status
- `GET /feature-summary` — feature catalog
- `POST /extract-features` — analyze one text
- `POST /extract-features-batch` — analyze multiple texts

## 3) Start the Frontend (Streamlit)

Open a second terminal:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run frontend.py
```

Frontend URL: usually `http://localhost:8501`

The UI expects backend at:
- `http://127.0.0.1:8000/extract-features`
- `http://127.0.0.1:8000/health`

## Quick API Example

```bash
curl -X POST "http://127.0.0.1:8000/extract-features" \
	-H "Content-Type: application/json" \
	-d '{
		"text": "This is a short sample paragraph for feature extraction.",
		"include_lftk": true,
		"include_cohmetrix": false
	}'
```

Response shape:
- `feature_count`: total number of extracted features
- `features`: feature dictionary (`name -> value`)
- `categories`: count of features by top-level prefix

## Coh-Metrix Module (Optional)

To include Coh-Metrix indices:
1. Keep `CohMetrixCore/` in project root.
2. Ensure `CohMetrixCore/CohMetrixCoreCLI.exe` exists.
3. Set `include_cohmetrix=true` in API request (or enable in Streamlit UI).

Notes:
- If CLI is missing/unavailable, extraction still works; Coh-Metrix features are skipped.
- Temporary files are written to `cohmetrix_output/` and cleaned up after processing.

## First-Run Behavior

On first backend startup, the app loads/downloads:
- NLTK tokenizers/taggers
- spaCy model: `en_core_web_sm`
- Stanza English pipeline
- Sentence-BERT model: `all-mpnet-base-v2`
- fastcoref model: `biu-nlp/f-coref`

This may take a few minutes depending on network and hardware.



- Backend defaults:
	- `include_lftk=True`
	- `include_cohmetrix=False`
- Frontend and backend should run in separate terminals.
- Recommended workflow: start backend first, then launch Streamlit.











