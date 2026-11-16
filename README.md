
# PredictMed: Agentic AI for early CKD deterioration detection

Team AI Innovix. This folder is a runnable prototype of the pipeline in your slides.
It starts with **no keys and no internet** (offline mode, synthetic data) and gets more real as you connect things.

## Run it

```bash
cd predictmed
./run.sh                       # Mac/Linux. Opens on http://localhost:8000
```
Windows or manual:
```bash
cd backend
python -m venv .venv && .venv\Scripts\activate      # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy ..\.env.example .env                           # Mac/Linux: cp ../.env.example .env
uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000 (dashboard) or http://localhost:8000/docs (API explorer).
Tests: `cd backend && python tests/test_core.py`. Docker: `docker compose up --build`.

The top bar of the dashboard shows what is connected (model, RAG mode, Groq, local LLM).

## What you need to connect

| # | Connection | Required? | Without it | How to connect |
|---|------------|-----------|------------|----------------|
| 1 | **Groq API key** (Llama 3.3 70B) | Optional | Explanations use a built-in template | Create a key at console.groq.com, put `GROQ_API_KEY=` in `backend/.env`. Confirm the model ID in `GROQ_MODEL`. Only de-identified text is sent. |
| 2 | **Local Hugging Face LLM** (PHI route) | Optional | Text containing identifiers is never sent anywhere, and templates are used | `pip install -r requirements-ai.txt`, set `HF_MODEL=` to a chat model your machine can run. `HF_TOKEN` only for gated models. Needs enough RAM or GPU. |
| 3 | **KDIGO knowledge base** (RAG) | Strongly recommended | Only the short starter notes are retrieved | Replace `backend/kb/kdigo_starter.md` with your own summaries of the current guideline (check its licence), one idea per paragraph. Install `faiss-cpu` and `sentence-transformers` to switch from keyword search to FAISS + MiniLM. |
| 4 | **Real patient data** | Required for real use | Synthetic members | Choose one: (a) CSV import in the API (`POST /api/members/import`), format below. (b) FHIR server: fill `FHIR_BASE_URL`, adapt `backend/connectors/fhir.py`. (c) Database or warehouse: write a loader that calls `store.add_member`. |
| 5 | **Real training labels** | Required for real use | Model trains on synthetic labels | Edit `training_frame()` in `backend/app/pipeline.py` to return your features plus a `label` column that your clinicians define (for example rapid progression within 12 months). Keep rows in time order. |
| 6 | **MLflow** | Optional | Retrain still works, nothing is logged | `pip install mlflow`, run `docker compose --profile mlops up`, set `MLFLOW_TRACKING_URI=http://localhost:5000`. |
| 7 | **Database** | Optional for demo | SQLite file in `backend/data` | Swap `store.py` to Postgres for multi-user use. |

CSV format (one row per member, 8 eGFR readings 3 months apart):
`name,age,uacr,nsaid,days_since_lab,e1,e2,e3,e4,e5,e6,e7,e8`

## Later-phase pieces from your roadmap (not built here)

| Slide item | Where it would plug in |
|------------|------------------------|
| Kafka (streaming ingestion) | A consumer that calls `store.add_member` or updates a member's `egfr` list |
| Airflow (scheduling) | A DAG that calls `POST /api/retrain` and `GET /api/drift` on a schedule |
| Iceberg / feature store | Replace `store.members()` and `training_frame()` with reads from your tables |
| FHIR/HL7 integration | `backend/connectors/fhir.py` is an untested starter, not a finished integration |
| Kubernetes | The `Dockerfile` is the unit to deploy. Add auth, TLS and secrets management first |
| Graph layer (NetworkX, node2vec) | The demo UI uses a simple nearest-neighbour idea. Build a member-similarity graph from your features |
| Federated learning | Not implemented |

## How each slide idea maps to code

| Slide idea | File |
|------------|------|
| CKD-EPI 2021, OLS slope, CUSUM, DVI | `app/trajectory.py` |
| LightGBM + isotonic calibration + SHAP-style contributions | `app/model.py` (LightGBM `pred_contrib` is TreeSHAP) |
| RAG over KDIGO (FAISS + MiniLM) | `app/rag.py` |
| Dual-LLM router, PHI scrubbing | `app/llm_router.py` |
| PSI drift watch, retrain | `app/drift.py`, `/api/drift`, `/api/retrain` |
| Human approval and audit trail | `/api/members/{id}/approve`, `/api/audit` |
| English, Telugu, Hindi drafts | `app/templates.py` |

## Be honest about what this is

- **Not a medical device and not clinically validated.** The default model trains on *synthetic* data with a made-up outcome rule, so its metrics only prove the plumbing works.
- The DVI formula, tier thresholds (`pipeline.TIERS`) and CUSUM settings are starting values. Agree them with clinicians.
- **Privacy:** the router keeps text with names, phones, dates, emails or record numbers on the local model and refuses to send it to Groq. Pattern matching can miss identifiers. Before using real data, get legal and security review for HIPAA and India's DPDP Act, add authentication, encryption and access logs.
- Telugu and Hindi drafts need review by native speakers and clinicians.
- Market figures in the slides are third-party estimates; values after each firm's base year were derived from its stated growth rate.
- The `fhir.py` starter has not been run against a real server.
- This project was tested with the core Python libraries only. The FastAPI layer, LightGBM, FAISS, MiniLM, Groq and Hugging Face paths were written to their documented APIs but have **not** been run end to end, so expect to fix small things on first launch.

## Folder map

```
backend/app/        trajectory, model, rag, llm_router, drift, store, pipeline, templates, main (API)
backend/kb/         knowledge-base text for RAG (replace the starter)
backend/connectors/ fhir.py starter
backend/tests/      core tests
frontend/index.html dashboard served by the API
docs/               your enhanced slides, study notes, standalone demo (no backend needed)
```
=======
# PredictMed
It is a dual LLM project for Virtusa Jatayu

