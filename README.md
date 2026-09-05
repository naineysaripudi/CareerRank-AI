# CareerRank AI

CareerRank AI is an explainable job recommendation system built around resume NLP, sentence embeddings, FAISS retrieval, transparent personalized ranking, skill-gap analysis, and optional LLM explanations.

## Live Demo

[Open CareerRank AI](https://careerrank-ai-qf76rbgebrwajhvxqqbyli.streamlit.app)

## Current status

The working prototype includes resume PDF extraction, controlled skill normalization, sentence-transformer embeddings, persistent FAISS retrieval, transparent personalized ranking, skill-gap analysis, deterministic explanations, FastAPI endpoints, Streamlit visualization, and a reproducible synthetic evaluation.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
Copy-Item .env.example .env
python scripts\prepare_data.py
python scripts\build_index.py
```

The generated development dataset contains 100 synthetic records across AI, data, software, backend, MLOps, and data engineering roles. It contains no copied job descriptions.

## Run commands

```powershell
uvicorn app.api.routes:app --reload
streamlit run frontend\streamlit_app.py
pytest
python scripts\evaluate.py
```

The first index build downloads the configured Sentence Transformer model. Subsequent recommendations reuse `vector_store/jobs.faiss` unless the job IDs change.

## Evaluation

```powershell
python scripts\evaluate.py
```

The evaluation compares semantic-only retrieval with personalized ranking using explicit synthetic relevance labels. See [EVALUATION.md](EVALUATION.md) for the measured results and limitations.
