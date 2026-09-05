"""Application pipeline wiring Phase 1-2 components."""

from pathlib import Path

from app.data.loader import load_jobs
from app.data.preprocessing import prepare_jobs
from app.embeddings.embedding_service import EmbeddingService
from app.core.config import get_settings
from app.nlp.profile_builder import build_candidate_profile
from app.recommendation.recommender import recommend
from app.retrieval.retriever import JobRetriever
from app.retrieval.vector_store import JobVectorStore


def run_demo_recommendation(request: dict[str, object], data_path: str | None = None) -> list[dict[str, object]]:
    settings = get_settings()
    data_path = data_path or settings.data_path
    jobs = prepare_jobs(load_jobs(data_path))
    profile = build_candidate_profile(
        str(request["resume_text"]),
        target_role=request.get("target_role", ""),
        location=request.get("location", ""),
        experience_level=request.get("experience_level", ""),
        employment_type=request.get("employment_type", ""),
    )
    profile.skills = sorted(set(profile.skills + [str(skill) for skill in request.get("additional_skills", [])]))
    embedding_service = EmbeddingService(settings.embedding_model)
    index_path = Path(settings.vector_store_path)
    ids_path = index_path.with_suffix(".ids.npy")
    expected_ids = jobs["job_id"].tolist()
    if index_path.exists() and ids_path.exists():
        store = JobVectorStore.load(index_path)
        if store.job_ids != expected_ids:
            store = _build_store(jobs, embedding_service, index_path)
    else:
        store = _build_store(jobs, embedding_service, index_path)
    retriever = JobRetriever(jobs, embedding_service, store)
    retrieved = retriever.retrieve(profile.to_embedding_text(), top_k=settings.top_k_retrieval)
    return recommend(profile, retrieved)


def _build_store(jobs, embedding_service: EmbeddingService, index_path: Path) -> JobVectorStore:
    job_embeddings = embedding_service.embed_documents(jobs["search_text"].tolist())
    store = JobVectorStore.build(job_embeddings, jobs["job_id"].tolist())
    store.save(index_path)
    return store
