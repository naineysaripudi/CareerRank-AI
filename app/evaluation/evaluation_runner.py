"""Reproducible evaluation of semantic and personalized ranking."""

from app.embeddings.embedding_service import EmbeddingService
from app.ranking.features import calculate_features
from app.ranking.ranker import rank_jobs
from app.evaluation.ranking_metrics import ndcg_at_k
from app.evaluation.retrieval_metrics import mean_reciprocal_rank, precision_at_k, recall_at_k


EVALUATION_PROFILES = [
    {"resume_text": "Python machine learning NLP SQL", "target_role": "AI Engineer", "location": "Remote"},
    {"resume_text": "Python SQL pandas statistics Power BI", "target_role": "Data Analyst", "location": "Bengaluru"},
    {"resume_text": "Python Docker Kubernetes MLflow", "target_role": "MLOps Engineer", "location": "Pune"},
]


def run_evaluation(jobs, top_k: int = 5) -> dict[str, dict[str, float]]:
    """Evaluate explicit role-based relevance labels on synthetic development data."""

    embedding_service = EmbeddingService()
    semantic_scores: list[list[int]] = []
    personalized_scores: list[list[int]] = []
    for profile_data in EVALUATION_PROFILES:
        from app.nlp.profile_builder import build_candidate_profile
        profile = build_candidate_profile(**profile_data)
        job_records = jobs.to_dict(orient="records")
        embeddings = embedding_service.embed_documents(jobs["search_text"].tolist())
        query_embedding = embedding_service.embed_text(profile.to_embedding_text())
        similarities = embeddings @ query_embedding
        for job, similarity in zip(job_records, similarities):
            job["semantic_similarity"] = float(similarity)
        labels = [int(job["job_title"].casefold() == profile_data["target_role"].casefold()) for job in job_records]
        semantic_order = sorted(range(len(job_records)), key=lambda index: job_records[index]["semantic_similarity"], reverse=True)
        semantic_relevance = [labels[index] for index in semantic_order]
        ranked = rank_jobs(profile, job_records)
        personalized_relevance = [int(job["job_title"].casefold() == profile_data["target_role"].casefold()) for job in ranked]
        semantic_scores.append(semantic_relevance)
        personalized_scores.append(personalized_relevance)

    return {"semantic_only": _summarize(semantic_scores, top_k), "personalized": _summarize(personalized_scores, top_k)}


def _summarize(relevance_rows: list[list[int]], k: int) -> dict[str, float]:
    total_relevant = sum(sum(row) for row in relevance_rows) / len(relevance_rows)
    return {
        f"precision_at_{k}": sum(precision_at_k(row, k) for row in relevance_rows) / len(relevance_rows),
        f"recall_at_{k}": sum(recall_at_k(row, total_relevant, k) for row in relevance_rows) / len(relevance_rows),
        "mrr": sum(mean_reciprocal_rank(row) for row in relevance_rows) / len(relevance_rows),
        f"ndcg_at_{k}": sum(ndcg_at_k(row, k) for row in relevance_rows) / len(relevance_rows),
    }
