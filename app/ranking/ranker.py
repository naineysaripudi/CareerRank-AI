"""Apply feature scoring and return ranked jobs."""

from app.nlp.profile_builder import CandidateProfile
from app.ranking.features import calculate_features
from app.ranking.scoring import RankingConfig, score_features


def rank_jobs(profile: CandidateProfile, retrieved_jobs: list[dict[str, object]], config: RankingConfig | None = None) -> list[dict[str, object]]:
    ranked = []
    for job in retrieved_jobs:
        features = calculate_features(profile, job, float(job.get("semantic_similarity", 0.0)))
        ranked.append({**job, **features, "final_score": score_features(features, config)})
    ranked.sort(key=lambda job: float(job["final_score"]), reverse=True)
    for rank, job in enumerate(ranked, start=1):
        job["rank"] = rank
    return ranked
