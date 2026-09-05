"""Orchestrate ranked recommendations and explanations."""

from app.llm.client import DemoLLMClient
from app.core.config import get_settings
from app.llm.client import OpenAICompatibleClient
from app.nlp.profile_builder import CandidateProfile
from app.ranking.ranker import rank_jobs
from app.recommendation.skill_gap import analyze_skill_gap


def recommend(profile: CandidateProfile, retrieved_jobs: list[dict[str, object]], top_k: int = 10) -> list[dict[str, object]]:
    ranked = rank_jobs(profile, retrieved_jobs)[:top_k]
    settings = get_settings()
    client = DemoLLMClient()
    if settings.llm_provider.casefold() == "openai" and settings.llm_api_key and settings.llm_model:
        client = OpenAICompatibleClient(settings.llm_api_key, settings.llm_model, settings.llm_base_url)
    for job in ranked:
        job["skill_gap"] = analyze_skill_gap(profile.skills, job)
        try:
            job["explanation"] = client.explain(job)
        except Exception:
            job["explanation"] = DemoLLMClient().explain(job)
    return ranked
