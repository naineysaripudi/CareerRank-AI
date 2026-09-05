"""Transparent configurable weighted ranking score."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RankingConfig:
    semantic_weight: float = 0.40
    skill_weight: float = 0.30
    experience_weight: float = 0.10
    role_weight: float = 0.10
    location_weight: float = 0.05
    employment_weight: float = 0.05


def score_features(features: dict[str, float], config: RankingConfig | None = None) -> float:
    """Return a UI-friendly score from 0 to 100."""

    config = config or RankingConfig()
    score = (
        config.semantic_weight * features["semantic_similarity"]
        + config.skill_weight * features["skill_match"]
        + config.experience_weight * features["experience_match"]
        + config.role_weight * features["role_match"]
        + config.location_weight * features["location_match"]
        + config.employment_weight * features["employment_match"]
    )
    return round(max(0.0, min(100.0, score * 100)), 2)
