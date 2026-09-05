"""Optional learned relevance ranker using scikit-learn."""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor

FEATURE_NAMES = [
    "semantic_similarity", "skill_match", "experience_match", "role_match",
    "location_match", "employment_match",
]


@dataclass
class MLRanker:
    """Small supervised ranker trained on labeled matching features."""

    model: RandomForestRegressor | None = None

    def fit(self, feature_rows: list[dict[str, float]], labels: list[float]) -> "MLRanker":
        matrix = np.array([[row[name] for name in FEATURE_NAMES] for row in feature_rows])
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=2)
        self.model.fit(matrix, labels)
        return self

    def predict(self, feature_rows: list[dict[str, float]]) -> list[float]:
        if self.model is None:
            raise RuntimeError("MLRanker must be fitted before prediction.")
        matrix = np.array([[row[name] for name in FEATURE_NAMES] for row in feature_rows])
        return self.model.predict(matrix).tolist()
