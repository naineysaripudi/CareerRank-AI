"""Configurable sentence-transformer embedding service."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Load one model per model name and expose normalized embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = _load_model(model_name)

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True)[0]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def similarity(self, first: str, second: str) -> float:
        return float(np.dot(self.embed_text(first), self.embed_text(second)))


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)
