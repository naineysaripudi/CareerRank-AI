"""FAISS index persistence for normalized job embeddings."""

from pathlib import Path

import faiss
import numpy as np


class JobVectorStore:
    """Store a FAISS inner-product index and matching job identifiers."""

    def __init__(self, index: faiss.Index, job_ids: list[str]) -> None:
        self.index = index
        self.job_ids = job_ids

    @classmethod
    def build(cls, embeddings: np.ndarray, job_ids: list[str]) -> "JobVectorStore":
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(np.asarray(embeddings, dtype="float32"))
        return cls(index, job_ids)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        scores, positions = self.index.search(np.asarray([query_embedding], dtype="float32"), top_k)
        return [(self.job_ids[position], float(score)) for position, score in zip(positions[0], scores[0]) if position >= 0]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        np.save(path.with_suffix(".ids.npy"), np.array(self.job_ids, dtype=str))

    @classmethod
    def load(cls, path: str | Path) -> "JobVectorStore":
        path = Path(path)
        index = faiss.read_index(str(path))
        ids = np.load(path.with_suffix(".ids.npy")).tolist()
        return cls(index, ids)
