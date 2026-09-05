"""Retrieve job metadata using candidate semantic similarity."""

import pandas as pd

from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.vector_store import JobVectorStore


class JobRetriever:
    def __init__(self, jobs: pd.DataFrame, embeddings: EmbeddingService, store: JobVectorStore) -> None:
        self.jobs = jobs.set_index("job_id", drop=False)
        self.embeddings = embeddings
        self.store = store

    def retrieve(self, query_text: str, top_k: int = 20) -> list[dict[str, object]]:
        query_embedding = self.embeddings.embed_text(query_text)
        matches = self.store.search(query_embedding, top_k)
        results = []
        for job_id, similarity in matches:
            if job_id in self.jobs.index:
                result = self.jobs.loc[job_id].to_dict()
                result["semantic_similarity"] = similarity
                results.append(result)
        return results
