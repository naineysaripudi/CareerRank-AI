"""Build and persist the FAISS job index."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.data.loader import load_jobs
from app.data.preprocessing import prepare_jobs
from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.vector_store import JobVectorStore


def main() -> None:
    settings = get_settings()
    jobs = prepare_jobs(load_jobs(settings.data_path))
    service = EmbeddingService(settings.embedding_model)
    embeddings = service.embed_documents(jobs["search_text"].tolist())
    JobVectorStore.build(embeddings, jobs["job_id"].tolist()).save(settings.vector_store_path)
    print(f"Indexed {len(jobs)} jobs at {settings.vector_store_path}")


if __name__ == "__main__":
    main()
