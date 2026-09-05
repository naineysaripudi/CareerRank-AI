"""Environment-backed application settings.

Settings are intentionally centralized so later services can share the same
configuration without reading environment variables directly.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a .env file."""

    app_name: str = "CareerRank AI"
    environment: str = "development"
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k_retrieval: int = 20
    top_k_results: int = 10
    llm_provider: str = "demo"
    llm_model: str = ""
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    data_path: str = "data/sample_jobs.csv"
    vector_store_path: str = "vector_store/jobs.faiss"
    max_upload_size_mb: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings object for the process lifetime."""

    return Settings()
