from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime settings. Each one can be overridden by an environment variable of the same
    name (DATABASE_URL, LLM_PROVIDER, ...) or by a line in .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://locum:locum@localhost:5432/locum"
    data_dir: Path = Path("data")

    # Local embedding model (ONNX via fastembed): no API key, runs on CPU, 384 dimensions.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Retrieval
    top_k: int = 5
    candidates: int = 20  # how many hits each retriever returns before fusion
    max_chunks_per_doc: int = 2  # stops one long document filling every slot

    # LLM: "fake" runs the agent with a scripted model, for tests and for working without a key.
    llm_provider: str = "fake"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5-5"


@lru_cache
def get_settings() -> Settings:
    return Settings()
