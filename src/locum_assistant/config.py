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

    # LLM: "anthropic" calls Claude directly; "openrouter" calls any model through OpenRouter's
    # OpenAI-compatible API; "extractive" needs no key and quotes the top passages.
    llm_provider: str = "extractive"
    openrouter_api_key: str | None = None
    openrouter_model: str = "anthropic/claude-opus-5.5"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: str = "medium"  # Opus 5.5 always thinks; effort sets how hard
    max_agent_steps: int = 6

    # API
    api_keys: str = ""  # comma-separated; empty means no auth (local development only)
    rate_limit_per_minute: int = 20
    daily_question_limit: int = 200  # whole-service cap for the public demo
    log_json: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
