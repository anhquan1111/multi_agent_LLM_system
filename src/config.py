"""Application settings, loaded once from environment / .env."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Credentials. Empty by default so the app still boots without a key —
    # /health reports the gap instead of the process crashing on import.
    google_api_key: str = ""

    # Models
    llm_model: str = "gemini-3.6-flash"
    embedding_model: str = "gemini-embedding-2"
    embedding_dim: int = 768

    # Guardrails
    max_agent_steps: int = 8
    max_tokens_per_request: int = 8000
    request_timeout_seconds: int = 60

    # Tracing (optional). LangSmith reads these from os.environ, not from here — see
    # src/observability.py for why they have to be copied across explicitly.
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "multi-agent-llm-system"

    # Storage
    chroma_dir: Path = Path("data/chroma")
    sqlite_path: Path = Path("data/shop.db")
    checkpoint_path: Path = Path("checkpoints/graph.sqlite")

    @property
    def has_api_key(self) -> bool:
        return bool(self.google_api_key.strip())


settings = Settings()
