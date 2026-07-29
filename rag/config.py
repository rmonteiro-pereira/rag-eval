"""Single source of truth for paths and service settings.

Everything is overridable through environment variables (see `.env.example`),
so the same code runs against the local docker-compose stack or, later, CI.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- paths ---
    data_dir: Path = REPO_ROOT / "data"

    # --- vector store ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "bacen_copom"

    # --- embeddings (local, CPU is fine) ---
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 4

    # --- chunking (M1 baseline: deliberately naive fixed-size) ---
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # --- retrieval ---
    top_k: int = 5

    # --- generation ---
    # "auto" -> use Ollama when reachable, otherwise fall back to extractive.
    llm_mode: str = "auto"  # auto | ollama | extractive
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_timeout_s: float = 180.0

    # --- tracing ---
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = "pk-lf-rag-eval-local"
    langfuse_secret_key: str = "sk-lf-rag-eval-local"
    langfuse_enabled: bool = True

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"


settings = Settings()
