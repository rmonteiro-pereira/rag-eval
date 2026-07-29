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
    #: Named arm from `retrieval.configs` used by the CLI and serving layer.
    retrieval_config: str = "hybrid+rerank+metadata"

    # --- reranking (local cross-encoder, CPU) ---
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_device: str = "cpu"
    reranker_max_length: int = 512

    # --- generation ---
    # "auto" -> use Ollama when reachable, otherwise fall back to extractive.
    llm_mode: str = "auto"  # auto | ollama | extractive
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_timeout_s: float = 180.0

    # --- LLM-as-judge (M3) ---
    #: Defaults to a *different* model from `ollama_model` where possible, so the
    #: judge is not grading its own output. `eval.run_generation` records when
    #: generator and judge coincide and flags those rows.
    judge_model: str = "llama3.1"
    #: Arms compared by the generation suite. Extractive is the groundedness
    #: floor, not a competitor: a verbatim quote cannot hallucinate.
    generation_arms: str = "extractive,qwen2.5:3b,llama3.1"

    # --- agent mode (M6) ---
    #: Read-only DuckDB export of the Open-Finance-LakeHouse gold marts. Produced
    #: by that project, lives outside this repo, and is never committed here.
    gold_duckdb_path: Path = REPO_ROOT.parent / "_artifacts" / "ofl_gold.duckdb"
    agent_model: str = "llama3.1"
    agent_max_steps: int = 4

    # --- governance (M5) ---
    #: How many of the most recent meetings are treated as embargoed. The
    #: classification is SYNTHETIC — these are public BACEN documents — and every
    #: report that uses it says so. See `governance/acl.py`.
    acl_restricted_count: int = 5
    audit_log_path: Path | None = None

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
