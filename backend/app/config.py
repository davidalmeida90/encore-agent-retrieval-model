from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str
    database_url: str

    openai_api_key: str

    # --- Gemini (generation). Embeddings stay on OpenAI. ---
    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-3.1-flash-lite"
    gemini_grounding_model: str = "gemini-3.1-flash-lite"
    gemini_keyword_model: str = "gemini-3.1-flash-lite"
    # OpenAI-compatible shim, lets chat.completions.parse keep working
    gemini_openai_base_url: str = (
        "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536
    # Gates the agent loop regardless of provider; the old name said
    # "openai" while the agent has been running on Gemini.
    agent_request_limit: int = 20

    retrieval_candidate_k: int = 50
    # 10 passages plus a neighbour each = 12,030 chars per search (~3,007 tokens),
    # re-sent on every later round. With 5 searches that payload IS the cost of a
    # question. 6 without neighbours is ~5,300 chars, a 56% cut. Raise again if
    # answers start missing evidence rather than merely getting cheaper.
    retrieval_top_k: int = 6
    retrieval_rrf_k: int = 60
    # Neighbours mattered more before table rows carried their own table_title;
    # each one adds ~325 chars to every passage for context now largely duplicated.
    retrieval_neighbor_radius: int = 0

    # read_surrounding_chunks exists precisely to fetch context on demand, so it
    # keeps its own radius. Tying it to the search radius above set it to 0 and
    # broke the tool that makes a lean search payload safe in the first place:
    # the model can now pull context for the one passage that needs it, instead
    # of every passage carrying context it will never read.
    retrieval_read_neighbor_radius: int = 1
    retrieval_fts_config: str = "english"

    # Reranking. The pool is what the cross-encoder sees; top_k is what survives.
    # Pool of 30 costs ~0.25s on CPU and gives the reranker enough to choose from;
    # fusion alone was picking 6 of 50 on rank position, which carries no relevance.
    retrieval_rerank_enabled: bool = True
    retrieval_rerank_pool: int = 30
    retrieval_rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    retrieval_rerank_max_tokens: int = 512
    # Torch CPU threads for reranking. Was pinned to 1 to dodge a Windows
    # "paging file is too small" error, which turned out to be the disk at 0.6%
    # free rather than anything about torch. Re-measured on 16 cores: 1 thread
    # 1.87s, 4 threads 0.99s, 8 threads 0.48s for the same 30 passages, and 8
    # loads cleanly inside the full app. Lower it if the paging error returns.
    retrieval_rerank_threads: int = 8
    # "auto" uses the GPU when torch reports one, else CPU. Pin to "cpu" to keep
    # the GPU free, or "cuda" to fail loudly rather than silently falling back.
    retrieval_rerank_device: str = "auto"

    # Vendored loaders cache settled price ranges to parquet, but they read the
    # flag from os.environ directly (their own config accessor was de-vendored),
    # so a value in .env alone never reaches them. Exported below.
    loader_data_cache: bool = True
    retrieval_fts_keyword_min: int = 3
    retrieval_fts_keyword_max: int = 5
    retrieval_fts_keyword_fast_path_tokens: int = 5

    # Comma-separated in .env; use `cors_origins` for the parsed list.
    allowed_origins: str = "http://localhost:5173"

    @computed_field
    @property
    def sqlalchemy_database_url(self) -> str:
        """Normalize Supabase-style URLs for SQLAlchemy + psycopg v3."""
        url = self.database_url
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+psycopg://", 1)
        if url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql+psycopg://", 1)
        return url

    @computed_field
    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]


settings = Settings()


def _export_loader_env() -> None:
    """Bridge settings into os.environ for the vendored loaders.

    They predate this app's config and read os.getenv directly. Without this the
    price cache silently stays off no matter what .env says, and every question
    refetches a full year from Yahoo.
    """
    import os

    os.environ.setdefault("LOADER_DATA_CACHE", "1" if settings.loader_data_cache else "0")


_export_loader_env()
