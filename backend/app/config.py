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
    #: Reasoning effort for the grounding judge only. "none" everywhere else:
    #: the agent loop was measured at 30x the output tokens with thinking on and
    #: an identical tool call, so it buys nothing there. Deciding whether an
    #: excerpt actually supports a claim is the one call in the turn that is a
    #: judgement rather than a lookup, and it is the call that decides whether
    #: the user sees an answer at all.
    #: Measured on six labelled cases: "none" and "low" both falsely REJECTED a
    #: citation whose excerpt was verbatim and did support the claim; "medium"
    #: scored 6/6 and "high" added latency without adding accuracy. A false
    #: rejection is expensive out of proportion to the judge itself, because the
    #: orchestrator answers one by re-running the entire agent.
    grounding_judge_reasoning: str = "medium"

    #: reason_about_assumptions thinks before it answers, and thinking has no
    #: natural stopping point. Unbounded, a Microsoft DCF hung for 20+ minutes
    #: with the GPU showing one request running and the UI showing nothing.
    #: 1,500 tokens is enough for every assumption plus its justification;
    #: 180 seconds is well past the slowest good run and short enough to fail
    #: visibly rather than look like a crash.
    assumptions_max_tokens: int = 1500
    assumptions_timeout_seconds: float = 180.0
    gemini_keyword_model: str = "gemini-3.1-flash-lite"
    # OpenAI-compatible shim, lets chat.completions.parse keep working
    gemini_openai_base_url: str = (
        "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    # --- Self-hosted model, any OpenAI-compatible server (vLLM, Ollama, ...) ---
    # Only read when a model whose provider is "openai_compatible" is selected,
    # so a Gemini-only deployment can leave both unset.
    local_llm_base_url: str = "http://localhost:8001/v1"
    local_llm_api_key: str = ""
    # What the SERVER calls the model, when it differs from the registry's
    # served_name. vLLM answers to whatever --served-model-name it was given;
    # Ollama answers to the full pull tag, e.g.
    # "hf.co/unsloth/Qwen-AgentWorld-35B-A3B-GGUF:Q8_0". Empty means use the
    # registry value.
    local_llm_model: str = ""

    # --- Agentic retrieval (index-free; see retrieval/modes.py) ---
    # SEC asks for a contact address rather than a key. Without one it rate
    # limits hard, and the failure looks like the tool being broken.
    sec_edgar_ua: str = ""
    # Optional. sec-api.io returns richer metadata than SEC's free full-text
    # index; when absent the tools fall back to efts.sec.gov, which needs no key
    # so the mode still works for anyone cloning this repo.
    sec_api_key: str = ""
    # Optional. Web search for questions filings cannot answer.
    tavily_api_key: str = ""
    # Characters handed to the sub-model per extract call. A 10-K runs to
    # megabytes, so the model reads it in windows rather than whole.
    agentic_extract_window: int = 40_000

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
    # How many fused candidates the cross-encoder is allowed to judge. Measured
    # on the retrieval eval (52 questions, ground truth by construction):
    #
    #     fused@30   37%      <- the old value
    #     fused@50   46%      +9 points, +101 ms of GPU rerank
    #     fused@100  54%      +17 points, +351 ms
    #
    # 30 was discarding more recall than the entire lexical leg contributes:
    # semantic search alone reached 46% at candidate_k=50, so the pool cut threw
    # away chunks the retriever had already found. 50 restores that for a tenth
    # of a second, against questions that take twenty to thirty.
    retrieval_rerank_pool: int = 50

    # Embed a hypothetical answer alongside the question. Measured on the
    # retrieval eval: dense recall 46% -> 58% overall, and 31% -> 38% on
    # paraphrased questions, which were the weak case. Costs one cheap model call
    # per search, so it trades roughly a second against a question that takes
    # twenty to thirty. See retrieval/hyde.py.
    retrieval_hyde_enabled: bool = True
    retrieval_hyde_timeout_seconds: float = 8.0

    # Score the lexical leg with BM25 instead of ts_rank_cd. Measured on the
    # retrieval eval over an IDENTICAL candidate set, so only the ranking
    # differs: lexical recall@50 went 23% -> 42% overall, and 46% -> 81% on
    # questions using the filing's own wording, where it beats even the vector
    # leg. IDF is why: "azure" scores 6.59 against "million" at 0.94, and
    # ts_rank_cd has no notion of either. See retrieval/bm25.py.
    retrieval_bm25_enabled: bool = True
    #: Candidates pulled from the index before BM25 reorders them. Bigger than
    #: candidate_k on purpose: rescoring can only promote what it was given.
    retrieval_bm25_candidates: int = 300
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
