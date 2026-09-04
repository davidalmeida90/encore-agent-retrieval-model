"""## Three ways to find what a filing says

## Why this is a choice, not a setting
There are three defensible designs for "get the passage that answers this", and
they fail in different directions. Encore was built on the first; the second is
what Vals AI's finance-agent-v2 uses for its benchmark; the third is the CAG
paper. All three are worth being able to run against the same questions.

    rag        index everything ahead of time, then search the index
    agentic    index nothing, let the model search SEC and read what it finds
    cag        search nothing, put one whole filing in the context window

## What actually differs
RAG needs an ingestion pipeline: download, convert, chunk, embed, store. Queries
are then cheap and, crucially, every answer can point at a chunk id that a
validator can check. The cost is that the corpus is a fixed set of companies and
years, and anything outside it simply does not exist.

Agentic retrieval needs no pipeline at all. The model queries SEC's full-text
search, fetches a document, and reads it. Any public company is reachable and
nothing goes stale. The cost is per-question: whole filings pass through the
model, so input tokens are large, and there is no chunk id to verify a claim
against, so the grounding gate has much less to work with.

CAG removes the selection step rather than improving it, which matters because
selection is where this system loses most of its answers: 62% of answers reach
the fused pool and about 30% survive the reranker. Recall becomes 100% by
construction. The cost is that one filing is all that fits, so cross-company
questions are impossible, and the first question pays ~90s of prefill.

    rag       ~7-30k input tokens/question   verifiable citations   10 companies
    agentic   whole documents per question   weaker grounding       every filer
    cag       ~190k once, then cached        nothing is discarded   one filing

## How it is wired
A mode is a set of tool names. The agent is built with a FilteredToolset, so
selecting a mode changes which tools the model is offered and nothing else: the
instructions, the output schema, the guards and the validator are shared.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RetrievalMode:
    id: str
    label: str
    hint: str
    #: Tools this mode adds. Everything not listed here is hidden from the model.
    tools: frozenset[str]
    default: bool = False
    #: Shown in the UI when the mode has a caveat worth knowing before you pick it.
    caveat: str | None = None
    #: Prompt fragment appended to the shared instructions for this mode. It has
    #: to be per-mode rather than shared: the RAG prompt tells the model the
    #: corpus holds two companies, which is true there and false here, and a
    #: model that believes it in agentic mode refuses questions it could answer.
    prompt_file: str = ""


# Tools that are not about finding filing text, and so belong to every mode:
# XBRL figures, market data, valuation, skills.
SHARED_TOOLS: frozenset[str] = frozenset()

MODES: list[RetrievalMode] = [
    RetrievalMode(
        id="rag",
        label="RAG",
        hint=(
            "Hybrid search over the indexed corpus: pgvector KNN and Postgres "
            "full-text, fused by reciprocal rank, then reordered by a "
            "cross-encoder. Every claim cites a chunk the validator can check."
        ),
        tools=frozenset(
            {"search_filings", "read_chunk", "read_chunks", "read_surrounding_chunks"}
        ),
        default=True,
        caveat="Limited to the 10 companies and years actually ingested.",
        prompt_file="retrieval_rag.md",
    ),
    RetrievalMode(
        id="agentic",
        label="Agentic retrieval",
        hint=(
            "No index. The model searches SEC EDGAR full-text directly, fetches "
            "the filing, and reads it with a sub-model call. Reaches any public "
            "filer and never goes stale."
        ),
        tools=frozenset(
            {
                "web_search",
                "edgar_search",
                "parse_html_page",
                "retrieve_information",
            }
        ),
        caveat="Whole documents pass through the model, so questions cost more "
               "and citations cannot be verified against an index.",
        prompt_file="retrieval_agentic.md",
    ),
    RetrievalMode(
        id="cag",
        label="CAG (whole filing in context)",
        hint=(
            "No search at all. One complete 10-K is loaded into the context "
            "window and the model answers from it directly, so nothing can be "
            "missed by a ranker. Best for many questions about one company."
        ),
        # No retrieval tools, deliberately. The filing is already in the
        # instructions, so there is nothing to fetch and nothing to call.
        # Everything else the agent has - XBRL, prices, the valuation
        # engines - is untouched by the mode.
        tools=frozenset(),
        caveat="One company and one year per conversation, from the ingested "
               "corpus only, and needs a context window of ~160k tokens.",
        prompt_file="retrieval_cag.md",
    ),
]

_BY_ID = {m.id: m for m in MODES}
DEFAULT_MODE = next(m.id for m in MODES if m.default)


def resolve(mode_id: str | None) -> str:
    """A known mode id, falling back to the default rather than raising.

    Same reasoning as models.resolve: a stale value in a browser tab should not
    make the chat unusable.
    """
    return mode_id if mode_id in _BY_ID else DEFAULT_MODE


def mode(mode_id: str | None) -> RetrievalMode:
    return _BY_ID[resolve(mode_id)]


def retrieval_tool_names() -> frozenset[str]:
    """Every tool owned by some mode, so the gate knows what to hide."""
    names: set[str] = set()
    for m in MODES:
        names |= m.tools
    return frozenset(names)


_PROMPTS = Path(__file__).resolve().parents[1] / "agent" / "prompts"


@lru_cache(maxsize=8)
def instructions(mode_id: str | None) -> str:
    """The prompt fragment for a mode, read once and cached."""
    path = _PROMPTS / mode(mode_id).prompt_file
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""
