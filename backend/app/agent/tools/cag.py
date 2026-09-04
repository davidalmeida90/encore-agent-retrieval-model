"""## Cache-augmented retrieval: put the filing in context, retrieve nothing

## Why this is a third mode and not a worse RAG
RAG and agentic retrieval both *select*, and selection is where this system loses
most of its answers. Measured on the retrieval eval, 62% of answers reach the
fused candidate pool and roughly 30% survive the cross-encoder into the six
passages the agent is shown. Every retrieval improvement made so far is throttled
by that last step.

CAG deletes the step. The whole filing enters the context window, so recall is
100% by construction: nothing is chosen, nothing is discarded, and there is no
ranker to get it wrong. What it trades away is scale and cross-company reach.

## Why one filing
An NVIDIA 10-K is about 155,000 tokens. The model advertises 262,144, so one
fits with room to spare and two do not. Loading the whole company history is not
a tuning decision that could go either way; it does not fit.

## Why the filing is instructions, not a tool result
This began as a tool, and that was wrong. A tool result arrives AFTER the
question in the message order, so two questions about the same company share only
the system prompt and all 156,000 tokens are re-prefilled every time. Persisting
the cache across turns is the paper's entire contribution.

In the instructions the order becomes `system(rules + filing) -> question`, which
is a stable prefix that vLLM's `--enable-prefix-caching` reuses. The server does
by itself what the paper does by hand.

The consequence is that CAG has NO retrieval tool. Nothing needs fetching, so
there is nothing to call. The tools that remain are for other work entirely:
XBRL figures, market data, the valuation engines.

## KV cache cost, on the model actually being served
Qwen3.8-27B is a hybrid: only 16 of its 64 layers keep a KV cache, the other 48
are linear attention with fixed-size state. So

    2 x 16 layers x 4 kv_heads x 256 head_dim x 2 bytes = 64 KB per token
    155,000 tokens -> ~10.2 GB

against roughly 53 GB free on an H100 NVL after the weights. A pure-attention
model of the same size would need about four times that.

## What it cannot do
Cross-company questions, and anything outside the ingested corpus. It is a
complement to the other two modes, not a replacement, which is why all three are
offered rather than one being chosen for you.
"""

from __future__ import annotations

import json
from typing import Annotated

from pydantic import Field
from pydantic_ai import RunContext
from sqlalchemy import text

from app.agent.deps import DocumentAgentDeps
from app.agent.status import emit_tool_start
from app.agent.tools._guards import _record_failure
from app.database.session import get_session
from app.retrieval.types import RetrievedPassage


def preload(ticker: str, fiscal_year: int = 0, registry=None) -> tuple[str, int]:
    """The filing as prompt text, plus how many passages were registered.

    Returns ("", 0) when there is nothing to load, so the caller can say so
    rather than let an empty context pass for an answer.
    """
    if not ticker:
        return "", 0
    try:
        with get_session() as session:
            document = session.execute(text(
                "SELECT id::text, ticker, company_name, form, fiscal_year, "
                "filing_date::text, accession_number FROM source_documents "
                "WHERE ticker = :ticker AND (:fy = 0 OR fiscal_year = :fy) "
                "ORDER BY fiscal_year DESC LIMIT 1"
            ), {"ticker": ticker.strip().upper(), "fy": fiscal_year}).mappings().first()
            if document is None:
                return "", 0
            chunks = session.execute(text(
                "SELECT id::text AS id, chunk_index, page, section, text "
                "FROM document_chunks WHERE document_id = :doc ORDER BY chunk_index"
            ), {"doc": document["id"]}).mappings().all()
    except Exception:
        return "", 0
    if not chunks:
        return "", 0

    from datetime import date as _date
    from uuid import UUID

    body: list[str] = []
    registered = 0
    for chunk in chunks:
        if registry is not None:
            try:
                registry.register(RetrievedPassage(
                    chunk_id=UUID(chunk["id"]),
                    document_id=UUID(document["id"]),
                    chunk_index=chunk["chunk_index"],
                    text=chunk["text"],
                    page=chunk["page"],
                    section=chunk["section"],
                    fusion_score=0.0,
                    ticker=document["ticker"],
                    company_name=document["company_name"],
                    form=document["form"],
                    filing_date=_date.fromisoformat(document["filing_date"]),
                    fiscal_year=document["fiscal_year"],
                    accession_number=document["accession_number"],
                ))
                registered += 1
            except Exception:
                pass
        body.append("[[" + chunk["id"] + "]]\n" + chunk["text"])

    header = (
        "## The filing, in full\n\n"
        + document["company_name"] + " (" + document["ticker"] + ") "
        + document["form"] + " FY" + str(document["fiscal_year"])
        + ", filed " + document["filing_date"] + ", accession "
        + document["accession_number"] + ". " + str(len(chunks))
        + " passages, complete and in order. Each is preceded by [[chunk_id]]; "
        "cite that id and quote from the text beneath it.\n\n"
    )
    return header + "\n\n".join(body), registered


def available_tickers() -> list[str]:
    """Companies that can be preloaded, for the UI picker."""
    try:
        with get_session() as session:
            return list(session.execute(text(
                "SELECT DISTINCT ticker FROM source_documents ORDER BY 1"
            )).scalars().all())
    except Exception:
        return []