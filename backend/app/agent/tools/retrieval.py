"""## Retrieval tools: "what did management SAY"

Everything here reads indexed filing text. Anything answerable from a number
belongs in fundamentals.py or market.py, because arithmetic over retrieved prose
is the documented FinanceBench failure mode.

## Examples

    search_filings("Azure competitive advantage", ticker="MSFT")
      -> 6 passages, each headed
         MSFT 10-K FY2026 (Item 1) relevance=5.2 chunk_id=<uuid>: <text>
                                   ^ the cross-encoder score. Above +2 means the
                                     passage answers the question; below -2 means
                                     the filing does not discuss it and the model
                                     should say so rather than search again.

    search_filings("why Apple spends on capex", ticker="AAPL")
      -> best relevance -4.7, because Apple's 10-K never explains its capex level.
         Absence is a finding, not a failure.

    read_chunks([uuid, uuid])          full text of specific passages
    read_surrounding_chunks(uuid)      the lines either side, for table rows

## Why chunk ids are printed as chunk_id=<uuid> and never [<uuid>]
Square brackets mean a citation index in an answer, and the grounding validator
rejects anything else. Printing "[uuid]" here taught the model to copy that shape
into its prose, which failed validation and forced a full re-run of the agent.
"""

from __future__ import annotations

import asyncio
import functools
import time
from uuid import UUID

from pydantic_ai import RunContext

from app.agent import tuning
from app.agent.deps import DocumentAgentDeps
from app.agent.progress import report_progress
from app.agent.status import emit_tool_start
from app.config import settings
from app.database.documents import (
    get_chunk_with_document,
    get_chunks_by_ids,
    get_surrounding_chunks,
)
from app.database.models import DocumentChunk, SourceDocument
from app.database.session import get_session
from app.retrieval.types import RetrievedPassage, SearchFilters, format_passages_for_agent


def _passage_from_chunk(
    chunk: DocumentChunk,
    document: SourceDocument,
    *,
    fusion_score: float = 0.0,
) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        page=chunk.page,
        section=chunk.section,
        fusion_score=fusion_score,
        ticker=document.ticker,
        company_name=document.company_name,
        form=document.form,
        filing_date=document.filing_date,
        fiscal_year=document.fiscal_year,
        accession_number=document.accession_number,
        neighbors=[],
    )


def _parse_fiscal_years(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    years = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return years or None


def _search_sync(
    deps: DocumentAgentDeps,
    query: str,
    *,
    ticker: str | None,
    form: str | None,
    fiscal_years: str | None,
) -> list[RetrievedPassage]:
    filters = SearchFilters(
        ticker=ticker,
        form=form,
        fiscal_years=_parse_fiscal_years(fiscal_years),
    )
    return deps.retriever.search(query, filters=filters)


def _read_chunk_sync(deps: DocumentAgentDeps, chunk_id: UUID) -> RetrievedPassage | None:
    with get_session() as session:
        result = get_chunk_with_document(session, chunk_id)
        if result is None:
            return None
        chunk, document = result
        return _passage_from_chunk(chunk, document)


def _read_chunks_sync(
    deps: DocumentAgentDeps,
    chunk_ids: list[UUID],
) -> list[RetrievedPassage]:
    with get_session() as session:
        chunks_by_id = get_chunks_by_ids(session, chunk_ids)
        passages: list[RetrievedPassage] = []
        for chunk_id in chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None or chunk.document is None:
                continue
            passages.append(_passage_from_chunk(chunk, chunk.document))
        return passages


def _read_surrounding_sync(
    deps: DocumentAgentDeps,
    chunk_id: UUID,
    radius: int,
) -> list[RetrievedPassage]:
    with get_session() as session:
        anchor = get_chunk_with_document(session, chunk_id)
        if anchor is None:
            return []
        anchor_chunk, _ = anchor
        neighbor_chunks = get_surrounding_chunks(session, chunk_id, radius)
        passages: list[RetrievedPassage] = []
        for neighbor_chunk in neighbor_chunks:
            if neighbor_chunk.document is None:
                continue
            passages.append(
                _passage_from_chunk(neighbor_chunk, neighbor_chunk.document)
            )
        if anchor_chunk.document is not None:
            passages.insert(
                0,
                _passage_from_chunk(anchor_chunk, anchor_chunk.document),
            )
        return passages


async def _run_tool(
    deps: DocumentAgentDeps,
    name: str,
    detail: str,
    fn,
    /,
    *args,
    **kwargs,
):
    emit_tool_start(deps, name, detail)
    started = time.perf_counter()
    result = await asyncio.to_thread(functools.partial(fn, *args, **kwargs))
    if isinstance(result, list):
        summary = f"{len(result)} results"
    elif result is None:
        summary = "not found"
    else:
        summary = "1 result"
    report_progress(
        f"tool {name} done ({summary}) in {time.perf_counter() - started:.2f}s"
    )
    return result


# --------------------------------------------------------------- search budget
# The generic repeat guard in _guards.py keys on EXACT arguments, so it never
# fires for search: the model rephrases slightly every time. Observed on a
# question asking why Apple spends what it does on capex, which Apple's 10-K
# simply does not say. Six near-identical AAPL searches, 174,000 tokens, before
# it concluded absence. The same question shape where the text does exist cost
# 80,000.
#
# Counting per TICKER catches the rephrasing that per-argument counting misses.
# Absence is a legitimate, reportable finding; the model just has to be told it
# is allowed to stop looking.
_searches: dict[tuple[str, str], int] = {}


def _search_budget_exceeded(ctx, ticker: str | None) -> str | None:
    turn = str(getattr(ctx.deps, "thread_id", "?"))
    total_key = (turn, "__total__")
    _searches[total_key] = _searches.get(total_key, 0) + 1
    if _searches[total_key] > tuning.MAX_SEARCHES_PER_TURN:
        return (
            f"STOP: {tuning.MAX_SEARCHES_PER_TURN} searches have already run for "
            "this question, which is the budget. Answer now from what you have, "
            "and say plainly which parts the filings did not cover."
        )

    key = (turn, (ticker or "*").upper())
    _searches[key] = _searches.get(key, 0) + 1
    if _searches[key] > tuning.MAX_SEARCHES_PER_TICKER:
        target = ticker.upper() if ticker else "this corpus"
        return (
            f"STOP: you have already searched {target} "
            f"{tuning.MAX_SEARCHES_PER_TICKER} times this turn with different "
            "wordings. Rephrasing again will not surface text that is not there. "
            "If you have not found it by now, the filing very likely does not "
            "discuss it. That is a real and useful finding: say plainly that the "
            "company does not disclose it, and answer the rest of the question "
            "from what you already have."
        )
    return None


async def search_filings(
    ctx: RunContext[DocumentAgentDeps],
    query: str,
    ticker: str | None = None,
    form: str | None = None,
    fiscal_years: str | None = None,
) -> str:
    """Search SEC filings with hybrid retrieval. Optional filters: ticker, form, fiscal_years (comma-separated)."""
    if (stop := _search_budget_exceeded(ctx, ticker)):
        return stop
    filter_bits = [
        bit
        for bit in (
            f"ticker={ticker}" if ticker else None,
            f"form={form}" if form else None,
            f"fiscal_years={fiscal_years}" if fiscal_years else None,
        )
        if bit
    ]
    detail = ", ".join(filter_bits) if filter_bits else "no filters"
    passages = await _run_tool(
        ctx.deps,
        "search_filings",
        detail,
        _search_sync,
        ctx.deps,
        query,
        ticker=ticker,
        form=form,
        fiscal_years=fiscal_years,
    )
    ctx.deps.registry.register_many(passages)
    return format_passages_for_agent(passages)


async def read_chunk(ctx: RunContext[DocumentAgentDeps], chunk_id: str) -> str:
    """Read the full text of a specific document chunk by UUID."""
    try:
        parsed_id = UUID(chunk_id)
    except ValueError:
        return f"Error: invalid chunk_id {chunk_id!r}."

    passage = await _run_tool(
        ctx.deps,
        "read_chunk",
        f"chunk_id={chunk_id}",
        _read_chunk_sync,
        ctx.deps,
        parsed_id,
    )
    if passage is None:
        return f"Error: chunk {chunk_id} not found."

    ctx.deps.registry.register(passage)
    return format_passages_for_agent([passage])


async def read_chunks(ctx: RunContext[DocumentAgentDeps], chunk_ids: list[str]) -> str:
    """Read the full text of multiple document chunks in one call."""
    parsed_ids: list[UUID] = []
    for chunk_id in chunk_ids:
        try:
            parsed_ids.append(UUID(chunk_id))
        except ValueError:
            return f"Error: invalid chunk_id {chunk_id!r}."

    if not parsed_ids:
        return "Error: chunk_ids must include at least one UUID."

    passages = await _run_tool(
        ctx.deps,
        "read_chunks",
        f"count={len(parsed_ids)}",
        _read_chunks_sync,
        ctx.deps,
        parsed_ids,
    )
    if not passages:
        return "Error: none of the requested chunks were found."

    ctx.deps.registry.register_many(passages)
    return format_passages_for_agent(passages)


async def read_surrounding_chunks(
    ctx: RunContext[DocumentAgentDeps],
    chunk_id: str,
    radius: int | None = None,
) -> str:
    """Read chunks before and after a given chunk within the same filing."""
    try:
        parsed_id = UUID(chunk_id)
    except ValueError:
        return f"Error: invalid chunk_id {chunk_id!r}."

    resolved_radius = (
        radius if radius is not None else settings.retrieval_read_neighbor_radius
    )
    if resolved_radius < 1:
        return "Error: radius must be 1 or greater."

    passages = await _run_tool(
        ctx.deps,
        "read_surrounding_chunks",
        f"chunk_id={chunk_id} radius={resolved_radius}",
        _read_surrounding_sync,
        ctx.deps,
        parsed_id,
        resolved_radius,
    )
    if not passages:
        return f"Error: chunk {chunk_id} not found."

    ctx.deps.registry.register_many(passages)
    return format_passages_for_agent(passages)
