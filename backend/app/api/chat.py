"""FastAPI routes for chat threads and stubbed streaming."""

from __future__ import annotations

import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.auth.dependencies import CurrentUser, get_access_token, get_current_user
from app.turn.messages import extract_last_user_message
from app.turn.orchestrator import run_turn
from app.database.chats import (
    create_thread,
    delete_thread,
    list_threads,
    load_messages,
    require_thread_access,
)
from app.database.documents import get_chunk_context
from app.database.models import DocumentChunk
from app.database.session import get_session
from app.database.supabase import create_user_client
from app.database.users import ensure_user
from app.retrieval.retriever import DocumentRetriever
from app.schemas.chat import (
    CitationContextChunk,
    CitationContextResponse,
    CitationContextTable,
    CreateThreadRequest,
    MessageHistoryResponse,
    StreamRequest,
    ThreadListResponse,
    ThreadResponse,
)

router = APIRouter(prefix="/chat", tags=["chat"])


def _chunk_role(
    chunk: DocumentChunk,
    anchor: DocumentChunk,
) -> Literal["previous", "anchor", "next"]:
    if chunk.id == anchor.id:
        return "anchor"
    if chunk.chunk_index < anchor.chunk_index:
        return "previous"
    return "next"


def citation_context_response(
    chunks: list[DocumentChunk],
    *,
    anchor_chunk_id: uuid.UUID,
) -> CitationContextResponse:
    anchor = next(chunk for chunk in chunks if chunk.id == anchor_chunk_id)
    document = anchor.document
    return CitationContextResponse(
        anchor_chunk_id=anchor.id,
        document_id=anchor.document_id,
        ticker=document.ticker,
        company_name=document.company_name,
        form=document.form,
        filing_date=document.filing_date,
        source_url=document.source_url,
        table=_table_context_from_chunk(anchor),
        chunks=[
            CitationContextChunk(
                chunk_id=chunk.id,
                chunk_index=chunk.chunk_index,
                role=_chunk_role(chunk, anchor),
                text=chunk.text,
                page=chunk.page,
                section=chunk.section,
            )
            for chunk in chunks
        ],
    )


def _table_context_from_chunk(chunk: DocumentChunk) -> CitationContextTable | None:
    metadata = chunk.chunk_metadata or {}
    if metadata.get("chunk_kind") != "table_row":
        return None
    table_data = metadata.get("table")
    if not isinstance(table_data, dict):
        return None
    return CitationContextTable(
        table_index=table_data["table_index"],
        title=table_data.get("title"),
        units=table_data.get("units"),
        markdown=table_data["markdown"],
        table_data=table_data,
    )


def load_citation_context(
    chunk_id: uuid.UUID,
    radius: int,
) -> CitationContextResponse | None:
    with get_session() as session:
        chunks = get_chunk_context(session, chunk_id, radius)
        if chunks is None:
            return None
        return citation_context_response(chunks, anchor_chunk_id=chunk_id)


@router.get("/threads")
async def get_threads(
    user: CurrentUser = Depends(get_current_user),
    access_token: str = Depends(get_access_token),
) -> ThreadListResponse:
    await ensure_user(user)
    client = await create_user_client(access_token)
    threads = await list_threads(client, user)
    return ThreadListResponse(threads=threads)


@router.post("/threads")
async def post_thread(
    body: CreateThreadRequest,
    user: CurrentUser = Depends(get_current_user),
    access_token: str = Depends(get_access_token),
) -> ThreadResponse:
    await ensure_user(user)
    client = await create_user_client(access_token)
    return await create_thread(client, user, title=body.title)


@router.get("/threads/{thread_id}/messages")
async def get_thread_messages(
    thread_id: uuid.UUID,
    user: CurrentUser = Depends(get_current_user),
    access_token: str = Depends(get_access_token),
) -> MessageHistoryResponse:
    await require_thread_access(thread_id, user)
    client = await create_user_client(access_token)
    messages = await load_messages(client, thread_id)
    return MessageHistoryResponse(messages=messages)


@router.get("/citations/{chunk_id}/context")
async def get_citation_context(
    chunk_id: uuid.UUID,
    radius: int = Query(default=1, ge=0, le=3),
    _user: CurrentUser = Depends(get_current_user),
) -> CitationContextResponse:
    context = await run_in_threadpool(load_citation_context, chunk_id, radius)
    if context is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Citation chunk not found",
        )
    return context


@router.delete("/threads/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_thread_route(
    thread_id: uuid.UUID,
    user: CurrentUser = Depends(get_current_user),
    access_token: str = Depends(get_access_token),
) -> None:
    await require_thread_access(thread_id, user)
    client = await create_user_client(access_token)
    await delete_thread(client, thread_id)


@router.get("/models")
async def list_models(_: CurrentUser = Depends(get_current_user)) -> list[dict]:
    """Models the UI can offer, cheapest first."""
    from app.agent.models import MODELS

    return [
        {
            "id": m.id,
            "label": m.label,
            "hint": m.hint,
            "default": m.default,
            # Where it runs, so the picker can say so. A self-hosted model with
            # no server up fails with a bare connection error, which reads as a
            # broken app rather than a machine that is switched off.
            "self_hosted": m.provider == "openai_compatible",
        }
        for m in MODELS
    ]


@router.get("/cag-companies")
async def list_cag_companies(_: CurrentUser = Depends(get_current_user)) -> list[str]:
    """Companies CAG can preload. It cannot infer one from the question: the
    filing has to be in the prompt before the question is asked."""
    from app.agent.tools.cag import available_tickers

    return available_tickers()


@router.post("/cag-warm")
async def warm_cag_cache(
    ticker: str,
    _: CurrentUser = Depends(get_current_user),
) -> dict:
    """Prefill the model's cache with one filing, before any question is asked.

    CAG's cost is front-loaded: the first question pays ~90 seconds to prefill
    191,000 tokens and every later one reuses the cached prefix in ~2 seconds.
    Left alone, that wait lands after the user presses send, which is the worst
    possible moment for it and looks like a hang.

    This moves it to the moment they pick the company, so the loading happens
    while they are still typing. It asks the model for a single token; the
    answer is discarded and only the cache it leaves behind matters.
    """
    import time

    from openai import AsyncOpenAI

    from app.agent.agent import INSTRUCTIONS
    from app.agent.tools.cag import preload
    from app.retrieval import modes

    started = time.time()
    filing, passages = preload(ticker)
    if not filing:
        return {"ok": False, "error": f"No filing held for {ticker.upper()}."}

    prompt = INSTRUCTIONS + "\n\n" + modes.instructions("cag") + "\n\n" + filing
    if not (settings.local_llm_base_url and settings.local_llm_model):
        # Hosted models have no prefix cache we control, so there is nothing to
        # warm. Report the size so the UI can still set expectations honestly.
        return {"ok": True, "warmed": False, "passages": passages,
                "characters": len(prompt),
                "note": "No local server configured; the filing is sent with each question."}

    client = AsyncOpenAI(api_key=settings.local_llm_api_key or "none",
                         base_url=settings.local_llm_base_url)
    try:
        await client.chat.completions.create(
            model=settings.local_llm_model,
            messages=[{"role": "system", "content": prompt},
                      {"role": "user", "content": "Reply with the single word ready."}],
            max_tokens=1,
            temperature=0,
        )
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}

    return {"ok": True, "warmed": True, "passages": passages,
            "seconds": round(time.time() - started, 1)}


@router.get("/retrieval-modes")
async def list_retrieval_modes(_: CurrentUser = Depends(get_current_user)) -> list[dict]:
    """How the agent may look for filing text. See retrieval/modes.py."""
    from app.retrieval import modes

    return [
        {
            "id": m.id,
            "label": m.label,
            "hint": m.hint,
            "default": m.default,
            "caveat": m.caveat,
        }
        for m in modes.MODES
    ]


@router.post("/stream")
async def post_stream(
    body: StreamRequest,
    user: CurrentUser = Depends(get_current_user),
    access_token: str = Depends(get_access_token),
) -> StreamingResponse:
    await ensure_user(user)
    thread = await require_thread_access(body.thread_id, user)
    user_message = extract_last_user_message(body.messages)
    client = await create_user_client(access_token)

    retriever = DocumentRetriever()
    return StreamingResponse(
        run_turn(
            client=client,
            thread_id=body.thread_id,
            user=user,
            user_message=user_message,
            thread_title=thread.title,
            retriever=retriever,
            model_name=body.model,
            retrieval_mode=body.retrieval_mode,
            cag_ticker=body.cag_ticker,
            thinking=body.thinking,
        ),
        media_type="text/event-stream",
    )
