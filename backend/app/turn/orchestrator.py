"""Coordinates one chat turn: agent → validate → stream → persist."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

from supabase import AsyncClient

from app import telemetry
from app.agent.agent import run_document_agent
from app.agent import tuning
from app.agent.deps import DocumentAgentDeps, TurnRegistry
from app.agent.outputs import GroundedAnswer
from app.auth.dependencies import CurrentUser
from app.turn.messages import text_from_parts
from app.turn.streaming import (
    stream_grounded_turn_and_persist,
    stream_error,
    stream_status,
)
from app.verification.validator import GroundingValidator, prune_unreferenced_citations
from app.retrieval.retriever import DocumentRetriever
from app.schemas.chat import UIMessage

MAX_VALIDATION_ATTEMPTS = tuning.MAX_VALIDATION_ATTEMPTS


async def _yield_status_updates(
    status_queue: asyncio.Queue[tuple[str, str]],
    agent_task: asyncio.Task[GroundedAnswer],
) -> AsyncIterator[str]:
    while not agent_task.done():
        try:
            stage, message = await asyncio.wait_for(status_queue.get(), timeout=0.3)
        except TimeoutError:
            continue
        async for event in stream_status(stage, message):
            yield event

    while not status_queue.empty():
        stage, message = status_queue.get_nowait()
        async for event in stream_status(stage, message):
            yield event


async def run_turn(
    *,
    client: AsyncClient,
    thread_id: uuid.UUID,
    user: CurrentUser,
    user_message: UIMessage,
    thread_title: str,
    retriever: DocumentRetriever,
    model_name: str | None = None,
    retrieval_mode: str | None = None,
    thinking: bool | None = None,
    cag_ticker: str = "",
) -> AsyncIterator[str]:
    loop = asyncio.get_running_loop()
    query = text_from_parts(user_message.parts).strip()
    if not query:
        async for event in stream_error("User message is empty."):
            yield event
        return

    async for event in stream_status("analyzing", "Analyzing your question…"):
        yield event

    grounded: GroundedAnswer | None = None
    validation = None
    for attempt in range(1, MAX_VALIDATION_ATTEMPTS + 1):
        registry = TurnRegistry()
        status_queue = asyncio.Queue()

        def on_status(stage: str, message: str) -> None:
            loop.call_soon_threadsafe(status_queue.put_nowait, (stage, message))

        deps = DocumentAgentDeps(
            retriever=retriever,
            registry=registry,
            thread_id=thread_id,
            user_id=user.id,
            on_status=on_status,
            retrieval_mode=retrieval_mode,
            thinking=thinking,
            cag_ticker=cag_ticker,
        )
        counters = telemetry.start_turn()
        # Attempt 2 previously re-sent the identical query with no hint about what
        # failed, so it was a blind re-roll paying full price for a fresh sample.
        # Appending the rejection reason turns it into a correction.
        attempt_query = query if validation is None or validation.ok else (
            f"{query}\n\n[Your previous answer was rejected by the grounding "
            f"check: {validation.error} Fix exactly that and answer again. "
            f"Cite with [1], [2] markers that match your citations list.]"
        )
        agent_task = asyncio.create_task(
            asyncio.to_thread(run_document_agent, attempt_query, deps, model_name)
        )

        async for event in _yield_status_updates(status_queue, agent_task):
            yield event

        try:
            grounded = await agent_task
        except Exception as exc:
            async for event in stream_error(f"Assistant run failed: {exc}"):
                yield event
            return

        async for event in stream_status("verifying", "Verifying citations…"):
            yield event

        grounded = prune_unreferenced_citations(grounded)
        validation = await GroundingValidator().validate(
            grounded, registry, tool_outputs=deps.tool_outputs
        )
        telemetry.log_turn(counters, thread_id=str(thread_id), query=query)

        # Report the verdict either way. The gate decides whether an answer is
        # shown at all, so "it passed" is as worth seeing as "it failed" — and
        # until now a pass was silent, which made a working grounding check
        # indistinguishable from one that never ran.
        if validation.ok:
            cited = len(grounded.citations)
            verdict = (
                f"Grounding passed · {cited} citation{'' if cited == 1 else 's'} verified"
                if cited
                else "Grounding passed · answer from tool data, nothing to cite"
            )
            async for event in stream_status("grounding", verdict):
                yield event
        else:
            async for event in stream_status(
                "error",
                f"Grounding FAILED (attempt {attempt}/{MAX_VALIDATION_ATTEMPTS}): "
                f"{validation.error}",
            ):
                yield event

        if validation.ok or attempt == MAX_VALIDATION_ATTEMPTS:
            break

        async for event in stream_status(
            "retrying",
            "Could not fully verify citations; retrying with stricter grounding…",
        ):
            yield event

    if grounded is None or validation is None:
        async for event in stream_error("Assistant run failed before producing an answer."):
            yield event
        return

    if validation.ok:
        async for event in stream_status("streaming", "Preparing answer…"):
            yield event

    async for event in stream_grounded_turn_and_persist(
        client=client,
        thread_id=thread_id,
        user_message=user_message,
        thread_title=thread_title,
        answer=grounded,
        registry=registry,
        validation=validation,
    ):
        yield event
