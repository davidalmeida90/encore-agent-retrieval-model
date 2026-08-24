"""AI SDK-compatible SSE streaming for grounded assistant replies."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

from supabase import AsyncClient

from app.agent.deps import TurnRegistry
from app.agent.outputs import GroundedAnswer
from app.turn.messages import build_assistant_message
from app.database.chats import append_grounded_turn
from app.verification.validator import ValidationResult
from app.schemas.chat import CitationPart, StatusPart, StatusPayload, UIMessage

GROUNDING_FAILURE_MESSAGE = (
    "I found relevant source passages, but I could not fully verify the answer "
    "against them. Try asking a narrower question or breaking it into smaller parts."
)


def _sse_event(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'), default=str)}\n\n"


async def stream_status(stage: str, message: str) -> AsyncIterator[str]:
    part = StatusPart(data=StatusPayload(stage=stage, message=message))
    yield _sse_event(part.model_dump(by_alias=True, mode="json"))


async def _text_events(
    text: str,
    *,
    message_id: str,
) -> AsyncIterator[str]:
    yield _sse_event({"type": "text-start", "id": message_id})

    for word in text.split(" "):
        yield _sse_event({"type": "text-delta", "id": message_id, "delta": f"{word} "})

    yield _sse_event({"type": "text-end", "id": message_id})


async def _citation_events(citation_parts: list[CitationPart]) -> AsyncIterator[str]:
    for part in citation_parts:
        payload = part.model_dump(by_alias=True, mode="json")
        yield _sse_event(payload)


async def stream_grounded_answer(
    answer: GroundedAnswer,
    registry: TurnRegistry,
    *,
    message_id: str,
) -> AsyncIterator[str]:
    yield _sse_event({"type": "start", "messageId": message_id})

    async for event in _text_events(answer.answer, message_id=message_id):
        yield event

    assistant_message = build_assistant_message(answer, registry, message_id=uuid.UUID(message_id))
    citation_parts = [
        part for part in assistant_message.parts if isinstance(part, CitationPart)
    ]
    async for event in _citation_events(citation_parts):
        yield event

    yield _sse_event({"type": "finish"})


async def stream_error(error_text: str) -> AsyncIterator[str]:
    yield _sse_event({"type": "error", "errorText": error_text})


async def stream_grounded_turn_and_persist(
    *,
    client: AsyncClient,
    thread_id: uuid.UUID,
    user_message: UIMessage,
    thread_title: str,
    answer: GroundedAnswer,
    registry: TurnRegistry,
    validation: ValidationResult,
) -> AsyncIterator[str]:
    if not validation.ok:
        async for event in stream_error(GROUNDING_FAILURE_MESSAGE):
            yield event
        return

    message_id = str(uuid.uuid4())
    assistant_message = build_assistant_message(
        answer,
        registry,
        message_id=uuid.UUID(message_id),
    )

    try:
        async for event in stream_grounded_answer(
            answer,
            registry,
            message_id=message_id,
        ):
            yield event
    finally:
        await append_grounded_turn(
            client,
            thread_id=thread_id,
            user_message=user_message,
            assistant_message=assistant_message,
            thread_title=thread_title,
        )
