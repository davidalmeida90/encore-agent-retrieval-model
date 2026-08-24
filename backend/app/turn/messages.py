"""Convert between AI SDK UI messages and chat_messages rows."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import HTTPException, status

from app.agent.deps import TurnRegistry
from app.agent.outputs import GroundedAnswer
from app.database.models.message_role import MessageRole
from app.schemas.chat import (
    CitationPart,
    CitationPayload,
    MessagePart,
    TextPart,
    UIMessage,
)

DEFAULT_THREAD_TITLE = "New chat"
MAX_TITLE_LENGTH = 255


def text_from_parts(parts: list[MessagePart]) -> str:
    return "".join(part.text for part in parts if isinstance(part, TextPart))


def extract_last_user_message(messages: list[UIMessage]) -> UIMessage:
    for message in reversed(messages):
        if message.role == "user":
            return message
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail="Request must include at least one user message",
    )


def ui_message_to_insert(
    message: UIMessage,
    *,
    thread_id: uuid.UUID,
    sequence: int,
    message_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    parts = [part.model_dump(by_alias=True, mode="json") for part in message.parts]
    return {
        "id": str(message_id or uuid.uuid4()),
        "thread_id": str(thread_id),
        "role": MessageRole(message.role).value,
        "content": text_from_parts(message.parts) or None,
        "parts": parts,
        "sequence": sequence,
    }


def _parse_part(raw: dict[str, Any]) -> MessagePart:
    part_type = raw.get("type")
    if part_type == "text":
        return TextPart.model_validate(raw)
    if part_type == "data-citation":
        return CitationPart.model_validate(raw)
    raise ValueError(f"Unsupported message part type: {part_type!r}")


def row_to_ui_message(row: dict[str, Any]) -> UIMessage:
    raw_parts = row.get("parts") or []
    parts: list[MessagePart] = []
    for part in raw_parts:
        parts.append(_parse_part(part))
    if not parts and row.get("content"):
        parts = [TextPart(text=row["content"])]

    return UIMessage(
        id=str(row["id"]),
        role=row["role"],
        parts=parts,
    )


def citation_parts_from_grounded_answer(
    answer: GroundedAnswer,
    registry: TurnRegistry,
) -> list[CitationPart]:
    parts: list[CitationPart] = []
    for citation in answer.citations:
        passage = registry.passages_by_chunk_id[citation.chunk_id]
        parts.append(
            CitationPart(
                id=str(citation.chunk_id),
                data=CitationPayload(
                    citation_index=citation.citation_index,
                    chunk_id=citation.chunk_id,
                    excerpt=citation.excerpt,
                    ticker=passage.ticker,
                    company_name=passage.company_name,
                    form=passage.form,
                    filing_date=passage.filing_date,
                    page=passage.page,
                    section=passage.section,
                ),
            )
        )
    return parts


def build_assistant_message(
    answer: GroundedAnswer,
    registry: TurnRegistry,
    *,
    message_id: uuid.UUID | None = None,
) -> UIMessage:
    parts: list[MessagePart] = [TextPart(text=answer.answer)]
    parts.extend(citation_parts_from_grounded_answer(answer, registry))
    return UIMessage(
        id=str(message_id or uuid.uuid4()),
        role="assistant",
        parts=parts,
    )


def title_from_user_message(message: UIMessage) -> str:
    text = text_from_parts(message.parts).strip()
    if not text:
        return DEFAULT_THREAD_TITLE
    if len(text) <= MAX_TITLE_LENGTH:
        return text
    return text[: MAX_TITLE_LENGTH - 3] + "..."
