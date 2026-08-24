"""Chat thread and message persistence via Supabase."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException, status
from supabase import AsyncClient

from app.auth.dependencies import CurrentUser
from app.turn.messages import (
    DEFAULT_THREAD_TITLE,
    row_to_ui_message,
    title_from_user_message,
    ui_message_to_insert,
)
from app.database.supabase import get_service_role_client
from app.schemas.chat import CitationPart, CitationPayload, ThreadResponse, UIMessage, thread_row_to_response


@dataclass(frozen=True, slots=True)
class ThreadRow:
    id: uuid.UUID
    user_id: uuid.UUID
    title: str


async def require_thread_access(thread_id: uuid.UUID, user: CurrentUser) -> ThreadRow:
    client = await get_service_role_client()
    response = await (
        client.table("chat_threads")
        .select("id,user_id,title")
        .eq("id", str(thread_id))
        .maybe_single()
        .execute()
    )

    if response.data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found",
        )

    row = response.data
    owner_id = uuid.UUID(str(row["user_id"]))
    if owner_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden",
        )

    return ThreadRow(
        id=uuid.UUID(str(row["id"])),
        user_id=owner_id,
        title=row["title"],
    )


async def list_threads(client: AsyncClient, user: CurrentUser) -> list[ThreadResponse]:
    response = await (
        client.table("chat_threads")
        .select("id,title,created_at,updated_at")
        .eq("user_id", str(user.id))
        .order("updated_at", desc=True)
        .execute()
    )
    return [thread_row_to_response(row) for row in response.data]


async def create_thread(
    client: AsyncClient,
    user: CurrentUser,
    *,
    title: str | None = None,
) -> ThreadResponse:
    thread_id = uuid.uuid4()
    response = await (
        client.table("chat_threads")
        .insert(
            {
                "id": str(thread_id),
                "user_id": str(user.id),
                "title": title or DEFAULT_THREAD_TITLE,
            }
        )
        .select("id,title,created_at,updated_at")
        .execute()
    )
    return thread_row_to_response(response.data[0])


async def delete_thread(client: AsyncClient, thread_id: uuid.UUID) -> None:
    await client.table("chat_threads").delete().eq("id", str(thread_id)).execute()


def _citation_rows_from_message(
    assistant_message: UIMessage,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    message_id = assistant_message.id
    if message_id is None:
        return rows

    for part in assistant_message.parts:
        if not isinstance(part, CitationPart):
            continue
        data: CitationPayload = part.data
        rows.append(
            {
                "id": str(uuid.uuid4()),
                "message_id": message_id,
                "chunk_id": str(data.chunk_id),
                "citation_index": data.citation_index,
                "excerpt": data.excerpt,
                "ticker": data.ticker,
                "company_name": data.company_name,
                "form": data.form,
                "filing_date": data.filing_date.isoformat(),
                "page": data.page,
                "section": data.section,
            }
        )
    return rows


async def load_messages(client: AsyncClient, thread_id: uuid.UUID) -> list[UIMessage]:
    response = await (
        client.table("chat_messages")
        .select("id,role,content,parts,sequence")
        .eq("thread_id", str(thread_id))
        .order("sequence")
        .execute()
    )
    messages = [row_to_ui_message(row) for row in response.data]
    assistant_ids = [message.id for message in messages if message.role == "assistant" and message.id]
    if not assistant_ids:
        return messages

    citations_response = await (
        client.table("message_citations")
        .select(
            "message_id,citation_index,excerpt,chunk_id,ticker,company_name,form,filing_date,page,section"
        )
        .in_("message_id", assistant_ids)
        .order("citation_index")
        .execute()
    )
    citations_by_message: dict[str, list[dict[str, Any]]] = {}
    for row in citations_response.data:
        message_id = str(row["message_id"])
        citations_by_message.setdefault(message_id, []).append(row)

    hydrated: list[UIMessage] = []
    for message in messages:
        if message.role != "assistant" or message.id is None:
            hydrated.append(message)
            continue

        citation_rows = citations_by_message.get(message.id, [])
        if not citation_rows:
            hydrated.append(message)
            continue

        existing_citation_ids = {
            part.data.chunk_id
            for part in message.parts
            if isinstance(part, CitationPart)
        }
        parts = list(message.parts)
        for row in citation_rows:
            chunk_id = uuid.UUID(str(row["chunk_id"]))
            if chunk_id in existing_citation_ids:
                continue
            parts.append(
                CitationPart(
                    id=str(chunk_id),
                    data=CitationPayload(
                        citation_index=int(row["citation_index"]),
                        chunk_id=chunk_id,
                        excerpt=row["excerpt"],
                        ticker=row["ticker"],
                        company_name=row.get("company_name"),
                        form=row["form"],
                        filing_date=row["filing_date"],
                        page=row.get("page"),
                        section=row.get("section"),
                    ),
                )
            )
        hydrated.append(UIMessage(id=message.id, role=message.role, parts=parts))

    return hydrated


async def get_next_sequence(client: AsyncClient, thread_id: uuid.UUID) -> int:
    response = await (
        client.table("chat_messages")
        .select("sequence")
        .eq("thread_id", str(thread_id))
        .order("sequence", desc=True)
        .limit(1)
        .execute()
    )
    if not response.data:
        return 0
    return int(response.data[0]["sequence"]) + 1


async def append_grounded_turn(
    client: AsyncClient,
    *,
    thread_id: uuid.UUID,
    user_message: UIMessage,
    assistant_message: UIMessage,
    thread_title: str,
) -> None:
    next_sequence = await get_next_sequence(client, thread_id)
    rows = [
        ui_message_to_insert(
            user_message,
            thread_id=thread_id,
            sequence=next_sequence,
        ),
        ui_message_to_insert(
            assistant_message,
            thread_id=thread_id,
            sequence=next_sequence + 1,
            message_id=uuid.UUID(assistant_message.id) if assistant_message.id else None,
        ),
    ]
    await client.table("chat_messages").insert(rows).execute()

    citation_rows = _citation_rows_from_message(assistant_message)
    if citation_rows:
        await client.table("message_citations").insert(citation_rows).execute()

    updates: dict[str, Any] = {"updated_at": datetime.now(UTC).isoformat()}
    if thread_title == DEFAULT_THREAD_TITLE:
        updates["title"] = title_from_user_message(user_message)

    await (
        client.table("chat_threads")
        .update(updates)
        .eq("id", str(thread_id))
        .execute()
    )
