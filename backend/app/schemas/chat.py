"""Pydantic models for chat API request and response bodies."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class CitationPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    citation_index: int = Field(alias="citationIndex")
    chunk_id: uuid.UUID = Field(alias="chunkId")
    excerpt: str
    ticker: str
    company_name: str | None = Field(default=None, alias="companyName")
    form: str
    filing_date: date = Field(alias="filingDate")
    page: str | None = None
    section: str | None = None


class CitationPart(BaseModel):
    type: Literal["data-citation"] = "data-citation"
    id: str | None = None
    data: CitationPayload


class CitationContextChunk(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    chunk_id: uuid.UUID = Field(alias="chunkId")
    chunk_index: int = Field(alias="chunkIndex")
    role: Literal["previous", "anchor", "next"]
    text: str
    page: str | None = None
    section: str | None = None


class CitationContextTable(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    table_index: int = Field(alias="tableIndex")
    title: str | None = None
    units: str | None = None
    markdown: str
    table_data: dict[str, Any] = Field(alias="tableData")


class CitationContextResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    anchor_chunk_id: uuid.UUID = Field(alias="anchorChunkId")
    document_id: uuid.UUID = Field(alias="documentId")
    ticker: str
    company_name: str | None = Field(default=None, alias="companyName")
    form: str
    filing_date: date = Field(alias="filingDate")
    source_url: str = Field(alias="sourceUrl")
    chunks: list[CitationContextChunk]
    table: CitationContextTable | None = None


class StatusPayload(BaseModel):
    stage: str
    message: str


class StatusPart(BaseModel):
    type: Literal["data-status"] = "data-status"
    data: StatusPayload


# StatusPart is emitted by the backend during streaming, kept by the client in
# its message list, and echoed back on the next turn. Leaving it out of this
# union made every request after the first fail with union_tag_invalid.
MessagePart = Annotated[
    TextPart | CitationPart | StatusPart, Field(discriminator="type")
]


class UIMessage(BaseModel):
    id: str | None = None
    role: Literal["user", "assistant", "system"]
    parts: list[MessagePart]


class CreateThreadRequest(BaseModel):
    title: str | None = None


class ThreadResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: uuid.UUID
    title: str
    created_at: datetime = Field(serialization_alias="createdAt")
    updated_at: datetime = Field(serialization_alias="updatedAt")


class ThreadListResponse(BaseModel):
    threads: list[ThreadResponse]


class MessageHistoryResponse(BaseModel):
    messages: list[UIMessage]


class StreamRequest(BaseModel):
    # Which model answers this question. Unknown or missing values fall back to
    # the default rather than erroring, so a stale browser tab still works.
    model: str | None = None
    model_config = ConfigDict(populate_by_name=True)

    thread_id: uuid.UUID = Field(validation_alias="threadId")
    messages: list[UIMessage]


def thread_row_to_response(row: dict[str, Any]) -> ThreadResponse:
    return ThreadResponse(
        id=uuid.UUID(str(row["id"])),
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
