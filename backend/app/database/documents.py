"""Chunk and source-document lookups for retrieval and agent tools."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.database.models import DocumentChunk, SourceDocument


def get_chunks_by_ids(
    session: Session,
    chunk_ids: list[UUID],
) -> dict[UUID, DocumentChunk]:
    if not chunk_ids:
        return {}

    rows = session.scalars(
        select(DocumentChunk)
        .options(joinedload(DocumentChunk.document))
        .where(DocumentChunk.id.in_(chunk_ids))
    ).all()
    return {row.id: row for row in rows}


def get_chunk_with_document(
    session: Session,
    chunk_id: UUID,
) -> tuple[DocumentChunk, SourceDocument] | None:
    chunk = session.scalar(
        select(DocumentChunk)
        .options(joinedload(DocumentChunk.document))
        .where(DocumentChunk.id == chunk_id)
    )
    if chunk is None or chunk.document is None:
        return None
    return chunk, chunk.document


def get_surrounding_chunks(
    session: Session,
    chunk_id: UUID,
    radius: int,
) -> list[DocumentChunk]:
    if radius < 1:
        return []

    anchor = session.scalar(
        select(DocumentChunk).where(DocumentChunk.id == chunk_id)
    )
    if anchor is None:
        return []

    min_index = anchor.chunk_index - radius
    max_index = anchor.chunk_index + radius
    return list(
        session.scalars(
            select(DocumentChunk)
            .where(
                DocumentChunk.document_id == anchor.document_id,
                DocumentChunk.chunk_index >= min_index,
                DocumentChunk.chunk_index <= max_index,
                DocumentChunk.id != chunk_id,
            )
            .order_by(DocumentChunk.chunk_index)
        ).all()
    )


def get_chunk_context(
    session: Session,
    chunk_id: UUID,
    radius: int,
) -> list[DocumentChunk] | None:
    anchor = session.scalar(
        select(DocumentChunk)
        .options(joinedload(DocumentChunk.document))
        .where(DocumentChunk.id == chunk_id)
    )
    if anchor is None:
        return None

    min_index = anchor.chunk_index - radius
    max_index = anchor.chunk_index + radius
    return list(
        session.scalars(
            select(DocumentChunk)
            .options(joinedload(DocumentChunk.document))
            .where(
                DocumentChunk.document_id == anchor.document_id,
                DocumentChunk.chunk_index >= min_index,
                DocumentChunk.chunk_index <= max_index,
            )
            .order_by(DocumentChunk.chunk_index)
        ).all()
    )
