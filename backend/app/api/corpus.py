"""## Browsing the corpus directly

The agent's answers are only as good as what was indexed, and until now the only
way to see what that was is to query Postgres by hand. These endpoints back the
Documents panel: pick a ticker, see its filings, read the actual chunks the
retriever searches over.

Worth having for the same reason the Tools panel shows real source: when an
answer looks wrong, the first question is what the model could possibly have
found, and that should be inspectable rather than inferred.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import func, select

from app.auth.dependencies import CurrentUser, get_current_user
from app.database.models import DocumentChunk, SourceDocument
from app.database.session import get_session

router = APIRouter(prefix="/corpus", tags=["corpus"])


class CompanySummary(BaseModel):
    ticker: str
    company_name: str | None
    filings: int
    chunks: int


class DocumentSummary(BaseModel):
    id: uuid.UUID
    ticker: str
    company_name: str | None
    form: str
    fiscal_year: int | None
    filing_date: str | None
    accession_number: str
    chunks: int


class ChunkView(BaseModel):
    id: uuid.UUID
    chunk_index: int
    page: str | None
    section: str | None
    token_count: int | None
    kind: str | None
    text: str


@router.get("/companies", response_model=list[CompanySummary])
async def list_companies(
    _: CurrentUser = Depends(get_current_user),
) -> list[CompanySummary]:
    """Every company in the corpus, with how much of it is indexed."""
    with get_session() as session:
        rows = session.execute(
            select(
                SourceDocument.ticker,
                func.min(SourceDocument.company_name),
                func.count(func.distinct(SourceDocument.id)),
                func.count(DocumentChunk.id),
            )
            .join(DocumentChunk, DocumentChunk.document_id == SourceDocument.id, isouter=True)
            .group_by(SourceDocument.ticker)
            .order_by(SourceDocument.ticker)
        ).all()
    return [
        CompanySummary(ticker=t, company_name=n, filings=f, chunks=c)
        for t, n, f, c in rows
    ]


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents(
    ticker: str | None = Query(default=None),
    _: CurrentUser = Depends(get_current_user),
) -> list[DocumentSummary]:
    """Filings, newest first. Optionally narrowed to one ticker."""
    with get_session() as session:
        stmt = (
            select(SourceDocument, func.count(DocumentChunk.id))
            .join(DocumentChunk, DocumentChunk.document_id == SourceDocument.id, isouter=True)
            .group_by(SourceDocument.id)
            .order_by(SourceDocument.ticker, SourceDocument.filing_date.desc())
        )
        if ticker:
            stmt = stmt.where(SourceDocument.ticker == ticker.upper())
        rows = session.execute(stmt).all()
        return [
            DocumentSummary(
                id=doc.id,
                ticker=doc.ticker,
                company_name=doc.company_name,
                form=doc.form,
                fiscal_year=doc.fiscal_year,
                filing_date=doc.filing_date.isoformat() if doc.filing_date else None,
                accession_number=doc.accession_number,
                chunks=count,
            )
            for doc, count in rows
        ]


@router.get("/documents/{document_id}/chunks", response_model=list[ChunkView])
async def list_chunks(
    document_id: uuid.UUID,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    _: CurrentUser = Depends(get_current_user),
) -> list[ChunkView]:
    """Chunks in document order.

    Paged because a single filing can hold thousands: JPMorgan's runs to about
    4,600, which is why this is not a "load everything" endpoint.
    """
    with get_session() as session:
        exists = session.get(SourceDocument, document_id)
        if exists is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No such document")
        rows = session.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
            .offset(offset)
            .limit(limit)
        ).all()
    return [
        ChunkView(
            id=c.id,
            chunk_index=c.chunk_index,
            page=c.page,
            section=c.section,
            token_count=c.token_count,
            kind=(c.chunk_metadata or {}).get("chunk_kind"),
            text=c.text,
        )
        for c in rows
    ]
