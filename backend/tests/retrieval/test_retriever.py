from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch
from uuid import UUID

from app.database.models import DocumentChunk, SourceDocument
from app.retrieval.retriever import DocumentRetriever
from app.retrieval.types import RankedChunkHit, SearchFilters

ID_A = UUID("00000000-0000-0000-0000-000000000001")
ID_B = UUID("00000000-0000-0000-0000-000000000002")
ID_N1 = UUID("00000000-0000-0000-0000-000000000011")
DOC_ID = UUID("00000000-0000-0000-0000-000000000100")


def _document() -> SourceDocument:
    return SourceDocument(
        id=DOC_ID,
        ticker="AAPL",
        cik="0000320193",
        company_name="Apple Inc.",
        form="10-K",
        filing_date=date(2024, 10, 31),
        fiscal_year=2024,
        accession_number="0000320193-24-000123",
        primary_document="aapl-10k.htm",
        source_url="https://example.com/aapl",
    )


def _chunk(
    chunk_id: UUID,
    *,
    chunk_index: int,
    text: str,
    document: SourceDocument,
) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id=document.id,
        chunk_index=chunk_index,
        page="42",
        section="Item 1",
        text=text,
        chunk_metadata={},
        document=document,
    )


@patch("app.retrieval.retriever.get_surrounding_chunks")
@patch("app.retrieval.retriever.get_chunks_by_ids")
@patch("app.retrieval.retriever.full_text_search")
@patch("app.retrieval.retriever.semantic_search")
@patch("app.retrieval.retriever.extract_fts_keywords")
@patch("app.retrieval.retriever.embed_query")
def test_document_retriever_fuses_and_hydrates(
    mock_embed_query: MagicMock,
    mock_extract_fts_keywords: MagicMock,
    mock_semantic_search: MagicMock,
    mock_full_text_search: MagicMock,
    mock_get_chunks_by_ids: MagicMock,
    mock_get_surrounding_chunks: MagicMock,
) -> None:
    mock_embed_query.return_value = [0.1] * 3
    mock_extract_fts_keywords.return_value = "iPhone Services revenue"
    mock_semantic_search.return_value = [
        RankedChunkHit(chunk_id=ID_A, rank=1, score=0.9),
        RankedChunkHit(chunk_id=ID_B, rank=2, score=0.8),
    ]
    mock_full_text_search.return_value = [
        RankedChunkHit(chunk_id=ID_B, rank=1, score=0.7),
        RankedChunkHit(chunk_id=ID_A, rank=2, score=0.6),
    ]

    document = _document()
    chunk_a = _chunk(ID_A, chunk_index=5, text="iPhone revenue mix", document=document)
    chunk_b = _chunk(ID_B, chunk_index=8, text="Services growth", document=document)
    neighbor = _chunk(ID_N1, chunk_index=4, text="prior context", document=document)

    mock_get_chunks_by_ids.return_value = {ID_A: chunk_a, ID_B: chunk_b}
    mock_get_surrounding_chunks.return_value = [neighbor]

    session = MagicMock()
    retriever = DocumentRetriever()
    passages = retriever.search(
        "Apple iPhone Services revenue mix",
        filters=SearchFilters(ticker="AAPL"),
        top_k=2,
        candidate_k=50,
        session=session,
    )

    assert len(passages) == 2
    assert {p.chunk_id for p in passages} == {ID_A, ID_B}
    assert passages[0].fusion_score >= passages[1].fusion_score
    assert passages[0].ticker == "AAPL"
    assert passages[0].neighbors
    assert passages[0].neighbors[0].chunk_id == ID_N1
    assert passages[0].neighbors[0].fusion_score == 0.0

    mock_extract_fts_keywords.assert_called_once_with(
        "Apple iPhone Services revenue mix",
        filters=SearchFilters(ticker="AAPL"),
    )
    mock_semantic_search.assert_called_once()
    mock_full_text_search.assert_called_once()
    semantic_kwargs = mock_semantic_search.call_args.kwargs
    assert semantic_kwargs["filters"] == SearchFilters(ticker="AAPL")
    assert semantic_kwargs["limit"] == 50
    fts_args = mock_full_text_search.call_args.args
    assert fts_args[1] == "iPhone Services revenue"


@patch("app.retrieval.retriever.full_text_search")
@patch("app.retrieval.retriever.semantic_search")
@patch("app.retrieval.retriever.extract_fts_keywords")
@patch("app.retrieval.retriever.embed_query")
def test_document_retriever_returns_empty_when_no_hits(
    mock_embed_query: MagicMock,
    mock_extract_fts_keywords: MagicMock,
    mock_semantic_search: MagicMock,
    mock_full_text_search: MagicMock,
) -> None:
    mock_embed_query.return_value = [0.1] * 3
    mock_extract_fts_keywords.return_value = "nothing here"
    mock_semantic_search.return_value = []
    mock_full_text_search.return_value = []

    retriever = DocumentRetriever()
    passages = retriever.search("nothing here", session=MagicMock())

    assert passages == []
