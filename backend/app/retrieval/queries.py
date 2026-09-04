"""pgvector semantic search and Postgres full-text search over document_chunks."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.retrieval.types import RankedChunkHit, SearchFilters


@dataclass(frozen=True, slots=True)
class _FilterClause:
    sql: str
    params: dict[str, object]


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(v) for v in values) + "]"


def _build_filters(filters: SearchFilters | None) -> _FilterClause:
    if filters is None:
        return _FilterClause("", {})

    clauses: list[str] = []
    params: dict[str, object] = {}

    if filters.ticker is not None:
        clauses.append("sd.ticker = :ticker")
        params["ticker"] = filters.ticker
    if filters.fiscal_years:
        clauses.append("sd.fiscal_year = ANY(:fiscal_years)")
        params["fiscal_years"] = filters.fiscal_years
    if filters.form is not None:
        clauses.append("sd.form = :form")
        params["form"] = filters.form

    if not clauses:
        return _FilterClause("", {})

    return _FilterClause(" AND " + " AND ".join(clauses), params)


def _rows_to_hits(rows: list) -> list[RankedChunkHit]:
    return [
        RankedChunkHit(
            chunk_id=UUID(str(row.id)),
            rank=index,
            score=float(row.score) if row.score is not None else None,
        )
        for index, row in enumerate(rows, start=1)
    ]


def semantic_search(
    session: Session,
    query_vec: list[float],
    *,
    limit: int,
    filters: SearchFilters | None = None,
) -> list[RankedChunkHit]:
    filter_clause = _build_filters(filters)
    sql = f"""
        SELECT dc.id,
               1 - (dc.embedding <=> CAST(:query_vec AS vector)) AS score
        FROM document_chunks dc
        JOIN source_documents sd ON sd.id = dc.document_id
        WHERE dc.embedding IS NOT NULL
        {filter_clause.sql}
        ORDER BY dc.embedding <=> CAST(:query_vec AS vector)
        LIMIT :limit
    """
    params: dict[str, object] = {
        "query_vec": _vector_literal(query_vec),
        "limit": limit,
        **filter_clause.params,
    }
    rows = session.execute(text(sql), params).all()
    return _rows_to_hits(rows)


def full_text_candidates(
    session: Session,
    query_text: str,
    *,
    limit: int,
    filters: SearchFilters | None = None,
) -> list[tuple[UUID, str]]:
    """FTS hits WITH their stored tsvector, in a single round trip.

    BM25 needs a term frequency and a document length per candidate, both of
    which are already inside `search_vector`. Fetching them separately cost a
    second query to a remote Postgres: measured at 321 ms of BM25's 387 ms,
    against 1 ms for the arithmetic itself. The database was never the slow
    part; the round trip was.

    Returns (id, tsvector-as-text) so the caller can score without another
    query. See retrieval/bm25.py for the parsing.
    """
    fts_config = settings.retrieval_fts_config
    filter_clause = _build_filters(filters)
    sql = f"""
        SELECT dc.id, dc.search_vector::text
        FROM document_chunks dc
        JOIN source_documents sd ON sd.id = dc.document_id,
             to_tsquery('{fts_config}',
                        replace(plainto_tsquery('{fts_config}', :query_text)::text,
                                ' & ', ' | ')) query
        WHERE dc.search_vector @@ query
        {filter_clause.sql}
        ORDER BY ts_rank_cd(dc.search_vector, query) DESC
        LIMIT :limit
    """
    rows = session.execute(
        text(sql),
        {"query_text": query_text, "limit": limit, **filter_clause.params},
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def full_text_search(
    session: Session,
    query_text: str,
    *,
    limit: int,
    filters: SearchFilters | None = None,
) -> list[RankedChunkHit]:
    # plainto_tsquery joins every term with AND, which for a natural-language
    # question means the chunk must contain ALL of them - including "describe",
    # "10" and "k". Measured on this corpus, "How does Microsoft describe Azure's
    # competitive advantage in its 10-K?" matched 0 chunks of 27,222, and a
    # retrieval eval put full-text recall at 2% against 46% for the vector leg,
    # with ZERO chunks found by full text that the vector leg missed. The lexical
    # half of a hybrid retriever was contributing nothing at all.
    #
    # Swapping the operator to OR gives ts_rank_cd something to rank: the same
    # query then matches 5,351 chunks, and cover-density scoring rewards the ones
    # carrying more of the rare terms. The cast is safe because plainto_tsquery
    # has already parsed, stemmed and escaped the input; only the operator
    # between the lexemes changes.
    fts_config = settings.retrieval_fts_config
    filter_clause = _build_filters(filters)
    sql = f"""
        SELECT dc.id,
               ts_rank_cd(dc.search_vector, query) AS score
        FROM document_chunks dc
        JOIN source_documents sd ON sd.id = dc.document_id,
             to_tsquery('{fts_config}',
                        replace(plainto_tsquery('{fts_config}', :query_text)::text,
                                ' & ', ' | ')) query
        WHERE dc.search_vector @@ query
        {filter_clause.sql}
        ORDER BY score DESC
        LIMIT :limit
    """
    params: dict[str, object] = {
        "query_text": query_text,
        "limit": limit,
        **filter_clause.params,
    }
    rows = session.execute(text(sql), params).all()
    return _rows_to_hits(rows)


def build_semantic_search_sql(filters: SearchFilters | None = None) -> str:
    """Expose SQL shape for unit tests."""
    return _semantic_sql_template(_build_filters(filters))


def build_full_text_search_sql(filters: SearchFilters | None = None) -> str:
    """Expose SQL shape for unit tests."""
    fts_config = settings.retrieval_fts_config
    return _fts_sql_template(fts_config, _build_filters(filters))


def _semantic_sql_template(filter_clause: _FilterClause) -> str:
    return f"""
        SELECT dc.id,
               1 - (dc.embedding <=> CAST(:query_vec AS vector)) AS score
        FROM document_chunks dc
        JOIN source_documents sd ON sd.id = dc.document_id
        WHERE dc.embedding IS NOT NULL
        {filter_clause.sql}
        ORDER BY dc.embedding <=> CAST(:query_vec AS vector)
        LIMIT :limit
    """


def _fts_sql_template(fts_config: str, filter_clause: _FilterClause) -> str:
    return f"""
        SELECT dc.id,
               ts_rank_cd(dc.search_vector, query) AS score
        FROM document_chunks dc
        JOIN source_documents sd ON sd.id = dc.document_id,
             to_tsquery('{fts_config}',
                        replace(plainto_tsquery('{fts_config}', :query_text)::text,
                                ' & ', ' | ')) query
        WHERE dc.search_vector @@ query
        {filter_clause.sql}
        ORDER BY score DESC
        LIMIT :limit
    """
