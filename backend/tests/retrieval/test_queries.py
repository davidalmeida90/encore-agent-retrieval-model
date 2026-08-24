from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from app.retrieval.queries import (
    build_full_text_search_sql,
    build_semantic_search_sql,
    full_text_search,
    semantic_search,
)
from app.retrieval.types import SearchFilters

ID_A = UUID("00000000-0000-0000-0000-000000000001")
ID_B = UUID("00000000-0000-0000-0000-000000000002")


def test_build_semantic_search_sql_without_filters() -> None:
    sql = build_semantic_search_sql()
    assert "dc.embedding <=> CAST(:query_vec AS vector)" in sql
    assert "JOIN source_documents sd" in sql
    assert "sd.ticker" not in sql


def test_build_semantic_search_sql_with_filters() -> None:
    sql = build_semantic_search_sql(
        SearchFilters(ticker="AAPL", fiscal_years=[2021, 2022], form="10-K")
    )
    assert "sd.ticker = :ticker" in sql
    assert "sd.fiscal_year = ANY(:fiscal_years)" in sql
    assert "sd.form = :form" in sql


def test_build_full_text_search_sql_with_filters() -> None:
    sql = build_full_text_search_sql(SearchFilters(ticker="NVDA"))
    assert "plainto_tsquery('english', :query_text)" in sql
    assert "dc.search_vector @@ query" in sql
    assert "sd.ticker = :ticker" in sql


def test_semantic_search_maps_rows_to_hits() -> None:
    session = MagicMock()
    session.execute.return_value.all.return_value = [
        SimpleNamespace(id=ID_A, score=0.91),
        SimpleNamespace(id=ID_B, score=0.82),
    ]

    hits = semantic_search(session, [0.1, 0.2, 0.3], limit=2)

    assert len(hits) == 2
    assert hits[0].chunk_id == ID_A
    assert hits[0].rank == 1
    assert hits[0].score == pytest.approx(0.91)
    assert hits[1].chunk_id == ID_B
    assert hits[1].rank == 2


def test_full_text_search_maps_rows_to_hits() -> None:
    session = MagicMock()
    session.execute.return_value.all.return_value = [
        SimpleNamespace(id=ID_B, score=0.55),
    ]

    hits = full_text_search(session, "data center demand", limit=1)

    assert len(hits) == 1
    assert hits[0].chunk_id == ID_B
    assert hits[0].score == pytest.approx(0.55)
