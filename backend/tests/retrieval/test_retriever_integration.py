from __future__ import annotations

import pytest

from app.retrieval.retriever import DocumentRetriever
from app.retrieval.types import SearchFilters, format_passages_for_agent

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("query", "filters", "expected_ticker"),
    [
        (
            "Apple iPhone Services revenue mix",
            SearchFilters(ticker="AAPL"),
            "AAPL",
        ),
        (
            "NVIDIA Data Center demand drivers customer concentration",
            SearchFilters(ticker="NVDA"),
            "NVDA",
        ),
    ],
)
def test_retriever_returns_relevant_company_passages(
    query: str,
    filters: SearchFilters,
    expected_ticker: str,
) -> None:
    retriever = DocumentRetriever()
    passages = retriever.search(query, filters=filters, top_k=5)

    assert passages, f"No passages returned for query: {query!r}"
    assert all(p.ticker == expected_ticker for p in passages)

    combined = " ".join(p.text.lower() for p in passages)
    if expected_ticker == "AAPL":
        assert any(term in combined for term in ("iphone", "services", "revenue"))
    if expected_ticker == "NVDA":
        assert any(term in combined for term in ("data center", "datacenter", "demand"))

    formatted = format_passages_for_agent(passages)
    assert expected_ticker in formatted
    # chunk ids are rendered as chunk_id=<uuid>, never in square brackets:
    # square brackets are reserved for [n] citation markers in the answer, and
    # showing "[uuid]" here taught the model to emit ids the validator rejects
    assert "chunk_id=" in formatted
    assert "[" not in formatted
