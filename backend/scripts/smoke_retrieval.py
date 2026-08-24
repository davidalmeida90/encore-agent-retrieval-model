"""Print top retrieval hits for client-brief-style questions."""

from __future__ import annotations

from app.retrieval.retriever import DocumentRetriever
from app.retrieval.types import SearchFilters, format_passages_for_agent

SMOKE_QUERIES: list[tuple[str, SearchFilters | None]] = [
    (
        "Across Apple's 10-Ks, how did the revenue mix between iPhone, Services, Mac, iPad, and Wearables change?",
        SearchFilters(ticker="AAPL", form="10-K"),
    ),
    (
        "How did NVIDIA describe demand drivers and customer concentration for its Data Center business?",
        SearchFilters(ticker="NVDA", form="10-K"),
    ),
    (
        "What changed in the way Microsoft describes Azure, AI infrastructure, and cloud capacity constraints?",
        SearchFilters(ticker="MSFT", form="10-K"),
    ),
]


def main() -> None:
    retriever = DocumentRetriever()
    for query, filters in SMOKE_QUERIES:
        print("\n" + "=" * 80)
        print(f"Query: {query}")
        if filters is not None:
            print(f"Filters: {filters.model_dump_json()}")
        passages = retriever.search(query, filters=filters, top_k=5)
        print(format_passages_for_agent(passages))


if __name__ == "__main__":
    main()
