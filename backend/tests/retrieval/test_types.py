from __future__ import annotations

from datetime import date
from uuid import UUID

from app.retrieval.types import RetrievedPassage, format_passages_for_agent

CHUNK_ID = UUID("00000000-0000-0000-0000-000000000001")


def _passage(**overrides: object) -> RetrievedPassage:
    defaults = {
        "chunk_id": CHUNK_ID,
        "document_id": UUID("00000000-0000-0000-0000-000000000100"),
        "chunk_index": 3,
        "text": "iPhone and Services revenue increased year over year.",
        "page": "24",
        "section": "Item 7",
        "fusion_score": 0.032,
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "form": "10-K",
        "filing_date": date(2024, 10, 31),
        "fiscal_year": 2024,
        "accession_number": "0000320193-24-000123",
        "neighbors": [],
    }
    defaults.update(overrides)
    return RetrievedPassage(**defaults)  # type: ignore[arg-type]


def test_format_passages_for_agent_includes_metadata_and_chunk_id() -> None:
    output = format_passages_for_agent([_passage()])
    assert "AAPL 10-K FY2024 p.24 (Item 7)" in output
    assert str(CHUNK_ID) in output
    assert "iPhone and Services revenue" in output


def test_format_passages_for_agent_empty() -> None:
    assert format_passages_for_agent([]) == "No matching passages found in the filing corpus."
