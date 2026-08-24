from __future__ import annotations

from pathlib import Path

import pytest

from ingest.chunking import (
    CHUNK_MAX_TOKENS,
    build_hierarchical_chunker,
    build_hybrid_chunker,
    chunk_document,
    convert_html_to_document,
    map_chunk_record,
)

FILING_METADATA = {
    "ticker": "TEST",
    "cik": "0000000000",
    "company_name": "Test Corp",
    "form": "10-K",
    "filing_date": "2021-01-01",
    "report_date": "2020-12-31",
    "fiscal_year": 2020,
    "accession_number": "0000000000-21-000001",
    "primary_document": "test.htm",
    "source_url": "https://example.com/test.htm",
}


@pytest.fixture(scope="module")
def sample_html_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    fixture_dir = tmp_path_factory.mktemp("fixtures")
    html_path = fixture_dir / "sample_filing.htm"
    html_path.write_text(
        """
        <html><body>
        <h1>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</h1>
        <h2>Item 1. Business</h2>
        <p>We design and sell consumer electronics products worldwide.</p>
        <h2>Item 1A. Risk Factors</h2>
        <p>Our business depends on global supply chains and may be affected by shortages.</p>
        <p>Products and Services Performance</p>
        <p>The following table shows net sales by category for 2025, 2024 and 2023 (dollars in millions):</p>
        <table>
          <tr>
            <td colspan="3"></td><td colspan="3">2025</td><td colspan="3">Change</td>
            <td colspan="3">2024</td><td colspan="3">Change</td><td colspan="3">2023</td>
          </tr>
          <tr>
            <td colspan="3">iPhone</td><td>$</td><td>209,586</td><td></td><td>4</td><td>4</td><td>%</td>
            <td>$</td><td>201,183</td><td></td><td>&#8212;</td><td>&#8212;</td><td>%</td>
            <td>$</td><td>200,583</td><td></td>
          </tr>
          <tr>
            <td colspan="3">Services (1)</td><td></td><td>109,158</td><td></td><td>14</td><td>14</td><td>%</td>
            <td></td><td>96,169</td><td></td><td>13</td><td>13</td><td>%</td>
            <td></td><td>85,200</td><td></td>
          </tr>
        </table>
        <p>(1) Services net sales include amortization of deferred value.</p>
        <h2>Item 2. Properties</h2>
        <p>We operate offices, data centers, and retail stores worldwide after the table.</p>
        </body></html>
        """,
        encoding="utf-8",
    )
    return html_path


def test_hierarchical_chunker_produces_chunks(sample_html_path: Path) -> None:
    doc = convert_html_to_document(sample_html_path)
    chunker = build_hierarchical_chunker()
    chunks = list(chunker.chunk(dl_doc=doc))
    assert chunks
    combined = "\n".join(chunk.text for chunk in chunks)
    assert "consumer electronics" in combined.lower()


def test_hybrid_chunker_respects_token_limit(sample_html_path: Path) -> None:
    records = chunk_document(sample_html_path, FILING_METADATA)
    assert records
    assert all(record.token_count <= CHUNK_MAX_TOKENS for record in records)


def test_map_chunk_record_extracts_section(sample_html_path: Path) -> None:
    doc = convert_html_to_document(sample_html_path)
    chunker = build_hybrid_chunker()
    chunk = next(chunker.chunk(dl_doc=doc))
    record = map_chunk_record(
        chunk_index=0,
        chunk=chunk,
        chunker=chunker,
        filing_metadata=FILING_METADATA,
    )
    assert record.chunk_index == 0
    assert record.text
    assert record.token_count > 0
    assert record.chunk_metadata["ticker"] == "TEST"
    assert record.chunk_metadata["raw_text"]
    assert record.chunk_metadata["accession_number"] == FILING_METADATA["accession_number"]


def test_chunk_document_adds_semantic_table_row_chunks(sample_html_path: Path) -> None:
    records = chunk_document(sample_html_path, FILING_METADATA)

    table_records = [
        record
        for record in records
        if record.chunk_metadata.get("chunk_kind") == "table_row"
    ]

    assert [record.chunk_metadata["row_label"] for record in table_records] == [
        "iPhone",
        "Services (1)",
    ]
    assert table_records[0].section == "Products and Services Performance"
    assert "| iPhone | $209,586 | 4% | $201,183 | - | $200,583 |" in table_records[0].text
    assert "iPhone | iPhone | iPhone" not in "\n".join(record.text for record in records)
    assert table_records[0].chunk_metadata["table"]["title"] == "Products and Services Performance"
    assert table_records[-1].chunk_index < records[-1].chunk_index
