from __future__ import annotations

import uuid
from unittest.mock import MagicMock

from sqlalchemy import Text

from app.database.models import DocumentChunk, DocumentTable, MessageCitation
from ingest.chunk_and_embed import _delete_chunks, _document_tables_from_records
from ingest.chunking import ChunkRecord


def test_document_table_title_allows_full_sec_captions() -> None:
    assert isinstance(DocumentTable.__table__.c.title.type, Text)


def test_document_chunk_section_allows_full_sec_captions() -> None:
    assert isinstance(DocumentChunk.__table__.c.section.type, Text)


def test_message_citation_section_allows_full_sec_captions() -> None:
    assert isinstance(MessageCitation.__table__.c.section.type, Text)


def test_delete_chunks_removes_dependent_citations_before_chunks() -> None:
    session = MagicMock()
    document_id = uuid.uuid4()

    _delete_chunks(session, document_id)

    deleted_tables = [call.args[0].table.name for call in session.execute.call_args_list]
    assert deleted_tables == [
        "message_citations",
        "document_chunks",
        "document_tables",
    ]


def test_document_tables_from_records_deduplicates_table_metadata() -> None:
    document_id = uuid.uuid4()
    table = {
        "table_index": 2,
        "title": "Products and Services Performance",
        "units": "dollars in millions",
        "markdown": "| Category | 2025 Sales |\n| --- | --- |\n| iPhone | $209,586 |",
        "source_html_hash": "abc123",
        "rows": [],
        "columns": [],
        "footnotes": [],
    }
    records = [
        ChunkRecord(
            chunk_index=0,
            text="row one",
            page=None,
            section="Products and Services Performance",
            token_count=10,
            chunk_metadata={
                "chunk_kind": "table_row",
                "table_index": 2,
                "table": table,
            },
        ),
        ChunkRecord(
            chunk_index=1,
            text="row two",
            page=None,
            section="Products and Services Performance",
            token_count=10,
            chunk_metadata={
                "chunk_kind": "table_row",
                "table_index": 2,
                "table": table,
            },
        ),
    ]

    tables = _document_tables_from_records(document_id, records)

    assert len(tables) == 1
    assert tables[0].document_id == document_id
    assert tables[0].table_index == 2
    assert tables[0].title == "Products and Services Performance"
    assert tables[0].units == "dollars in millions"
    assert tables[0].markdown.startswith("| Category |")
    assert tables[0].table_data["rows"] == []
