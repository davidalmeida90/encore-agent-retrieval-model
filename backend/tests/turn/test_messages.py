import uuid
from datetime import date

import pytest
from fastapi import HTTPException

from app.agent.deps import TurnRegistry
from app.agent.outputs import Citation, GroundedAnswer
from app.turn.messages import (
    build_assistant_message,
    citation_parts_from_grounded_answer,
    extract_last_user_message,
    row_to_ui_message,
    text_from_parts,
    title_from_user_message,
    ui_message_to_insert,
)
from app.retrieval.types import RetrievedPassage
from app.schemas.chat import CitationPart, TextPart, UIMessage


def test_text_from_parts_joins_text_segments() -> None:
    parts = [
        TextPart(text="Hello "),
        TextPart(text="world"),
    ]
    assert text_from_parts(parts) == "Hello world"


def test_extract_last_user_message_returns_latest_user() -> None:
    messages = [
        UIMessage(role="user", parts=[TextPart(text="First")]),
        UIMessage(role="assistant", parts=[TextPart(text="Reply")]),
        UIMessage(role="user", parts=[TextPart(text="Second")]),
    ]
    result = extract_last_user_message(messages)
    assert text_from_parts(result.parts) == "Second"


def test_extract_last_user_message_raises_when_missing() -> None:
    messages = [
        UIMessage(role="assistant", parts=[TextPart(text="Only assistant")]),
    ]
    with pytest.raises(HTTPException) as exc_info:
        extract_last_user_message(messages)
    assert exc_info.value.status_code == 422


def test_ui_message_to_insert_maps_fields() -> None:
    thread_id = uuid.uuid4()
    message = UIMessage(
        id="client-id",
        role="user",
        parts=[TextPart(text="Hello")],
    )
    row = ui_message_to_insert(message, thread_id=thread_id, sequence=3)
    assert row["thread_id"] == str(thread_id)
    assert row["role"] == "user"
    assert row["content"] == "Hello"
    assert row["parts"] == [{"type": "text", "text": "Hello"}]
    assert row["sequence"] == 3


def test_row_to_ui_message_round_trip() -> None:
    message_id = uuid.uuid4()
    row = {
        "id": str(message_id),
        "role": "assistant",
        "content": "Saved reply",
        "parts": [{"type": "text", "text": "Saved reply"}],
        "sequence": 1,
    }
    message = row_to_ui_message(row)
    assert message.id == str(message_id)
    assert message.role == "assistant"
    assert text_from_parts(message.parts) == "Saved reply"


def test_row_to_ui_message_falls_back_to_content() -> None:
    row = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": "Legacy content",
        "parts": None,
        "sequence": 0,
    }
    message = row_to_ui_message(row)
    assert text_from_parts(message.parts) == "Legacy content"


def test_build_assistant_message_includes_citation_parts() -> None:
    chunk_id = uuid.uuid4()
    passage = RetrievedPassage(
        chunk_id=chunk_id,
        document_id=uuid.uuid4(),
        chunk_index=0,
        text="iPhone revenue was $201 billion.",
        page="22",
        section="Segments",
        fusion_score=0.9,
        ticker="AAPL",
        company_name="Apple Inc.",
        form="10-K",
        filing_date=date(2024, 10, 31),
        fiscal_year=2024,
        accession_number="0000320193-24-000123",
    )
    registry = TurnRegistry()
    registry.register(passage)
    answer = GroundedAnswer(
        answer="iPhone revenue was large [1].",
        citations=[
            Citation(
                citation_index=1,
                chunk_id=chunk_id,
                excerpt="iPhone revenue was $201 billion.",
            )
        ],
    )
    message = build_assistant_message(answer, registry)
    assert message.role == "assistant"
    assert text_from_parts(message.parts) == "iPhone revenue was large [1]."
    citation_parts = [p for p in message.parts if isinstance(p, CitationPart)]
    assert len(citation_parts) == 1
    assert citation_parts[0].data.ticker == "AAPL"


def test_citation_parts_from_grounded_answer() -> None:
    chunk_id = uuid.uuid4()
    passage = RetrievedPassage(
        chunk_id=chunk_id,
        document_id=uuid.uuid4(),
        chunk_index=0,
        text="AWS operating income grew.",
        page="8",
        section=None,
        fusion_score=0.4,
        ticker="AMZN",
        company_name="Amazon.com, Inc.",
        form="10-K",
        filing_date=date(2024, 2, 2),
        fiscal_year=2023,
        accession_number="0001018724-24-000008",
    )
    registry = TurnRegistry()
    registry.register(passage)
    answer = GroundedAnswer(
        answer="AWS income grew [1].",
        citations=[
            Citation(
                citation_index=1,
                chunk_id=chunk_id,
                excerpt="AWS operating income grew.",
            )
        ],
    )
    parts = citation_parts_from_grounded_answer(answer, registry)
    assert parts[0].data.form == "10-K"
    assert parts[0].data.citation_index == 1


def test_title_from_user_message_truncates_long_text() -> None:
    message = UIMessage(
        role="user",
        parts=[TextPart(text="x" * 300)],
    )
    title = title_from_user_message(message)
    assert len(title) == 255
    assert title.endswith("...")
