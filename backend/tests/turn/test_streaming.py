import uuid
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.deps import TurnRegistry
from app.agent.outputs import Citation, GroundedAnswer
from app.turn.streaming import (
    stream_grounded_answer,
    stream_grounded_turn_and_persist,
    stream_status,
)
from app.verification.validator import ValidationResult
from app.retrieval.types import RetrievedPassage
from app.schemas.chat import TextPart, UIMessage


def _grounded_answer() -> tuple[GroundedAnswer, TurnRegistry]:
    chunk_id = uuid.uuid4()
    passage = RetrievedPassage(
        chunk_id=chunk_id,
        document_id=uuid.uuid4(),
        chunk_index=0,
        text="Cloud revenue increased significantly.",
        page="15",
        section="MD&A",
        fusion_score=0.6,
        ticker="MSFT",
        company_name="Microsoft Corporation",
        form="10-K",
        filing_date=date(2024, 7, 30),
        fiscal_year=2024,
        accession_number="0000789019-24-000012",
    )
    registry = TurnRegistry()
    registry.register(passage)
    answer = GroundedAnswer(
        answer="Cloud revenue increased [1].",
        citations=[
            Citation(
                citation_index=1,
                chunk_id=chunk_id,
                excerpt="Cloud revenue increased significantly.",
            )
        ],
    )
    return answer, registry


@pytest.mark.anyio
async def test_stream_status_emits_data_status_event() -> None:
    events = [event async for event in stream_status("searching", "Searching SEC filings…")]

    assert len(events) == 1
    assert '"type":"data-status"' in events[0]
    assert '"stage":"searching"' in events[0]
    assert "Searching SEC filings" in events[0]


@pytest.mark.anyio
async def test_stream_grounded_answer_emits_text_and_citation_events() -> None:
    answer, registry = _grounded_answer()
    events: list[str] = []

    async for event in stream_grounded_answer(
        answer,
        registry,
        message_id=str(uuid.uuid4()),
    ):
        events.append(event)

    assert events[0].startswith('data: {"type":"start"')
    assert any('"type":"text-start"' in event for event in events)
    assert any('"type":"text-delta"' in event for event in events)
    assert any('"type":"data-citation"' in event for event in events)
    assert events[-1].startswith('data: {"type":"finish"')
    assert all(event.endswith("\n\n") for event in events)


@pytest.mark.anyio
async def test_stream_grounded_turn_persists_on_success() -> None:
    answer, registry = _grounded_answer()
    user_message = UIMessage(role="user", parts=[TextPart(text="Hello")])
    mock_append = AsyncMock()

    with patch("app.turn.streaming.append_grounded_turn", mock_append):
        async for _ in stream_grounded_turn_and_persist(
            client=MagicMock(),
            thread_id=uuid.uuid4(),
            user_message=user_message,
            thread_title="New chat",
            answer=answer,
            registry=registry,
            validation=ValidationResult(ok=True),
        ):
            pass

    mock_append.assert_awaited_once()
    persisted = mock_append.await_args.kwargs["assistant_message"]
    part_types = {part.type for part in persisted.parts}
    assert "data-status" not in part_types


@pytest.mark.anyio
async def test_stream_grounded_turn_skips_persist_on_validation_failure() -> None:
    answer, registry = _grounded_answer()
    user_message = UIMessage(role="user", parts=[TextPart(text="Hello")])
    mock_append = AsyncMock()

    with patch("app.turn.streaming.append_grounded_turn", mock_append):
        events = [
            event
            async for event in stream_grounded_turn_and_persist(
                client=MagicMock(),
                thread_id=uuid.uuid4(),
                user_message=user_message,
                thread_title="New chat",
                answer=answer,
                registry=registry,
                validation=ValidationResult(ok=False, error="bad citations"),
            )
        ]

    assert any('"type":"error"' in event for event in events)
    mock_append.assert_not_awaited()


@pytest.mark.anyio
async def test_stream_grounded_turn_explains_validation_failure_to_user() -> None:
    answer, registry = _grounded_answer()
    user_message = UIMessage(role="user", parts=[TextPart(text="Hello")])

    events = [
        event
        async for event in stream_grounded_turn_and_persist(
            client=MagicMock(),
            thread_id=uuid.uuid4(),
            user_message=user_message,
            thread_title="New chat",
            answer=answer,
            registry=registry,
            validation=ValidationResult(ok=False, error="bad citations"),
        )
    ]

    payload = "".join(events)
    assert "found relevant source passages" in payload
    assert "could not fully verify" in payload
    assert "bad citations" not in payload
