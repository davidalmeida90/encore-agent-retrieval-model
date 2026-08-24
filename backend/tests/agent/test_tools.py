import uuid
from dataclasses import dataclass
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from app.agent.deps import DocumentAgentDeps, TurnRegistry
from app.agent.tools.retrieval import read_chunk, read_chunks, read_surrounding_chunks, search_filings
from app.retrieval.types import RetrievedPassage


def _passage(chunk_id: uuid.UUID | None = None) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id or uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_index=0,
        text="Data Center revenue increased materially.",
        page="42",
        section="MD&A",
        fusion_score=0.5,
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        form="10-K",
        filing_date=date(2024, 1, 28),
        fiscal_year=2024,
        accession_number="0001045810-24-000012",
    )


@dataclass
class _FakeCtx:
    deps: DocumentAgentDeps


def _ctx(retriever: MagicMock | None = None) -> _FakeCtx:
    registry = TurnRegistry()
    deps = DocumentAgentDeps(
        retriever=retriever or MagicMock(),
        registry=registry,
        thread_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
    )
    return _FakeCtx(deps=deps)


@pytest.mark.anyio
async def test_search_filings_registers_passages() -> None:
    passage = _passage()
    retriever = MagicMock()
    retriever.search.return_value = [passage]
    ctx = _ctx(retriever)

    with patch("app.agent.tools.retrieval._search_sync", return_value=[passage]):
        result = await search_filings(ctx, "NVIDIA Data Center demand")

    assert passage.chunk_id in ctx.deps.registry.passages_by_chunk_id
    assert "NVDA" in result


@pytest.mark.anyio
async def test_read_chunk_returns_error_for_missing_chunk() -> None:
    ctx = _ctx()

    with patch("app.agent.tools.retrieval._read_chunk_sync", return_value=None):
        result = await read_chunk(ctx, str(uuid.uuid4()))

    assert result.startswith("Error:")
    assert not ctx.deps.registry.passages_by_chunk_id


@pytest.mark.anyio
async def test_read_chunk_registers_found_passage() -> None:
    passage = _passage()
    ctx = _ctx()

    with patch("app.agent.tools.retrieval._read_chunk_sync", return_value=passage):
        result = await read_chunk(ctx, str(passage.chunk_id))

    assert passage.chunk_id in ctx.deps.registry.passages_by_chunk_id
    assert str(passage.chunk_id) in result


@pytest.mark.anyio
async def test_read_surrounding_chunks_registers_neighbors() -> None:
    anchor = _passage()
    neighbor = _passage()
    ctx = _ctx()

    with patch(
        "app.agent.tools.retrieval._read_surrounding_sync",
        return_value=[anchor, neighbor],
    ):
        result = await read_surrounding_chunks(ctx, str(anchor.chunk_id))

    assert anchor.chunk_id in ctx.deps.registry.passages_by_chunk_id
    assert neighbor.chunk_id in ctx.deps.registry.passages_by_chunk_id
    assert "NVDA" in result


@pytest.mark.anyio
async def test_read_chunk_rejects_invalid_uuid() -> None:
    ctx = _ctx()
    result = await read_chunk(ctx, "not-a-uuid")
    assert "invalid chunk_id" in result


@pytest.mark.anyio
async def test_read_chunks_registers_all_found_passages() -> None:
    first = _passage()
    second = _passage()
    ctx = _ctx()

    with patch(
        "app.agent.tools.retrieval._read_chunks_sync",
        return_value=[first, second],
    ):
        result = await read_chunks(
            ctx,
            [str(first.chunk_id), str(second.chunk_id)],
        )

    assert first.chunk_id in ctx.deps.registry.passages_by_chunk_id
    assert second.chunk_id in ctx.deps.registry.passages_by_chunk_id
    assert "NVDA" in result


@pytest.mark.anyio
async def test_read_chunks_rejects_invalid_uuid() -> None:
    ctx = _ctx()
    result = await read_chunks(ctx, ["not-a-uuid"])
    assert "invalid chunk_id" in result
