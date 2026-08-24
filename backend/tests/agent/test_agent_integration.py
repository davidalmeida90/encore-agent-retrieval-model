import asyncio
import uuid

import pytest

from app.agent.agent import run_document_agent
from app.agent.deps import DocumentAgentDeps, TurnRegistry
from app.verification.validator import GroundingValidator
from app.retrieval.retriever import DocumentRetriever


@pytest.mark.integration
def test_agent_answers_apple_question_with_citations() -> None:
    registry = TurnRegistry()
    deps = DocumentAgentDeps(
        retriever=DocumentRetriever(),
        registry=registry,
        thread_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
    )
    answer = run_document_agent(
        "How did Apple describe iPhone and Services revenue in its recent 10-K filings?",
        deps,
    )
    validation = asyncio.run(GroundingValidator().validate(answer, registry))

    assert validation.ok
    if not answer.insufficient_evidence:
        assert answer.citations
        tickers = {
            registry.passages_by_chunk_id[c.chunk_id].ticker
            for c in answer.citations
        }
        assert "AAPL" in tickers


@pytest.mark.integration
def test_agent_refuses_underspecified_stock_pick_question() -> None:
    registry = TurnRegistry()
    deps = DocumentAgentDeps(
        retriever=DocumentRetriever(),
        registry=registry,
        thread_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
    )
    answer = run_document_agent("What is the best stock to buy right now?", deps)
    validation = asyncio.run(GroundingValidator().validate(answer, registry))

    assert validation.ok
    assert answer.insufficient_evidence or not answer.citations
