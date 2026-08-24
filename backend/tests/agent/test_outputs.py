import uuid

import pytest
from pydantic import ValidationError

from app.agent.outputs import Citation, GroundedAnswer


def test_grounded_answer_defaults_insufficient_evidence_false() -> None:
    answer = GroundedAnswer(answer="No data [1]", citations=[])
    assert answer.insufficient_evidence is False


def test_citation_requires_chunk_id() -> None:
    with pytest.raises(ValidationError):
        Citation(citation_index=1, excerpt="text")  # type: ignore[call-arg]


def test_grounded_answer_with_citation() -> None:
    chunk_id = uuid.uuid4()
    answer = GroundedAnswer(
        answer="Revenue grew [1].",
        citations=[
            Citation(citation_index=1, chunk_id=chunk_id, excerpt="Revenue grew 10%"),
        ],
    )
    assert answer.citations[0].chunk_id == chunk_id
