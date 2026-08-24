import asyncio
import uuid
from datetime import date

from app.agent.deps import TurnRegistry
from app.agent.outputs import Citation, GroundedAnswer
from app.verification.validator import (
    CitationGroundingCase,
    CitationGroundingDecision,
    GroundingValidator,
    prune_unreferenced_citations,
)
from app.retrieval.types import RetrievedPassage


def _passage(text: str = "Services revenue grew 12% year over year.") -> RetrievedPassage:
    chunk_id = uuid.uuid4()
    return RetrievedPassage(
        chunk_id=chunk_id,
        document_id=uuid.uuid4(),
        chunk_index=0,
        text=text,
        page="10",
        section="Item 7",
        fusion_score=0.8,
        ticker="AAPL",
        company_name="Apple Inc.",
        form="10-K",
        filing_date=date(2024, 10, 31),
        fiscal_year=2024,
        accession_number="0000320193-24-000123",
    )


class FakeGroundingJudge:
    def __init__(self, *, supported: bool = True, fail: bool = False) -> None:
        self.supported = supported
        self.fail = fail
        self.calls: list[list[CitationGroundingCase]] = []

    async def judge(
        self,
        cases: list[CitationGroundingCase],
    ) -> list[CitationGroundingDecision]:
        self.calls.append(cases)
        if self.fail:
            raise RuntimeError("judge unavailable")
        return [
            CitationGroundingDecision(
                citation_index=case.citation_index,
                supported=self.supported,
                reason="supported" if self.supported else "unsupported",
            )
            for case in cases
        ]


class MissingBatchDecisionJudge:
    def __init__(self) -> None:
        self.calls: list[list[CitationGroundingCase]] = []

    async def judge(
        self,
        cases: list[CitationGroundingCase],
    ) -> list[CitationGroundingDecision]:
        self.calls.append(cases)
        if len(cases) > 1:
            return [
                CitationGroundingDecision(
                    citation_index=cases[0].citation_index,
                    supported=True,
                    reason="supported",
                )
            ]

        return [
            CitationGroundingDecision(
                citation_index=cases[0].citation_index,
                supported=True,
                reason="supported",
            )
        ]


def _validate(
    answer: GroundedAnswer,
    registry: TurnRegistry,
    *,
    judge: FakeGroundingJudge | None = None,
):
    return asyncio.run(
        GroundingValidator(judge=judge or FakeGroundingJudge()).validate(
            answer,
            registry,
        )
    )


def test_valid_grounded_answer_passes() -> None:
    passage = _passage()
    registry = TurnRegistry()
    registry.register(passage)
    answer = GroundedAnswer(
        answer="Services revenue grew [1].",
        citations=[
            Citation(
                citation_index=1,
                chunk_id=passage.chunk_id,
                excerpt="Services revenue grew 12% year over year.",
            )
        ],
    )
    result = _validate(answer, registry)
    assert result.ok


def test_insufficient_evidence_requires_empty_citations() -> None:
    registry = TurnRegistry()
    answer = GroundedAnswer(
        answer="The corpus does not contain enough evidence to answer.",
        citations=[],
        insufficient_evidence=True,
    )
    result = _validate(answer, registry)
    assert result.ok


def test_insufficient_evidence_rejects_citations() -> None:
    passage = _passage()
    registry = TurnRegistry()
    registry.register(passage)
    answer = GroundedAnswer(
        answer="Not enough evidence.",
        citations=[
            Citation(citation_index=1, chunk_id=passage.chunk_id, excerpt="x"),
        ],
        insufficient_evidence=True,
    )
    result = _validate(answer, registry)
    assert not result.ok


def test_unknown_chunk_id_fails() -> None:
    passage = _passage("Registered chunk text for another id.")
    registry = TurnRegistry()
    registry.register(passage)
    unknown_id = uuid.uuid4()
    answer = GroundedAnswer(
        answer="Claim [1].",
        citations=[
            Citation(citation_index=1, chunk_id=unknown_id, excerpt="Claim"),
        ],
    )
    judge = FakeGroundingJudge()
    result = _validate(answer, registry, judge=judge)
    assert not result.ok
    assert "not retrieved" in (result.error or "")
    assert judge.calls == []


def test_unsupported_claim_fails() -> None:
    passage = _passage()
    registry = TurnRegistry()
    registry.register(passage)
    answer = GroundedAnswer(
        answer="Claim [1].",
        citations=[
            Citation(
                citation_index=1,
                chunk_id=passage.chunk_id,
                excerpt="Fabricated quote not in chunk.",
            )
        ],
    )
    result = _validate(answer, registry, judge=FakeGroundingJudge(supported=False))
    assert not result.ok
    assert "not supported" in (result.error or "")


def test_table_claim_passes_when_judge_finds_support() -> None:
    passage = _passage(
        "| Services (3) |  |  | 68,425 | 68,425 |\n"
        "| Total net sales | Total net sales | $ | 365,817 |"
    )
    registry = TurnRegistry()
    registry.register(passage)
    answer = GroundedAnswer(
        answer="Services revenue was about $68.4 billion [1].",
        citations=[
            Citation(
                citation_index=1,
                chunk_id=passage.chunk_id,
                excerpt="Services revenue was about $68.4 billion.",
            )
        ],
    )
    judge = FakeGroundingJudge(supported=True)

    result = _validate(answer, registry, judge=judge)

    assert result.ok
    assert len(judge.calls) == 1
    assert judge.calls[0][0].answer == answer.answer
    assert judge.calls[0][0].source_text == passage.text


def test_judge_error_fails_closed() -> None:
    passage = _passage()
    registry = TurnRegistry()
    registry.register(passage)
    answer = GroundedAnswer(
        answer="Services revenue grew [1].",
        citations=[
            Citation(
                citation_index=1,
                chunk_id=passage.chunk_id,
                excerpt="Services revenue grew 12% year over year.",
            )
        ],
    )

    result = _validate(answer, registry, judge=FakeGroundingJudge(fail=True))

    assert not result.ok
    assert "Grounding judge failed" in (result.error or "")


def test_malformed_batch_judge_decisions_retry_per_citation() -> None:
    first = _passage("Revenue increased.")
    second = _passage("Costs decreased.")
    registry = TurnRegistry()
    registry.register(first)
    registry.register(second)
    answer = GroundedAnswer(
        answer="Revenue increased [1]. Costs decreased [2].",
        citations=[
            Citation(
                citation_index=1,
                chunk_id=first.chunk_id,
                excerpt="Revenue increased.",
            ),
            Citation(
                citation_index=2,
                chunk_id=second.chunk_id,
                excerpt="Costs decreased.",
            ),
        ],
    )
    judge = MissingBatchDecisionJudge()

    result = _validate(answer, registry, judge=judge)

    assert result.ok
    assert [len(call) for call in judge.calls] == [2, 1, 1]


def test_missing_marker_fails() -> None:
    passage = _passage()
    registry = TurnRegistry()
    registry.register(passage)
    answer = GroundedAnswer(
        answer="Services revenue grew without marker.",
        citations=[
            Citation(
                citation_index=1,
                chunk_id=passage.chunk_id,
                excerpt="Services revenue grew 12% year over year.",
            )
        ],
    )
    result = _validate(answer, registry)
    assert not result.ok


def test_answer_with_no_citations_and_no_markers_is_allowed() -> None:
    """A tool-derived answer has nothing to cite, and that is legitimate.

    This validator was written when every answer came from retrieved filing text.
    Numbers now also come from live tools (SEC XBRL, prices, the valuation
    engines) which have no chunk behind them, exactly as instructions.md tells
    the model. Requiring a citation here rejected correct answers: "compare
    Apple and Microsoft capex" was retried twice and then shown to the user as a
    grounding error.
    """
    registry = TurnRegistry()
    registry.register(_passage())
    answer = GroundedAnswer(answer="Apple spent $12.7B on capex; Microsoft $115.9B.")
    result = _validate(answer, registry)
    assert result.ok


def test_markers_without_citations_still_fails() -> None:
    """The loosening above is narrow: placing a marker still demands a citation."""
    registry = TurnRegistry()
    registry.register(_passage())
    answer = GroundedAnswer(answer="Azure grew strongly [1].")
    result = _validate(answer, registry)
    assert not result.ok


def test_prune_unreferenced_citations_removes_extra_metadata() -> None:
    passage = _passage()
    extra_passage = _passage("Mac revenue declined.")
    registry = TurnRegistry()
    registry.register(passage)
    registry.register(extra_passage)
    answer = GroundedAnswer(
        answer="Services revenue grew [1].",
        citations=[
            Citation(
                citation_index=1,
                chunk_id=passage.chunk_id,
                excerpt="Services revenue grew 12% year over year.",
            ),
            Citation(
                citation_index=2,
                chunk_id=extra_passage.chunk_id,
                excerpt="Mac revenue declined.",
            ),
        ],
    )

    pruned = prune_unreferenced_citations(answer)

    assert [citation.citation_index for citation in pruned.citations] == [1]
    assert _validate(pruned, registry).ok


def test_combined_markers_are_seen_by_the_validator() -> None:
    r""""[1, 2]" must count as markers 1 AND 2, not be invisible.

    The regex used to be r"\[(\d+)\]", which matched only the single form. Models
    write combined markers constantly, and every one was skipped: an answer
    showing [1, 2], [3, 4], [2, 5] had all five citations pruned as unreferenced
    and reported "Grounding passed - 1 citation verified". It passed by throwing
    the evidence away.
    """
    from app.verification.validator import _citation_markers

    assert _citation_markers("[1] and [2, 3] and [4,5]") == {1, 2, 3, 4, 5}
    assert _citation_markers("no markers here") == set()


def test_pruning_keeps_combined_markers_combined() -> None:
    registry = TurnRegistry()
    registry.register(_passage())
    answer = GroundedAnswer(
        answer="Alpha [1, 2] beta [4] gamma.",
        citations=[
            Citation(citation_index=i, chunk_id=uuid.uuid4(), excerpt="x")
            for i in (1, 2, 3, 4)
        ],
    )
    pruned = prune_unreferenced_citations(answer)
    # 3 was never referenced, so 4 renumbers to 3 and the group stays a group.
    assert pruned.answer == "Alpha [1, 2] beta [3] gamma."
    assert [c.citation_index for c in pruned.citations] == [1, 2, 3]
