"""## Run several questions end to end and report what each one cost

## Why this exists as a script rather than a loop in a notebook
Each question is measured the way the app runs it: same agent, same tools, same
grounding gate. What comes back is one line per question with the two numbers
that actually matter — how many model requests it took, and how many tokens.

    python -u scripts/benchmark.py            # every question
    python -u scripts/benchmark.py 2 4        # only these

## One event loop, and why that is not a detail
Every question runs inside a SINGLE `asyncio.run`, using `await agent.run(...)`
rather than `agent.run_sync(...)` per question.

`run_sync` creates a fresh event loop for each call and closes it afterwards. The
agent is cached (one per model), and it holds an HTTP client whose connection
pool has sockets bound to whichever loop was open when they were created. Ask
three questions with `run_sync` and the third one writes into a socket belonging
to a loop that no longer exists:

    1. PASS      3s  reqs=2   What was Apple's capital expenditure...
    2. PASS     78s  reqs=2   How does Microsoft describe Azure's...
    3. ERROR   120s  WriteTimeout

Question 3 is not the problem. Run it first in its own process and it passes in
42 seconds. Before a request timeout existed, this same stall presented as a run
that hung for forty minutes with no output and nothing obviously wrong.

The server has one long-lived loop, so this never bites there. It bites exactly
here, in the scripts used to measure whether the server is any good.

## Grounding
The validator's LLM judge is stubbed to approve, so a run costs no extra quota
beyond the answers themselves. The free structural checks still run in full, and
those are the ones that catch a citation pointing at a passage that was never
retrieved.

This measures the FIRST attempt only. The app retries once on a grounding
failure, so a FAIL here is not what a user would have seen: it is the raw
first-attempt rate, which is the number worth watching.

## Read the same question several times before believing any of it
Cost is not a property of the question. It is a property of what the model
decides to do, and that changes run to run. Four runs of question 3, unchanged:

    11 requests   83,366 tokens   PASS
     7 requests   32,652 tokens   PASS
     6 requests   43,979 tokens   PASS
     2 requests    9,888 tokens   FAIL  (a [4] marker with no citation 4)

An 8x spread on identical input, and the cheapest run is the one that failed:
it answered in two requests without doing the work. A single measurement of a
question like this is not a measurement.
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid

from pydantic_ai import UsageLimits

from app.agent.agent import get_document_agent
from app.agent.deps import DocumentAgentDeps, TurnRegistry
from app.config import settings
from app.retrieval.retriever import DocumentRetriever
from app.verification.validator import (
    CitationGroundingDecision,
    GroundingValidator,
    prune_unreferenced_citations,
)

QUESTIONS = [
    "What was Apple's capital expenditure in fiscal 2025?",
    "How does Microsoft describe Azure's competitive advantage in its 10-K?",
    "Compare Apple's and Microsoft's R&D spending in their most recent fiscal "
    "year, and what each says about its R&D priorities.",
    "Run a DCF valuation for Apple and state every assumption.",
]


class _ApproveEverything:
    """Stand-in for the judge, so measuring cost does not spend quota judging."""

    async def judge(self, cases):
        return [
            CitationGroundingDecision(
                citation_index=case.citation_index, supported=True, reason="stubbed"
            )
            for case in cases
        ]


async def ask(index: int, question: str) -> None:
    started = time.time()
    deps = DocumentAgentDeps(
        retriever=DocumentRetriever(),
        registry=TurnRegistry(),
        user_id=uuid.uuid4(),
        thread_id=uuid.uuid4(),
        on_status=lambda *args, **kwargs: None,
    )
    try:
        result = await get_document_agent().run(
            question,
            deps=deps,
            usage_limits=UsageLimits(request_limit=settings.agent_request_limit),
        )
        answer = prune_unreferenced_citations(result.output)
        verdict = await GroundingValidator(_ApproveEverything()).validate(
            answer, deps.registry
        )
        print(
            f"{index}. {'PASS' if verdict.ok else 'FAIL'} {time.time() - started:5.0f}s  "
            f"reqs={result.usage.requests:<3} cites={len(answer.citations)} "
            f"tok={result.usage.input_tokens:>7,}  {question[:44]}",
            flush=True,
        )
        if not verdict.ok:
            print(f"      -> {(verdict.error or '')[:110]}", flush=True)
            # A grounding failure is only useful if you can see WHICH markers
            # disagreed with which citations. Without this the message names the
            # rule that fired and nothing about why.
            from app.verification.validator import _citation_markers

            markers = sorted(_citation_markers(answer.answer))
            indices = sorted(c.citation_index for c in answer.citations)
            print(f"         markers in text: {markers}", flush=True)
            print(f"         citation_index : {indices}", flush=True)
    except Exception as error:  # a failure is a result, not a crash
        print(
            f"{index}. ERROR {time.time() - started:5.0f}s  "
            f"{type(error).__name__}: {str(error)[:90]}",
            flush=True,
        )


async def main() -> None:
    picked = [int(arg) for arg in sys.argv[1:]] or list(range(1, len(QUESTIONS) + 1))
    for index, question in enumerate(QUESTIONS, 1):
        if index in picked:
            await ask(index, question)
    print("DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
