"""Small regression suite for the finance agent, sized for a free-tier quota.

Eight cases, chosen to cover one capability each rather than to be exhaustive.
Every evaluator is deterministic string/structure checking, so grading itself
costs zero model calls; only the agent runs consume quota.

Corpus under test: AAPL and MSFT 10-Ks, FY2024 and FY2025.

Run:  python -m evals.finance_agent_eval            (all 8)
      python -m evals.finance_agent_eval 1 3 5      (selected case numbers)
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.agent.agent import get_document_agent
from app.agent.deps import DocumentAgentDeps, TurnRegistry
from app.config import settings
from app.verification.validator import (
    CitationGroundingDecision,
    GroundingValidator,
    prune_unreferenced_citations,
)
from app.retrieval.retriever import DocumentRetriever
from pydantic_ai import UsageLimits


@dataclass
class Result:
    """What one agent run produced, flattened for checking."""

    answer: str
    tools: list[str] = field(default_factory=list)
    citations: int = 0
    insufficient: bool = False
    requests: int = 0
    grounding: str = "ok"


PACE_SECONDS = 20      # gap between cases: free tier is 15 req/min, a case spends 3-8
MAX_429_RETRIES = 2


def run_with_backoff(q: str) -> Result:
    """Free-tier quotas are per-minute, so a 429 is a wait, not a failure."""
    for attempt in range(MAX_429_RETRIES + 1):
        try:
            return run_question(q)
        except Exception as e:
            if "429" not in str(e) or attempt == MAX_429_RETRIES:
                raise
            wait = 60 * (attempt + 1)
            print(f"   ... 429, waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def run_question(q: str) -> Result:
    deps = DocumentAgentDeps(
        retriever=DocumentRetriever(), registry=TurnRegistry(),
        user_id=uuid.uuid4(), thread_id=uuid.uuid4(),
        on_status=lambda *a, **k: None,
    )
    res = get_document_agent().run_sync(
        q, deps=deps,
        usage_limits=UsageLimits(request_limit=settings.agent_request_limit),
    )
    msgs = res.all_messages()
    tools = [
        p.tool_name
        for m in msgs
        for p in m.parts
        if type(p).__name__ == "ToolCallPart" and getattr(p, "tool_name", "") != "final_result"
    ]
    # run_sync alone skips the orchestrator, so the structural half of grounding
    # never ran and the suite scored answers production would have rejected.
    # Stub the judge so the paid LLM check is skipped but every marker, index and
    # chunk-id rule still fires, at zero quota cost.
    class _ApproveAll:
        async def judge(self, cases):
            return [CitationGroundingDecision(citation_index=c.citation_index,
                                              supported=True, reason="stub")
                    for c in cases]

    pruned = prune_unreferenced_citations(res.output)
    verdict = asyncio.run(GroundingValidator(_ApproveAll()).validate(pruned, deps.registry))

    return Result(
        grounding="ok" if verdict.ok else (verdict.error or "failed"),
        answer=res.output.answer,
        tools=tools,
        citations=len(res.output.citations),
        insufficient=res.output.insufficient_evidence,
        requests=sum(1 for m in msgs if type(m).__name__ == "ModelResponse"),
    )


# --------------------------------------------------------------------------- checks
def contains_all(text: str, *needles: str) -> tuple[bool, str]:
    missing = [n for n in needles if n.lower() not in text.lower()]
    return (not missing, "" if not missing else f"missing {missing}")


def used(r: Result, *names: str) -> tuple[bool, str]:
    absent = [n for n in names if n not in r.tools]
    return (not absent, "" if not absent else f"never called {absent}")


def grounded(r: Result) -> tuple[bool, str]:
    """Would the orchestrator have shown this answer, or errored out?"""
    return (r.grounding == "ok", f"grounding rejected: {r.grounding[:70]}")


def no_uuid_citations(r: Result) -> tuple[bool, str]:
    """A bracketed UUID is invisible to the grounding validator's [\\d+] regex."""
    import re

    bad = re.findall(r"\[[0-9a-f]{8}-[0-9a-f]{4}", r.answer)
    return (not bad, "" if not bad else f"{len(bad)} UUID citation markers")


# --------------------------------------------------------------------------- cases
CASES: list[tuple[str, str, Any]] = [
    (
        "numeric-single",
        "What was Apple's capital expenditure in fiscal 2025?",
        lambda r: [used(r, "get_sec_financials"), contains_all(r.answer, "12.7"),
                   no_uuid_citations(r)],
    ),
    (
        "numeric-comparison",
        "Compare Apple's and Microsoft's capital expenditure in their most recent fiscal year.",
        lambda r: [used(r, "get_sec_financials"), contains_all(r.answer, "apple", "microsoft"),
                   no_uuid_citations(r)],
    ),
    (
        "qualitative-retrieval",
        "How does Microsoft describe Azure's competitive advantage in its 10-K?",
        lambda r: [used(r, "search_filings"), (r.citations > 0, "no citations"),
                   no_uuid_citations(r)],
    ),
    (
        "absence-fail-closed",
        "Does Apple disclose a specific artificial intelligence risk factor in its 10-K?",
        lambda r: [used(r, "search_filings"), no_uuid_citations(r)],
    ),
    (
        "market-data",
        "How has Apple's stock performed over the past year?",
        lambda r: [used(r, "get_stock_prices"), contains_all(r.answer, "%")],
    ),
    (
        "rates-not-invented",
        "What risk-free rate should be used to value a US company today, and why?",
        lambda r: [used(r, "get_risk_free_rate"), contains_all(r.answer, "4.")],
    ),
    (
        "dcf-cross-checks",
        "Run a DCF valuation for Apple and state every assumption.",
        lambda r: [used(r, "load_skill", "get_risk_free_rate", "run_dcf_valuation"),
                   contains_all(r.answer, "terminal")],
    ),
    (
        "comps-peer-warning",
        "Value Apple on trading comparables using Microsoft as the peer.",
        lambda r: [used(r, "get_comps_inputs", "run_comps_valuation"),
                   contains_all(r.answer, "peer")],
    ),
]


def main() -> None:
    picked = [int(a) for a in sys.argv[1:] if a.isdigit()]
    cases = [c for i, c in enumerate(CASES, 1) if not picked or i in picked]
    passed = total = 0
    print(f"model: {settings.gemini_chat_model}   cases: {len(cases)}\n")
    for i, (name, question, checks) in enumerate(cases, 1):
        try:
            r = run_with_backoff(question)
        except Exception as e:  # quota, provider outage, engine refusal
            print(f"{i}. {name:22} ERROR  {type(e).__name__}: {str(e)[:90]}")
            total += 1
            continue
        results = checks(r) + [grounded(r)]
        ok = all(p for p, _ in results)
        passed += ok
        total += 1
        why = "; ".join(m for p, m in results if not p)
        print(f"{i}. {name:22} {'PASS' if ok else 'FAIL'}  "
              f"reqs={r.requests} tools={len(r.tools)} cites={r.citations}"
              f"{'  <- ' + why if why else ''}")
        print(f"   tools: {r.tools}")
        if i < len(cases):
            time.sleep(PACE_SECONDS)
    print(f"\n{passed}/{total} passed")


if __name__ == "__main__":
    main()
