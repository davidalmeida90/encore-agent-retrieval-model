"""## The agent: one definition, assembled from four pieces

## What actually runs a question
PydanticAI owns the loop. It sends the conversation to the model, receives tool
calls, validates their arguments against each tool's signature, runs them,
appends the results, and repeats until the model returns a `GroundedAnswer` that
passes output validation. Nothing in this repo re-implements that loop.

                        your question
                              |
                              v
        +------------->  [  model  ]  -------------------+
        |                     |                          |
        |            wants a tool?                        | done, emits
        |                     | yes                       | GroundedAnswer
        |                     v                           v
        |             [ tools/ registry ]          output validation
        |          retrieval | fundamentals         (schema, retried
        |          market    | valuation             if malformed)
        |          skills                                 |
        |                     |                           v
        |             result appended             GROUNDING VALIDATOR
        +---- back into  <----+                   app/grounding/validator.py
              history                                     |
                                              +-----------+-----------+
                                              |                       |
                                         markers match           mismatch, or
                                         chunk retrieved         judge says the
                                         judge says supported    excerpt is not
                                              |                  supported
                                              v                       |
                                          streamed                    v
                                         to the user           re-run the WHOLE
                                                               agent (max 2), then
                                                               show an error and
                                                               NO answer

Note where the loop closes. Every tool result is appended to the conversation and
re-sent on the next request, so round 6 pays for rounds 1-5 again. That, not the
per-call price, is what makes long questions expensive.

Note also that the validator sits OUTSIDE the loop, after the model believes it
has finished. The agent has no authority to publish: if grounding fails twice,
the answer is discarded rather than shown.

## The four pieces assembled below
    1. model         which LLM, and the key it authenticates with
    2. instructions  the system prompt, plus one line per available skill
    3. capabilities  harness behaviours: payload limits, compaction, budgets
    4. tools         the registry in tools/__init__.py

Every number used here lives in `tuning.py`, not inline, so testing means editing
one file. See that file for the observation behind each value.

## Three real runs, to show what the loop actually does

    "What was Apple's capital expenditure in fiscal 2025?"
        round 1  get_sec_financials(AAPL, [capex], annual)
        round 2  answer: "$12.715 billion"
        2 requests, ~7,000 tokens, 4 seconds
        No search: the figure is an XBRL fact, so retrieval never runs.

    "How does Microsoft describe Azure's competitive advantage?"
        round 1  search_filings(ticker=MSFT)
        round 2  answer with 5 citations
        2 requests, ~8,000 tokens, 15 seconds
        One search was enough because reranking put the right passages first.

    "Compare Apple and Microsoft capex, and what each says about why"
        rounds   get_sec_financials x2, then search_filings x6
        5 requests, ~41,000 tokens, 60 seconds
        Expensive because Apple's 10-K contains no capex rationale, so several
        searches went looking for text that does not exist.

## Why cost grows faster than round count
Every round re-sends the whole conversation so far, including every earlier tool
result. So round 6 pays for rounds 1-5 again. Measured:

    3 requests ->  18,435 tokens  ->  6,145 per request
    10 requests -> 174,269 tokens -> 17,427 per request

Cutting a round therefore saves more than shrinking a payload.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.messages import ModelResponse
import httpx
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.toolsets import FilteredToolset, FunctionToolset
from pydantic_ai_harness.compaction import ClearToolResults, TieredCompaction
from pydantic_ai_harness.spend import Budget, SpendLimits
from pydantic_ai_harness.tool_output_limits import (
    Band,
    Spill,
    ToolOutputLimits,
    Truncate,
)

from app.agent import tuning
from app.agent.models import DEFAULT_MODEL, resolve as resolve_model
from app.agent.deps import DocumentAgentDeps
from app.agent.outputs import GroundedAnswer
from app.agent.status import emit_agent_done, emit_agent_start, emit_usage_breakdown
from app.agent.toolgate import include_tool
from app.agent.tools import TOOLS, skill_descriptions
from app.config import settings
from app import telemetry

_INSTRUCTIONS_PATH = Path(__file__).parent / "prompts" / "instructions.md"
INSTRUCTIONS = _INSTRUCTIONS_PATH.read_text(encoding="utf-8")

# One agent per model, built on first use. Construction reads the prompt from
# disk and assembles the toolset, so it is worth caching, but there is no reason
# to limit it to a single model.
_agents: dict[str, Agent[DocumentAgentDeps, GroundedAnswer]] = {}


def _one_unit_per_request(_response: ModelResponse) -> Decimal:
    """Price every model response at 1 notional unit, so "usd" counts requests.

    SpendLimits meters money, and Gemini's free tier costs nothing, so a real
    price would hold the budget at zero forever and gate nothing. Charging a flat
    1 per response turns a `usd` budget into an exact request counter, which is
    the unit the daily free-tier quota is actually expressed in.
    """
    return Decimal(1)


def _capabilities() -> list[object]:
    """## Harness behaviours wrapped around the loop.

    Each one addresses a specific way an observed run wasted tokens.
    """
    return [
        # 1. OVERSIZED TOOL RETURNS
        # A tool return persists in history and is re-sent on every later round,
        # so a big one is paid for repeatedly. search_filings returns a measured
        # 12,030 characters; everything else is under 2,000.
        #
        # Spill is LOSSLESS: the payload is stored, the model gets a preview plus
        # a read-back handle. Removing it was tried and measured worse (132,594 ->
        # 174,269 tokens on the same question), so it stays. Preview is 4,000
        # rather than the original 1,500, which was thin enough to nudge the model
        # into re-searching instead of reading back.
        ToolOutputLimits(
            bands=[
                Band(
                    over=tuning.SPILL_OVER_CHARS,
                    action=Spill(
                        preview_chars=tuning.SPILL_PREVIEW_CHARS,
                        then=Truncate(max_chars=tuning.SPILL_FALLBACK_CHARS),
                    ),
                ),
                Band(
                    over=tuning.TRUNCATE_OVER_CHARS,
                    action=Truncate(max_chars=tuning.TRUNCATE_TO_CHARS),
                ),
            ],
        ),
        # 2. LONG CONVERSATIONS
        # Once history grows past the trigger, blank OLD tool results while
        # keeping the last few exchanges. Costs no LLM call, and anything cleared
        # can be re-fetched by calling the tool again.
        TieredCompaction(
            tiers=[
                ClearToolResults(
                    max_tokens=tuning.COMPACTION_TRIGGER_TOKENS,
                    keep_pairs=tuning.COMPACTION_KEEP_PAIRS,
                )
            ],
            target_tokens=tuning.COMPACTION_TARGET_TOKENS,
        ),
        # 3. SPENDING MORE THAN THE QUOTA ALLOWS
        # Without this, exhausting the daily free tier produces twenty minutes of
        # confusing 429s instead of one clear stop.
        SpendLimits(
            budgets=[
                Budget(
                    usd=Decimal(tuning.GEMINI_FREE_TIER_DAILY_REQUESTS),
                    window="day",
                    name="daily-requests",
                    warn_at=0.8,
                ),
                Budget(
                    tokens=tuning.PER_QUESTION_TOKEN_CEILING,
                    window="run",
                    name="per-question-tokens",
                ),
            ],
            price=_one_unit_per_request,
            # Surfaces the running daily count in the logs, so the free-tier
            # allowance never drains invisibly again.
            on_spend=telemetry.record_spend,
        ),
    ]


def get_document_agent(
    model_name: str | None = None,
) -> Agent[DocumentAgentDeps, GroundedAnswer]:
    """Build (once per model) and return the agent.

    Everything except the model itself is identical across models: same
    instructions, same tools, same capabilities. Only the thing doing the
    reasoning changes.
    """
    name = resolve_model(model_name)
    if name not in _agents:
        # An explicit timeout, because the default is none: a hung socket
        # otherwise blocks the whole turn forever with no error to show. See
        # tuning.GEMINI_REQUEST_TIMEOUT_SECONDS for the run that proved it.
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                tuning.GEMINI_REQUEST_TIMEOUT_SECONDS,
                connect=tuning.GEMINI_CONNECT_TIMEOUT_SECONDS,
            )
        )
        model = GoogleModel(
            name,
            provider=GoogleProvider(
                api_key=settings.gemini_api_key,
                http_client=http_client,
            ),
        )
        _agents[name] = Agent(
            model,
            deps_type=DocumentAgentDeps,
            output_type=GroundedAnswer,
            # Skill DESCRIPTIONS are always in the prompt (one line each) so the
            # model knows what exists; the full method is fetched by load_skill
            # only when needed. Progressive disclosure: cheap to know, paid to read.
            instructions=(
                INSTRUCTIONS
                + "\n\n## Skills (call load_skill to read the full method)\n"
                + skill_descriptions()
            ),
            capabilities=_capabilities(),
            # Tool schemas are re-sent on EVERY round, so an unused tool is not
            # free: the two valuation engines are 1,337 of the 2,707-token schema
            # budget. FilteredToolset hides them unless the question sounds like
            # valuation. See toolgate.py for why the vocabulary errs wide.
            toolsets=[FilteredToolset(FunctionToolset(TOOLS), include_tool)],
        )
    return _agents[name]


def run_document_agent(
    query: str,
    deps: DocumentAgentDeps,
    model_name: str | None = None,
) -> GroundedAnswer:
    """Run one question to completion, emitting UI status events around it."""
    name = resolve_model(model_name)
    emit_agent_start(
        deps,
        model=name,
        request_limit=settings.agent_request_limit,
    )
    result = get_document_agent(name).run_sync(
        query,
        deps=deps,
        usage_limits=UsageLimits(request_limit=settings.agent_request_limit),
    )
    # Where this question's tokens went, for the Tokens panel.
    emit_usage_breakdown(deps, result)

    usage = result.usage
    emit_agent_done(
        deps,
        requests=usage.requests or 0,
        tool_calls=usage.tool_calls or 0,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )
    return result.output
