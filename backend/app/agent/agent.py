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

from pydantic_ai import Agent, RunContext, UsageLimits
from pydantic_ai.messages import ModelResponse, ToolReturnPart
from typing import Any

import httpx
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.toolsets import FilteredToolset, FunctionToolset
from pydantic_ai_harness.compaction import ClearToolResults, TieredCompaction
from pydantic_ai_harness.spend import Budget, SpendLimits
from pydantic_ai_harness.system_reminders import SystemReminders
from pydantic_ai_harness.tool_output_limits import (
    Band,
    Spill,
    ToolOutputLimits,
    Truncate,
)

from app.agent import tuning
from app.agent.models import DEFAULT_MODEL, choice as model_choice, resolve as resolve_model
from app.agent.deps import DocumentAgentDeps
from app.agent.toolgate import VALUATION_WORDS
from app.agent.outputs import GroundedAnswer
from app.agent.status import emit_agent_done, emit_agent_start, emit_usage_breakdown
from app.agent.toolgate import include_tool
from app.agent.tools import TOOLS, skill_descriptions
from app.retrieval import modes
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


def _answer_with_what_you_have(ctx: Any) -> str | None:
    """Remind a model that has gathered plenty that it is allowed to stop.

    Not a loop guard. The guards in tools/_guards.py catch a tool called twice
    with the same arguments; this catches the opposite, a model whose every call
    is distinct and reasonable and which simply never decides it has enough.
    Measured: thirteen distinct calls to answer a question two calls settle.
    """
    gathered = sum(
        1
        for message in ctx.messages
        for part in getattr(message, "parts", ())
        if isinstance(part, ToolReturnPart)
    )
    if gathered < tuning.REMIND_TO_ANSWER_AFTER_TOOL_RESULTS:
        return None
    return (
        f"You have {gathered} tool results already. That is normally enough. "
        "Produce your final answer now from what you have. Do not search or read "
        "further to double-check a figure a tool has already returned: if "
        "something looks uncertain, say so in the answer and explain why, which "
        "is more useful than another lookup."
    )


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
            # CAG is exempt, and the exemption IS the mode. `load_filing` returns
            # a whole 10-K, roughly 594,000 characters, and the point of the mode
            # is that all of it sits in context so nothing can be lost to a
            # ranker. Spilling it hands the model a 4,000-character preview and a
            # read-back handle, which is retrieval again wearing a different hat:
            # measured, the first CAG question arrived at 58,976 input tokens
            # instead of ~156,000 and took five rounds instead of two.
            #
            # This is safe only because the mode enforces one filing per turn.
            # An empty band list here with two filings loaded would blow any
            # context window.
            per_tool={"load_filing": []},
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
                    tokens=max(tuning.PER_QUESTION_TOKEN_CEILING,
                               tuning.CAG_QUESTION_TOKEN_CEILING),
                    window="run",
                    name="per-question-tokens",
                ),
            ],
            price=_one_unit_per_request,
            # Surfaces the running daily count in the logs, so the free-tier
            # allowance never drains invisibly again.
            on_spend=telemetry.record_spend,
        ),
        # 4. NEVER DECIDING IT HAS ENOUGH
        # Distinct from the loop guards: every call here is reasonable on its
        # own, and only the pattern is wrong. Reminders ride at the tail behind
        # a CachePoint, so the cached prefix is untouched.
        SystemReminders(dynamic_reminders=[_answer_with_what_you_have]),
    ]


def _build_model(entry) -> Model:
    """One registry entry in, one configured model out.

    Both branches set an explicit timeout, because the default is none: a hung
    socket otherwise blocks the whole turn forever with no error to show. See
    tuning.GEMINI_REQUEST_TIMEOUT_SECONDS for the run that proved it.

    Nothing else differs. The same TOOLS list, the same GroundedAnswer schema
    and the same capabilities apply to both; PydanticAI translates the tool
    definitions into whichever wire format the provider speaks.
    """
    if entry.provider == "openai_compatible":
        # Deliberately imported here. The openai extra is only needed when a
        # self-hosted model is actually selected, so a Gemini-only deployment
        # does not have to carry it.
        from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIChatModel(
            settings.local_llm_model or entry.served_name or entry.id,
            provider=OpenAIProvider(
                base_url=settings.local_llm_base_url,
                # vLLM ignores the key unless started with --api-key, but the
                # OpenAI client refuses to send a request without one.
                api_key=settings.local_llm_api_key or "not-needed",
                http_client=httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        tuning.LOCAL_REQUEST_TIMEOUT_SECONDS,
                        connect=tuning.LOCAL_CONNECT_TIMEOUT_SECONDS,
                    )
                ),
            ),
            # vLLM reads chat_template_kwargs and hands them to the model's own
            # Jinja template, which is where Qwen's thinking switch lives. It is
            # not an OpenAI field, so it rides in extra_body.
            settings=OpenAIChatModelSettings(
                # Not a style preference: see tuning.LOCAL_PARALLEL_TOOL_CALLS
                # for the 21-tool-calls-in-2-requests run that set it.
                parallel_tool_calls=tuning.LOCAL_PARALLEL_TOOL_CALLS,
                # vLLM hands chat_template_kwargs to the model's own Jinja
                # template, which is where Qwen's thinking switch lives. Only
                # sent when the model actually has one.
                # Two providers, two spellings of the same switch. vLLM passes
                # chat_template_kwargs into the model's Jinja template; Ollama
                # takes a top-level `think`. Sending both is harmless, since each
                # ignores the other's field, and it means one setting controls
                # thinking whichever server is behind the URL.
                extra_body=(
                    {
                        "chat_template_kwargs": {
                            "enable_thinking": tuning.LOCAL_THINKING_ENABLED
                        },
                        "think": tuning.LOCAL_THINKING_ENABLED,
                    }
                    if tuning.LOCAL_THINKING_ENABLED is not None
                    else {}
                ),
            ),
        )

    return GoogleModel(
        entry.id,
        provider=GoogleProvider(
            api_key=settings.gemini_api_key,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(
                    tuning.GEMINI_REQUEST_TIMEOUT_SECONDS,
                    connect=tuning.GEMINI_CONNECT_TIMEOUT_SECONDS,
                )
            ),
        ),
    )


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
        entry = model_choice(name)
        model = _build_model(entry)

        # How the FINAL answer is requested, which is not the same question as
        # how tools are called.
        #
        # By default PydanticAI asks for GroundedAnswer as a `final_result` tool
        # call. Gemini obliges. A self-hosted Qwen traced on 2026-08-29 did not:
        # it called its real tool correctly on the first round, then answered in
        # prose, took a json_invalid retry, and emitted the JSON as plain TEXT.
        # Three rounds and 5,194 output tokens to deliver an answer it already
        # had after round one.
        #
        # Structured output stays on the default tool-based path, and the two
        # alternatives were both tried against a self-hosted Qwen on
        # 2026-08-29 and both failed:
        #
        #   PromptedOutput  injects the schema as a system message, and that
        #                   model's chat template raises "System message must be
        #                   at the beginning"
        #   NativeOutput    asks for a JSON-schema response_format, which
        #                   Ollama's OpenAI endpoint rejects with a 400 when
        #                   tools are also present
        #
        # The default costs that model two extra rounds (it answers in prose,
        # takes a json_invalid retry, then emits JSON as text) but it does
        # finish. Worth revisiting when either server grows the support.
        output: Any = GroundedAnswer
        _agents[name] = Agent(
            model,
            deps_type=DocumentAgentDeps,
            output_type=output,
            # Skill DESCRIPTIONS are always in the prompt (one line each) so the
            # model knows what exists; the full method is fetched by load_skill
            # only when needed. Progressive disclosure: cheap to know, paid to read.
            capabilities=_capabilities(),
            # Tool schemas are re-sent on EVERY round, so an unused tool is not
            # free: the two valuation engines are 1,337 of the 2,707-token schema
            # budget. FilteredToolset hides them unless the question sounds like
            # valuation. See toolgate.py for why the vocabulary errs wide.
            toolsets=[FilteredToolset(FunctionToolset(TOOLS), include_tool)],
        )

        # ONE instructions source, deliberately. Registering the mode block as a
        # second source is the obvious way to write this, and it breaks the local
        # model outright: pydantic-ai emits each source as its own system
        # message, and Qwen3.8's chat template rejects anything but a single
        # system message at the front ("System message must be at the
        # beginning"). Gemini accepts several, so the fault only shows on the pod.
        #
        # It has to be dynamic rather than static because agents are cached by
        # model name while the mode is chosen per question: the RAG text asserts
        # a two-company corpus, and a model reading it in agentic mode refuses
        # questions about every other filer even though it can reach them.
        @_agents[name].instructions
        def _instructions(ctx: RunContext[DocumentAgentDeps]) -> str:
            skills = skill_descriptions()
            mode = getattr(ctx.deps, "retrieval_mode", None)
            # CAG puts the whole filing HERE rather than in a tool result, so it
            # sits before the question and forms a stable prefix the KV cache can
            # reuse across every question about that company. That ordering IS
            # the mode; as a tool result it would be re-prefilled each turn.
            filing = ""
            if modes.resolve(mode) == "cag":
                from app.agent.tools.cag import preload

                filing, _count = preload(
                    getattr(ctx.deps, "cag_ticker", ""),
                    getattr(ctx.deps, "cag_fiscal_year", 0),
                    registry=ctx.deps.registry,
                )
                if not filing:
                    filing = ("## No filing is loaded\n\nCAG needs a company "
                            "chosen before the question. Say so and ask which "
                            "company to load; do not answer from memory.")
            return "\n\n".join(
                part
                for part in (
                    INSTRUCTIONS,
                    modes.instructions(mode),
                    "## Skills (call load_skill to read the full method)\n" + skills,
                    filing,
                )
                if part
            )

    return _agents[name]


def _thinking_settings(name: str, thinking: bool | None):
    """Model settings that turn reasoning on or off for one run.

    The switch is spelled differently by each server, and sending the wrong one
    is not harmless: a kwarg a chat template never reads breaks the request.

      local vLLM   chat_template_kwargs.enable_thinking, plus Ollama's `think`
      Gemini       reasoning_effort

    None returns None, which leaves the model exactly as configured.
    """
    if thinking is None:
        return None
    entry = model_choice(name)
    if entry.provider == "openai_compatible":
        from pydantic_ai.models.openai import OpenAIChatModelSettings

        return OpenAIChatModelSettings(
            extra_body={
                "chat_template_kwargs": {"enable_thinking": thinking},
                "think": thinking,
            }
        )
    from pydantic_ai.models.google import GoogleModelSettings

    # Gemini has no on/off switch, only an effort dial. "none" is the honest
    # translation of off; "medium" matches what the grounding judge was measured
    # to need before it stopped rejecting valid citations.
    return GoogleModelSettings(
        google_thinking_config={"thinking_budget": 0} if not thinking else {}
    )


def _thinking_for(name: str, query: str, deps: DocumentAgentDeps) -> bool | None:
    """Whether the agent LOOP should think. Almost never, and here is why.

    The loop does routing and retrieval: pick a tool, read what came back, pick
    the next one. That is lookup work, and thinking makes it worse rather than
    slower-but-better. Measured on the local 27B it broke the loop outright,
    because vLLM emits the tool call as XML inside the reasoning block, the tool
    parser never sees it, the client gets an empty tool_calls array and retries,
    and the run repeats identical calls until the repeat guard aborts it
    (vllm#42021, vllm#39056). On Gemini it merely multiplies output tokens on
    every round of a question whose hard part is not in the loop at all.

    Judgement in this system is deliberately not in the loop. It sits in two
    calls that offer no tools, so neither can hit that failure:

        reason_about_assumptions   thinking ON   picks the DCF assumptions
        the grounding judge        medium        decides if an excerpt supports a claim

    So this returns whatever the user chose in the UI, and otherwise nothing.
    `query` is unused, kept because the signature reads better with the question
    in view of anyone tempted to switch on thinking by keyword again.
    """
    return deps.thinking


def run_document_agent(
    query: str,
    deps: DocumentAgentDeps,
    model_name: str | None = None,
) -> GroundedAnswer:
    """Run one question to completion, emitting UI status events around it."""
    name = resolve_model(model_name)
    deps.model_name = name
    emit_agent_start(
        deps,
        model=name,
        request_limit=settings.agent_request_limit,
    )
    result = get_document_agent(name).run_sync(
        query,
        deps=deps,
        usage_limits=UsageLimits(request_limit=settings.agent_request_limit),
        # Per run, not per agent: agents are cached by model name, so baking the
        # switch in at construction would make the first question of a session
        # decide it for every later one.
        model_settings=_thinking_settings(name, _thinking_for(name, query, deps)),
    )
    # What the tools actually returned this turn. The grounding gate lets an
    # answer cite nothing on the grounds that its figures came from a tool; this
    # is what lets it check that claim instead of assuming it.
    try:
        deps.tool_outputs = "\n".join(
            f"{part.tool_name}: {part.content}"
            for message in result.all_messages()
            for part in getattr(message, "parts", [])
            if isinstance(part, ToolReturnPart)
        )
    except Exception:  # never fail a good answer over bookkeeping
        deps.tool_outputs = ""

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
