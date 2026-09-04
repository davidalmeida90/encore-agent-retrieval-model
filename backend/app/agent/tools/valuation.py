"""## Valuation: DCF and trading comparables

The model never does valuation arithmetic. It gathers inputs and calls a
deterministic engine vendored from HKUDS/Vibe-Trading (MIT, see vendor/quantlib),
which owns WACC, the FCFF bridge, both terminal values, and the EV-to-equity
bridge.

## The expensive lesson encoded here
Every failure in this family came from a FREE-TEXT parameter, and prose in a
docstring never fixed one of them. Types did:

    field names guessed wrong  ->  Literal enum
    peers passed as JSON text  ->  list[Peer], a real pydantic model
    units silently mixed       ->  magnitude guard + a billions-only contract

`get_comps_inputs` exists for the same reason: one call assembles every input the
comps engine needs, so the model is never left hunting for market cap or EBITDA
and deriving them inconsistently.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from app.agent.deps import DocumentAgentDeps
from app.config import settings
from app.agent.status import emit_tool_start
from app.agent.tools.skills import (
    has_reasoned,
    loader as skill_loader,
    mark_reasoned,
    skill_was_loaded,
)
from app.agent.tools._guards import (
    _record_failure,
    _safe,
    _too_many_failures,
    _too_many_repeats,
)
from app.agent.tools.fundamentals import CONCEPT_TAGS, _annual_series


class Peer(BaseModel):
    """One comparable company. Money in USD billions; price and EPS per share."""

    name: str
    market_cap: float = Field(description="price_per_share x diluted shares, USD bn")
    price_per_share: float
    total_debt: float
    cash_and_equivalents: float
    ebitda: float
    ebit: float
    revenue: float
    diluted_eps: float
    book_value_of_equity: float

def _sensitivity(inputs: dict, result: dict, discounting_convention: str) -> dict:
    """Value per share across a WACC x terminal-growth grid.

    ## Why this is here and was not
    instructions.md tells the model to structure a valuation as "assumptions,
    output and sensitivity", and the valuation skill calls the sensitivity table
    mandatory. The engine has shipped `sensitivity_grid` all along and nothing
    ever called it, so the table was demanded, promised, and never produced -
    the same shape of gap as `cross_checks`.

    It matters more than a nicety here. This model came in 48-68% below market
    on four large caps, and a single point estimate cannot show whether that is
    a real disagreement or a WACC a quarter-point too high. A grid can: it shows
    the whole neighbourhood at once, and the reader can find their own view in it.

    The axes are centred on the assumptions actually used, plus or minus one and
    two steps, so the chosen cell sits in the middle. Cells where growth is at or
    above WACC come back as null rather than a huge number, because the
    perpetuity is undefined there rather than merely large.
    """
    from vendor.quantlib.valuation.dcf import fcff_bridge, sensitivity_grid

    wacc = float((result.get("wacc_build") or {}).get("wacc") or 0.0)
    growth_input = inputs.get("terminal_growth")
    growth = float(getattr(growth_input, "value", growth_input) or 0.0)
    if not wacc:
        return {}

    waccs = [round(wacc + step, 4) for step in (-0.01, -0.005, 0.0, 0.005, 0.01)]
    growths = [round(growth + step, 4) for step in (-0.01, -0.005, 0.0, 0.005, 0.01)]
    waccs = [w for w in waccs if w > 0]
    growths = [g for g in growths if g >= 0]

    try:
        years = fcff_bridge(
            ebit=inputs["ebit"],
            tax_rate=float(getattr(inputs["tax_rate"], "value",
                                   inputs["tax_rate"])),
            depreciation_amortization=inputs["depreciation_amortization"],
            capex=inputs["capex"],
            delta_nwc=inputs["delta_nwc"],
        )
        grid = sensitivity_grid(
            years,
            wacc_values=waccs,
            growth_values=growths,
            discounting_convention=discounting_convention,
            total_debt=inputs["total_debt"],
            cash_and_equivalents=inputs["cash_and_equivalents"],
            minority_interest=inputs["minority_interest"],
            preferred_equity=inputs["preferred_equity"],
            associate_investments=inputs["associate_investments"],
            diluted_shares=inputs["diluted_shares"],
        )
    except Exception as exc:  # a grid is a nicety; never lose the valuation over it
        return {"error": f"{type(exc).__name__}: {exc}"}

    rows = []
    for wacc_value, row in grid.iterrows():
        rows.append({
            "wacc": f"{float(wacc_value):.2%}",
            **{f"g={float(g):.2%}": (None if row[g] != row[g] else round(float(row[g]), 2))
               for g in grid.columns},
        })
    return {
        "axes": "rows are WACC, columns are terminal growth; cells are value per share",
        "centre": {"wacc": f"{wacc:.2%}", "terminal_growth": f"{growth:.2%}"},
        "grid": rows,
        "null_cells": "growth at or above WACC: the perpetuity is undefined, not large",
        "REPORT_AS": "Render this as a markdown table. It is the sensitivity the "
                     "valuation skill requires, and a point estimate without it is "
                     "not decision-grade.",
    }


def _cross_checks(result: dict, terminal_growth: float, market_cap: float,
                  diluted_shares: float, risk_free_rate: float = 0.0) -> dict:
    """Reciprocal checks on a finished DCF, in the units the analyst argues in.

    ## Why this exists
    instructions.md has always told the model to report `cross_checks`, and the
    engine has never produced them, so the sanity checks the valuation skill
    calls mandatory were never actually computed. A Microsoft DCF came back at
    $148 a share against a $513 price and nothing in the output said why.

    The arithmetic in that run was correct; the assumptions were incoherent, and
    incoherent in a way a single ratio exposes. Terminal capex was $95B against
    $56B of D&A, so the company reinvests 29% of NOPAT forever, while terminal
    growth was set to 2.5%. Growth equals reinvestment times return, so those two
    together imply a return on new capital of 8.6% against a 10.2% WACC: every
    dollar Microsoft invests destroys value, in perpetuity. Nobody would defend
    that in a committee, but nobody was shown it either.

    These are checks, not corrections. A DCF with a flagged ratio is still
    returned; the flag goes next to it so the number is argued with rather than
    quoted.
    """
    bridge = result.get("fcff_bridge") or []
    wacc = ((result.get("wacc_build") or {}).get("wacc")) or 0.0
    if not bridge or not wacc:
        return {}

    final = bridge[-1]
    nopat = float(final.get("nopat") or 0.0)
    da = float(final.get("depreciation_amortization") or 0.0)
    capex = float(final.get("capex") or 0.0)
    nwc = float(final.get("delta_nwc") or 0.0)
    ebit = float(final.get("ebit") or 0.0)
    ev = float(result.get("enterprise_value") or 0.0)

    checks: dict = {}
    warnings: list[str] = []

    # 1. Reinvestment must be consistent with growth: g = reinvestment rate x ROIC.
    reinvestment = capex - da + nwc
    if nopat > 0:
        rate = reinvestment / nopat
        checks["terminal_reinvestment_rate"] = round(rate, 4)
        if rate > 0.01:
            implied_roic = terminal_growth / rate
            checks["implied_roic_on_new_capital"] = round(implied_roic, 4)
            checks["wacc"] = round(wacc, 4)
            if implied_roic < wacc:
                warnings.append(
                    f"Terminal assumptions imply a return on new capital of "
                    f"{implied_roic:.1%} against a WACC of {wacc:.1%}: the company "
                    f"would destroy value on every dollar reinvested, forever. "
                    f"Either terminal growth is too low for this level of capex, "
                    f"or terminal capex is too high for this growth."
                )

    # 2. Capex should converge towards D&A in a steady state.
    if da > 0:
        ratio = capex / da
        checks["terminal_capex_to_da"] = round(ratio, 2)
        if ratio > 1.35:
            warnings.append(
                f"Terminal capex is {ratio:.2f}x D&A. A perpetuity assumes a "
                f"steady state, where the two converge; sustained excess capex "
                f"depresses terminal FCFF without buying the growth that would "
                f"justify it."
            )

    # 3. What multiple is this valuation implicitly paying?
    ebitda = ebit + da
    if ebitda > 0 and ev:
        checks["implied_ev_ebitda"] = round(ev / ebitda, 2)

    # 4. Terminal growth cannot exceed the economy it grows in.
    checks["terminal_growth"] = round(terminal_growth, 4)
    if terminal_growth >= wacc:
        warnings.append("Terminal growth is at or above WACC: the perpetuity does "
                        "not converge and the terminal value is meaningless.")

    # 5. The gap the analyst has to defend.
    if market_cap and diluted_shares:
        market_price = market_cap / diluted_shares
        value = float(result.get("value_per_share") or 0.0)
        checks["market_price_per_share"] = round(market_price, 2)
        if market_price > 0 and value:
            gap = value / market_price - 1.0
            checks["gap_to_market"] = f"{gap:+.0%}"
            if abs(gap) > 0.5:
                warnings.append(
                    f"Valuation is {gap:+.0%} away from the market price. A gap "
                    f"this size is usually an assumption error rather than an "
                    f"opportunity: check the reinvestment and terminal ratios "
                    f"above before presenting it."
                )

    # 6. The reverse question, which is the one an analyst actually asks.
    #
    # Across AAPL, MSFT, WMT and KO this model came in 48-68% below market with
    # internally consistent assumptions, because a 9-10% WACC against 2.5-3%
    # terminal growth implies roughly 14-17x NOPAT while those names trade at
    # 29-43x earnings. Reporting "the market is 60% too high" four times is not
    # analysis. Solving for the growth the market is paying for is: it converts a
    # disagreement about price into a testable statement about expectations.
    #
    # From EV = PV(explicit) + TV x discount_factor and
    # TV = FCFF_final x (1+g) / (wacc - g), solving for g gives
    #     g = (TV x wacc - FCFF) / (TV + FCFF)
    equity = float(result.get("equity_value") or 0.0)
    pv_explicit = float(result.get("pv_of_explicit_fcff") or 0.0)
    tdf = float(result.get("terminal_discount_factor") or 0.0)
    fcff_final = float(final.get("fcff") or 0.0)
    if market_cap and ev and equity and tdf > 0 and fcff_final > 0:
        net_debt = ev - equity
        ev_at_market = market_cap + net_debt
        tv_needed = (ev_at_market - pv_explicit) / tdf
        if tv_needed > 0:
            implied_g = (tv_needed * wacc - fcff_final) / (tv_needed + fcff_final)
            checks["terminal_growth_implied_by_market"] = round(implied_g, 4)
            checks["risk_free_rate"] = round(risk_free_rate, 4)
            if implied_g >= risk_free_rate:
                warnings.append(
                    f"To reach the market price this model needs terminal growth "
                    f"of {implied_g:.1%}, above the {risk_free_rate:.1%} risk-free "
                    f"rate. No company outgrows its economy forever, so either the "
                    f"market is pricing growth that cannot persist, or the WACC "
                    f"here is too high for a business of this quality. Say which "
                    f"you believe rather than presenting the gap as a call."
                )
            else:
                warnings.append(
                    f"The market is priced for terminal growth of about "
                    f"{implied_g:.1%} against the {terminal_growth:.1%} assumed "
                    f"here. That difference, not the price gap, is the thing to "
                    f"defend."
                )

    checks["warnings"] = warnings
    checks["READ_THIS"] = (
        "Report these alongside the valuation. A warning means the inputs are "
        "internally inconsistent, not that the arithmetic failed."
    )
    return checks


async def run_dcf_valuation(
    ctx: RunContext[DocumentAgentDeps],
    ticker: str,
    ebit: str,
    depreciation_amortization: str,
    capex: str,
    delta_nwc: str,
    tax_rate: float,
    risk_free_rate: float,
    beta: float,
    equity_risk_premium: float,
    pretax_cost_of_debt: float,
    market_value_of_equity: float,
    market_value_of_debt: float,
    terminal_growth: float,
    exit_multiple: float,
    total_debt: float,
    cash_and_equivalents: float,
    diluted_shares: float,
    minority_interest: float = 0.0,
    preferred_equity: float = 0.0,
    associate_investments: float = 0.0,
    terminal_value_method: str = "perpetuity_growth",
    discounting_convention: str = "mid_year",
) -> str:
    """Run a full DCF using the vendored engine's own `run_dcf` entry point.

    This wrapper does NO valuation arithmetic. It marshals named inputs into the
    engine's contract and returns what the engine produces. Every intermediate
    figure (WACC build, FCFF bridge, discount factors, both terminal-value
    estimates and their cross-checks, the net-debt bridge) comes from the engine.

    Projection inputs (ebit, depreciation_amortization, capex, delta_nwc) are
    comma-separated per-year values in the SAME unit, equal length, e.g.
    "141.1,149.5,158.5". Rates are decimals (0.045 not 4.5).

    Balance-sheet inputs feed the engine's net-debt bridge. Use
    `get_sec_financials` for total_debt, cash (add marketable_securities_current
    and marketable_securities_noncurrent to cash for cash-rich filers) and
    shares_diluted.

    terminal_value_method: "perpetuity_growth" or "exit_multiple". Both estimates
    are always computed and cross-checked regardless of which is selected.
    discounting_convention: "mid_year" or "year_end".

    A missing or invalid input makes the model NOT RUNNABLE rather than being
    silently defaulted. Report the error to the user; do not invent a substitute.
    """
    emit_tool_start(ctx.deps, "run_dcf_valuation",
                    f"{ticker.upper()} ebit={ebit[:40]} capex={capex[:24]} "
                    f"debt={total_debt} cash={cash_and_equivalents} sh={diluted_shares}")
    from vendor.quantlib.valuation.contracts import Assumption
    from vendor.quantlib.valuation.dcf import run_dcf

    def nums(raw: str) -> list[float]:
        return [float(x.strip()) for x in raw.split(",") if x.strip()]

    # The instructions ask the model to read us-equity-valuation first, and on
    # 2026-08-20 it simply did not: it invented beta 1.1, an ERP of 5.0% and FLAT
    # EBIT for five years, producing $88.33/share for a stock trading above $300
    # and never mentioning the gap. The skill is what supplies the reinvestment
    # rule, the terminal-growth ceiling, the sensitivity table and the mandatory
    # comparison to market price. Prose asked; this insists.
    # Same insistence, one step later. Prose told the model to reason about its
    # assumptions before running the engine, and compliance was intermittent: the
    # runs that skipped it came back with terminal capex at 1.5x to 3.4x D&A and
    # a return on new capital below WACC, which is the incoherence that produced
    # $76 for Microsoft. The reasoning step is where that gets caught, so it is
    # required rather than encouraged.
    if not has_reasoned(ctx, "dcf"):
        return json.dumps({
            "error": "STOP: reason about the assumptions before running the engine.",
            "instruction": (
                "Call reason_about_assumptions with every figure you have sourced. "
                "It reasons over the valuation skill and returns proposals with "
                "their basis. You may overrule any of them; you may not skip it."
            ),
        })

    if not skill_was_loaded(ctx, "us-equity-valuation"):
        return json.dumps({
            "error": "STOP: read the valuation method before running the engine.",
            "instruction": (
                "Call load_skill('us-equity-valuation') first. It defines how to "
                "source beta, the equity risk premium, cost of debt, tax rate and "
                "terminal growth, and it requires reinvestment to be consistent "
                "with the growth you assume. Running without it produces a number "
                "with invented inputs."
            ),
        })

    if (stop := _too_many_failures(ctx, "run_dcf_valuation")):
        return stop
    try:
        ebit_v = nums(ebit)
        _scale_inputs = {"total_debt": total_debt, "cash_and_equivalents": cash_and_equivalents,
                         "diluted_shares": diluted_shares, "market_value_of_equity": market_value_of_equity}
        _ref = max(abs(v) for v in ebit_v) or 1.0
        _bad = {k: v for k, v in _scale_inputs.items() if abs(v) > _ref * 1000}
        if _bad:
            return json.dumps({
                "error": "UNIT MISMATCH — refusing to run",
                "detail": (f"ebit is on the order of {_ref:.4g}, but {list(_bad)} are "
                           f"{ {k: f'{v:.4g}' for k, v in _bad.items()} }. Mixing billions with "
                           "raw dollars silently produces a wrong per-share value."),
                "fix": ("Pass EVERY numeric input in the SAME unit, USD billions. "
                        "get_sec_financials returns value_billions for exactly this purpose."),
            })
        inputs = {
            "ebit": ebit_v,
            "depreciation_amortization": nums(depreciation_amortization),
            "capex": nums(capex),
            "delta_nwc": nums(delta_nwc),
            "tax_rate": tax_rate,
            "risk_free_rate": risk_free_rate,
            "beta": beta,
            "equity_risk_premium": equity_risk_premium,
            "pretax_cost_of_debt": pretax_cost_of_debt,
            "market_value_of_equity": market_value_of_equity,
            "market_value_of_debt": market_value_of_debt,
            "terminal_growth": Assumption(
                name="terminal_growth", value=terminal_growth,
                basis="analyst estimate", source="agent"),
            "exit_multiple": Assumption(
                name="exit_multiple", value=exit_multiple,
                basis="peer EV/EBITDA", source="agent"),
            "total_debt": total_debt,
            "cash_and_equivalents": cash_and_equivalents,
            "diluted_shares": diluted_shares,
            "minority_interest": minority_interest,
            "preferred_equity": preferred_equity,
            "associate_investments": associate_investments,
        }
        result = run_dcf(
            inputs,
            capital_structure_basis="current",
            discounting_convention=discounting_convention,
            terminal_value_method=terminal_value_method,
        )
        safe = _safe(result)
        return json.dumps(
            {"ticker": ticker.upper(), "result": safe,
             "sensitivity": _sensitivity(inputs, safe if isinstance(safe, dict) else {},
                                         discounting_convention),
             "cross_checks": _cross_checks(
                 safe if isinstance(safe, dict) else {},
                 terminal_growth, market_value_of_equity, diluted_shares,
                 risk_free_rate),
             "engine": "quantlib run_dcf (HKUDS/Vibe-Trading, MIT), unmodified"},
            default=str,
        )
    except Exception as e:
        _record_failure(ctx, "run_dcf_valuation")
        return json.dumps({"error": f"{type(e).__name__}: {e}",
                           "note": "engine refused the inputs; do not substitute a guess. "
                                   "Fix the inputs or report the failure; do not retry blindly."})

async def reason_about_assumptions(
    ctx: RunContext[DocumentAgentDeps],
    ticker: Annotated[str, Field(description="Ticker being valued, e.g. 'MSFT'.")],
    sourced_facts: Annotated[str, Field(description=(
        "Every figure you have already SOURCED, with its origin, one per line. "
        "For example: 'FY2026 EBIT 155.237 (get_sec_financials)', 'capex 115.948 "
        "(get_sec_financials)', 'D&A 34.3 (get_xbrl_tag, tag Depreciation)', "
        "'risk-free 4.72% (get_risk_free_rate)', 'shares 7.453'. Do not put "
        "anything here you have not actually retrieved."
    ))],
    question: Annotated[str, Field(description=(
        "What to decide, e.g. 'terminal growth, terminal capex and beta for a "
        "5-year DCF'."
    ))] = "every DCF assumption",
) -> str:
    """Think through DCF assumptions against the valuation skill, before running it.

    Call this AFTER gathering figures and BEFORE `run_dcf_valuation`. It returns
    reasoning and proposed assumptions; you still choose, and you still run the
    engine.

    This is where a valuation is won or lost. The engine is arithmetic and cannot
    be wrong; the assumptions are judgements, and a DCF built on incoherent ones
    produced $148 for Microsoft against a $513 price.
    """
    emit_tool_start(ctx.deps, "reason_about_assumptions", f"{ticker.upper()}: {question[:40]}")

    try:
        method = skill_loader().get_content("us-equity-valuation")
    except Exception:
        method = ""

    # In CAG mode the parent already has the whole filing in context, but this
    # is a SEPARATE model call and does not inherit it. Without this the filing
    # contributes nothing to a valuation, which is the wrong half of the right
    # rule: instructions.md deliberately routes reported FIGURES to XBRL,
    # because tagged facts are exact and prose is not. It never routed the
    # JUDGEMENTS anywhere, and those are exactly what a filing is good for -
    # management's own capex guidance, growth commentary, the concentration and
    # risk language that argues a beta up or down.
    # Loaded here regardless of retrieval mode, and that is the point. A DCF in
    # CAG mode is not merely expensive but impossible: the filing rides in the
    # parent context on EVERY round, and three tool calls came to 589,810 tokens
    # against a 400,000 ceiling. Yet the filing is only wanted by this one step.
    #
    # So it is fetched here instead, for a single sub-model call. A DCF can then
    # run in RAG mode -- cheap parent context, six rounds, no filing re-sent --
    # while its assumptions are still argued from what management actually wrote.
    # ONLY in CAG mode. The filing is 191,000 tokens, and loading it when the
    # user has not asked for CAG is a surprise on their bill and their latency.
    # RAG and agentic already carry their own way of reaching filing text.
    #
    # The cost of this restraint is real and worth knowing: a DCF run in RAG mode
    # sets beta and growth from sector priors rather than from what management
    # actually wrote. In CAG mode it argues them from the filing. That is the
    # trade, and it is the user's to make by choosing the mode.
    filing = ""
    if getattr(ctx.deps, "retrieval_mode", "") == "cag":
        from app.agent.tools.cag import preload

        filing, _n = preload(getattr(ctx.deps, "cag_ticker", "") or ticker,
                             getattr(ctx.deps, "cag_fiscal_year", 0))

    prompt = (
        "You are setting the assumptions for a discounted cash flow valuation of "
        f"{ticker.upper()}. Decide: {question}.\n\n"
        "## The method you must follow\n" + method + "\n\n"
        "## Figures already sourced\n" + sourced_facts + "\n\n"
        "## What to return\n"
        "For EACH assumption: the value, and one sentence of justification tied "
        "to the method above or to a sourced figure. Then state, explicitly:\n"
        "  - the terminal reinvestment rate your capex and D&A imply,\n"
        "  - the return on new capital that rate implies at your terminal growth "
        "(growth = reinvestment rate x return), and\n"
        "  - whether that return is above the WACC you are proposing.\n"
        "If it is below WACC, your assumptions say the company destroys value "
        "forever. Fix them here rather than defending them later. Terminal capex "
        "must converge towards D&A unless you can say what the extra buys.\n"
        "Plain text. No tool calls."
    )
    if filing:
        prompt += (
            "\n\n## The full 10-K\n\n"
            "Numbers come from the tools above, not from this text: tagged XBRL "
            "is exact and prose is not. Use the filing for the JUDGEMENTS "
            "instead. Quote it where management's own words bear on growth, "
            "on the capex trajectory, or on the risk that should move beta, "
            "and say which sentence moved you.\n\n" + filing
        )

    from openai import AsyncOpenAI

    # Follow the model answering this turn, not the .env file. A stale
    # LOCAL_LLM_BASE_URL from a terminated pod otherwise sends this call to a
    # dead endpoint while the agent itself is happily running on Gemini.
    from app.agent.models import choice as model_choice

    try:
        on_local = model_choice(ctx.deps.model_name).provider == "openai_compatible"
    except Exception:
        on_local = False
    if on_local and settings.local_llm_base_url:
        client = AsyncOpenAI(api_key=settings.local_llm_api_key or "none",
                             base_url=settings.local_llm_base_url)
        model_name = settings.local_llm_model
        # Thinking is safe HERE and nowhere else on a self-hosted Qwen. In the
        # agent loop it breaks tool calling outright: vLLM emits the call as XML
        # inside the reasoning block, the tool parser never sees it, the client
        # gets an empty tool_calls array and retries, and the run loops until the
        # repeat guard kills it (vllm#42021, vllm#39056). This call offers no
        # tools and wants prose, so none of that machinery is in the path.
        extra = {"chat_template_kwargs": {"enable_thinking": True}, "think": True}
    else:
        client = AsyncOpenAI(api_key=settings.gemini_api_key,
                             base_url=settings.gemini_openai_base_url)
        model_name = settings.gemini_grounding_model
        extra = {"reasoning_effort": "medium"}

    # Bounded, and on a clock. Thinking is on for this call because setting DCF
    # assumptions is judgement rather than lookup, but an unbounded reasoning
    # chain at ~50 tok/s on a self-hosted 27B is not a slow answer, it is a hung
    # product: a Microsoft valuation sat at this step for over twenty minutes
    # with the GPU busy and nothing to show. A cap turns that into a worse answer
    # instead of no answer, which is the right way round.
    try:
        completion = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=settings.assumptions_max_tokens,
            timeout=settings.assumptions_timeout_seconds,
            extra_body=extra,
        )
        message = completion.choices[0].message
        # With thinking on, vLLM may put the answer in reasoning_content and
        # leave content empty. Take whichever is populated.
        text = (message.content or "").strip()
        reasoning = (getattr(message, "reasoning_content", None) or "").strip()
        answer = text or reasoning
    except Exception as exc:
        _record_failure(ctx, "reason_about_assumptions")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}",
                           "note": "Proceed with the skill yourself; do not skip it."})

    usage = getattr(completion, "usage", None)
    if usage is not None:
        ctx.deps.extract_calls += 1
        ctx.deps.extract_input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        ctx.deps.extract_output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

    mark_reasoned(ctx, "dcf")
    return json.dumps({
        "ticker": ticker.upper(),
        "thinking_enabled": True,
        "model": model_name,
        "proposed_assumptions": answer,
        "USE_AS": ("Proposals, not results. Pass what you accept into "
                   "run_dcf_valuation, and report each one's basis in your answer."),
    })


async def run_comps_valuation(
    ctx: RunContext[DocumentAgentDeps],
    target_name: str,
    target_revenue: float,
    target_ebitda: float,
    target_ebit: float,
    target_total_debt: float,
    target_cash_and_equivalents: float,
    target_diluted_eps: float,
    target_diluted_shares_outstanding: float,
    target_book_value_of_equity: float,
    peers: list[Peer],
    calendarisation_policy: str = "ltm",
    eps_basis: str = "gaap",
) -> str:
    """Trading comparables valuation using the vendored engine's `run_comps`.

    Does NO arithmetic here: it builds the engine's TargetCompany / PeerCompany
    records and returns peer multiples, their distributions, and the implied
    valuations the engine computes.

    ALL money figures in USD billions, consistent with `run_dcf_valuation`.
    EPS and price_per_share stay in dollars per share.

    peers: list of peer records. Get every field from `get_comps_inputs`.
    Use 4 to 6 genuinely comparable peers and say which you chose and why.
    The engine warns when fewer than 3 peers are supplied: a median of one point
    is just that input, not a cross-sectional statistic. Repeat that warning.

    calendarisation_policy: "ltm" or "calendar_year".  eps_basis: "gaap" or "adjusted".
    """
    emit_tool_start(ctx.deps, "run_comps_valuation", f"{target_name} vs peers")
    from vendor.quantlib.valuation.comps import (
        FlowMetricPeriods,
        PeerCompany,
        TargetCompany,
        run_comps,
    )

    # The engine calendarises flow metrics as
    #   LTM = last_full_fiscal_year + current_YTD - prior_YTD
    # Our XBRL tool only exposes ANNUAL (10-K) figures, with no interim YTD data,
    # so YTD legs are set equal and LTM collapses to the last full fiscal year.
    # That is a stated simplification of the INPUTS, not a change to the engine.
    def flow(annual: float, fy_end_month: int = 12) -> FlowMetricPeriods:
        return FlowMetricPeriods(
            fiscal_year_end_month=fy_end_month,
            last_full_fiscal_year=float(annual),
            current_year_to_date=0.0,
            prior_year_to_date=0.0,
            next_full_fiscal_year=float(annual),
        )

    if (stop := _too_many_failures(ctx, "run_comps_valuation")):
        return stop
    try:
        target = TargetCompany(
            name=target_name, revenue=flow(target_revenue), ebitda=flow(target_ebitda),
            ebit=flow(target_ebit), total_debt=target_total_debt,
            cash_and_equivalents=target_cash_and_equivalents,
            diluted_eps=flow(target_diluted_eps),
            diluted_shares_outstanding=target_diluted_shares_outstanding,
            book_value_of_equity=target_book_value_of_equity,
            eps_basis=eps_basis, minority_interest=0.0,
            preferred_stock=0.0, investments_in_associates=0.0,
        )
        peers = [
            PeerCompany(
                name=p.name, market_cap=p.market_cap,
                price_per_share=p.price_per_share,
                total_debt=p.total_debt,
                cash_and_equivalents=p.cash_and_equivalents,
                ebitda=flow(p.ebitda), ebit=flow(p.ebit),
                revenue=flow(p.revenue), diluted_eps=flow(p.diluted_eps),
                book_value_of_equity=p.book_value_of_equity,
                eps_basis=eps_basis, minority_interest=0.0,
                preferred_stock=0.0, investments_in_associates=0.0,
            )
            for p in peers
        ]
        result = run_comps(target, peers, calendarisation_policy=calendarisation_policy)
        return json.dumps(
            {"target": target_name, "result": _safe(result),
             "engine": "quantlib run_comps (HKUDS/Vibe-Trading, MIT), unmodified"},
            default=str,
        )
    except Exception as e:
        _record_failure(ctx, "run_comps_valuation")
        return json.dumps({"error": f"{type(e).__name__}: {e}",
                           "note": "engine refused the inputs; do not substitute a guess. "
                                   "Fix the inputs or report the failure; do not retry blindly."})

async def get_comps_inputs(
    ctx: RunContext[DocumentAgentDeps],
    tickers: str,
) -> str:
    """Assemble everything `run_comps_valuation` needs, for one or more tickers.

    Call this FIRST for any comparables work. It removes all guesswork: the
    output fields map one-to-one onto the target and peer arguments.

    Sourcing, per ticker:
      revenue, ebit, total_debt, cash, shares_diluted, book_value_of_equity,
      net_income          -> vendored PIT-safe fundamentals loader (TTM)
      depreciation_amortization -> SEC XBRL tag DepreciationDepletionAndAmortization
      price_per_share     -> vendored price loader (market fallback chain)

    Two figures are DERIVED here, because neither the vendored fundamentals
    schema nor its tools expose them:
      ebitda      = ebit + depreciation_amortization
      market_cap  = price_per_share x shares_diluted
    Both are labelled `derived` in the output. State that in your answer.

    All money figures in USD billions; price and EPS in dollars per share.
    """
    tl = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    emit_tool_start(ctx.deps, "get_comps_inputs", ",".join(tl))

    import datetime as _dt

    from vendor.loaders.fundamentals_loader import load_fundamental_panel
    from vendor.loaders.registry import FALLBACK_CHAINS, get_loader_cls_with_fallback
    from vendor.loaders.sec_edgar_client import cik_for, get_company_facts

    FIELDS = ["revenue", "operating_income", "total_debt", "cash",
              "shares_diluted", "total_equity", "net_income"]
    end = _dt.date.today()
    start = end.replace(year=end.year - 2)
    out: dict[str, Any] = {
        "UNITS": "money in USD billions; price_per_share and diluted_eps in USD",
        "note": "ebitda and market_cap are DERIVED (see tool docs), not filed figures",
    }

    for tick in tl:
        row: dict[str, Any] = {"name": tick}
        try:
            panel = load_fundamental_panel([tick], FIELDS, start.isoformat(),
                                           end.isoformat(), freq="ttm", pit=True)
            for f in FIELDS:
                fr = panel.get(f)
                if fr is None or getattr(fr, "empty", True):
                    row[f] = None
                    continue
                col = fr[tick].dropna()
                row[f] = round(float(col.iloc[-1]) / 1e9, 4) if not col.empty else None
        except Exception as e:
            out[tick] = {"error": f"fundamentals: {type(e).__name__}: {e}"}
            continue

        # D&A is not in the vendored schema; read the raw us-gaap tag.
        try:
            facts = get_company_facts(cik_for(tick))
            da = _annual_series(facts, ["DepreciationDepletionAndAmortization",
                                        "DepreciationAmortizationAndAccretionNet"], 1)
            if da:
                row["depreciation_amortization"] = da[0]["value_billions"]
                row["da_basis"] = "combined us-gaap tag"
            else:
                # Some filers (Microsoft) report the legs separately rather than
                # a combined tag, so D&A is summed from its components.
                dep = _annual_series(facts, ["Depreciation"], 1)
                amo = _annual_series(facts, ["AmortizationOfIntangibleAssets"], 1)
                parts = [x[0]["value_billions"] for x in (dep, amo) if x]
                row["depreciation_amortization"] = round(sum(parts), 4) if parts else None
                row["da_basis"] = ("summed: Depreciation + AmortizationOfIntangibleAssets"
                                   if parts else "not reported")
        except Exception:
            row["depreciation_amortization"] = None

        # price via the registry fallback chain
        px = None
        for source in FALLBACK_CHAINS.get("us_equity", ["yfinance"]):
            try:
                cls = get_loader_cls_with_fallback(source)
                loader = cls() if isinstance(cls, type) else cls
                fr = (loader.fetch([tick], (end - _dt.timedelta(days=10)).isoformat(),
                                   end.isoformat(), interval="1D") or {}).get(tick)
                if fr is not None and not getattr(fr, "empty", True):
                    col = next((c for c in ("close", "Close") if c in fr.columns), None)
                    if col:
                        px = round(float(fr[col].dropna().iloc[-1]), 2)
                        break
            except Exception:
                continue
        row["price_per_share"] = px

        ebit, da_v, sh, ni = (row.get("operating_income"), row.get("depreciation_amortization"),
                              row.get("shares_diluted"), row.get("net_income"))
        row["ebit"] = ebit
        row["ebitda_DERIVED"] = round(ebit + da_v, 4) if (ebit is not None and da_v is not None) else None
        row["market_cap_DERIVED"] = round(px * sh, 2) if (px is not None and sh) else None
        row["diluted_eps_DERIVED"] = round(ni / sh, 2) if (ni is not None and sh) else None
        row["cash_and_equivalents"] = row.get("cash")
        row["book_value_of_equity"] = row.get("total_equity")
        out[tick] = row

    return json.dumps(out, default=str)

async def project_three_statement_model(
    ctx: RunContext[DocumentAgentDeps],
    company: str,
    opening_revenue: float,
    opening_cash: float,
    opening_net_working_capital: float,
    opening_ppe: float,
    opening_revolver_balance: float,
    opening_paid_in_capital: float,
    opening_retained_earnings: float,
    revenue_growth: str,
    gross_margin: str,
    opex_pct_revenue: str,
    capex_pct_revenue: str,
    nwc_pct_revenue: str,
    tax_rate: str,
    dividend_payout_ratio: str,
    depreciation_amortization: str,
    interest_rate: str,
    minimum_cash: str,
) -> str:
    """Build a linked 3-statement projection with the engine's `project_three_statement`.

    Income statement, balance sheet and cash flow are linked, the revolver plugs
    the cash shortfall, and the engine solves the interest/cash circularity and
    checks that the balance sheet balances. It refuses rather than defaulting.

    Opening figures: USD billions. Driver arguments are comma-separated per-year
    values of EQUAL length, one entry per projection year:
      revenue_growth, gross_margin, opex_pct_revenue, capex_pct_revenue,
      nwc_pct_revenue, tax_rate, dividend_payout_ratio, interest_rate
        -> decimals (0.05 not 5)
      depreciation_amortization, minimum_cash
        -> USD billions

    State the basis for every driver. Growth is not free: capex and working
    capital must be consistent with the revenue growth you assume.
    """
    emit_tool_start(ctx.deps, "project_three_statement_model", company)
    from vendor.quantlib.valuation.threestatement import project_three_statement

    def nums(raw: str) -> list[float]:
        return [float(x.strip()) for x in raw.split(",") if x.strip()]

    if (stop := _too_many_failures(ctx, "project_three_statement_model")):
        return stop
    try:
        opening = {
            "revenue": opening_revenue, "cash": opening_cash,
            "net_working_capital": opening_net_working_capital,
            "ppe": opening_ppe, "revolver_balance": opening_revolver_balance,
            "paid_in_capital": opening_paid_in_capital,
            "retained_earnings": opening_retained_earnings,
        }
        drivers = {
            "revenue_growth": nums(revenue_growth), "gross_margin": nums(gross_margin),
            "opex_pct_revenue": nums(opex_pct_revenue),
            "capex_pct_revenue": nums(capex_pct_revenue),
            "nwc_pct_revenue": nums(nwc_pct_revenue), "tax_rate": nums(tax_rate),
            "dividend_payout_ratio": nums(dividend_payout_ratio),
            "depreciation_amortization": nums(depreciation_amortization),
            "interest_rate": nums(interest_rate), "minimum_cash": nums(minimum_cash),
        }
        proj = project_three_statement(opening, drivers)
        return json.dumps(
            {"company": company, "projection": _safe(proj),
             "engine": "quantlib project_three_statement (HKUDS/Vibe-Trading, MIT), unmodified"},
            default=str,
        )
    except Exception as e:
        _record_failure(ctx, "project_three_statement_model")
        return json.dumps({"error": f"{type(e).__name__}: {e}",
                           "note": "engine refused the inputs; do not substitute a guess. "
                                   "Fix the inputs or report the failure; do not retry blindly."})

async def get_three_statement_opening(
    ctx: RunContext[DocumentAgentDeps],
    ticker: str,
    balance_mode: Literal["strict", "plug_equity"] = "strict",
) -> str:
    """Assemble the opening balance sheet `project_three_statement_model` requires.

    balance_mode:
      "strict"      report the residual and let the engine refuse. Correct when
                    you want to know the sheet does not fit the model's shape.
      "plug_equity" fold marketable securities into cash, long-term debt into
                    the revolver, and put any remainder into paid_in_capital as
                    an explicit balancing plug. Use ONLY for an illustrative
                    projection, and say in your answer that the opening position
                    was adjusted to fit a stylised 7-line balance sheet.

    Call this FIRST for any three-statement work. Output maps one-to-one onto the
    tool's `opening_*` arguments, so no guessing is needed.

    Sourcing:
      revenue, cash              -> vendored PIT-safe fundamentals loader (annual)
      ppe                        -> XBRL PropertyPlantAndEquipmentNet
      retained_earnings          -> XBRL RetainedEarningsAccumulatedDeficit
      paid_in_capital            -> XBRL CommonStocksIncludingAdditionalPaidInCapital
                                    (falls back to AdditionalPaidInCapital)
      net_working_capital        -> DERIVED: AssetsCurrent - LiabilitiesCurrent
      revolver_balance           -> DERIVED: 0.0 unless the filer reports a revolver;
                                    most large caps do not draw one

    All figures USD billions. The engine checks that the sheet balances and
    refuses if it does not, so report any imbalance rather than forcing it.
    """
    tick = ticker.strip().upper()
    emit_tool_start(ctx.deps, "get_three_statement_opening", tick)

    import datetime as _dt

    from vendor.loaders.fundamentals_loader import load_fundamental_panel
    from vendor.loaders.sec_edgar_client import cik_for, get_company_facts

    out: dict[str, Any] = {"ticker": tick, "UNITS": "USD billions",
                           "note": "net_working_capital and revolver_balance are DERIVED"}
    try:
        end = _dt.date.today()
        panel = load_fundamental_panel([tick], ["revenue", "cash"],
                                       end.replace(year=end.year - 2).isoformat(),
                                       end.isoformat(), freq="annual", pit=True)
        for f in ("revenue", "cash"):
            fr = panel.get(f)
            col = fr[tick].dropna() if fr is not None and not getattr(fr, "empty", True) else None
            out[f"opening_{f}"] = round(float(col.iloc[-1]) / 1e9, 4) if col is not None and not col.empty else None
    except Exception as e:
        out["fundamentals_error"] = f"{type(e).__name__}: {e}"

    try:
        facts = get_company_facts(cik_for(tick))

        def one(tags: list[str]) -> float | None:
            r = _annual_series(facts, tags, 1)
            return r[0]["value_billions"] if r else None

        ppe = one(["PropertyPlantAndEquipmentNet"])
        re_ = one(["RetainedEarningsAccumulatedDeficit"])
        pic = one(["CommonStocksIncludingAdditionalPaidInCapital", "AdditionalPaidInCapital"])
        ca, cl = one(["AssetsCurrent"]), one(["LiabilitiesCurrent"])
        out["opening_ppe"] = ppe
        out["opening_retained_earnings"] = re_
        out["opening_paid_in_capital"] = pic
        out["opening_net_working_capital"] = round(ca - cl, 4) if (ca is not None and cl is not None) else None
        out["_components"] = {"AssetsCurrent": ca, "LiabilitiesCurrent": cl}
        out["opening_revolver_balance"] = 0.0

        if balance_mode == "plug_equity":
            # A real filer does not fit cash + NWC + PPE = revolver + equity.
            # Fold the big unmodelled items in, then plug the remainder into
            # paid_in_capital and SAY SO. This is an illustrative adjustment.
            msc = one(["MarketableSecuritiesCurrent"]) or 0.0
            msn = one(["MarketableSecuritiesNoncurrent"]) or 0.0
            ltd = one(["LongTermDebtNoncurrent", "LongTermDebt"]) or 0.0
            out["opening_cash"] = round((out.get("opening_cash") or 0.0) + msc + msn, 4)
            out["opening_revolver_balance"] = round(ltd, 4)
            assets = (out["opening_cash"] + (out.get("opening_net_working_capital") or 0.0)
                      + (out.get("opening_ppe") or 0.0))
            liab_eq = (out["opening_revolver_balance"] + (out.get("opening_paid_in_capital") or 0.0)
                       + (out.get("opening_retained_earnings") or 0.0))
            plug = round(assets - liab_eq, 4)
            out["opening_paid_in_capital"] = round((out.get("opening_paid_in_capital") or 0.0) + plug, 4)
            out["balancing_plug_applied"] = plug
            out["balance_mode"] = "plug_equity"
            out["disclosure"] = (
                "Opening position adjusted to a stylised 7-line balance sheet: "
                f"marketable securities ({round(msc + msn, 3)}) folded into cash, "
                f"long-term debt ({round(ltd, 3)}) treated as revolver, and {plug} "
                "plugged into paid_in_capital. Report this adjustment in your answer.")
        else:
            assets = ((out.get("opening_cash") or 0.0) + (out.get("opening_net_working_capital") or 0.0)
                      + (out.get("opening_ppe") or 0.0))
            liab_eq = ((out.get("opening_paid_in_capital") or 0.0)
                       + (out.get("opening_retained_earnings") or 0.0))
            out["balance_mode"] = "strict"
            out["residual_assets_minus_liab_equity"] = round(assets - liab_eq, 4)
            out["warning"] = (
                "A real filer's balance sheet does not fit this engine's 7 slots "
                "(no place for marketable securities, long-term debt or AOCI). "
                "The engine will refuse. Either report that, or call again with "
                "balance_mode='plug_equity' and disclose the adjustment.")
    except Exception as e:
        out["xbrl_error"] = f"{type(e).__name__}: {e}"
    return json.dumps(out, default=str)
