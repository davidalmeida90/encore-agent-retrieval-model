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
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from app.agent.deps import DocumentAgentDeps
from app.agent.status import emit_tool_start
from app.agent.tools.skills import skill_was_loaded
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
        return json.dumps(
            {"ticker": ticker.upper(), "result": _safe(result),
             "engine": "quantlib run_dcf (HKUDS/Vibe-Trading, MIT), unmodified"},
            default=str,
        )
    except Exception as e:
        _record_failure(ctx, "run_dcf_valuation")
        return json.dumps({"error": f"{type(e).__name__}: {e}",
                           "note": "engine refused the inputs; do not substitute a guess. "
                                   "Fix the inputs or report the failure; do not retry blindly."})

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
