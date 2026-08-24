"""## Company fundamentals, straight from SEC XBRL

Answers "what is the number" for anything a company files with the SEC. Nothing
here reads prose: every figure is a tagged XBRL fact, because letting a model do
arithmetic over retrieved text is the documented FinanceBench failure mode.

Two tools, deliberately layered:

* `get_sec_financials` - the canonical schema (revenue, capex, ebit, ...), point-
  in-time safe, values labelled with the date they became public.
* `get_xbrl_tag` - the escape hatch, any of the ~500 raw us-gaap tags a large
  filer reports, for anything the schema does not name.

## Examples

    get_sec_financials("AAPL", ["capex"], freq="annual")
      -> latest_value_billions: 12.715
         history: [..., {fiscal_year: 2025, period_end: "2025-09-27",
                         value_billions: 12.715, first_visible: "2025-10-31"}]

    get_sec_financials("MSFT", ["capex"], freq="annual")
      -> 115.948, fiscal_year 2026, first_visible 2026-07-29
         ^ XBRL runs AHEAD of the indexed filings. Microsoft filed FY2026 three
           weeks ago; the corpus holds text only to FY2025. Each value carries
           its real fiscal_year for exactly this reason: without it the model
           labelled 115.948 as "June 30, 2025", taking the number from here and
           the period from whichever filing it happened to be quoting.

    get_xbrl_tag("MSFT", "PaymentsToAcquirePropertyPlantAndEquipment")
      -> the same figure, straight from the raw tag, for fields the 12-field
         schema does not name.

## freq defaults to "annual"
It used to default to "ttm", which cost a wasted round on nearly every question:
the model asked a fiscal-year question, got a trailing-twelve-month number, and
had to call again. Six calls became two when the default changed.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Literal

from pydantic_ai import RunContext

from app.agent.deps import DocumentAgentDeps
from app.agent.status import emit_tool_start
from app.agent.tools._guards import _record_failure, _too_many_repeats
from vendor.loaders._fundamental_schema import SEC_CONCEPT_MAP


_EXTRA_TAGS: dict[str, list[str]] = {
    "marketable_securities_current": [
        "MarketableSecuritiesCurrent",
        "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
    ],
    "marketable_securities_noncurrent": [
        "MarketableSecuritiesNoncurrent",
        "AvailableForSaleSecuritiesDebtSecuritiesNoncurrent",
    ],
    "rnd": ["ResearchAndDevelopmentExpense"],
    "ebitda_proxy_da": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
    ],
}

CONCEPT_TAGS: dict[str, list[str]] = {**SEC_CONCEPT_MAP, **_EXTRA_TAGS}

@lru_cache(maxsize=16)
def _facts_for(ticker: str) -> dict:
    from vendor.loaders.sec_edgar_client import cik_for, get_company_facts
    cik = cik_for(ticker)
    return get_company_facts(cik) if cik else {}


def _annual_series(facts: dict, tags: list[str], years: int) -> list[dict[str, Any]]:
    """Pull annual (FY, 10-K) values for the first tag that has them."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        node = gaap.get(tag)
        if not node:
            continue
        rows: dict[str, dict[str, Any]] = {}
        for unit_rows in node.get("units", {}).values():
            for r in unit_rows:
                if r.get("fp") != "FY" or r.get("form") not in ("10-K", "10-K/A"):
                    continue
                fy = r.get("fy")
                if fy is None:
                    continue
                # later filings supersede earlier restatements of the same year
                prev = rows.get(str(fy))
                if prev is None or str(r.get("end", "")) >= str(prev["end"]):
                    rows[str(fy)] = {
                        "fy": fy,
                        "end": r.get("end"),
                        "val": r.get("val"),
                        "unit": "USD",
                    }
        if rows:
            out = sorted(rows.values(), key=lambda x: x["fy"], reverse=True)[:years]
            return [{"fiscal_year": r["fy"],
                     "value_billions": round(r["val"] / 1e9, 4) if isinstance(r["val"], (int, float)) else None,
                     "value_raw": r["val"],
                     "period_end": r["end"]}
                    for r in out]
    return []

async def get_sec_financials(
    ctx: RunContext[DocumentAgentDeps],
    ticker: str,
    fields: list[
        Literal[
            "revenue", "cogs", "gross_profit", "operating_income", "net_income",
            "total_assets", "total_equity", "total_debt", "cash", "cfo",
            "capex", "shares_diluted",
        ]
    ],
    # Defaults to annual: questions here ask "in fiscal 2025" or "most recent
    # fiscal year" far more often than they ask for trailing twelve months, and
    # the old ttm default cost a wasted round on nearly every numeric question
    # (call once, get TTM, call again with freq="annual"). Each of those rounds
    # re-sends the whole history, search results included. TTM is now the
    # explicit choice rather than the accidental one.
    freq: Literal["annual", "ttm"] = "annual",
    years_back: int = 4,
) -> str:
    """US fundamentals via the vendored PIT-safe loader (`load_fundamental_panel`).

    Point-in-time safe: a value only becomes visible on its SEC **filed** date,
    never on period end, and the original as-reported figure is kept rather than
    a later restatement. Period spans are validated so an annual field can never
    silently return a single quarter.

    fields: comma-separated, from the loader's canonical schema:
      revenue, cogs, gross_profit, operating_income, net_income,
      total_assets, total_equity, total_debt, cash, cfo, capex, shares_diluted

    freq: "ttm" (trailing twelve months, the basis comps expect) or "annual".

    Returns the latest observation per field in USD billions, plus a short
    history. For anything outside this schema (balance-sheet detail such as
    PropertyPlantAndEquipmentNet or AssetsCurrent) use `get_xbrl_tag`.
    """
    if (stop := _too_many_repeats(ctx, "get_sec_financials",
                                  {"t": ticker, "f": sorted(fields), "q": freq,
                                   "y": years_back})):
        return stop
    emit_tool_start(ctx.deps, "get_sec_financials", f"{ticker.upper()} {freq}: {','.join(fields)}")
    import datetime as _dt

    from vendor.loaders.fundamentals_loader import load_fundamental_panel

    tick = ticker.strip().upper()
    wanted = [f.strip().lower() for f in fields]
    end = _dt.date.today()
    start_d = end.replace(year=end.year - max(1, years_back))
    try:
        panel = load_fundamental_panel(
            [tick], wanted, start_d.isoformat(), end.isoformat(),
            freq=freq, pit=True,
        )
    except Exception as e:
        _record_failure(ctx, "get_sec_financials")
        return json.dumps({"error": f"{type(e).__name__}: {e}",
                           "hint": "check the field names against the canonical schema"})

    out: dict[str, Any] = {
        "ticker": tick, "freq": freq, "point_in_time": True,
        "source": "SEC XBRL via vendored fundamentals_loader (PIT-safe)",
        "UNITS": "value_billions is USD billions. Use these for run_dcf_valuation "
                 "and run_comps_valuation, and keep EVERY input in billions.",
    }
    for f in wanted:
        frame = panel.get(f) if isinstance(panel, dict) else None
        if frame is None or getattr(frame, "empty", True):
            out[f] = {"error": "no data"}
            continue
        col = frame[tick].dropna()
        if col.empty:
            out[f] = {"error": "no data"}
            continue
        # The panel is daily and forward-filled, so the index is calendar days,
        # not fiscal periods. Reporting index[-1] labelled the newest figure with
        # today's date, which left no way to tell which fiscal year it belonged
        # to. Collapse to the step changes instead: each distinct value with the
        # date it first became visible, which is the filing date under pit=True.
        # Attach the real fiscal year and period end to each value.
        # Telling the model to INFER the year from a filing date was tried and
        # failed: it kept labelling Microsoft's FY2026 capex as "June 30, 2025",
        # because it took the number from here and the period from whichever
        # filing it happened to be quoting. XBRL runs a year ahead of the indexed
        # corpus, so those two genuinely disagree, and only one of them is right.
        labels: dict[float, dict[str, Any]] = {}
        try:
            raw = _annual_series(_facts_for(tick), CONCEPT_TAGS.get(f, []), 8)
            for item in raw or []:
                labels[round(float(item["value_raw"]), 2)] = {
                    "fiscal_year": item.get("fiscal_year"),
                    "period_end": item.get("period_end"),
                }
        except Exception:  # labelling is a nicety; never fail the figure over it
            labels = {}

        steps: list[dict[str, Any]] = []
        prev = None
        for ts, v in col.items():
            if prev is None or abs(float(v) - prev) > 1e-6:
                label = labels.get(round(float(v), 2), {})
                steps.append({**label,
                              "value_billions": round(float(v) / 1e9, 4),
                              # the first step is the value already standing when
                              # the window opened, so its date is the window start
                              "first_visible": ("on or before " if prev is None else "")
                                               + str(ts.date())})
                prev = float(v)
        basis = ("TRAILING TWELVE MONTHS, not a fiscal year. Do NOT describe this "
                 "as 'fiscal year X'. Call get_sec_financials with freq='annual' "
                 "if you need the reported fiscal-year figure."
                 if freq == "ttm" else
                 "Reported fiscal year as filed.")
        out[f] = {
            "BASIS": basis,

            "latest_value_billions": steps[-1]["value_billions"],
            "history": steps[-4:],
            "NOTE": "first_visible is the SEC filing date, not the fiscal period "
                    "end. A value appearing recently belongs to the fiscal year "
                    "that had just closed. Indexed filing text may be older than "
                    "this, so a mismatch means a newer fiscal year, not an error.",
        }
    return json.dumps(out, default=str)

async def get_xbrl_tag(
    ctx: RunContext[DocumentAgentDeps],
    ticker: str,
    tags: str,
    years: int = 3,
) -> str:
    """Read ANY exact us-gaap XBRL tag straight from SEC company facts.

    Escape hatch for figures outside the canonical schema. A large filer reports
    roughly 500 tags, e.g. PropertyPlantAndEquipmentNet,
    RetainedEarningsAccumulatedDeficit, AssetsCurrent, LiabilitiesCurrent,
    InventoryNet, CommonStocksIncludingAdditionalPaidInCapital.

    Annual (10-K, FY) values only, newest first, in USD billions. Derived figures
    are yours to compute, e.g. net working capital = AssetsCurrent - LiabilitiesCurrent.
    Prefer `get_sec_financials` when the field exists there: it is point-in-time safe.
    """
    emit_tool_start(ctx.deps, "get_xbrl_tag", f"{ticker.upper()}: {tags}")
    from vendor.loaders.sec_edgar_client import cik_for, get_company_facts

    tick = ticker.strip().upper()
    cik = cik_for(tick)
    if not cik:
        return json.dumps({"error": f"no CIK found for {tick}"})
    facts = get_company_facts(cik)
    out: dict[str, Any] = {"ticker": tick, "cik": cik,
                           "source": "SEC XBRL companyfacts (raw tags, annual)",
                           "UNITS": "USD billions"}
    for tag in [t.strip() for t in tags.split(",") if t.strip()]:
        series = _annual_series(facts, [tag], years)
        out[tag] = series or {"error": f"tag '{tag}' not reported by {tick}"}
    return json.dumps(out, default=str)
