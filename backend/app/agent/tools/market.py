"""## Market data: prices and the risk-free rate

Everything here is priced by the market rather than filed with the SEC, so it
moves daily and must never be typed from memory.

`get_stock_prices` walks the loader registry's fallback chain (yfinance, then
Stooq) so one provider outage does not end the answer, and computes drawdown /
VaR / CVaR with the vendored risk module rather than by hand.

`get_risk_free_rate` exists because the model does not reliably know today's
yield, and that number is the ceiling on terminal growth in every DCF. Dates are
computed server-side from today: an earlier version let the model pass dates and
it confidently asked for a window a year stale.

## Examples

    get_stock_prices("AAPL", period="1y")
      -> total_return_pct : 40.18
         low / high       : 224.90 / 340.08
         max_drawdown     : -13.82% (2025-12-02 -> 2026-03-30, recovered 05-06)
         historical_var   : 2.07% daily at 95%

    get_risk_free_rate("10y")
      -> {"symbol": "^TNX", "yield_pct": 4.653, "risk_free_rate": 0.04653,
          "as_of": "2026-08-19"}
         ^ the valuation skill requires this call rather than a typed rate,
           because it is the ceiling on terminal growth: invent it and the error
           propagates through the entire DCF.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic_ai import RunContext

from app.agent.deps import DocumentAgentDeps
from app.agent.status import emit_tool_start
from app.agent.tools._guards import _record_failure, _safe


async def get_stock_prices(
    ctx: RunContext[DocumentAgentDeps],
    ticker: str,
    period: str = "1y",
    start_date: str = "",
    end_date: str = "",
    market: str = "us_equity",
) -> str:
    """Price history and risk metrics, using the vendored loader registry.

    Retrieval goes through the registry's market fallback chain, so a US ticker
    tries yahoo, then stooq, then the rest, rather than one hardcoded provider.
    Risk figures come from the vendored quantlib, not from arithmetic here.

    period: "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd". Dates are computed
      server-side from TODAY, so prefer this. Do NOT guess calendar dates: you do
      not reliably know the current date, and a stale window silently returns the
      wrong year's performance.
    start_date / end_date: YYYY-MM-DD, only for an explicit historical window.
    market: us_equity (default), hk_equity, cn_equity.
    """
    emit_tool_start(ctx.deps, "get_stock_prices", f"{ticker.upper()} {period or start_date+'..'+end_date}")
    import datetime as _dt

    from vendor.loaders.registry import FALLBACK_CHAINS, get_loader_cls_with_fallback
    from vendor.quantlib.risk import historical_cvar, historical_var, max_drawdown_analysis

    tick = ticker.strip().upper()
    if not (start_date and end_date):
        today = _dt.date.today()
        months = {"1mo": 1, "3mo": 3, "6mo": 6, "1y": 12, "2y": 24, "5y": 60}
        if period == "ytd":
            begin = today.replace(month=1, day=1)
        else:
            m = months.get(period, 12)
            y, mo = today.year - m // 12, today.month - m % 12
            if mo <= 0:
                y, mo = y - 1, mo + 12
            begin = today.replace(year=y, month=mo, day=min(today.day, 28))
        start_date, end_date = begin.isoformat(), today.isoformat()
    # Walk the registry's own market fallback chain. Several sources in the chain
    # need credentials or are geo-blocked and return empty rather than raising,
    # so "empty" is treated as a miss and the next source is tried.
    # SPLIT THE RANGE so history can actually be cached.
    #
    # The vendored loaders already cache to parquet, and already refuse to cache
    # any range whose end_date is today (loader_cache_range_is_final), because a
    # bar that is still forming must not be pinned. Asking for [start .. today]
    # therefore defeated the cache completely: every call refetched a full year
    # from Yahoo, which is why one stock question took 14s and the next took 4s.
    #
    # So fetch two blocks:
    #   history  [start .. yesterday]  settled, cacheable, served from disk after
    #                                  the first call
    #   tail     [yesterday .. today]  a couple of bars, always live, never cached
    #
    # Net effect: "what happened over the past year" is a disk read plus two bars,
    # while "what is it trading at now" is still genuinely live.
    yesterday = (_dt.date.today() - _dt.timedelta(days=1)).isoformat()
    hist_end = min(end_date, yesterday)
    wants_today = end_date > yesterday

    def _fetch(loader, begin: str, finish: str):
        if begin > finish:
            return None
        frames = loader.fetch([tick], begin, finish, interval="1D")
        return frames.get(tick) if isinstance(frames, dict) else None

    df, used, tried = None, None, []
    for source in FALLBACK_CHAINS.get(market, ["yfinance"]):
        try:
            cls = get_loader_cls_with_fallback(source)
            loader = cls() if isinstance(cls, type) else cls
            cand = _fetch(loader, start_date, hist_end)
            if wants_today:
                tail = _fetch(loader, yesterday, end_date)
                if tail is not None and not getattr(tail, "empty", True):
                    if cand is None or getattr(cand, "empty", True):
                        cand = tail
                    else:
                        import pandas as _pd

                        cand = _pd.concat([cand, tail])
                        cand = cand[~cand.index.duplicated(keep="last")].sort_index()
            tried.append(source)
            if cand is not None and not getattr(cand, "empty", True):
                df, used = cand, source
                break
        except Exception:
            tried.append(f"{source}(err)")
            continue
    if df is None:
        _record_failure(ctx, "get_stock_prices")
        return json.dumps({"error": f"no price history for {tick}", "sources_tried": tried})
    if df is None or getattr(df, "empty", True):
        return json.dumps({"error": f"no price history for {tick} in {market}"})

    col = next((c for c in ("close", "Close", "adj_close", "Adj Close") if c in df.columns), None)
    if col is None:
        return json.dumps({"error": f"no close column; got {list(df.columns)[:8]}"})
    close = df[col].dropna()
    rets = close.pct_change().dropna()

    dd = max_drawdown_analysis(close)
    return json.dumps(
        {
            "ticker": tick, "market": market, "source_used": used, "sources_tried": tried,
            "start": str(close.index[0])[:10], "end": str(close.index[-1])[:10],
            "price_start": round(float(close.iloc[0]), 2),
            "price_last": round(float(close.iloc[-1]), 2),
            "total_return_pct": round(float(close.iloc[-1] / close.iloc[0] - 1) * 100, 2),
            "high": round(float(close.max()), 2), "low": round(float(close.min()), 2),
            "max_drawdown": _safe(dd),
            "historical_var_95_daily": round(float(historical_var(rets, confidence=0.95)), 5),
            "historical_cvar_95_daily": round(float(historical_cvar(rets, confidence=0.95)), 5),
            "observations": int(len(close)),
            "source": "vendored loader registry fallback chain + quantlib.risk",
        },
        default=str,
    )

async def get_risk_free_rate(
    ctx: RunContext[DocumentAgentDeps],
    tenor: Literal["10y", "5y", "3m"] = "10y",
) -> str:
    """Current US Treasury yield. Use this instead of typing a rate from memory.

    Call this BEFORE any DCF. The risk-free rate feeds WACC and, per the
    valuation skill, is the ceiling on nominal perpetual growth, so an invented
    value propagates through the whole model.

    tenor: "10y" (default, matches long-horizon USD cash flows), "5y", "3m".
    Returns a decimal rate (0.0465, not 4.65) plus the observation date.
    """
    emit_tool_start(ctx.deps, "get_risk_free_rate", tenor)
    import datetime as _dt

    from vendor.loaders.registry import FALLBACK_CHAINS, get_loader_cls_with_fallback

    symbol = {"10y": "^TNX", "5y": "^FVX", "3m": "^IRX"}[tenor]
    end = _dt.date.today()
    start = end - _dt.timedelta(days=21)
    for source in FALLBACK_CHAINS.get("us_equity", ["yfinance"]):
        try:
            cls = get_loader_cls_with_fallback(source)
            loader = cls() if isinstance(cls, type) else cls
            fr = (loader.fetch([symbol], start.isoformat(), end.isoformat(),
                               interval="1D") or {}).get(symbol)
            if fr is None or getattr(fr, "empty", True):
                continue
            col = next((c for c in ("close", "Close") if c in fr.columns), None)
            if not col:
                continue
            series = fr[col].dropna()
            pct = float(series.iloc[-1])
            return json.dumps({
                "tenor": tenor, "symbol": symbol,
                "yield_pct": round(pct, 3),
                "risk_free_rate": round(pct / 100.0, 5),
                "as_of": str(series.index[-1])[:10],
                "source": f"{source} via vendored loader registry",
                "note": "risk_free_rate is a decimal, pass it directly to run_dcf_valuation",
            })
        except Exception:
            continue
    _record_failure(ctx, "get_risk_free_rate")
    return json.dumps({"error": f"no yield data for {symbol}"})
