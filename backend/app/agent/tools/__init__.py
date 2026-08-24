"""## The tool registry

## How to add, remove, or disable a tool
Edit `TOOLS` below. Nothing else. `agent.py` imports this list and passes it
straight to the Agent, so a tool exists exactly when it appears here.

To disable one during testing, comment out its line. The implementation stays on
disk and keeps working; it simply stops being offered to the model. That is how
the three-statement tools are parked right now: they work, but every registered
tool costs schema tokens on every single request, and they were not earning it.

## What each group is for
Grouping is by the question each answers, because that is how the model chooses:

    retrieval     "what did management SAY"      prose from indexed filings
    fundamentals  "what is the NUMBER"           tagged SEC XBRL facts
    market        "what is it worth TODAY"       prices and yields
    valuation     "what SHOULD it be worth"      deterministic engines
    skills        "how do I do this properly"    methodology, loaded on demand

## Cost note
Registered tools are described in the system prompt on every request, whether
called or not. Twelve tools cost about 2,550 tokens per request. Adding a tool is
not free, so a tool earns its place by being called often enough to justify the
rent it charges on every other question.
"""

from __future__ import annotations

from app.agent.tools.fundamentals import get_sec_financials, get_xbrl_tag
from app.agent.tools.market import get_risk_free_rate, get_stock_prices
from app.agent.tools.retrieval import (
    read_chunk,
    read_chunks,
    read_surrounding_chunks,
    search_filings,
)
from app.agent.tools.skills import load_skill, skill_descriptions
from app.agent.tools.valuation import (
    Peer,
    get_comps_inputs,
    get_three_statement_opening,
    project_three_statement_model,
    run_comps_valuation,
    run_dcf_valuation,
)

# --------------------------------------------------------------------- registry
TOOLS = [
    # -- retrieval over the indexed filings ---------------------------------
    search_filings,
    read_chunk,
    read_chunks,
    read_surrounding_chunks,
    # -- company fundamentals, from SEC XBRL --------------------------------
    get_sec_financials,
    get_xbrl_tag,
    # -- market data --------------------------------------------------------
    get_stock_prices,
    get_risk_free_rate,
    # -- input assemblers: one call each, so the model never hunts ----------
    get_comps_inputs,
    # -- deterministic valuation engines (vendored quantlib) ----------------
    run_dcf_valuation,
    run_comps_valuation,
    # -- methodology, loaded on demand --------------------------------------
    load_skill,
    # -- parked: working, but not worth its schema rent on every request ----
    # project_three_statement_model,
    # get_three_statement_opening,
]

__all__ = [
    "TOOLS",
    "Peer",
    "skill_descriptions",
    # exported individually so tests and the eval suite can call one directly
    "search_filings",
    "read_chunk",
    "read_chunks",
    "read_surrounding_chunks",
    "get_sec_financials",
    "get_xbrl_tag",
    "get_stock_prices",
    "get_risk_free_rate",
    "get_comps_inputs",
    "run_dcf_valuation",
    "run_comps_valuation",
    "load_skill",
    "project_three_statement_model",
    "get_three_statement_opening",
]
