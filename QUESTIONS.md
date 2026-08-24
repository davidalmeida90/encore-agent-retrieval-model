# Questions to try

Scoped to what is actually indexed: ten companies, 23 annual reports, 27,222
chunks. Asking about a company outside this list, or about a year before FY2024,
returns "not in the corpus" rather than an answer, which is the behaviour worth
testing.

| | | |
|---|---|---|
| AAPL Apple FY2024-25 | AMZN Amazon FY2024-25 | GOOGL Alphabet FY2024-25 |
| JPM JPMorgan FY2024-25 | META Meta FY2024-25 | MSFT Microsoft FY2024-**26** |
| NFLX Netflix FY2024-25 | NVDA NVIDIA FY2024-**26** | TSLA Tesla FY2024-25 |
| WMT Walmart FY2024-**26** | | |

Three companies have a FY2026 filing and seven stop at FY2025, because fiscal
years end at different times. That mismatch is not a bug to hide: comparing
"most recent year" across two companies nine months apart is exactly the trap
the harder questions below are built to expose.

Cost figures are measured, not estimated. The pattern behind them:

| Costs | Because |
|---|---|
| ~4s, ~7k tokens | answered from XBRL or market data, no filing search |
| ~15s, ~10k tokens | one search of the filings, one rerank |
| 60s+, 40k+ tokens | several searches, or a valuation |

Reranking runs once per search and takes about 9s on CPU, so wall-clock tracks
the number of searches more than anything else.

---

## Basic — no search, answered from tagged data

Fast and cheap. These never touch retrieval.

- What was Apple's capital expenditure in fiscal 2025?
- What was Microsoft's revenue in its most recent fiscal year?
- How much cash did Apple generate from operations in FY2025?
- What is Apple's total debt, and how does it compare to its cash?
- How has Apple's stock performed over the past year?
- What is Microsoft's diluted share count, and how has it changed since FY2024?
- What risk-free rate should be used to value a US company today, and why?
- Show Microsoft's capital expenditure for every year you have.

Available fields: revenue, cogs, gross_profit, operating_income, net_income,
total_assets, total_equity, total_debt, cash, cfo, capex, shares_diluted. For
anything outside that list the agent falls back to raw XBRL tags.

---

## Intermediate — one search, cited from the filings

These ask what a company *said*, so they retrieve and cite.

- How does Microsoft describe Azure's competitive advantage in its 10-K?
- What does Microsoft say about the risks of its AI investments?
- How does Apple describe its dependence on manufacturing in China?
- What does Microsoft say about datacenter capacity constraints?
- How does Apple describe competition in its risk factors?
- What does Microsoft disclose about its capital commitments for construction?
- How does Apple describe its services business and its growth drivers?
- What does Microsoft say about the environmental impact of its datacenters?

---

## Harder — comparisons and valuation

Several searches or a valuation engine. Slower and materially more expensive.

- Compare Apple's and Microsoft's R&D spending in their most recent fiscal year,
  and what each says about its R&D priorities.
- Compare Apple's and Microsoft's capital expenditure in FY2025 on a like-for-like
  basis.
- How did Microsoft's description of AI infrastructure change between its FY2025
  and FY2026 10-K?
- Compare gross margin for Apple and Microsoft, and explain what each says drives it.
- Run a DCF valuation for Apple and state every assumption.
- Value Apple on trading comparables using Microsoft as the peer.

---

## Deliberately hard, worth trying to see the failure modes

These are honest tests rather than demos.

- Why does Apple spend what it does on capital expenditure?
  Apple's 10-K contains no such rationale. A good answer says so plainly instead
  of paraphrasing the nearest paragraph. Watch the relevance scores in the log.

- Compare Apple's and Microsoft's most recent fiscal year capex.
  Apple's most recent is FY2025 (ended September 2025), Microsoft's is FY2026
  (ended June 2026) — nine months apart during an AI capex boom. A good answer
  flags the mismatch rather than presenting the two as comparable.

- What was Amazon's revenue last year?
  Not in the corpus. Should decline, not improvise.

---

## Watching the cost

Open the **Log** panel before asking. The `usage` line reports model requests,
tool calls and tokens for that question, and every `tool` line shows exactly what
ran with which arguments. That is the fastest way to understand why one question
cost 7,000 tokens and another 40,000.
