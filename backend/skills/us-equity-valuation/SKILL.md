---
name: us-equity-valuation
description: US equity DCF assumption framework following Damodaran. Load BEFORE calling run_dcf_valuation. Defines how to set risk-free rate, equity risk premium, beta (bottom-up unlevered then relevered), cost of debt, tax rate, terminal growth and exit multiple, plus the mandatory sanity checks. Use whenever asked to value, price, or run a DCF on a US-listed company.
category: analysis
---

# US Equity Valuation Assumptions (Damodaran framework)

Every input below must be either **sourced** (a tool, a filing, a published dataset)
or **declared as a judgement** with a stated basis. Never state a number without
saying where it came from. If an input cannot be sourced or justified, say so and
let `run_dcf_valuation` refuse rather than substituting a plausible-looking value.

## 1. Risk-free rate

**Call `get_risk_free_rate` and use what it returns.** Do not type a rate from
memory: you do not reliably know today's yield, and this number is the ceiling on
terminal growth, so an invented value propagates through the entire model.

Use the 10-year tenor for long-horizon USD cash flows. Do not use a historical
average.

Damodaran's rule that matters most downstream: in a mature currency, the risk-free
rate is the ceiling on nominal perpetual growth. See section 6.

## 2. Equity risk premium

Prefer Damodaran's **implied ERP** for the S&P 500 over historical averages, because
the implied number is forward-looking and reprices with the market.

- Typical implied ERP range for the US: **4.0% to 5.5%**
- Historical arithmetic averages (6%+) overstate it; do not default to them
- Source: `pages.stern.nyu.edu/~adamodar/` (updated monthly)

For a company with material foreign revenue, weight in country risk premiums by
revenue exposure rather than by country of listing.

## 3. Beta: bottom-up, not regression

Regression betas are noisy and backward-looking. Damodaran's method:

1. Take the **unlevered (asset) beta** for the company's industry
2. Relever at the company's own capital structure:

```
levered beta = unlevered beta x (1 + (1 - tax rate) x D/E)
```

Indicative unlevered betas, US, technology and adjacent:

| Sector | Unlevered beta | Typical levered beta at low leverage |
|---|---|---|
| Software (system and application) | 1.05 - 1.25 | 1.05 - 1.30 |
| Semiconductors | 1.30 - 1.60 | 1.35 - 1.65 |
| Computers and peripherals | 1.00 - 1.20 | 1.00 - 1.25 |
| Internet and advertising | 1.10 - 1.35 | 1.15 - 1.40 |
| Retail (general) | 0.85 - 1.05 | 0.90 - 1.15 |

Rules of thumb worth stating explicitly:
- A large, cash-rich, diversified US mega-cap rarely justifies a beta **below 1.0**
- If your relevered beta lands under 1.0 for a technology company, say why
- Do not pick a beta that makes the answer come out where you want it

## 4. Cost of debt

Use the company's **synthetic rating** approach when a market yield is unavailable:
interest coverage ratio (EBIT / interest expense) implies a rating, which implies a
default spread over the risk-free rate.

```
pretax cost of debt = risk-free rate + default spread
```

For a AA/AAA-rated US mega-cap the spread is typically **0.4% to 1.0%**. Never set
cost of debt below the risk-free rate.

## 5. Tax rate

Two different rates, used in two different places:

- **Effective tax rate** (from the filing) for near-term forecast years
- **Marginal tax rate** (US federal 21% plus state, so roughly **24% to 26%**) in
  the terminal year and for the WACC after-tax cost of debt

Using a low effective rate in perpetuity assumes tax structuring lasts forever.
State which rate you used where.

## 6. Terminal growth: the hard constraint

**Perpetual nominal growth must not exceed the risk-free rate.** This is
Damodaran's cap, and it is stricter than "must not exceed GDP growth". Reason: the
risk-free rate embeds expected nominal growth for the economy, so a company growing
faster forever eventually becomes the economy.

- Practical range: **1.5% to 3.0%** for USD cash flows in a normal rate environment
- If your terminal growth is above the current 10-year yield, it is wrong
- `run_dcf_valuation` returns `terminal_growth_exceeds_gdp_ceiling`; treat a True
  there as a hard stop, not a note

## 7. Exit multiple

Only use as a **cross-check**, never as the primary method for a going concern. An
exit multiple imports today's market pricing into perpetuity, which is circular if
you are trying to decide whether today's price is right.

- Set it from peer EV/EBITDA, and say which peers
- Then read `implied_growth_from_exit_multiple` from the tool output. If that implied
  growth exceeds the risk-free rate, the multiple is too high, and you must say so

## 8. Reinvestment must be consistent with growth

Growth is not free. If you project revenue or EBIT growing at g, capex net of
depreciation and the change in working capital must be consistent with it:

```
reinvestment rate = g / return on invested capital
```

Common failure: projecting 5% growth while holding capex flat at maintenance level.
That manufactures free cash flow out of nothing. If you cannot estimate working
capital from the data available, say the delta_nwc figure is an assumption and give
its basis.

## 9. Mandatory reporting

Every valuation must state:

1. Each assumption, its **value**, and its **source or basis**
2. Both terminal value estimates and the tool's three `cross_checks`
3. `terminal_share_of_ev` — if the terminal value is more than about **75%** of
   enterprise value, say so plainly; the model is mostly an opinion about perpetuity
4. Sensitivity: at minimum how value per share moves with WACC +/- 1% and terminal
   growth +/- 0.5%. A 1% move in WACC can shift value by 20% or more
5. The gap to the current market price, and which assumption would have to change to
   close it

## 10. What not to do

- Do not invent a beta, an ERP or a risk-free rate when a tool can source it
- Do not use a terminal growth rate above the risk-free rate
- Do not present a single point estimate without a range
- Do not describe the output as a target price or a recommendation
