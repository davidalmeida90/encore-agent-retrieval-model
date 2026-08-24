/**
 * Starter questions.
 *
 * These must match what is actually indexed, or the first thing a new user sees
 * is the agent declining. Ten companies are in the corpus, FY2024 onward, so a
 * question about 2021 correctly returns nothing and makes a poor first click.
 *
 * One per capability, so the first click also demonstrates what the agent can do:
 * a tool-sourced figure, a cited narrative, a mixed comparison, and a valuation.
 * They are deliberately the same four questions `scripts/benchmark.py` measures,
 * so the cost of every starter question is a published number.
 */
export const EXAMPLE_QUESTIONS = [
  "What was Apple's capital expenditure in fiscal 2025?",
  "How does Microsoft describe Azure's competitive advantage in its 10-K?",
  "Compare Apple's and Microsoft's R&D spending in their most recent fiscal year, and what each says about its R&D priorities.",
  'Run a DCF valuation for Apple and state every assumption.',
] as const
