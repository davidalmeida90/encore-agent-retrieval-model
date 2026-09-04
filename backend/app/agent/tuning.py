"""## Every knob worth turning during testing, in one file

## Why this file exists
These numbers used to be scattered across four modules, so changing "how much can
one question spend" meant editing three files and hoping none was missed. Testing
means changing them often, so they live here and nowhere else. Each one carries
the observation that set it, because a number without its reason gets tuned back
to a wrong value six weeks later.

Nothing here imports from the rest of the app, so it is safe to import anywhere.
"""

from __future__ import annotations

# ---------------------------------------------------------------- quota ceilings
# Observed live on 2026-08-20 as the quota that actually stopped a test run:
#   GenerateRequestsPerDayPerProjectPerModel-FreeTier, value 500
# Raise this only when the key stops being free tier.
GEMINI_FREE_TIER_DAILY_REQUESTS = 500

# How long to wait for one Gemini HTTP response before giving up.
#
# The default is no timeout at all, which is not a theoretical problem: on
# 2026-08-21 a benchmark run sat inside a single model request for over ten
# minutes with a py-spy stack showing an idle event loop and nothing in flight
# but that one socket. In the browser that is a spinner that never resolves and
# gives the user nothing to act on.
#
# Set above the slowest legitimate response seen (a DCF on the reasoning model,
# about 45s) with room to spare, so this only ever fires on a genuinely stuck
# connection rather than a slow-but-working one. Connect gets its own much
# shorter budget: reaching the host is either fast or broken.
GEMINI_REQUEST_TIMEOUT_SECONDS = 120.0
GEMINI_CONNECT_TIMEOUT_SECONDS = 10.0

# A self-hosted model gets a longer leash. Decode is bandwidth bound and its
# time is linear in output tokens, so a thinking-mode answer of several thousand
# tokens legitimately takes minutes on a mid-range card. The measured DCF on
# Gemini produced 1,123 output tokens; thinking could be 5x that.
#
# Connect stays short: a pod that is up answers immediately, and one that is
# still loading weights should fail fast rather than look like a slow answer.
LOCAL_REQUEST_TIMEOUT_SECONDS = 600.0
LOCAL_CONNECT_TIMEOUT_SECONDS = 10.0

# Qwen3.x ships thinking mode ON. Measured on this pod, asking for one tool call:
#
#   thinking off      61 output tokens   1.2s
#   thinking on    1,833 output tokens   9.6s     -> identical tool call
#
# Decode is bandwidth bound and linear in output tokens, so thinking is a 30x
# multiplier on the part that actually costs time. Off by default for the same
# reason flash-lite is the default model: most questions here are lookups and
# retrieval, where the reasoning buys nothing.
#
# Worth turning on for valuation, where models.py records flash-lite skipping
# the sensitivity table and market comparison that the skill mandates.
# None means "do not send the switch at all". Qwen3.6 has enable_thinking in its
# chat template; Qwen3-30B-Instruct-2507 does not, and passing a kwarg the
# template never reads is a needless way to break a request.
LOCAL_THINKING_ENABLED: bool | None = False

# Let a self-hosted model emit several tool calls in one round?
#
# Gemini's wire format has no such switch and emits one call per turn, so the
# harness runs one round per call. OpenAI's format has the switch and it
# defaults to ALLOWED, which Qwen takes full advantage of: measured on the
# question "what was Apple's capital expenditure in fiscal 2025", it produced
#
#     2 model requests   and   21 tool calls
#
# Every result lands in history, so on retrieval questions, where each result is
# a passage, a single round is enough to blow the per-question ceiling. The
# three questions that failed at 100k+ tokens all failed this way.
#
# TESTED, and False is worse. Forcing one call per round does not stop the model
# wanting N calls; it spreads them over N rounds, and every round re-sends the
# whole history. Same four questions, concurrently:
#
#     parallel ON    capex 2 requests, 7,908 tokens, PASS
#     parallel OFF   capex blew the 100k ceiling,    FAIL  (all four failed)
#
# One round carrying twenty results is far cheaper than twenty rounds each
# re-sending everything before them. Left ON, because the real fault is that it
# calls at all after the first result answers the question, and that is a model
# behaviour no wire-protocol switch fixes.
LOCAL_PARALLEL_TOOL_CALLS = True

# After how many tool results should the model be reminded that it can answer?
#
# Measured on Qwen3.6-35B, "What was Apple's capital expenditure in fiscal 2025":
# it got the figure from XBRL in round 1 and then spent twelve more rounds
# verifying it against the filing text. Two searches, four read_chunk, four
# read_surrounding_chunks, two more searches. Thirteen distinct calls, ZERO
# repeats: not a loop, and no guard catches it, because every single call is
# individually reasonable.
#
# What it lacks is a sense of enough. Gemini answers the same question in two
# rounds. This is instruction fade, which SystemReminders exists to counter:
# a reminder re-injected at the tail of the request, behind a CachePoint so the
# cached prefix stays byte-identical.
#
# 4 is above what a legitimate two-company comparison needs (one XBRL call plus
# one search each) and well below the runaway.
REMIND_TO_ANSWER_AFTER_TOOL_RESULTS = 4

# Ceiling for a SINGLE question. Observed costs: a comps answer about 4,000
# tokens, a simple XBRL lookup about 15,000, a DCF on a reasoning model about
# 40,000, and a two-company question mixing XBRL figures with cited 10-K prose
# 67,000 (four tool calls, two of them retrieval).
#
# Was 60,000, which refused that last one as a runaway when it was simply a bigger
# question. Set from the largest LEGITIMATE run seen, with room above it. The
# actual runaway this guards against spent 112,000 retrying a broken comps call,
# and MAX_TOOL_FAILURES / MAX_IDENTICAL_CALLS below now catch that class directly
# and far earlier, so this ceiling does not have to sit tight enough to catch it.
PER_QUESTION_TOKEN_CEILING = 100_000

# CAG breaks that ceiling by design rather than by accident: the whole filing is
# the point, and one NVIDIA 10-K measured 196,676 input tokens on the first
# question. 100k is right for RAG, where anything approaching it means the model
# is looping. Applying it to CAG just refuses the mode.
#
# Set from the model's context window, not from thrift: 262,144 is what
# Qwen3.8-27B advertises and what Gemini comfortably exceeds, and a turn that
# wants more than that has loaded a second filing, which the mode forbids.
CAG_QUESTION_TOKEN_CEILING = 400_000

# No "minute" budget window exists in the harness, so the 250,000
# input-tokens-per-minute free-tier ceiling cannot be enforced here. Pacing
# between questions is what handles that; see PACE_SECONDS in the eval runner.
#
# LIMITATION worth knowing before trusting the daily budget: SpendLimits defaults
# to an in-memory store, so the counter lives and dies with the PROCESS. Inside
# the long-running FastAPI server the daily ceiling works as intended. For CLI
# runs (scripts/ask.py, the eval suite) every invocation starts the count at zero,
# so the budget cannot stop you from exhausting the real quota across many short
# runs. Fixing that properly needs a persistent store (the harness ships a Redis
# one) or a small file-backed store.


# ---------------------------------------------------------------- loop guards
# How many times one tool may FAIL in a turn before it refuses to run again.
# Set to 2 because a third attempt has never once succeeded in observed runs.
MAX_TOOL_FAILURES = 2

# How many times one tool may be called with IDENTICAL arguments and SUCCEED.
# A successful repeat is invisible to the failure guard above, and one run made
# seven identical calls because the model disbelieved a (correct) number.
MAX_IDENTICAL_CALLS = 2


# ---------------------------------------------------------------- tool payloads
# Tool returns persist in message history and are re-sent on every later round,
# so an oversized return is paid for repeatedly rather than once.
#
# Measured: search_filings returns 12,030 characters, every time (the retrieval
# layer caps itself there). Everything else is under 2,000.
#
# CAREFUL: these thresholds count CHARACTERS, not tokens, unless the capability is
# built with over_tokens=True. An earlier config paired a 6,000 trigger with a
# 12,000 clamp, so the clamp was larger than its own trigger and removed 30
# characters out of 12,030. Always keep the clamp BELOW the trigger.
#
# Spill is LOSSLESS: the payload is stored and the model gets a preview plus a
# handle it can read back. That matters for retrieval, where grounding needs
# verbatim excerpts and blind truncation can cut away the sentence a citation
# depends on.
#
# Measured on 2026-08-20, removing Spill entirely made things WORSE, not better:
# the same question went 132,594 -> 174,269 tokens, because each 12,030-char
# result then sat in history and was re-sent every round. It was earning its keep.
#
# Preview raised 1,500 -> 4,000 all the same. At 1,500 the model saw 12% of a
# search result, which is thin enough to encourage re-searching rather than
# reading back. Repeated searching has a separate, better fix directly below.
SPILL_OVER_CHARS = 8_000        # above this, store the payload, hand back a handle
SPILL_PREVIEW_CHARS = 4_000     # a third of a search result, not a twelfth
SPILL_FALLBACK_CHARS = 8_000    # used only when no store accepts the write
TRUNCATE_OVER_CHARS = 40_000    # pure safety net for a pathological payload
TRUNCATE_TO_CHARS = 30_000

# How many times search_filings may run for ONE ticker in a single question.
# The generic repeat guard keys on exact arguments, so it never fires here: the
# model rephrases slightly every time. Observed on a question asking why Apple
# spends what it does on capex, which Apple's 10-K simply does not say. The model
# issued six near-identical AAPL searches before concluding absence, and the run
# cost 174,000 tokens against 80,000 for the same question shape where the text
# actually exists. Absence is a legitimate finding; it just has to be allowed.
# Was 3. With two tickers that allowed six searches plus any unfiltered ones, and
# since each search puts ~12,000 characters into history that is re-sent every
# later round, the cost is front-loaded: the guard fired on the 4th AAPL search
# but 170,000 tokens were already spent by then. A guard that only trims the tail
# does not help.
MAX_SEARCHES_PER_TICKER = 2

# Hard cap across ALL tickers in one question, because per-ticker counting still
# lets a two-company question spend double, and searches without a ticker filter
# escape per-ticker counting entirely.
MAX_SEARCHES_PER_TURN = 5

# History compaction: blank OLD tool results once the conversation grows, keeping
# the most recent exchanges intact. Costs no LLM call, and anything cleared can be
# re-fetched by calling the tool again.
# 40_000 meant compaction only engaged near the end of a long run. Real runs land
# at 60,000-90,000, so clearing older tool results earlier keeps the re-sent
# history smaller through the middle rounds, which is where it is paid for most.
COMPACTION_TRIGGER_TOKENS = 25_000
COMPACTION_TARGET_TOKENS = 60_000
COMPACTION_KEEP_PAIRS = 3


# ---------------------------------------------------------------- elsewhere
# Retrieval knobs deliberately do NOT live here. They are env-overridable
# deployment config and sit in `app/config.py`:
#
#   retrieval_top_k             how many fused chunks reach the model
#   retrieval_candidate_k       how many each search returns before fusion
#   retrieval_rrf_k             the k in 1/(k + rank); higher = flatter weighting
#   retrieval_neighbor_radius   how many chunks either side ride along
#   agent_request_limit         hard stop on loop iterations per question
#
# Change those in config.py or override them from .env.


# ---------------------------------------------------------------- grounding
# How many times the whole agent re-runs when the grounding check rejects an
# answer. Each attempt costs a full duplicate run, so this is deliberately small.
MAX_VALIDATION_ATTEMPTS = 2
