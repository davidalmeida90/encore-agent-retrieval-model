import { useState } from 'react'
import { ChevronRight } from 'lucide-react'

/**
 * Where the last question's tokens went.
 *
 * Totals are exact, reported by the provider. Components are counted server-side
 * with tiktoken, and whatever is left over is shown as "Unattributed" rather than
 * folded into the others: the output schema, message framing and provider-side
 * wrapping are real input tokens that nothing reports separately.
 *
 * An earlier version scaled the components up to fill the total. It summed neatly
 * and misreported the system prompt as 80% of input when the true figure was 45%.
 * A visible remainder is more useful than a tidy wrong one.
 *
 * System prompt and tool schemas are re-sent on EVERY round, which is why they
 * dominate short questions. Both rows expand to show the literal text.
 */

export type UsageBreakdown = {
  requests: number
  tool_calls: number
  input_tokens: number
  output_tokens: number
  total_tokens: number
  system_prompt: number
  tool_schemas: number
  conversation: number
  tool_results: number
  keyword_calls: number
  validator_calls: number
  unattributed: number
  per_tool: Record<string, number>
  system_prompt_text: string
  tool_schemas_text: string
  notes: string
}

const fmt = (n: number) => n.toLocaleString()

function Bar({ value, total }: { value: number; total: number }) {
  const pct = total > 0 ? Math.round((value / total) * 100) : 0
  return (
    <div className="mt-1 flex items-center gap-2">
      <div className="h-1 flex-1 overflow-hidden rounded-sm bg-slate-200">
        <div className="h-full bg-brand-700" style={{ width: `${pct}%` }} />
      </div>
      <span className="w-8 shrink-0 text-right font-mono text-[10px] text-slate-400">
        {pct}%
      </span>
    </div>
  )
}

function Row({ label, value, total, hint, text }: {
  label: string
  value: number
  total: number
  hint?: string
  /** When present the row expands to show the literal text being paid for. */
  text?: string
}) {
  const [open, setOpen] = useState(false)
  const expandable = Boolean(text)

  return (
    <li className="rounded-md border border-slate-200 bg-white">
      <div
        className={`px-3 py-2 ${expandable ? 'cursor-pointer' : ''}`}
        onClick={expandable ? () => setOpen((v) => !v) : undefined}
        role={expandable ? 'button' : undefined}
        aria-expanded={expandable ? open : undefined}
      >
        <div className="flex items-baseline justify-between gap-2">
          <span className="flex items-center gap-1 text-xs font-medium text-slate-800">
            {expandable ? (
              <ChevronRight
                className={`size-3 text-slate-400 transition-transform ${open ? 'rotate-90' : ''}`}
                aria-hidden
              />
            ) : null}
            {label}
          </span>
          <span className="font-mono text-xs text-slate-900">{fmt(value)}</span>
        </div>
        {hint ? <p className="mt-0.5 text-[11px] leading-snug text-slate-500">{hint}</p> : null}
        <Bar value={value} total={total} />
      </div>
      {open && text ? (
        <pre className="max-h-80 overflow-auto border-t border-slate-100 bg-slate-900 p-3 text-[11px] leading-relaxed whitespace-pre-wrap text-slate-100 show-scrollbar">
          <code>{text}</code>
        </pre>
      ) : null}
    </li>
  )
}

export function TokensTab({ usage }: { usage: UsageBreakdown | null }) {
  return (
    <>
      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3 show-scrollbar">
        {!usage ? (
          <p className="px-1 text-sm text-slate-500">
            Ask a question and its token spend appears here, split by what the
            tokens were spent on.
          </p>
        ) : (
          <>
            <div className="mb-3 rounded-md border border-slate-200 bg-white px-3 py-2">
              <div className="grid grid-cols-2 gap-y-1 text-xs">
                <span className="text-slate-500">in</span>
                <span className="text-right font-mono text-slate-900">{fmt(usage.input_tokens)}</span>
                <span className="text-slate-500">out</span>
                <span className="text-right font-mono text-slate-900">{fmt(usage.output_tokens)}</span>
                <span className="font-medium text-slate-700">total</span>
                <span className="text-right font-mono font-medium text-slate-900">
                  {fmt(usage.total_tokens)}
                </span>
              </div>
              <p className="mt-1.5 border-t border-slate-100 pt-1.5 text-[11px] text-slate-500">
                {usage.requests} model {usage.requests === 1 ? 'request' : 'requests'} ·{' '}
                {usage.tool_calls} tool {usage.tool_calls === 1 ? 'call' : 'calls'}
              </p>
            </div>

            <h3 className="mb-1.5 px-1 text-xs font-medium tracking-wide text-slate-500 uppercase">
              Input, by what it paid for
            </h3>
            <ul className="space-y-1.5">
              <Row
                label="System prompt"
                value={usage.system_prompt}
                total={usage.input_tokens}
                hint="Instructions plus skill descriptions. Re-sent on every round. Click to read it."
                text={usage.system_prompt_text}
              />
              <Row
                label="Tool schemas"
                value={usage.tool_schemas}
                total={usage.input_tokens}
                hint="Names, docstrings and signatures. Charged whether called or not. Click to read them."
                text={usage.tool_schemas_text}
              />
              <Row
                label="Tool results"
                value={usage.tool_results}
                total={usage.input_tokens}
                hint="What tools returned. Persists in history, so later rounds pay again."
              />
              <Row
                label="Conversation"
                value={usage.conversation}
                total={usage.input_tokens}
                hint="Your question and the model's own replies."
              />
              <Row
                label="Unattributed"
                value={usage.unattributed}
                total={usage.input_tokens}
                hint="Output schema, message framing and provider wrapping. Not separately reported, so it is shown rather than folded into the rows above."
              />
            </ul>

            {Object.keys(usage.per_tool).length > 0 ? (
              <>
                <h3 className="mt-4 mb-1.5 px-1 text-xs font-medium tracking-wide text-slate-500 uppercase">
                  Per tool
                </h3>
                <ul className="space-y-1">
                  {Object.entries(usage.per_tool)
                    .sort((a, b) => b[1] - a[1])
                    .map(([name, value]) => (
                      <li
                        key={name}
                        className="flex items-baseline justify-between gap-2 rounded-md border border-slate-200 bg-white px-3 py-1.5"
                      >
                        <span className="truncate font-mono text-[11px] text-slate-700">{name}</span>
                        <span className="font-mono text-[11px] text-slate-900">{fmt(value)}</span>
                      </li>
                    ))}
                </ul>
              </>
            ) : null}

            <h3 className="mt-4 mb-1.5 px-1 text-xs font-medium tracking-wide text-slate-500 uppercase">
              Calls outside the agent
            </h3>
            <ul className="space-y-1">
              <li className="flex items-baseline justify-between gap-2 rounded-md border border-slate-200 bg-white px-3 py-1.5">
                <span className="text-[11px] text-slate-700">Keyword extraction</span>
                <span className="font-mono text-[11px] text-slate-900">{usage.keyword_calls}</span>
              </li>
              <li className="flex items-baseline justify-between gap-2 rounded-md border border-slate-200 bg-white px-3 py-1.5">
                <span className="text-[11px] text-slate-700">Grounding validator</span>
                <span className="font-mono text-[11px] text-slate-900">{usage.validator_calls}</span>
              </li>
            </ul>
            <p className="mt-1.5 px-1 text-[11px] leading-snug text-slate-500">
              Separate model calls the agent's own usage never sees.
            </p>

            <p className="mt-4 px-1 text-[11px] leading-snug text-slate-400">{usage.notes}</p>
          </>
        )}
      </div>
    </>
  )
}

