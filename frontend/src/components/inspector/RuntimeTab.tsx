import { useEffect, useState } from 'react'

import { api } from '@/lib/api'

/**
 * ## What the model is, and where its time went
 *
 * The Tokens tab answers "what did that cost" in tokens. Tokens are not time,
 * and the two behave in opposite directions:
 *
 *   prefill   whole prompt at once, compute bound, very cheap per token
 *   decode    one token at a time, bandwidth bound, ~100x dearer per token
 *
 * So a question with 48,683 tokens in and 1,123 out spends most of its GPU time
 * on the 1,123. This tab exists to make that visible, because no amount of
 * staring at token counts reveals it.
 *
 * Everything under "measured" is scraped from the server's own metrics.
 * Everything under "derived" is arithmetic on those numbers plus the model's
 * active parameter count, and is labelled so the two are never confused.
 */

type RuntimeInfo = {
  configured: boolean
  reachable: boolean
  base_url?: string | null
  detail?: string | null
  model?: Record<string, unknown> | null
  serving?: Record<string, unknown> | null
  totals?: Record<string, unknown> | null
  derived?: Record<string, unknown> | null
}

function Row({ label, value, hint }: { label: string; value: React.ReactNode; hint?: string }) {
  if (value === null || value === undefined || value === '') return null
  return (
    <li className="flex items-baseline justify-between gap-3 border-b border-border/60 py-1.5 last:border-0">
      <span className="text-[11px] text-muted-foreground">
        {label}
        {hint ? <span className="ml-1 text-muted-foreground/60">{hint}</span> : null}
      </span>
      <span className="text-right font-mono text-[11px] text-foreground">{value}</span>
    </li>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mb-4">
      <h3 className="mb-1 px-1 text-xs font-medium tracking-wide text-muted-foreground uppercase">
        {title}
      </h3>
      <ul className="rounded-md border border-border bg-background px-3 py-1">{children}</ul>
    </section>
  )
}

const fmt = (n: unknown) => (typeof n === 'number' ? n.toLocaleString() : String(n ?? ''))

/** Same key ModelSelect persists to. The layout does not own the model choice,
 *  and threading it through four components to reach one panel is worse than
 *  reading the value where it is already stored. */
const MODEL_KEY = 'encore.model'

export function RuntimeTab() {
  const [info, setInfo] = useState<RuntimeInfo | null>(null)
  const [error, setError] = useState<string | null>(null)
  const model = (() => {
    try {
      return localStorage.getItem(MODEL_KEY)
    } catch {
      return null
    }
  })()

  useEffect(() => {
    let cancelled = false
    setInfo(null)
    setError(null)
    const query = model ? `?model=${encodeURIComponent(model)}` : ''
    api
      .get<RuntimeInfo>(`/runtime${query}`)
      .then((data) => !cancelled && setInfo(data))
      .catch((err: unknown) =>
        !cancelled && setError(err instanceof Error ? err.message : 'Could not read runtime'),
      )
    return () => {
      cancelled = true
    }
  }, [model])

  if (error) return <div className="px-4 py-3 text-sm text-rose-600">{error}</div>
  if (!info) return <div className="px-4 py-3 text-sm text-muted-foreground">Loading…</div>

  // A hosted model, or a local one with nothing listening. Both are ordinary
  // states rather than errors, so they read as an explanation.
  if (!info.reachable) {
    return (
      <div className="px-4 py-3">
        <p className="text-sm leading-relaxed text-muted-foreground">{info.detail}</p>
        {info.base_url ? (
          <p className="mt-2 font-mono text-[11px] text-muted-foreground/70">{info.base_url}</p>
        ) : null}
      </div>
    )
  }

  const m = info.model ?? {}
  const s = info.serving ?? {}
  const t = info.totals ?? {}
  const d = info.derived ?? {}
  const hitRate = s.prefix_cache_hit_rate as number | null

  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3 show-scrollbar">
      <Section title="Model">
        <Row label="Serving" value={String(m.checkpoint ?? m.served_as ?? '')} />
        <Row label="Quantisation" value={m.quantization as string} />
        <Row label="Architecture" value={m.architecture as string} />
        <Row
          label="Experts"
          value={m.experts ? `${m.active_experts} active of ${m.experts}` : null}
        />
        <Row
          label="Parameters"
          value={
            m.total_params_b
              ? `${m.total_params_b}B total, ${m.active_params_b}B active`
              : null
          }
        />
        <Row label="Modality" value={m.modality as string} />
        <Row label="Context window" value={fmt(m.context_window)} />
      </Section>

      <Section title="Serving">
        <Row label="Prefix caching" value={s.prefix_caching ? 'on' : 'off'} />
        <Row
          label="Cache hit rate"
          hint="prompt tokens reused"
          value={hitRate != null ? `${(hitRate * 100).toFixed(1)}%` : null}
        />
        <Row label="KV cache" value={`${fmt(s.kv_cache_tokens)} tokens`} />
        <Row label="GPU memory" value={s.gpu_memory_utilization as string} />
        <Row label="In flight" value={fmt(s.requests_running)} />
      </Section>

      <Section title="Measured, since the server started">
        <Row label="Requests" value={fmt(t.requests)} />
        <Row label="Prompt tokens" value={fmt(t.prompt_tokens)} />
        <Row label="Generated tokens" value={fmt(t.generation_tokens)} />
        <Row label="Prefill, average" value={`${t.avg_prefill_seconds}s`} />
        <Row label="Decode rate" value={`${t.decode_tokens_per_second} tok/s`} />
        <Row label="Time in prefill" value={`${d.measured_prefill_seconds}s`} />
        <Row label="Time in decode" value={`${d.measured_decode_seconds}s`} />
      </Section>

      {Object.keys(d).length > 0 ? (
        <Section title="Derived from active parameters">
          <Row label="Read per output token" value={`${d.gb_read_per_output_token} GB`} />
          <Row label="Compute per output token" value={`${d.gflops_per_output_token} GFLOP`} />
          <Row
            label="Arithmetic intensity"
            hint="FLOPs per byte"
            value={String(d.arithmetic_intensity_flops_per_byte)}
          />
          <Row label="Total read" value={`${fmt(d.total_gb_read)} GB`} />
          <Row label="Total compute" value={`${fmt(d.total_tflops)} TFLOP`} />
        </Section>
      ) : null}

      {d.note ? (
        <p className="px-1 text-[11px] leading-snug text-muted-foreground">{d.note as string}</p>
      ) : null}
    </div>
  )
}
