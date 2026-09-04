import { useEffect, useState } from 'react'
import { Check, ChevronDown, FileText, Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { api } from '@/lib/api'

/**
 * Which company's filing CAG loads. Only shown when CAG is the chosen mode.
 *
 * CAG cannot work this out from the question. The filing has to be in the
 * prompt *before* the question is asked, because that ordering is the whole
 * mode: as a stable prefix the server caches it and reuses it across every
 * later question, and 191,000 tokens are prefilled once instead of every time.
 * Put after the question and it would be re-read on each turn.
 *
 * Hence a picker rather than inference, and hence the warning beside it: the
 * first question on a company is slow and the rest are fast, which is the
 * opposite of what people expect and worth saying before they wait.
 */

type CagFilingSelectProps = {
  value: string
  onChange: (ticker: string) => void
  disabled?: boolean
}

const STORAGE_KEY = 'encore.cagTicker'

export function CagFilingSelect({ value, onChange, disabled }: CagFilingSelectProps) {
  const [tickers, setTickers] = useState<string[]>([])

  useEffect(() => {
    let cancelled = false
    api
      .get<string[]>('/chat/cag-companies')
      .then((list) => {
        if (cancelled) return
        setTickers(list)
        if (!value) {
          const saved = localStorage.getItem(STORAGE_KEY)
          onChange(saved && list.includes(saved) ? saved : (list[0] ?? ''))
        }
      })
      .catch(() => {
        /* the mode will say plainly that no filing is loaded */
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function select(ticker: string) {
    localStorage.setItem(STORAGE_KEY, ticker)
    onChange(ticker)
  }

  if (tickers.length === 0) return null

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          disabled={disabled}
          className="h-7 gap-1.5 px-2 text-xs text-muted-foreground hover:text-foreground"
        >
          <FileText className="size-3.5" aria-hidden />
          {value || 'Pick a filing'}
          <ChevronDown className="size-3" aria-hidden />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="max-h-80 w-56 overflow-y-auto">
        {tickers.map((ticker) => (
          <DropdownMenuItem
            key={ticker}
            onClick={() => select(ticker)}
            className="flex items-center gap-2 py-1.5 text-sm"
          >
            {ticker}
            <span className="text-[11px] text-muted-foreground">most recent 10-K</span>
            {ticker === value ? (
              <Check className="ml-auto size-3.5 text-brand-600" aria-hidden />
            ) : null}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

/**
 * The one thing a first-time CAG user needs told: the first question is slow.
 *
 * Measured on an H100 with the NVDA 10-K: question one pays ~93s to prefill
 * 191,000 tokens; questions two and three came back in 45s and 33s with 99.96%
 * of the context served from cache. Without this notice the first question
 * looks like a hang.
 */
type WarmState = 'idle' | 'warming' | 'ready' | 'failed'

export function CagNotice({ ticker }: { ticker: string }) {
  const [state, setState] = useState<WarmState>('idle')
  const [seconds, setSeconds] = useState(0)

  // Warm on ticker change, not on send. The cost of CAG is entirely front
  // loaded -- ~90s to prefill 191,000 tokens, then ~2s per question -- so the
  // only question is whether the user waits before typing or after pressing
  // send. Before is better: they are reading the notice anyway, and a wait
  // after send is indistinguishable from a hang.
  useEffect(() => {
    if (!ticker) return
    let cancelled = false
    setState('warming')
    const started = Date.now()
    const tick = setInterval(() => setSeconds(Math.round((Date.now() - started) / 1000)), 1000)
    api
      .post<{ ok: boolean; warmed?: boolean }>(`/chat/cag-warm?ticker=${ticker}`, {})
      .then((r) => {
        if (!cancelled) setState(r?.ok ? 'ready' : 'failed')
      })
      .catch(() => {
        if (!cancelled) setState('failed')
      })
      .finally(() => clearInterval(tick))
    return () => {
      cancelled = true
      clearInterval(tick)
    }
  }, [ticker])

  if (state === 'warming') {
    return (
      <p className="px-1 pt-1 text-[11px] leading-snug text-amber-700 dark:text-amber-500">
        <Loader2 className="mr-1 inline size-3 animate-spin" aria-hidden />
        Loading {ticker}&apos;s 10-K into the model, about 190,000 tokens. {seconds}s
        so far, usually around 90. You can type your question now; it will be
        answered as soon as this finishes.
      </p>
    )
  }
  if (state === 'ready') {
    return (
      <p className="px-1 pt-1 text-[11px] leading-snug text-emerald-700 dark:text-emerald-500">
        {ticker}&apos;s 10-K is cached. Questions now answer in seconds, and the whole
        filing is in view, so nothing can be missed by a search. One company at a
        time, and no valuations here, they need too many rounds.
      </p>
    )
  }
  if (state === 'failed') {
    return (
      <p className="px-1 pt-1 text-[11px] leading-snug text-amber-700 dark:text-amber-500">
        Could not preload {ticker}. The first question will still work, it will
        just take around 90 seconds while the filing is cached.
      </p>
    )
  }
  return (
    <p className="px-1 pt-1 text-[11px] leading-snug text-muted-foreground">
      Pick a company to load its 10-K into the model.
    </p>
  )
}
