import { useEffect, useState } from 'react'
import { Check, ChevronDown } from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { api } from '@/lib/api'

/**
 * How the next question looks for filing text.
 *
 * Sits beside the model picker because it is the same kind of choice: it
 * belongs to the question you are about to ask, and it changes the answer
 * rather than the appearance of one.
 *
 * The two modes fail in opposite directions, which is why both are offered
 * rather than one being chosen for you. Measured over the same six questions:
 *
 *   RAG      98k input tokens   21 citations   limited to the indexed corpus
 *   agentic  798k input tokens   9 citations   reaches any public filer
 *
 * Agentic costs roughly seven times as much per question, because whole filings
 * pass through a sub-model rather than being looked up in an index. It earns
 * that when the company you are asking about was never ingested, and only then.
 */

export type RetrievalChoice = {
  id: string
  label: string
  hint: string
  default: boolean
  /** The thing worth knowing before picking this one. */
  caveat: string | null
}

const STORAGE_KEY = 'encore.retrieval'

type RetrievalSelectProps = {
  value: string | null
  onChange: (id: string) => void
  disabled?: boolean
}

export function RetrievalSelect({ value, onChange, disabled }: RetrievalSelectProps) {
  const [modes, setModes] = useState<RetrievalChoice[]>([])

  useEffect(() => {
    let cancelled = false
    api
      .get<RetrievalChoice[]>('/chat/retrieval-modes')
      .then((list) => {
        if (cancelled) return
        setModes(list)
        if (!value) {
          const saved = localStorage.getItem(STORAGE_KEY)
          const known = list.find((m) => m.id === saved)
          onChange(known?.id ?? list.find((m) => m.default)?.id ?? list[0]?.id)
        }
      })
      .catch(() => {
        /* selector is optional; the backend falls back to its default */
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function select(id: string) {
    localStorage.setItem(STORAGE_KEY, id)
    onChange(id)
  }

  const active = modes.find((m) => m.id === value)
  if (modes.length === 0) return null

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          disabled={disabled}
          className="h-7 gap-1.5 px-2 text-xs text-muted-foreground hover:text-foreground"
        >
          {active?.label ?? 'Retrieval'}
          <ChevronDown className="size-3" aria-hidden />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-80">
        {modes.map((mode) => (
          <DropdownMenuItem
            key={mode.id}
            onClick={() => select(mode.id)}
            className="flex-col items-start gap-0.5 py-2"
          >
            <span className="flex w-full items-center gap-2 text-sm font-medium">
              {mode.label}
              {mode.id === value ? (
                <Check className="ml-auto size-3.5 text-brand-600" aria-hidden />
              ) : null}
            </span>
            <span className="text-[11px] leading-snug text-muted-foreground">
              {mode.hint}
            </span>
            {mode.caveat ? (
              <span className="text-[11px] leading-snug text-amber-700 dark:text-amber-500">
                {mode.caveat}
              </span>
            ) : null}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
