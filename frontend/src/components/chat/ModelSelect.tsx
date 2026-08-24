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
 * Which model answers the next question.
 *
 * Sits under the composer rather than in the header because it belongs to the
 * question you are about to ask, not to the app.
 *
 * The choice is genuinely consequential rather than cosmetic: on the same DCF
 * question, Flash Lite spent 8,167 tokens and skipped the sensitivity table and
 * market comparison the loaded skill requires, while the reasoning model spent
 * 38,605 and produced both unprompted. Cheap is right for lookups; it is the
 * wrong default for judgement.
 */

export type ModelChoice = {
  id: string
  label: string
  hint: string
  default: boolean
}

const STORAGE_KEY = 'encore.model'

type ModelSelectProps = {
  value: string | null
  onChange: (id: string) => void
  disabled?: boolean
}

export function ModelSelect({ value, onChange, disabled }: ModelSelectProps) {
  const [models, setModels] = useState<ModelChoice[]>([])

  useEffect(() => {
    let cancelled = false
    api
      .get<ModelChoice[]>('/chat/models')
      .then((list) => {
        if (cancelled) return
        setModels(list)
        if (!value) {
          // Remembered choice wins, else whichever the backend marks default.
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

  const active = models.find((m) => m.id === value)
  if (models.length === 0) return null

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          disabled={disabled}
          className="h-7 gap-1.5 px-2 text-xs text-muted-foreground hover:text-foreground"
        >
          {active?.label ?? 'Model'}
          <ChevronDown className="size-3" aria-hidden />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-80">
        {models.map((model) => (
          <DropdownMenuItem
            key={model.id}
            onClick={() => select(model.id)}
            className="flex-col items-start gap-0.5 py-2"
          >
            <span className="flex w-full items-center gap-2 text-sm font-medium">
              {model.label}
              {model.id === value ? (
                <Check className="ml-auto size-3.5 text-brand-600" aria-hidden />
              ) : null}
            </span>
            <span className="text-[11px] leading-snug text-muted-foreground">
              {model.hint}
            </span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
