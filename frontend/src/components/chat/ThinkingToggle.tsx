import { Brain } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { PromptInputAction } from '@/components/ui/prompt-input'
import { cn } from '@/lib/utils'

/**
 * Whether the model reasons before answering.
 *
 * Off by default, and that is a measured default rather than a cautious one. On
 * the local 27B, asking for a single tool call:
 *
 *   thinking off      61 output tokens   1.2s
 *   thinking on    1,833 output tokens   9.6s   -> identical tool call
 *
 * Decode is bandwidth bound and linear in output length, so thinking is close to
 * a 30x multiplier on the part that actually costs time. Most questions here are
 * lookups and retrieval, where it buys nothing.
 *
 * It earns its cost on valuation and multi-step analysis, where Flash Lite was
 * recorded skipping the sensitivity table and market comparison the loaded skill
 * requires. Hence a switch on the composer rather than a setting in a menu: it
 * belongs to the question, and the right answer changes question to question.
 */

type ThinkingToggleProps = {
  value: boolean
  onChange: (next: boolean) => void
  disabled?: boolean
}

export function ThinkingToggle({ value, onChange, disabled }: ThinkingToggleProps) {
  return (
    <PromptInputAction
      tooltip={
        value
          ? 'Thinking on: slower and far more output tokens, better on valuation'
          : 'Thinking off: fastest, and enough for lookups and retrieval'
      }
    >
      <Button
        type="button"
        variant="ghost"
        size="sm"
        disabled={disabled}
        aria-pressed={value}
        onClick={() => onChange(!value)}
        className={cn(
          'h-7 gap-1.5 px-2 text-xs',
          value
            ? 'bg-brand-50 text-brand-700 hover:bg-brand-100 hover:text-brand-800 dark:bg-brand-950 dark:text-brand-300'
            : 'text-muted-foreground hover:text-foreground',
        )}
      >
        <Brain className="size-3.5" aria-hidden />
        Thinking
      </Button>
    </PromptInputAction>
  )
}
