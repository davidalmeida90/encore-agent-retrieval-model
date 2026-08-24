import { citationLabel, type CitationPayload } from '@/lib/citations'
import { cn } from '@/lib/utils'

type CitationChipProps = {
  citation: CitationPayload
  selected?: boolean
  onSelect: (citation: CitationPayload) => void
}

export function CitationChip({ citation, selected, onSelect }: CitationChipProps) {
  return (
    <button
      type="button"
      onClick={() => onSelect(citation)}
      className={cn(
        'inline-flex max-w-full items-center gap-1.5 rounded-full border py-1 pr-3 pl-1 text-left text-xs transition-colors',
        selected
          ? 'border-foreground bg-foreground/5 text-foreground'
          : 'border-border bg-background text-muted-foreground hover:border-foreground/40 hover:text-foreground',
      )}
    >
      <span className="flex size-4 shrink-0 items-center justify-center rounded-full bg-foreground text-[0.6rem] font-semibold text-background tabular-nums">
        {citation.citationIndex}
      </span>
      <span className="truncate font-medium">{citationLabel(citation)}</span>
    </button>
  )
}
