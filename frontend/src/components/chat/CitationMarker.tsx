import { cn } from '@/lib/utils'

type CitationMarkerProps = {
  index: number
  selected?: boolean
  onSelect: (index: number) => void
}

export function CitationMarker({ index, selected, onSelect }: CitationMarkerProps) {
  return (
    <button
      type="button"
      onClick={() => onSelect(index)}
      aria-label={`Show source ${index}`}
      className={cn(
        // Superscript numeral with a hairline underline, as a footnote is set in
        // print. The filled grey pill reads as a chat-app chip and competes with
        // the prose; a footnote should be findable without shouting.
        'mx-px inline-flex items-baseline align-super text-[0.7em] font-medium tabular-nums leading-none transition-colors',
        'border-b border-dotted',
        selected
          ? 'border-brand-600 text-brand-700'
          : 'border-brand-300 text-brand-600 hover:border-brand-700 hover:text-brand-800',
      )}
    >
      {index}
    </button>
  )
}
