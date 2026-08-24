import { useState } from 'react'
import type { UIMessage } from 'ai'
import { Check, Copy } from 'lucide-react'

import { AssistantMarkdown } from '@/components/chat/AssistantMarkdown'
import { CitationChip } from '@/components/chat/CitationChip'
import { Button } from '@/components/ui/button'
import {
  citationsFromMessage,
  textFromMessage,
  type CitationPayload,
} from '@/lib/citations'

type AssistantMessageProps = {
  message: UIMessage
  selectedCitationIndex: number | null
  onSelectCitation: (citation: CitationPayload) => void
  isStreaming?: boolean
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)

  async function handleCopy() {
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return (
    <Button
      type="button"
      variant="ghost"
      size="icon-sm"
      className="text-muted-foreground"
      onClick={() => void handleCopy()}
      aria-label="Copy answer"
    >
      {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
    </Button>
  )
}

export function AssistantMessage({
  message,
  selectedCitationIndex,
  onSelectCitation,
  isStreaming = false,
}: AssistantMessageProps) {
  const text = textFromMessage(message)
  const citations = citationsFromMessage(message)
  // An answer with no citations is NOT necessarily unsupported. Figures from
  // SEC XBRL, market data and the valuation engines are tool facts with no chunk
  // to cite, and the grounding validator accepts them deliberately. Calling that
  // "no filing evidence" told the user a correct, sourced number was unbacked.
  // Only a claim that CITES something and then loses it is worth warning about.
  const citesFilings = /\[\d+\]/.test(text)
  const hasNoEvidence =
    !isStreaming && text.length > 0 && citations.length === 0 && citesFilings

  // An assistant turn with no text and nothing streaming has nothing to show.
  // Rendering it anyway produced a bare "ANSWER" heading above the real one,
  // because a failed attempt leaves an empty assistant message in the thread.
  if (!text && !isStreaming && citations.length === 0) return null

  return (
    <div className="min-w-0 space-y-3">
      <div className="flex items-baseline gap-3">
        <span className="text-[11px] font-medium tracking-widest text-muted-foreground uppercase">
          Answer
        </span>
        {citations.length > 0 ? (
          <span className="text-[11px] text-muted-foreground">
            {citations.length} source{citations.length === 1 ? '' : 's'}
          </span>
        ) : null}
      </div>

      {citations.length > 0 ? (
        <div className="flex flex-wrap gap-1.5">
          {citations.map((citation) => (
            <CitationChip
              key={`${citation.chunkId}-${citation.citationIndex}`}
              citation={citation}
              selected={selectedCitationIndex === citation.citationIndex}
              onSelect={onSelectCitation}
            />
          ))}
        </div>
      ) : null}

      {text ? (
        <AssistantMarkdown
          text={text}
          citations={citations}
          selectedCitationIndex={selectedCitationIndex}
          onSelectCitation={onSelectCitation}
        />
      ) : null}

      {isStreaming && text ? (
        <span className="inline-block h-4 w-2 translate-y-0.5 animate-pulse rounded-sm bg-foreground" />
      ) : null}

      {hasNoEvidence ? (
        <p className="rounded-lg border border-dashed border-amber-300 bg-amber-50/60 px-3 py-2 text-xs text-amber-800">
          This answer references filings but no source passage was attached.
          Treat the cited claims as unverified.
        </p>
      ) : null}



      {!isStreaming && text ? (
        <div className="flex items-center gap-1 pt-0.5">
          <CopyButton text={text} />
        </div>
      ) : null}
    </div>
  )
}
