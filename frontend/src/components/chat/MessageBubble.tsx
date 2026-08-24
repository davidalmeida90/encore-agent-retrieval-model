import type { UIMessage } from 'ai'

import { AssistantMessage } from '@/components/chat/AssistantMessage'
import { textFromMessage, type CitationPayload } from '@/lib/citations'

type MessageBubbleProps = {
  message: UIMessage
  selectedCitationIndex: number | null
  onSelectCitation: (citation: CitationPayload) => void
  isStreaming?: boolean
}

export function MessageBubble({
  message,
  selectedCitationIndex,
  onSelectCitation,
  isStreaming,
}: MessageBubbleProps) {
  if (message.role === 'assistant') {
    return (
      <AssistantMessage
        message={message}
        selectedCitationIndex={selectedCitationIndex}
        onSelectCitation={onSelectCitation}
        isStreaming={isStreaming}
      />
    )
  }

  const text = textFromMessage(message)

  return (
    <div className="sticky top-0 z-10 -mx-4 space-y-1.5 bg-background/95 px-4 py-2 backdrop-blur">
      <span className="text-[11px] font-medium tracking-widest text-brand-600 uppercase">
        Question
      </span>
      <p className="text-base leading-relaxed font-medium whitespace-pre-wrap text-foreground">
        {text}
      </p>
    </div>
  )
}
