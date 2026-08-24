import type { ChatStatus, UIMessage } from 'ai'

import { MessageBubble } from '@/components/chat/MessageBubble'
import { PipelineStatus } from '@/components/chat/PipelineStatus'
import {
  ChatContainerContent,
  ChatContainerRoot,
} from '@/components/ui/chat-container'
import { ScrollButton } from '@/components/ui/scroll-button'
import {
  textFromMessage,
  type CitationPayload,
  type PipelineStatus as PipelineStatusState,
} from '@/lib/citations'
import { EXAMPLE_QUESTIONS } from '@/lib/suggestions'

type MessageListProps = {
  messages: UIMessage[]
  status: ChatStatus
  pipelineStatus: PipelineStatusState | null
  selectedCitationIndex: number | null
  onSelectCitation: (citation: CitationPayload) => void
  onSendSuggestion: (text: string) => void
}

export function MessageList({
  messages,
  status,
  pipelineStatus,
  selectedCitationIndex,
  onSelectCitation,
  onSendSuggestion,
}: MessageListProps) {
  const isBusy = status === 'submitted' || status === 'streaming'
  const lastMessage = messages[messages.length - 1]
  const lastIsStreamingAssistant =
    status === 'streaming' &&
    lastMessage?.role === 'assistant' &&
    textFromMessage(lastMessage).length > 0

  // Show the pipeline block while the model is working but before answer text arrives.
  const showPipeline = isBusy && !lastIsStreamingAssistant

  return (
    <ChatContainerRoot className="relative flex-1">
      <ChatContainerContent className="mx-auto w-full max-w-4xl gap-0 divide-y divide-border px-4 py-6 [&>*]:py-6">
        {messages.length === 0 ? (
          // Left-aligned and listed, matching the empty-app page. Centred text
          // with a row of pills is the shape every chat product ships, and it
          // was the last place this still looked like one.
          <div className="flex flex-1 flex-col justify-center gap-5 py-12">
            <div className="space-y-1.5">
              <span className="text-[11px] font-medium tracking-widest text-brand-600 uppercase">
                Start here
              </span>
              <h2 className="text-xl font-semibold tracking-tight text-foreground">
                Ask about a company
              </h2>
            </div>
            <ul className="divide-y divide-border border-y border-border">
              {EXAMPLE_QUESTIONS.map((question) => (
                <li key={question}>
                  <button
                    type="button"
                    onClick={() => onSendSuggestion(question)}
                    className="group flex w-full items-start gap-3 py-2.5 text-left text-sm text-foreground transition-colors hover:text-brand-700"
                  >
                    <span className="pt-0.5 font-mono text-[11px] text-muted-foreground group-hover:text-brand-600">
                      &rarr;
                    </span>
                    <span className="min-w-0">{question}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        {messages.map((message) => (
          <MessageBubble
            key={message.id}
            message={message}
            selectedCitationIndex={selectedCitationIndex}
            onSelectCitation={onSelectCitation}
            isStreaming={message === lastMessage && lastIsStreamingAssistant}
          />
        ))}

        {showPipeline ? (
          <PipelineStatus
            isSubmitted={status === 'submitted'}
            pipelineStatus={pipelineStatus}
          />
        ) : null}
      </ChatContainerContent>

      <div className="pointer-events-none absolute inset-x-0 bottom-4 flex justify-center">
        <div className="pointer-events-auto">
          <ScrollButton />
        </div>
      </div>
    </ChatContainerRoot>
  )
}
