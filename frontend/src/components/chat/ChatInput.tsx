import { useState } from 'react'
import type { ChatStatus } from 'ai'
import { ArrowUp, Square } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { ModelSelect } from '@/components/chat/ModelSelect'
import { RetrievalSelect } from '@/components/chat/RetrievalSelect'
import { ThinkingToggle } from '@/components/chat/ThinkingToggle'
import { CagFilingSelect, CagNotice } from '@/components/chat/CagFilingSelect'
import {
  PromptInput,
  PromptInputAction,
  PromptInputActions,
  PromptInputTextarea,
} from '@/components/ui/prompt-input'

type ChatInputProps = {
  status: ChatStatus
  model?: string | null
  onModelChange: (id: string) => void
  retrievalMode?: string | null
  onRetrievalModeChange: (id: string) => void
  thinking: boolean
  onThinkingChange: (next: boolean) => void
  cagTicker: string
  onCagTickerChange: (ticker: string) => void
  onSend: (text: string) => void
  onStop: () => void
}

export function ChatInput({
  status,
  onSend,
  onStop,
  model,
  onModelChange,
  retrievalMode,
  onRetrievalModeChange,
  thinking,
  onThinkingChange,
  cagTicker,
  onCagTickerChange,
}: ChatInputProps) {
  const [input, setInput] = useState('')
  const isBusy = status === 'submitted' || status === 'streaming'

  function submit() {
    const text = input.trim()
    if (!text || isBusy) return
    onSend(text)
    setInput('')
  }

  return (
    <div className="bg-background px-4 pb-4">
      <div className="mx-auto w-full max-w-4xl">
        <PromptInput
          value={input}
          onValueChange={setInput}
          isLoading={isBusy}
          onSubmit={submit}
          className="rounded-2xl"
        >
          <PromptInputTextarea placeholder="Ask about a company…" />
          {/* Both pickers sit inside the box, on the same row as send: the
              choices belong to the message you are about to send, not to the
              page. Which model answers, and how it looks things up. */}
          <PromptInputActions className="items-center justify-between pt-1">
            <div className="flex items-center gap-1">
              <ModelSelect value={model ?? null} onChange={onModelChange} disabled={isBusy} />
              <span className="text-muted-foreground/40" aria-hidden>
                &middot;
              </span>
              <RetrievalSelect
                value={retrievalMode ?? null}
                onChange={onRetrievalModeChange}
                disabled={isBusy}
              />
              <span className="text-muted-foreground/40" aria-hidden>
                &middot;
              </span>
              <ThinkingToggle
                value={thinking}
                onChange={onThinkingChange}
                disabled={isBusy}
              />
              {retrievalMode === 'cag' ? (
                <>
                  <span className="text-muted-foreground/40" aria-hidden>
                    &middot;
                  </span>
                  <CagFilingSelect
                    value={cagTicker}
                    onChange={onCagTickerChange}
                    disabled={isBusy}
                  />
                </>
              ) : null}
            </div>
            {isBusy ? (
              <PromptInputAction tooltip="Stop">
                <Button type="button" size="icon" className="rounded-md" onClick={onStop}>
                  <Square className="size-4 fill-current" />
                </Button>
              </PromptInputAction>
            ) : (
              <PromptInputAction tooltip="Send">
                <Button
                  type="button"
                  size="icon"
                  className="rounded-md"
                  onClick={submit}
                  disabled={input.trim() === ''}
                  aria-label="Send message"
                >
                  <ArrowUp className="size-4" />
                </Button>
              </PromptInputAction>
            )}
          </PromptInputActions>
        </PromptInput>
        {/* Only in CAG: the first question is slow and that needs saying. */}
        {retrievalMode === 'cag' ? <CagNotice ticker={cagTicker} /> : null}
        <p className="mt-2 text-center text-[11px] text-muted-foreground">
          Cited answers only. Check the source before relying on a number.
        </p>
      </div>
    </div>
  )
}
