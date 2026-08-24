import { useState } from 'react'
import type { ChatStatus } from 'ai'
import { ArrowUp, Square } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { ModelSelect } from '@/components/chat/ModelSelect'
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
  onSend: (text: string) => void
  onStop: () => void
}

export function ChatInput({ status, onSend, onStop, model, onModelChange }: ChatInputProps) {
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
          {/* Model picker sits inside the box, on the same row as send: the
              choice belongs to the message you are about to send, not to the
              page. Every serious chat product puts it here. */}
          <PromptInputActions className="items-center justify-between pt-1">
            <ModelSelect value={model ?? null} onChange={onModelChange} disabled={isBusy} />
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
        <p className="mt-2 text-center text-[11px] text-muted-foreground">
          Cited answers only. Check the source before relying on a number.
        </p>
      </div>
    </div>
  )
}
