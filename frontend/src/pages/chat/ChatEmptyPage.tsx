import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { ChatInput } from '@/components/chat/ChatInput'
import { PromptSuggestion } from '@/components/ui/prompt-suggestion'
import { useThreads } from '@/hooks/useThreads'
import { EXAMPLE_QUESTIONS } from '@/lib/suggestions'

/**
 * ## The landing screen is a composer, not a menu
 *
 * This page used to offer four suggestions and nothing else: there was no way to
 * type your own question until a thread already existed, so the first thing a new
 * user could do was pick from a list. The suggestions are examples of what the
 * corpus can answer, not the only things it can answer, and the layout said
 * otherwise.
 *
 * The box now comes first and the suggestions sit under it as examples. Both
 * paths run through `startConversation`, so a typed question and a clicked one
 * open a thread the same way.
 */
export function ChatEmptyPage() {
  const navigate = useNavigate()
  const { createNewThread } = useThreads()
  const [isStarting, setIsStarting] = useState(false)
  const [model, setModel] = useState<string | null>(null)

  async function startConversation(prompt?: string) {
    if (isStarting) return
    setIsStarting(true)
    try {
      const id = await createNewThread()
      // The model choice travels with the first message: it is a property of the
      // question being asked, not of a thread that does not exist yet.
      navigate(`/chats/${id}`, prompt ? { state: { initialPrompt: prompt, model } } : undefined)
    } finally {
      setIsStarting(false)
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col justify-center gap-6 p-6">
      <div className="flex flex-col gap-3">
        <span className="text-xs font-medium tracking-widest text-brand-600 uppercase">
          Grounded equity research
        </span>
        <h1 className="text-3xl font-semibold tracking-tight text-foreground">
          Ask a filing what it says.
        </h1>
        <p className="max-w-lg text-sm leading-relaxed text-muted-foreground">
          Ten companies, 23 annual reports. Every claim about a filing is cited
          back to the passage it came from, and every figure comes from SEC XBRL
          rather than from reading the prose.
        </p>
      </div>

      {/* Same composer as inside a thread, so the model picker and the send
          affordance are in the same place before and after the first message. */}
      <div className="-mx-4">
        <ChatInput
          status={isStarting ? 'submitted' : 'ready'}
          model={model}
          onModelChange={setModel}
          onSend={(text) => void startConversation(text)}
          onStop={() => undefined}
        />
      </div>

      <div className="flex flex-col gap-2">
        <span className="text-xs text-muted-foreground">Or start from an example</span>
        <div className="grid w-full gap-2 sm:grid-cols-2">
          {EXAMPLE_QUESTIONS.map((question) => (
            <PromptSuggestion
              key={question}
              variant="outline"
              size="default"
              className="h-auto justify-start rounded-md px-4 py-3 text-left text-sm font-normal whitespace-normal transition-colors hover:border-brand-300 hover:bg-brand-50"
              disabled={isStarting}
              onClick={() => void startConversation(question)}
            >
              {question}
            </PromptSuggestion>
          ))}
        </div>
      </div>
    </div>
  )
}
