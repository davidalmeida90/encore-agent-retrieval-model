import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useLocation, useNavigate, useParams } from 'react-router-dom'
import { useChat } from '@ai-sdk/react'
import type { UIMessage } from 'ai'

import { useActivityLog } from '@/components/chat/ChatLayout'
import { ChatError } from '@/components/chat/ChatError'
import { ChatInput } from '@/components/chat/ChatInput'
import { MessageList } from '@/components/chat/MessageList'
import { SourcePassageSheet } from '@/components/chat/SourcePassageSheet'
import { Loader } from '@/components/ui/loader'
import { useChatTransport } from '@/hooks/useChatTransport'
import { useThreads } from '@/hooks/useThreads'
import { classifyChatError } from '@/lib/chat-errors'
import { getThreadMessages } from '@/lib/chat'
import { type CitationPayload, type PipelineStatus as PipelineStatusState } from '@/lib/citations'
import { ApiError } from '@/lib/http'

type ChatThreadViewProps = {
  threadId: string
  initialMessages: UIMessage[]
}

function ChatThreadView({ threadId, initialMessages }: ChatThreadViewProps) {
  const navigate = useNavigate()
  const location = useLocation()
  const { refreshThreads } = useThreads()
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatusState | null>(null)
  const [selectedCitation, setSelectedCitation] = useState<CitationPayload | null>(null)
  const { push: pushLogEntry, clear: clearLog } = useActivityLog()
  // A model picked on the landing screen travels here with the first message.
  // Read as the INITIAL state rather than set in an effect: the transport is
  // built from `model` on the first render, and a later setState would send the
  // starter prompt through a transport still pointing at the default.
  const navState = location.state as {
    initialPrompt?: string
    model?: string | null
    retrievalMode?: string | null
    thinking?: boolean
    cagTicker?: string
  } | null
  const [model, setModel] = useState<string | null>(navState?.model ?? null)
  const [retrievalMode, setRetrievalMode] = useState<string | null>(
    navState?.retrievalMode ?? null,
  )
  // Off unless this question asks for it; see ThinkingToggle for the numbers.
  const [thinking, setThinking] = useState(navState?.thinking ?? false)
  const [cagTicker, setCagTicker] = useState(navState?.cagTicker ?? '')

  // Every status event does two jobs: it updates the transient line under the
  // composer, and it appends to the activity log so the whole run stays readable
  // after the fact.
  const handleStatus = useCallback(
    (status: PipelineStatusState) => {
      // Raw tool/usage events are for the log only; showing them under the
      // composer would replace a readable "Searching SEC filings…" with
      // "get_sec_financials(AAPL annual: capex)".
      if (!['tool', 'usage', 'grounding', 'usage_detail'].includes(status.stage)) {
        setPipelineStatus(status)
      }
      pushLogEntry(status)
    },
    [pushLogEntry],
  )

  const transport = useChatTransport(threadId, model, retrievalMode, thinking, cagTicker, handleStatus)

  const { messages, sendMessage, status, error, stop } = useChat({
    id: threadId,
    messages: initialMessages,
    transport,
    onFinish: () => {
      setPipelineStatus(null)
      void refreshThreads()
    },
  })

  function send(text: string) {
    clearLog()
    setPipelineStatus(null)
    setSelectedCitation(null)
    void sendMessage({ text })
  }

  // Auto-send a starter prompt forwarded from the empty page, once.
  const initialPrompt = navState?.initialPrompt
  const sentInitial = useRef(false)
  useEffect(() => {
    if (!initialPrompt || sentInitial.current) return
    sentInitial.current = true
    navigate(location.pathname, { replace: true, state: null })
    send(initialPrompt)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialPrompt])

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <MessageList
        messages={messages}
        status={status}
        pipelineStatus={pipelineStatus}
        selectedCitationIndex={selectedCitation?.citationIndex ?? null}
        onSelectCitation={setSelectedCitation}
        onSendSuggestion={send}
      />

      {error ? (
        <div className="mx-auto w-full max-w-3xl px-4 pb-2">
          <ChatError error={error} />
        </div>
      ) : null}

      <ChatInput
        status={status}
        onSend={send}
        onStop={stop}
        model={model}
        onModelChange={setModel}
        retrievalMode={retrievalMode}
        onRetrievalModeChange={setRetrievalMode}
        thinking={thinking}
        onThinkingChange={setThinking}
        cagTicker={cagTicker}
        onCagTickerChange={setCagTicker}
      />

      <SourcePassageSheet
        citation={selectedCitation}
        onOpenChange={(open) => {
          if (!open) setSelectedCitation(null)
        }}
      />
    </div>
  )
}

function ChatThreadLoader({ threadId }: { threadId: string }) {
  const navigate = useNavigate()
  const [initialMessages, setInitialMessages] = useState<UIMessage[] | null>(null)
  const [loadError, setLoadError] = useState<Error | null>(null)
  const [reloadKey, setReloadKey] = useState(0)

  useEffect(() => {
    let mounted = true

    async function load() {
      setLoadError(null)

      try {
        const messages = await getThreadMessages(threadId)
        if (mounted) {
          setInitialMessages(messages)
        }
      } catch (err) {
        if (!mounted) {
          return
        }

        const error = err instanceof Error ? err : new Error('Could not load this conversation.')

        if (err instanceof ApiError && err.status === 404) {
          navigate('/chats', { replace: true })
          return
        }

        if (err instanceof ApiError && err.status === 401) {
          navigate('/login', { replace: true })
          return
        }

        setLoadError(error)
      }
    }

    void load()

    return () => {
      mounted = false
    }
  }, [threadId, navigate, reloadKey])

  if (loadError) {
    const classified = classifyChatError(loadError)

    return (
      <div className="flex flex-1 items-center justify-center p-6">
        <div className="max-w-md space-y-3 text-center">
          <p className="text-sm font-medium text-destructive" role="alert">
            {classified.title}
          </p>
          <p className="text-sm text-muted-foreground">{classified.message}</p>
          {classified.showLoginLink ? (
            <Link to="/login" className="text-sm font-medium underline underline-offset-4">
              Sign in again
            </Link>
          ) : (
            <button
              type="button"
              className="text-sm font-medium underline underline-offset-4"
              onClick={() => setReloadKey((value) => value + 1)}
            >
              Try again
            </button>
          )}
        </div>
      </div>
    )
  }

  if (initialMessages === null) {
    return (
      <div className="flex flex-1 items-center justify-center p-6">
        <Loader variant="text-shimmer" text="Loading conversation…" />
      </div>
    )
  }

  return <ChatThreadView threadId={threadId} initialMessages={initialMessages} />
}

export function ChatThreadPage() {
  const { threadId } = useParams()

  if (!threadId) {
    return null
  }

  return <ChatThreadLoader key={threadId} threadId={threadId} />
}
