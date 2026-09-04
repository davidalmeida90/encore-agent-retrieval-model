import { useMemo } from 'react'
import { DefaultChatTransport } from 'ai'

import { getAccessToken } from '@/lib/api'
import { isStatusPart, type PipelineStatus } from '@/lib/citations'
import { env } from '@/lib/env'

async function consumeStatusStream(
  stream: ReadableStream<Uint8Array>,
  onStatus: (status: PipelineStatus) => void,
) {
  const reader = stream.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) {
        break
      }

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) {
          continue
        }

        try {
          const parsed: unknown = JSON.parse(line.slice(6))
          if (isStatusPart(parsed)) {
            onStatus(parsed.data)
          }
        } catch {
          // Ignore malformed SSE payloads.
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}

export function useChatTransport(
  threadId: string,
  model?: string | null,
  retrievalMode?: string | null,
  thinking?: boolean,
  cagTicker?: string,
  onStatus?: (status: PipelineStatus) => void,
) {
  return useMemo(
    () =>
      new DefaultChatTransport({
        api: `${env.apiBaseUrl}/chat/stream`,
        headers: async (): Promise<Record<string, string>> => {
          const token = await getAccessToken()
          return token ? { Authorization: `Bearer ${token}` } : {}
        },
        prepareSendMessagesRequest: ({ messages }) => ({
          body: { threadId, messages, model, retrievalMode, thinking, cagTicker },
        }),
        fetch: async (input, init) => {
          const response = await fetch(input, init)

          if (!response.ok || !response.body || !onStatus) {
            return response
          }

          const [clientStream, statusStream] = response.body.tee()
          void consumeStatusStream(statusStream, onStatus)

          return new Response(clientStream, {
            status: response.status,
            statusText: response.statusText,
            headers: response.headers,
          })
        },
      }),
    [threadId, model, retrievalMode, thinking, cagTicker, onStatus],
  )
}
