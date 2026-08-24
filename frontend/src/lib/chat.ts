import type { UIMessage } from 'ai'

import { api } from '@/lib/api'

export type { UIMessage }

export type ThreadSummary = {
  id: string
  title: string
  createdAt: string
  updatedAt: string
}

type ThreadListResponse = {
  threads: ThreadSummary[]
}

type MessageHistoryResponse = {
  messages: UIMessage[]
}

export type CitationContextChunk = {
  chunkId: string
  chunkIndex: number
  role: 'previous' | 'anchor' | 'next'
  text: string
  page?: string
  section?: string
}

export type CitationContextTable = {
  tableIndex: number
  title?: string
  units?: string
  markdown: string
  tableData: Record<string, unknown>
}

export type CitationContext = {
  anchorChunkId: string
  documentId: string
  ticker: string
  companyName?: string
  form: string
  filingDate: string
  sourceUrl: string
  chunks: CitationContextChunk[]
  table?: CitationContextTable
}

export async function listThreads(): Promise<ThreadSummary[]> {
  const response = await api.get<ThreadListResponse>('/chat/threads')
  return response.threads
}

export async function createThread(title?: string): Promise<ThreadSummary> {
  return api.post<ThreadSummary>('/chat/threads', title ? { title } : {})
}

export async function deleteThread(threadId: string): Promise<void> {
  await api.delete<void>(`/chat/threads/${threadId}`)
}

export async function getThreadMessages(threadId: string): Promise<UIMessage[]> {
  const response = await api.get<MessageHistoryResponse>(
    `/chat/threads/${threadId}/messages`,
  )
  return response.messages
}

export async function getCitationContext(
  chunkId: string,
  radius = 1,
): Promise<CitationContext> {
  const params = new URLSearchParams({ radius: String(radius) })
  return api.get<CitationContext>(
    `/chat/citations/${encodeURIComponent(chunkId)}/context?${params.toString()}`,
  )
}
