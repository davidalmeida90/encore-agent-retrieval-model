import { createContext } from 'react'

import type { ThreadSummary } from '@/lib/chat'

export type ThreadsContextValue = {
  threads: ThreadSummary[]
  isLoading: boolean
  error: string | null
  refreshThreads: () => Promise<void>
  createNewThread: () => Promise<string>
  deleteThread: (threadId: string) => Promise<void>
}

export const ThreadsContext = createContext<ThreadsContextValue | null>(null)
