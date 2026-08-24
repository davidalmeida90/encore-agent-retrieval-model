import { useContext } from 'react'

import { ThreadsContext } from '@/contexts/threads-context'

export function useThreads() {
  const context = useContext(ThreadsContext)
  if (!context) {
    throw new Error('useThreads must be used within ThreadsProvider')
  }
  return context
}
