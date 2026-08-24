import { ApiError } from '@/lib/http'

export type ChatErrorKind =
  | 'network'
  | 'auth'
  | 'grounding'
  | 'retrieval'
  | 'generic'

export type ClassifiedChatError = {
  kind: ChatErrorKind
  title: string
  message: string
  showLoginLink: boolean
}

function messageIncludesAny(text: string, needles: string[]): boolean {
  const lower = text.toLowerCase()
  return needles.some((needle) => lower.includes(needle.toLowerCase()))
}

export function classifyChatError(error: Error): ClassifiedChatError {
  if (error instanceof ApiError) {
    if (error.isNetworkError) {
      return {
        kind: 'network',
        title: 'Connection problem',
        message:
          "Can't reach the server. Check your connection and that the backend is running.",
        showLoginLink: false,
      }
    }

    if (error.status === 401) {
      return {
        kind: 'auth',
        title: 'Session expired',
        message: 'Your sign-in session has expired. Please sign in again.',
        showLoginLink: true,
      }
    }

    if (error.status === 403) {
      return {
        kind: 'auth',
        title: 'Access denied',
        message: 'You do not have access to this conversation.',
        showLoginLink: false,
      }
    }
  }

  const text = error.message || ''

  if (
    messageIncludesAny(text, [
      'grounding',
      'citation',
      'verified against source',
      'source passages',
      'fully verify',
    ])
  ) {
    return {
      kind: 'grounding',
      title: 'Answer not verified',
      message:
        text ||
        'The answer could not be verified against source documents. Try rephrasing your question.',
      showLoginLink: false,
    }
  }

  if (messageIncludesAny(text, ['assistant run failed', 'retrieval', 'search'])) {
    return {
      kind: 'retrieval',
      title: 'Search failed',
      message: 'Search or analysis failed. Please try again.',
      showLoginLink: false,
    }
  }

  if (messageIncludesAny(text, ['failed to fetch', 'network', 'cors'])) {
    return {
      kind: 'network',
      title: 'Connection problem',
      message:
        "Can't reach the server. Check your connection and that the backend is running.",
      showLoginLink: false,
    }
  }

  return {
    kind: 'generic',
    title: 'Something went wrong',
    message: text || 'Something went wrong while sending your message.',
    showLoginLink: false,
  }
}
