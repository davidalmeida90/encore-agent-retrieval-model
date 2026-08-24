import { isDataUIPart, type UIMessage } from 'ai'

export type CitationPayload = {
  citationIndex: number
  chunkId: string
  excerpt: string
  ticker: string
  companyName?: string
  form: string
  filingDate: string
  page?: string
  section?: string
}

export type PipelineStage =
  | 'analyzing'
  | 'searching'
  | 'reading'
  | 'verifying'
  | 'retrying'
  | 'streaming'
  // Log-only stages. The transient status line filters these out; the activity
  // log shows them, because "which tool ran" is the whole reason it exists.
  | 'tool'
  | 'usage'
  | 'grounding'
  | 'usage_detail'
  | 'error'

export type PipelineStatus = {
  stage: PipelineStage
  message: string
}

function isCitationData(data: unknown): data is CitationPayload {
  if (typeof data !== 'object' || data === null) {
    return false
  }

  const record = data as Record<string, unknown>
  return (
    typeof record.citationIndex === 'number' &&
    typeof record.chunkId === 'string' &&
    typeof record.excerpt === 'string' &&
    typeof record.ticker === 'string' &&
    typeof record.form === 'string' &&
    typeof record.filingDate === 'string'
  )
}

export function isCitationPart(
  part: UIMessage['parts'][number],
): part is UIMessage['parts'][number] & { type: 'data-citation'; data: CitationPayload } {
  return isDataUIPart(part) && part.type === 'data-citation' && isCitationData(part.data)
}

export function isStatusPart(
  part: unknown,
): part is { type: 'data-status'; data: PipelineStatus } {
  if (typeof part !== 'object' || part === null) {
    return false
  }

  const record = part as Record<string, unknown>
  if (record.type !== 'data-status' || typeof record.data !== 'object' || record.data === null) {
    return false
  }

  const data = record.data as Record<string, unknown>
  return typeof data.stage === 'string' && typeof data.message === 'string'
}

export function citationsFromMessage(message: UIMessage): CitationPayload[] {
  return message.parts
    .filter(isCitationPart)
    .map((part) => part.data)
    .sort((a, b) => a.citationIndex - b.citationIndex)
}

export function textFromMessage(message: UIMessage): string {
  return message.parts
    .filter((part) => part.type === 'text')
    .map((part) => part.text)
    .join('')
}

export function citationLabel(citation: CitationPayload): string {
  const parts = [citation.ticker, citation.form, citation.filingDate]
  if (citation.page) {
    parts.push(`p.${citation.page}`)
  }
  return parts.join(' · ')
}

export function citationHeader(citation: CitationPayload): string {
  const company = citation.companyName ?? citation.ticker
  return `${company} · ${citation.form} · filed ${citation.filingDate}`
}

export function citationSubtitle(citation: CitationPayload): string | null {
  const parts: string[] = []
  if (citation.page) {
    parts.push(`Page ${citation.page}`)
  }
  if (citation.section) {
    parts.push(citation.section)
  }
  return parts.length > 0 ? parts.join(' · ') : null
}

export function citationByIndex(
  citations: CitationPayload[],
  index: number,
): CitationPayload | undefined {
  return citations.find((citation) => citation.citationIndex === index)
}
