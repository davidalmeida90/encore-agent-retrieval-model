import { useEffect, useState } from 'react'
import { ChevronRight } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { api } from '@/lib/api'

/**
 * Browse what is actually indexed: ticker, then filing, then the real chunks.
 *
 * The agent can only answer from what was ingested, so when an answer looks
 * thin the first question is what it could possibly have found. Until now that
 * meant querying Postgres by hand.
 *
 * Chunks are paged rather than loaded whole: JPMorgan's 10-K alone runs to about
 * 4,600 of them.
 */

type Company = { ticker: string; company_name: string | null; filings: number; chunks: number }
type Document = {
  id: string
  ticker: string
  form: string
  fiscal_year: number | null
  filing_date: string | null
  chunks: number
}
type Chunk = {
  id: string
  chunk_index: number
  page: string | null
  section: string | null
  token_count: number | null
  kind: string | null
  text: string
}

const PAGE = 25

export function DocumentsTab() {
  const [companies, setCompanies] = useState<Company[] | null>(null)
  const [ticker, setTicker] = useState<string | null>(null)
  const [documents, setDocuments] = useState<Document[]>([])
  const [docId, setDocId] = useState<string | null>(null)
  const [chunks, setChunks] = useState<Chunk[]>([])
  const [offset, setOffset] = useState(0)

  useEffect(() => {
    if (companies) return
    api.get<Company[]>('/corpus/companies').then(setCompanies).catch(() => setCompanies([]))
  }, [companies])

  useEffect(() => {
    if (!ticker) return
    setDocId(null)
    setChunks([])
    api
      .get<Document[]>(`/corpus/documents?ticker=${ticker}`)
      .then(setDocuments)
      .catch(() => setDocuments([]))
  }, [ticker])

  useEffect(() => {
    if (!docId) return
    api
      .get<Chunk[]>(`/corpus/documents/${docId}/chunks?offset=${offset}&limit=${PAGE}`)
      .then(setChunks)
      .catch(() => setChunks([]))
  }, [docId, offset])

  const activeDoc = documents.find((d) => d.id === docId)

  return (
    <>
      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3 show-scrollbar">
        {/* breadcrumb back out of a drill-down */}
        {(ticker || docId) && (
          <button
            type="button"
            onClick={() => (docId ? setDocId(null) : setTicker(null))}
            className="mb-2 text-xs text-brand-600 hover:text-brand-800"
          >
            &larr; {docId ? ticker : 'All companies'}
          </button>
        )}

        {!ticker && (
          <ul className="divide-y divide-border border-y border-border">
            {(companies ?? []).map((c) => (
              <li key={c.ticker}>
                <button
                  type="button"
                  onClick={() => setTicker(c.ticker)}
                  className="flex w-full items-center gap-2 py-2 text-left text-sm hover:text-brand-700"
                >
                  <span className="w-14 shrink-0 font-mono text-xs font-medium">{c.ticker}</span>
                  <span className="min-w-0 flex-1 truncate text-xs text-muted-foreground">
                    {c.company_name ?? '—'}
                  </span>
                  <span className="shrink-0 font-mono text-[11px] text-muted-foreground">
                    {c.filings}f · {c.chunks.toLocaleString()}
                  </span>
                  <ChevronRight className="size-3 shrink-0 text-muted-foreground" aria-hidden />
                </button>
              </li>
            ))}
            {companies?.length === 0 && (
              <li className="py-2 text-sm text-muted-foreground">Nothing indexed yet.</li>
            )}
          </ul>
        )}

        {ticker && !docId && (
          <ul className="divide-y divide-border border-y border-border">
            {documents.map((d) => (
              <li key={d.id}>
                <button
                  type="button"
                  onClick={() => {
                    setDocId(d.id)
                    setOffset(0)
                  }}
                  className="flex w-full items-center gap-2 py-2 text-left text-sm hover:text-brand-700"
                >
                  <span className="font-medium">FY{d.fiscal_year ?? '?'}</span>
                  <span className="text-xs text-muted-foreground">{d.form}</span>
                  <span className="ml-auto font-mono text-[11px] text-muted-foreground">
                    {d.filing_date} · {d.chunks.toLocaleString()}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}

        {docId && (
          <>
            <p className="mb-2 text-[11px] text-muted-foreground">
              {activeDoc?.ticker} FY{activeDoc?.fiscal_year} · chunks {offset + 1}–
              {offset + chunks.length} of {activeDoc?.chunks.toLocaleString()}
            </p>
            <ul className="space-y-1.5">
              {chunks.map((c) => (
                <li key={c.id} className="rounded-md border border-border bg-background p-2.5">
                  <div className="mb-1 flex flex-wrap items-center gap-1.5 font-mono text-[10px] text-muted-foreground">
                    <span>#{c.chunk_index}</span>
                    {c.kind && <span className="rounded bg-muted px-1">{c.kind}</span>}
                    {c.section && <span className="truncate">{c.section}</span>}
                    {c.token_count != null && <span className="ml-auto">{c.token_count} tok</span>}
                  </div>
                  <p className="text-[11px] leading-relaxed whitespace-pre-wrap">{c.text}</p>
                </li>
              ))}
            </ul>
            <div className="mt-3 flex items-center justify-between">
              <Button
                variant="outline"
                size="sm"
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - PAGE))}
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={chunks.length < PAGE}
                onClick={() => setOffset(offset + PAGE)}
              >
                Next
              </Button>
            </div>
          </>
        )}
      </div>
    </>
  )
}
