import { useEffect, useMemo, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Loader } from '@/components/ui/loader'
import { Markdown } from '@/components/ui/markdown'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'
import {
  getCitationContext,
  type CitationContext,
  type CitationContextChunk,
  type CitationContextTable,
} from '@/lib/chat'
import { type CitationPayload } from '@/lib/citations'
import { cn } from '@/lib/utils'

const SOURCE_MARKDOWN_CLASSES = cn(
  'space-y-2 text-xs leading-relaxed text-foreground',
  '[&_p]:leading-relaxed',
  '[&_a]:font-medium [&_a]:underline [&_a]:underline-offset-4',
  '[&_ul]:list-disc [&_ul]:pl-4 [&_ol]:list-decimal [&_ol]:pl-4 [&_li]:my-1',
  '[&_table]:w-full [&_table]:min-w-max [&_table]:border-collapse [&_table]:text-left',
  '[&_th]:border [&_th]:bg-muted [&_th]:px-2 [&_th]:py-1.5 [&_th]:font-semibold',
  '[&_td]:border [&_td]:px-2 [&_td]:py-1.5 [&_td]:align-top',
  '[&_blockquote]:border-l-2 [&_blockquote]:border-border [&_blockquote]:pl-3 [&_blockquote]:text-muted-foreground',
)

const CHUNK_LABELS: Record<CitationContextChunk['role'], string> = {
  previous: 'Previous context',
  anchor: 'Cited passage',
  next: 'Next context',
}

function pipeTableCells(line: string): string[] {
  const trimmed = line.trim()
  if (!trimmed.startsWith('|')) {
    return []
  }
  return trimmed
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((cell) => cell.trim())
}

function isPipeTableRow(line: string): boolean {
  return pipeTableCells(line).length > 1
}

function isSeparatorRow(line: string): boolean {
  const cells = pipeTableCells(line)
  return cells.length > 1 && cells.every((cell) => /^:?-{3,}:?$/.test(cell))
}

function separatorFor(row: string): string {
  const cells = pipeTableCells(row)
  return `| ${cells.map(() => '---').join(' | ')} |`
}

function normalizeMarkdownTables(text: string): string {
  const lines = text.split('\n')
  const output: string[] = []

  for (let index = 0; index < lines.length; ) {
    if (!isPipeTableRow(lines[index])) {
      output.push(lines[index])
      index += 1
      continue
    }

    const tableLines: string[] = []
    while (index < lines.length && isPipeTableRow(lines[index])) {
      tableLines.push(lines[index])
      index += 1
    }

    if (tableLines.length > 1 && !isSeparatorRow(tableLines[1])) {
      output.push(tableLines[0], separatorFor(tableLines[0]), ...tableLines.slice(1))
    } else {
      output.push(...tableLines)
    }
  }

  return output.join('\n')
}

function fallbackContext(citation: CitationPayload): CitationContext {
  return {
    anchorChunkId: citation.chunkId,
    documentId: '',
    ticker: citation.ticker,
    companyName: citation.companyName,
    form: citation.form,
    filingDate: citation.filingDate,
    sourceUrl: '',
    chunks: [
      {
        chunkId: citation.chunkId,
        chunkIndex: 0,
        role: 'anchor',
        text: citation.excerpt,
        page: citation.page,
        section: citation.section,
      },
    ],
  }
}

function SourceChunkCard({ chunk }: { chunk: CitationContextChunk }) {
  const isAnchor = chunk.role === 'anchor'
  const markdown = useMemo(() => normalizeMarkdownTables(chunk.text), [chunk.text])

  return (
    <section
      className={cn(
        'rounded-xl border p-3 shadow-xs',
        isAnchor ? 'border-primary/40 bg-primary/5' : 'border-border bg-muted/20',
      )}
    >
      <div className="mb-2 flex flex-wrap items-center gap-1.5">
        <span
          className={cn(
            'rounded-md px-2 py-1 text-[0.65rem] font-semibold tracking-wide uppercase',
            isAnchor ? 'bg-primary text-primary-foreground' : 'bg-background text-muted-foreground',
          )}
        >
          {CHUNK_LABELS[chunk.role]}
        </span>
        <Badge variant="outline">Chunk {chunk.chunkIndex}</Badge>
        {chunk.page ? <Badge variant="outline">Page {chunk.page}</Badge> : null}
        {chunk.section ? <Badge variant="outline">{chunk.section}</Badge> : null}
      </div>
      <div className="overflow-x-auto">
        <Markdown className={SOURCE_MARKDOWN_CLASSES}>{markdown}</Markdown>
      </div>
    </section>
  )
}

function SourceTableCard({ table }: { table: CitationContextTable }) {
  const markdown = useMemo(() => normalizeMarkdownTables(table.markdown), [table.markdown])

  return (
    <section className="rounded-xl border border-primary/40 bg-primary/5 p-3 shadow-xs">
      <div className="mb-2 flex flex-wrap items-center gap-1.5">
        <span className="rounded-md bg-primary px-2 py-1 text-[0.65rem] font-semibold tracking-wide text-primary-foreground uppercase">
          Normalized table
        </span>
        <Badge variant="outline">Table {table.tableIndex}</Badge>
        {table.title ? <Badge variant="outline">{table.title}</Badge> : null}
        {table.units ? <Badge variant="outline">{table.units}</Badge> : null}
      </div>
      <div className="overflow-x-auto">
        <Markdown className={SOURCE_MARKDOWN_CLASSES}>{markdown}</Markdown>
      </div>
    </section>
  )
}

type SourcePassageSheetProps = {
  citation: CitationPayload | null
  onOpenChange: (open: boolean) => void
}

export function SourcePassageSheet({ citation, onOpenChange }: SourcePassageSheetProps) {
  const [context, setContext] = useState<CitationContext | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!citation) {
      return
    }

    const activeCitation = citation
    let mounted = true

    async function load() {
      setContext(null)
      setLoading(true)
      setError(null)

      try {
        const nextContext = await getCitationContext(activeCitation.chunkId)
        if (mounted) {
          setContext(nextContext)
        }
      } catch {
        if (mounted) {
          setContext(fallbackContext(activeCitation))
          setError('Could not load neighboring chunks. Showing the saved excerpt.')
        }
      } finally {
        if (mounted) {
          setLoading(false)
        }
      }
    }

    void load()

    return () => {
      mounted = false
    }
  }, [citation])

  const activeContext =
    citation && context?.anchorChunkId === citation.chunkId ? context : null
  const resolvedContext = activeContext ?? (citation ? fallbackContext(citation) : null)
  const activeError = activeContext ? error : null

  return (
    <Sheet open={citation !== null} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="flex w-full flex-col gap-0 sm:max-w-xl lg:max-w-2xl">
        {citation ? (
          <>
            <SheetHeader className="border-b">
              <div className="flex items-center gap-2">
                <span className="flex size-6 shrink-0 items-center justify-center rounded-md bg-foreground text-xs font-semibold text-background tabular-nums">
                  {citation.citationIndex}
                </span>
                <SheetTitle className="text-base">
                  {citation.companyName ?? citation.ticker}
                </SheetTitle>
              </div>
              <SheetDescription className="sr-only">
                Source passage for citation {citation.citationIndex}
              </SheetDescription>
              <div className="flex flex-wrap gap-1.5 pt-1">
                <Badge variant="secondary">{citation.ticker}</Badge>
                <Badge variant="outline">{citation.form}</Badge>
                <Badge variant="outline">Filed {citation.filingDate}</Badge>
                {citation.page ? <Badge variant="outline">Page {citation.page}</Badge> : null}
                {citation.section ? (
                  <Badge variant="outline">{citation.section}</Badge>
                ) : null}
              </div>
            </SheetHeader>

            <div className="flex-1 space-y-3 overflow-y-auto p-4">
              <div>
                <p className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
                  Source context
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Neighboring chunks are shown around the cited passage for continuity.
                </p>
              </div>

              {loading ? (
                <div className="rounded-xl border border-dashed bg-muted/30 p-4">
                  <Loader variant="loading-dots" text="Loading source context" size="sm" />
                </div>
              ) : null}

              {activeError ? (
                <p
                  className="rounded-lg border border-dashed bg-muted/40 px-3 py-2 text-xs text-muted-foreground"
                  role="alert"
                >
                  {activeError}
                </p>
              ) : null}

              {!loading && resolvedContext?.table ? (
                <SourceTableCard table={resolvedContext.table} />
              ) : null}

              {!loading &&
                resolvedContext?.chunks.map((chunk) => (
                  <SourceChunkCard key={chunk.chunkId} chunk={chunk} />
                ))}
            </div>
          </>
        ) : null}
      </SheetContent>
    </Sheet>
  )
}
