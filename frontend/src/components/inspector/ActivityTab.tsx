import { useEffect, useRef } from 'react'

import type { PipelineStatus } from '@/lib/citations'

/**
 * Running log of what the agent actually did.
 *
 * The backend already streams a status event for every stage and every tool
 * call; the header line only ever showed the latest one, so the sequence was
 * invisible. This keeps the whole run: which tools fired, in what order, how
 * long each took, and whether the grounding check sent it back for another try.
 */

export type LogEntry = PipelineStatus & { at: number }

/** Stage -> colour. Retry and error stand out; routine work stays quiet. */
const STAGE_TONE: Record<string, string> = {
  analyzing: 'text-brand-600',
  searching: 'text-brand-600',
  tool: 'text-slate-600',
  verifying: 'text-amber-600',
  retrying: 'text-amber-700 font-medium',
  streaming: 'text-emerald-600',
  grounding: 'text-emerald-700 font-medium',
  usage: 'text-slate-500',
  error: 'text-rose-600 font-medium',
}

function elapsed(entry: LogEntry, first: number) {
  const seconds = (entry.at - first) / 1000
  return `${seconds.toFixed(1)}s`
}

export function ActivityTab({ entries }: { entries: LogEntry[] }) {
  const endRef = useRef<HTMLDivElement>(null)

  // Follow the tail while a run is in progress.
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' })
  }, [entries.length])

  const first = entries[0]?.at ?? Date.now()

  return (
    <>
      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3 show-scrollbar">
        {entries.length === 0 ? (
          <p className="px-1 text-sm text-slate-500">
            Nothing yet. Ask a question and every retrieval, tool call and
            verification step will appear here in order.
          </p>
        ) : (
          <ol className="space-y-1.5">
            {entries.map((entry, i) => (
              <li key={`${entry.at}-${i}`} className="flex gap-2 text-xs leading-relaxed">
                <span className="w-10 shrink-0 pt-px text-right font-mono text-slate-400">
                  {elapsed(entry, first)}
                </span>
                <span className="min-w-0">
                  <span className={STAGE_TONE[entry.stage] ?? 'text-slate-600'}>
                    {entry.stage}
                  </span>
                  <span className="text-slate-500"> · {entry.message}</span>
                </span>
              </li>
            ))}
          </ol>
        )}
        <div ref={endRef} />
      </div>

      <footer className="border-t border-slate-200 px-4 py-2 text-xs text-slate-500">
        {entries.length} event{entries.length === 1 ? '' : 's'}
      </footer>
    </>
  )
}

