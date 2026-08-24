import { useEffect, useState } from 'react'
import { ChevronRight } from 'lucide-react'

import { api } from '@/lib/api'

/**
 * What the agent can call, and the actual Python behind each tool.
 *
 * Source is read from the .py file server-side at request time, so this cannot
 * drift from what is really running. That matters because the schema the model
 * sees is derived from these exact signatures and docstrings: when a tool
 * misbehaves, the first question is what it was told the tool does.
 */

export type ToolInfo = {
  name: string
  group: string
  groupLabel: string
  description: string
  parameters: string[]
  source: string
  sourcePath: string
  line: number | null
  valuationGated: boolean
}

type RawToolInfo = {
  name: string
  group: string
  group_label: string
  description: string
  parameters: string[]
  source: string
  source_path: string
  line: number | null
  valuation_gated: boolean
}

function normalise(raw: RawToolInfo): ToolInfo {
  return {
    name: raw.name,
    group: raw.group,
    groupLabel: raw.group_label,
    description: raw.description,
    parameters: raw.parameters,
    source: raw.source,
    sourcePath: raw.source_path,
    line: raw.line,
    valuationGated: raw.valuation_gated,
  }
}

export function ToolsTab() {
  const [tools, setTools] = useState<ToolInfo[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<string | null>(null)

  useEffect(() => {
    if (tools) return
    let cancelled = false
    api.get<RawToolInfo[]>('/tools')
      .then((raw: RawToolInfo[]) => {
        if (!cancelled) setTools(raw.map(normalise))
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Could not load tools')
      })
    return () => {
      cancelled = true
    }
  }, [tools])

  // Preserve the server's ordering, which groups by the question each answers.
  const groups: { label: string; tools: ToolInfo[] }[] = []
  for (const tool of tools ?? []) {
    const last = groups[groups.length - 1]
    if (last && last.label === tool.groupLabel) last.tools.push(tool)
    else groups.push({ label: tool.groupLabel, tools: [tool] })
  }

  return (
    <>
      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3 show-scrollbar">
        {error ? <p className="px-1 text-sm text-rose-600">{error}</p> : null}
        {!tools && !error ? <p className="px-1 text-sm text-slate-500">Loading…</p> : null}

        {groups.map((group) => (
          <section key={group.label} className="mb-4">
            <h3 className="mb-1.5 px-1 text-xs font-medium tracking-wide text-slate-500 uppercase">
              {group.label}
            </h3>
            <ul className="space-y-1">
              {group.tools.map((tool) => {
                const isOpen = expanded === tool.name
                return (
                  <li key={tool.name} className="rounded-lg border border-slate-200 bg-white">
                    <button
                      type="button"
                      onClick={() => setExpanded(isOpen ? null : tool.name)}
                      className="flex w-full items-start gap-2 px-3 py-2 text-left"
                      aria-expanded={isOpen}
                    >
                      <ChevronRight
                        className={`mt-0.5 size-3.5 shrink-0 text-slate-400 transition-transform ${isOpen ? 'rotate-90' : ''}`}
                        aria-hidden
                      />
                      <span className="min-w-0 flex-1">
                        <span className="block font-mono text-xs font-medium text-slate-900">
                          {tool.name}
                        </span>
                        <span className="mt-0.5 block text-xs leading-snug text-slate-500">
                          {tool.description.split('\n')[0]}
                        </span>
                        {tool.valuationGated ? (
                          <span className="mt-1 inline-block rounded bg-amber-50 px-1.5 py-px text-[10px] font-medium text-amber-700">
                            only offered on valuation questions
                          </span>
                        ) : null}
                      </span>
                    </button>

                    {isOpen ? (
                      <div className="border-t border-slate-100 px-3 py-2">
                        <p className="mb-2 font-mono text-[11px] text-slate-400">
                          {tool.sourcePath}
                          {tool.line ? `:${tool.line}` : ''}
                        </p>
                        {tool.parameters.length > 0 ? (
                          <p className="mb-2 text-[11px] text-slate-500">
                            <span className="font-medium">{tool.parameters.length} parameters:</span>{' '}
                            {tool.parameters.join(', ')}
                          </p>
                        ) : null}
                        <pre className="max-h-96 overflow-auto rounded-md bg-slate-900 p-3 text-[11px] leading-relaxed text-slate-100 show-scrollbar">
                          <code>{tool.source}</code>
                        </pre>
                      </div>
                    ) : null}
                  </li>
                )
              })}
            </ul>
          </section>
        ))}
      </div>
    </>
  )
}

