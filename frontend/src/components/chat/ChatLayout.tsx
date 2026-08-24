import { createContext, useCallback, useContext, useMemo, useState } from 'react'
import { Outlet, useParams } from 'react-router-dom'
import type { LogEntry } from '@/components/inspector/ActivityTab'
import type { UsageBreakdown } from '@/components/inspector/TokensTab'
import {
  INSPECTOR_TABS,
  InspectorPanel,
  type InspectorTab,
} from '@/components/inspector/InspectorPanel'
import { ThreadSidebar } from '@/components/chat/ThreadSidebar'
import { ThreadsProvider } from '@/components/chat/ThreadsProvider'
import { Button } from '@/components/ui/button'
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from '@/components/ui/sidebar'
import { useThreads } from '@/hooks/useThreads'
import type { PipelineStatus } from '@/lib/citations'

/**
 * The activity log lives at layout level rather than inside the thread page,
 * because the toggle sits in the header and the panel sits beside the whole
 * conversation. The thread page pushes events in; the layout owns them.
 */
type ActivityLogContextValue = {
  entries: LogEntry[]
  push: (status: PipelineStatus) => void
  clear: () => void
}

const ActivityLogContext = createContext<ActivityLogContextValue | null>(null)

export function useActivityLog() {
  const ctx = useContext(ActivityLogContext)
  if (!ctx) throw new Error('useActivityLog must be used inside ChatLayout')
  return ctx
}

function ChatHeader({
  inspector,
  onSelectInspector,
  logCount,
}: {
  inspector: InspectorTab | null
  onSelectInspector: (tab: InspectorTab) => void
  logCount: number
}) {
  const { threadId } = useParams()
  const { threads } = useThreads()
  const activeThread = threads.find((thread) => thread.id === threadId)

  return (
    <header className="flex h-14 shrink-0 items-center gap-2 border-b bg-background/80 px-3 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <SidebarTrigger className="text-muted-foreground" />
      <span className="truncate text-sm font-medium text-foreground">
        {activeThread?.title ?? 'Encore'}
      </span>
      {/* Four named buttons, where they have always been. They pick which view
          the one panel shows; clicking the active one closes it. */}
      <div className="ml-auto flex items-center gap-1">
        {INSPECTOR_TABS.map(({ id, label, icon: Icon }) => (
          <Button
            key={id}
            variant={inspector === id ? 'secondary' : 'ghost'}
            size="sm"
            onClick={() => onSelectInspector(id)}
            className="gap-2"
            aria-pressed={inspector === id}
          >
            <Icon className="size-4" aria-hidden />
            <span className="hidden sm:inline">{label}</span>
            {id === 'activity' && logCount > 0 && inspector !== 'activity' ? (
              <span className="rounded-full bg-brand-100 px-1.5 text-xs font-medium text-brand-700">
                {logCount}
              </span>
            ) : null}
          </Button>
        ))}
      </div>
    </header>
  )
}

export function ChatLayout() {
  const [entries, setEntries] = useState<LogEntry[]>([])
  const [inspector, setInspector] = useState<InspectorTab | null>(null)
  const [usage, setUsage] = useState<UsageBreakdown | null>(null)

  const push = useCallback((status: PipelineStatus) => {
    // The breakdown rides the status stream as JSON rather than needing its own
    // endpoint, so it lands at the same moment as the answer.
    if (status.stage === 'usage_detail') {
      try {
        setUsage(JSON.parse(status.message) as UsageBreakdown)
      } catch {
        /* a malformed breakdown must never break the chat */
      }
      return
    }
    setEntries((prev) => {
      // The backend re-emits the same stage while a step is still running;
      // collapse those so the log reads as a sequence of steps, not a heartbeat.
      const last = prev[prev.length - 1]
      if (last && last.stage === status.stage && last.message === status.message) {
        return prev
      }
      return [...prev, { ...status, at: Date.now() }]
    })
  }, [])

  const clear = useCallback(() => setEntries([]), [])

  const value = useMemo(() => ({ entries, push, clear }), [entries, push, clear])

  return (
    <ThreadsProvider>
      <ActivityLogContext.Provider value={value}>
        <SidebarProvider>
          <ThreadSidebar onOpenDocuments={() => setInspector('documents')} />
          <SidebarInset className="flex h-svh min-h-0 flex-col">
            <ChatHeader
              inspector={inspector}
              logCount={entries.length}
              onSelectInspector={(next) =>
                setInspector((tab) => (tab === next ? null : next))
              }
            />
            <div className="flex min-h-0 flex-1 flex-row">
              <div className="flex min-h-0 min-w-0 flex-1 flex-col">
                <Outlet />
              </div>
              <InspectorPanel
                tab={inspector}
                onClose={() => setInspector(null)}
                entries={entries}
                usage={usage}
              />
            </div>
          </SidebarInset>
        </SidebarProvider>
      </ActivityLogContext.Provider>
    </ThreadsProvider>
  )
}
