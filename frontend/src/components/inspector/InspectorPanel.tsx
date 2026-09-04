import { Activity, Coins, Cpu, FileText, Wrench, X } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { ActivityTab, type LogEntry } from '@/components/inspector/ActivityTab'
import { DocumentsTab } from '@/components/inspector/DocumentsTab'
import { TokensTab, type UsageBreakdown } from '@/components/inspector/TokensTab'
import { RuntimeTab } from '@/components/inspector/RuntimeTab'
import { ToolsTab } from '@/components/inspector/ToolsTab'

/**
 * ## One inspector, four views
 *
 * Activity, Tools, Tokens and Documents each began as their own `<aside>`, with
 * their own header, close button and width. Four copies of the same shell, and
 * opening two at once squeezed the conversation between them.
 *
 * They are one panel now, chosen by the four buttons in the chat header. The
 * buttons stay where they always were, top right, because that is where you
 * look for them; what changed is that they now share one panel instead of
 * fighting each other for width. The four views answer four questions about the
 * *same* run:
 *
 *   Activity   what did it do, in what order, did grounding pass
 *   Tools      what could it call, and what is the real Python
 *   Tokens     what did that cost, and what was the money spent on
 *   Runtime    what model is behind it, and where its time actually went
 *   Documents  what is actually indexed, and could it have found this
 *
 * Each tab file now renders only its body. Width, header, scrolling and the
 * close button live here, once.
 */

export type InspectorTab = 'activity' | 'tools' | 'tokens' | 'runtime' | 'documents'

type TabSpec = {
  id: InspectorTab
  label: string
  icon: LucideIcon
}

export const INSPECTOR_TABS: TabSpec[] = [
  { id: 'activity', label: 'Activity', icon: Activity },
  { id: 'tools', label: 'Tools', icon: Wrench },
  { id: 'tokens', label: 'Tokens', icon: Coins },
  { id: 'runtime', label: 'Runtime', icon: Cpu },
  { id: 'documents', label: 'Corpus', icon: FileText },
]

type InspectorPanelProps = {
  tab: InspectorTab | null
  onClose: () => void
  entries: LogEntry[]
  usage: UsageBreakdown | null
}

export function InspectorPanel({ tab, onClose, entries, usage }: InspectorPanelProps) {
  if (!tab) return null

  const active = INSPECTOR_TABS.find((entry) => entry.id === tab)

  return (
    <aside className="flex w-[22rem] shrink-0 flex-col border-l border-border bg-muted/30">
      {/* Which view you are in, and the way out. Switching happens in the
          chat header, so there is no second row of controls here. */}
      <header className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex items-center gap-2">
          {active ? <active.icon className="size-4 text-brand-600" aria-hidden /> : null}
          <h2 className="text-sm font-medium text-foreground">{active?.label}</h2>
          {tab === 'activity' && entries.length > 0 ? (
            <span className="text-xs text-muted-foreground">{entries.length}</span>
          ) : null}
        </div>
        <Button
          variant="ghost"
          size="icon"
          className="size-7"
          onClick={onClose}
          aria-label="Close inspector"
        >
          <X className="size-4" />
        </Button>
      </header>

      {/* Every tab mounts fresh on switch, which is wanted: Documents should not
          hold a half-scrolled filing from ten minutes ago. */}
      {tab === 'activity' ? <ActivityTab entries={entries} /> : null}
      {tab === 'tools' ? <ToolsTab /> : null}
      {tab === 'tokens' ? <TokensTab usage={usage} /> : null}
      {tab === 'runtime' ? <RuntimeTab /> : null}
      {tab === 'documents' ? <DocumentsTab /> : null}
    </aside>
  )
}
