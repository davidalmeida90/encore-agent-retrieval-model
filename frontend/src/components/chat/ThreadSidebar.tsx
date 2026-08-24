import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { FileText, Loader2, Plus, Search, Trash2 } from 'lucide-react'
import { toast } from 'sonner'

import { Logo } from '@/components/Logo'
import { UserMenu } from '@/components/chat/UserMenu'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogMedia,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { Button } from '@/components/ui/button'
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSkeleton,
  useSidebar,
} from '@/components/ui/sidebar'
import { useThreads } from '@/hooks/useThreads'
import type { ThreadSummary } from '@/lib/chat'
import { groupByRecency } from '@/lib/format'

export function ThreadSidebar({ onOpenDocuments }: { onOpenDocuments: () => void }) {
  const navigate = useNavigate()
  const { threadId } = useParams()
  const { setOpenMobile, isMobile } = useSidebar()
  const { threads, isLoading, error, createNewThread, deleteThread } = useThreads()
  const [isCreating, setIsCreating] = useState(false)
  const [threadToDelete, setThreadToDelete] = useState<ThreadSummary | null>(null)
  const [isDeleting, setIsDeleting] = useState(false)
  const [filter, setFilter] = useState('')

  // Filtering happens here rather than server-side: every thread is already
  // loaded for the list, so a query would cost a round trip and return the same
  // rows. Revisit if the list ever pages.
  const needle = filter.trim().toLowerCase()
  const visible = needle
    ? threads.filter((thread) => thread.title.toLowerCase().includes(needle))
    : threads
  const groups = groupByRecency(visible, (thread) => thread.updatedAt)

  async function handleNewChat() {
    setIsCreating(true)
    try {
      const id = await createNewThread()
      navigate(`/chats/${id}`)
      if (isMobile) setOpenMobile(false)
    } finally {
      setIsCreating(false)
    }
  }

  async function handleDeleteThread() {
    if (!threadToDelete) return

    setIsDeleting(true)
    try {
      await deleteThread(threadToDelete.id)
      toast.success('Conversation deleted')

      if (threadToDelete.id === threadId) {
        navigate('/chats', { replace: true })
      }

      if (isMobile) setOpenMobile(false)
      setThreadToDelete(null)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Could not delete conversation.'
      toast.error(message)
    } finally {
      setIsDeleting(false)
    }
  }

  return (
    <>
      <Sidebar>
        {/* Brand and the new-thread action share one row. A full-width button
            under the logo is the shape every chat app uses, and it spends a
            whole row on something used once a session. */}
        <SidebarHeader className="gap-0 border-b border-sidebar-border p-3">
          <div className="flex items-center justify-between gap-2">
            <Logo className="px-1" />
            <Button
              variant="ghost"
              size="icon"
              aria-label="New conversation"
              title="New conversation"
              className="size-7 shrink-0 text-sidebar-foreground/70 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
              onClick={() => void handleNewChat()}
              disabled={isCreating}
            >
              <Plus className="size-4" />
            </Button>
          </div>
        </SidebarHeader>

        {/* Search the history, then the corpus browser. Both are navigation:
            "find where I asked this" and "find what it could have read". */}
        <div className="px-3 pt-3">
          <div className="relative">
            <Search
              className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-sidebar-foreground/40"
              aria-hidden
            />
            <input
              type="search"
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
              placeholder="Search conversations"
              aria-label="Search conversations"
              className="h-8 w-full rounded-md border border-sidebar-border bg-sidebar-accent/40 pr-2 pl-8 text-xs text-sidebar-foreground placeholder:text-sidebar-foreground/40 focus:border-sidebar-ring focus:outline-none"
            />
          </div>
        </div>

        <div className="px-3 pt-1.5">
          <Button
            variant="ghost"
            size="sm"
            onClick={onOpenDocuments}
            className="w-full justify-start gap-2 text-sidebar-foreground/80 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
          >
            <FileText className="size-4" />
            Browse the corpus
          </Button>
        </div>

        <SidebarContent className="px-1">
          {isLoading ? (
            <SidebarGroup>
              <SidebarGroupContent>
                <SidebarMenu>
                  {Array.from({ length: 5 }).map((_, index) => (
                    <SidebarMenuItem key={index}>
                      <SidebarMenuSkeleton />
                    </SidebarMenuItem>
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          ) : null}

          {!isLoading && error ? (
            <p className="px-3 py-2 text-sm text-destructive" role="alert">
              {error}
            </p>
          ) : null}

          {!isLoading && !error && visible.length === 0 ? (
            <p className="px-3 py-2 text-sm text-muted-foreground">
              {needle ? `Nothing matching "${filter.trim()}".` : 'No conversations yet.'}
            </p>
          ) : null}

          {!isLoading && !error
            ? groups.map((group) => (
                <SidebarGroup key={group.label}>
                  <SidebarGroupLabel>{group.label}</SidebarGroupLabel>
                  <SidebarGroupContent>
                    <SidebarMenu>
                      {group.items.map((thread) => (
                        <SidebarMenuItem key={thread.id}>
                          <SidebarMenuButton
                            asChild
                            isActive={thread.id === threadId}
                            tooltip={thread.title}
                          >
                            <Link
                              to={`/chats/${thread.id}`}
                              onClick={() => {
                                if (isMobile) setOpenMobile(false)
                              }}
                            >
                              <span className="truncate">{thread.title}</span>
                            </Link>
                          </SidebarMenuButton>
                          <SidebarMenuAction
                            showOnHover
                            aria-label={`Delete conversation: ${thread.title}`}
                            title="Delete conversation"
                            className="text-muted-foreground hover:bg-destructive/10 hover:text-destructive focus-visible:text-destructive"
                            disabled={isDeleting}
                            onClick={(event) => {
                              event.preventDefault()
                              event.stopPropagation()
                              setThreadToDelete(thread)
                            }}
                          >
                            {isDeleting && threadToDelete?.id === thread.id ? (
                              <Loader2 className="animate-spin" />
                            ) : (
                              <Trash2 />
                            )}
                          </SidebarMenuAction>
                        </SidebarMenuItem>
                      ))}
                    </SidebarMenu>
                  </SidebarGroupContent>
                </SidebarGroup>
              ))
            : null}
        </SidebarContent>

        <SidebarFooter>
          <UserMenu />
        </SidebarFooter>
      </Sidebar>

      <AlertDialog
        open={threadToDelete !== null}
        onOpenChange={(open) => {
          if (!open && !isDeleting) setThreadToDelete(null)
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogMedia className="bg-destructive/10 text-destructive">
              <Trash2 />
            </AlertDialogMedia>
            <AlertDialogTitle>Delete this conversation?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete “{threadToDelete?.title}” and its message history.
              This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isDeleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={isDeleting}
              onClick={(event) => {
                event.preventDefault()
                void handleDeleteThread()
              }}
            >
              {isDeleting ? 'Deleting…' : 'Delete'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}
