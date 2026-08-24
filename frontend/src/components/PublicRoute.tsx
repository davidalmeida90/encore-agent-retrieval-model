import type { ReactNode } from 'react'
import { Navigate } from 'react-router-dom'

import { useSession } from '@/hooks/useSession'

type PublicRouteProps = {
  children: ReactNode
}

export function PublicRoute({ children }: PublicRouteProps) {
  const session = useSession()

  if (session === undefined) {
    return (
      <div className="flex min-h-screen items-center justify-center text-sm text-muted-foreground">
        Loading session…
      </div>
    )
  }

  if (session) {
    return <Navigate to="/chats" replace />
  }

  return children
}
