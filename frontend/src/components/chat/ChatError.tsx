import { Link } from 'react-router-dom'
import { AlertTriangle } from 'lucide-react'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { classifyChatError } from '@/lib/chat-errors'

type ChatErrorProps = {
  error: Error
}

export function ChatError({ error }: ChatErrorProps) {
  const classified = classifyChatError(error)

  return (
    <Alert variant="destructive">
      <AlertTriangle />
      <AlertTitle>{classified.title}</AlertTitle>
      <AlertDescription>
        {classified.message}
        {classified.showLoginLink ? (
          <Link to="/login" className="mt-1 inline-block font-medium text-foreground">
            Sign in again
          </Link>
        ) : null}
      </AlertDescription>
    </Alert>
  )
}
