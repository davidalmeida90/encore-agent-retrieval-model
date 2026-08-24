import { type PipelineStatus as PipelineStatusState } from '@/lib/citations'
import { cn } from '@/lib/utils'

type PipelineStatusProps = {
  isSubmitted: boolean
  pipelineStatus: PipelineStatusState | null
}

export function PipelineStatus({ isSubmitted, pipelineStatus }: PipelineStatusProps) {
  const message =
    isSubmitted && !pipelineStatus
      ? 'Analyzing your question…'
      : (pipelineStatus?.message ?? 'Researching filings…')

  return (
    <p
      aria-live="polite"
      className={cn(
        'w-fit bg-clip-text text-sm font-medium text-transparent',
        'bg-[linear-gradient(to_right,var(--muted-foreground)_35%,var(--foreground)_50%,var(--muted-foreground)_65%)]',
        'bg-size-[200%_auto]',
        'animate-[shimmer_2.5s_linear_infinite]',
      )}
    >
      {message}
    </p>
  )
}
