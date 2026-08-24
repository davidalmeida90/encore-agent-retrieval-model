import { cn } from '@/lib/utils'

type LogoProps = {
  className?: string
}

/**
 * Encore mark.
 *
 * A page with its corner turned, and a rising line drawn across it: a filing
 * being read, and the number pulled out of it. Those are the two halves of what
 * the app does, and the folded corner is what stops it reading as a generic
 * rounded square.
 *
 * Drawn rather than loaded, so there is no image asset to ship and it stays
 * crisp at any size. Replaces the inherited `/log.png` from the upstream project.
 */
export function LogoMark({ className }: LogoProps) {
  return (
    <span
      className={cn(
        'flex size-8 shrink-0 items-center justify-center rounded-md bg-brand-600',
        className,
      )}
    >
      <svg
        viewBox="0 0 24 24"
        fill="none"
        className="size-5"
        aria-hidden="true"
      >
        {/* A page with its corner turned, and a rising line across it: a filing
            being read, and the number pulled out of it. The corner fold is what
            makes it a document rather than a generic rounded square. */}
        <path
          d="M6 3.5h8.5L19 8v12.5a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1v-16a1 1 0 0 1 1-1Z"
          fill="white"
          fillOpacity="0.16"
        />
        <path
          d="M14.5 3.5 19 8h-3.5a1 1 0 0 1-1-1V3.5Z"
          fill="white"
          fillOpacity="0.55"
        />
        <path
          d="M7.8 16.4l2.9-3.3 2.4 2 3.1-4.4"
          stroke="white"
          strokeWidth="1.7"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </span>
  )
}

export function Logo({ className }: LogoProps) {
  return (
    <div className={cn('flex items-center gap-2.5', className)}>
      <LogoMark />
      <span className="text-sm font-semibold tracking-tight">
        Encore
      </span>
    </div>
  )
}
