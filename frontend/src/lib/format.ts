export function formatRelativeTime(isoDate: string): string {
  const date = new Date(isoDate)
  const diffSeconds = Math.round((date.getTime() - Date.now()) / 1000)
  const rtf = new Intl.RelativeTimeFormat('en', { numeric: 'auto' })

  const divisions: Array<{ amount: number; unit: Intl.RelativeTimeFormatUnit }> = [
    { amount: 60, unit: 'second' },
    { amount: 60, unit: 'minute' },
    { amount: 24, unit: 'hour' },
    { amount: 7, unit: 'day' },
    { amount: 4.34524, unit: 'week' },
    { amount: 12, unit: 'month' },
    { amount: Number.POSITIVE_INFINITY, unit: 'year' },
  ]

  let duration = diffSeconds
  for (const division of divisions) {
    if (Math.abs(duration) < division.amount) {
      return rtf.format(duration, division.unit)
    }
    duration = Math.round(duration / division.amount)
  }

  return rtf.format(duration, 'year')
}

export type RecencyGroup = 'Today' | 'Yesterday' | 'Previous 7 days' | 'Older'

const RECENCY_ORDER: RecencyGroup[] = ['Today', 'Yesterday', 'Previous 7 days', 'Older']

function startOfDay(date: Date): number {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime()
}

export function recencyGroup(isoDate: string): RecencyGroup {
  const date = new Date(isoDate)
  const today = startOfDay(new Date())
  const dayMs = 24 * 60 * 60 * 1000
  const dayStart = startOfDay(date)

  if (dayStart >= today) return 'Today'
  if (dayStart >= today - dayMs) return 'Yesterday'
  if (dayStart >= today - 7 * dayMs) return 'Previous 7 days'
  return 'Older'
}

export function groupByRecency<T>(
  items: T[],
  getDate: (item: T) => string,
): Array<{ label: RecencyGroup; items: T[] }> {
  const buckets = new Map<RecencyGroup, T[]>()
  for (const item of items) {
    const group = recencyGroup(getDate(item))
    const bucket = buckets.get(group) ?? []
    bucket.push(item)
    buckets.set(group, bucket)
  }

  return RECENCY_ORDER.filter((label) => buckets.has(label)).map((label) => ({
    label,
    items: buckets.get(label)!,
  }))
}
