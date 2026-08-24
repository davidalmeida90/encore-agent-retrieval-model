import { useMemo } from 'react'
import type { Components } from 'react-markdown'

import { CitationMarker } from '@/components/chat/CitationMarker'
import { CodeBlock, CodeBlockCode } from '@/components/ui/code-block'
import { Markdown } from '@/components/ui/markdown'
import { citationByIndex, type CitationPayload } from '@/lib/citations'
import { cn } from '@/lib/utils'

const CITE_PREFIX = '#citation-'

// Inline child-targeting utilities keep markdown readable without pulling in
// the typography plugin, and keep every color bound to a theme token.
const MARKDOWN_CLASSES = cn(
  'space-y-3 text-sm leading-relaxed text-foreground',
  '[&_p]:leading-relaxed [&_strong]:font-semibold',
  '[&_a]:font-medium [&_a]:underline [&_a]:underline-offset-4',
  '[&_ul]:list-disc [&_ul]:pl-5 [&_ol]:list-decimal [&_ol]:pl-5 [&_li]:my-1',
  '[&_h1]:text-base [&_h1]:font-semibold [&_h2]:text-base [&_h2]:font-semibold [&_h3]:text-sm [&_h3]:font-semibold',
  '[&_table]:w-full [&_table]:text-left [&_th]:border-b [&_th]:py-1.5 [&_th]:pr-3 [&_th]:font-semibold [&_td]:border-b [&_td]:py-1.5 [&_td]:pr-3 [&_td]:align-top',
  '[&_blockquote]:border-l-2 [&_blockquote]:border-border [&_blockquote]:pl-3 [&_blockquote]:text-muted-foreground',
)

function withCitationLinks(text: string, validIndices: Set<number>): string {
  // Handles combined markers as well as single ones: "[1, 2]" becomes two
  // separate superscript links, not literal text.
  //
  // The old pattern was /\[(\d+)\]/ and matched only "[1]". Models write "[1, 2]"
  // constantly, so half the citations in an answer rendered as raw brackets
  // beside properly-formatted ones. The validator had the same blind spot, where
  // it was worse than cosmetic: unmatched markers meant citations were pruned as
  // unreferenced and never verified.
  return text.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (match, group: string) => {
    const indices = group
      .split(',')
      .map((part) => Number(part.trim()))
      .filter((n) => Number.isInteger(n) && validIndices.has(n))
    if (indices.length === 0) return match
    return indices.map((n) => `[[${n}]](${CITE_PREFIX}${n})`).join('')
  })
}

function extractLanguage(className?: string): string {
  if (!className) return 'plaintext'
  const match = className.match(/language-(\w+)/)
  return match ? match[1] : 'plaintext'
}

type AssistantMarkdownProps = {
  text: string
  citations: CitationPayload[]
  selectedCitationIndex: number | null
  onSelectCitation: (citation: CitationPayload) => void
}

export function AssistantMarkdown({
  text,
  citations,
  selectedCitationIndex,
  onSelectCitation,
}: AssistantMarkdownProps) {
  const validIndices = useMemo(
    () => new Set(citations.map((citation) => citation.citationIndex)),
    [citations],
  )
  const source = useMemo(() => withCitationLinks(text, validIndices), [text, validIndices])

  const components: Partial<Components> = useMemo(
    () => ({
      a({ href, children, ...props }) {
        if (href?.startsWith(CITE_PREFIX)) {
          const index = Number(href.slice(CITE_PREFIX.length))
          const citation = citationByIndex(citations, index)
          if (citation) {
            return (
              <CitationMarker
                index={index}
                selected={selectedCitationIndex === index}
                onSelect={() => onSelectCitation(citation)}
              />
            )
          }
        }
        return (
          <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
            {children}
          </a>
        )
      },
      code({ className, children, ...props }) {
        const isInline =
          !props.node?.position?.start.line ||
          props.node?.position?.start.line === props.node?.position?.end.line

        if (isInline) {
          return (
            <span className="rounded-sm bg-muted px-1 py-0.5 font-mono text-[0.85em]">
              {children}
            </span>
          )
        }

        return (
          <CodeBlock>
            <CodeBlockCode
              code={String(children)}
              language={extractLanguage(className)}
            />
          </CodeBlock>
        )
      },
      pre({ children }) {
        return <>{children}</>
      },
    }),
    [citations, selectedCitationIndex, onSelectCitation],
  )

  return (
    <Markdown className={MARKDOWN_CLASSES} components={components}>
      {source}
    </Markdown>
  )
}
