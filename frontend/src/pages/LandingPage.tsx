import { Link } from 'react-router-dom'
import { ArrowRight, FileText, LineChart, Quote, ShieldCheck } from 'lucide-react'

import { Logo } from '@/components/Logo'
import { Button } from '@/components/ui/button'

/**
 * Public landing page.
 *
 * Deliberately louder than the app behind it: white over grey-50, soft blue
 * gradients, bordered cards that lift on hover, blue-100 icon tiles, large blue
 * figures. A landing page is allowed to sell; a research tool you stare at for
 * an hour is not, so the colour lives out here and the chat stays quiet.
 *
 * Every number below is real and measured, not marketing. Nothing claims an
 * accuracy rate, because none has been measured.
 */

const CAPABILITIES = [
  {
    icon: FileText,
    title: 'Reads the filings',
    body:
      'Hybrid retrieval over indexed 10-Ks: vector search and Postgres full-text run in parallel, then a cross-encoder reranks what comes back, so a passage has to answer the question rather than merely share its subject.',
  },
  {
    icon: LineChart,
    title: 'Uses the real numbers',
    body:
      'Figures come from SEC XBRL facts, market data and Treasury yields, never from arithmetic over retrieved prose. DCF and trading comparables run in a deterministic engine, not in the model.',
  },
  {
    icon: ShieldCheck,
    title: 'Fails closed',
    body:
      'Every claim about what a company said carries a citation, checked against the passages actually retrieved that turn. A citation that cannot be verified stops the answer rather than shipping it.',
  },
  {
    icon: Quote,
    title: 'Says when it does not know',
    body:
      'Relevance is scored, so silence is detectable. If a filing simply does not discuss something, the answer says so instead of paraphrasing the nearest paragraph.',
  },
]

const FACTS = [
  { figure: '3,681', label: 'indexed passages' },
  { figure: '5', label: '10-K filings, FY2024 to FY2026' },
  { figure: '2', label: 'companies: Apple and Microsoft' },
]

export function LandingPage() {
  return (
    <div className="min-h-svh bg-slate-50">
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/80 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
          <Logo />
          <div className="flex items-center gap-2">
            <Button asChild variant="ghost" size="sm">
              <Link to="/login">Sign in</Link>
            </Button>
            <Button asChild size="sm">
              <Link to="/signup">Get started</Link>
            </Button>
          </div>
        </div>
      </header>

      {/* Hero ------------------------------------------------------------- */}
      <section className="relative overflow-hidden bg-white">
        <div className="absolute inset-0 bg-gradient-to-br from-brand-50 via-white to-slate-50" />
        <div className="absolute right-0 top-0 h-full w-1/2 bg-gradient-to-l from-brand-100/50 to-transparent" />
        <div className="relative mx-auto max-w-6xl px-6 py-24 sm:py-32">
          <div className="max-w-2xl">
            <span className="inline-flex items-center rounded-full border border-brand-200 bg-brand-50 px-3 py-1 text-xs font-medium text-brand-700">
              Grounded equity research
            </span>
            <h1 className="mt-6 text-4xl font-bold tracking-tight text-slate-900 sm:text-5xl">
              Ask a filing what it actually says.
            </h1>
            <p className="mt-5 text-lg leading-relaxed text-slate-600">
              Encore answers questions about US public companies from two sources
              and refuses to answer from memory: the text of their SEC filings,
              cited passage by passage, and their reported figures, pulled live
              from XBRL.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Button asChild size="lg" className="gap-2">
                <Link to="/signup">
                  Start asking <ArrowRight className="size-4" />
                </Link>
              </Button>
              <Button asChild size="lg" variant="outline">
                <Link to="/login">Sign in</Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Capabilities ------------------------------------------------------ */}
      <section className="mx-auto max-w-6xl px-6 py-20">
        <h2 className="text-xl font-semibold text-slate-900">
          Built so the answer can be checked
        </h2>
        <div className="mt-8 grid gap-5 sm:grid-cols-2">
          {CAPABILITIES.map(({ icon: Icon, title, body }) => (
            <div
              key={title}
              className="group rounded-xl border border-slate-200 bg-white p-6 shadow-sm transition-all duration-300 hover:border-brand-300 hover:shadow-lg"
            >
              <div className="mb-4 flex size-12 items-center justify-center rounded-lg bg-brand-100 transition-colors group-hover:bg-brand-200">
                <Icon className="size-6 text-brand-600" aria-hidden />
              </div>
              <h3 className="mb-2 text-lg font-semibold text-slate-900">{title}</h3>
              <p className="text-sm leading-relaxed text-slate-600">{body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* What is loaded ---------------------------------------------------- */}
      <section className="border-y border-slate-200 bg-white">
        <div className="mx-auto grid max-w-6xl gap-8 px-6 py-16 sm:grid-cols-3">
          {FACTS.map(({ figure, label }) => (
            <div key={label}>
              <p className="mb-2 text-4xl font-bold text-brand-600 md:text-5xl">
                {figure}
              </p>
              <p className="text-sm text-slate-600">{label}</p>
            </div>
          ))}
        </div>
      </section>

      <footer className="mx-auto max-w-6xl px-6 py-10">
        <p className="text-xs text-slate-500">
          Encore is a research tool. Answers are grounded in SEC filings and
          should be verified against the source before being relied on. Nothing
          here is investment advice.
        </p>
      </footer>
    </div>
  )
}
