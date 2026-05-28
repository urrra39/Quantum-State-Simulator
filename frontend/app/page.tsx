import { CircuitBuilder } from "@/components/CircuitBuilder";

export default function HomePage(): JSX.Element {
  return (
    <main className="min-h-screen bg-void-900 text-graphite-200">
      <header className="border-b border-graphite-700/60 bg-void/80 backdrop-blur">
        <div className="mx-auto flex max-w-[1600px] items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="h-3 w-3 rounded-full bg-cyan-glow shadow-neon" aria-hidden />
            <h1 className="text-lg tracking-[0.2em] text-cyan-neon">
              TENSORQ&nbsp;&middot;&nbsp;ENGINE
            </h1>
            <span className="ml-2 text-xs text-graphite-500">
              state-vector / tensor-network simulator
            </span>
          </div>
          <a
            href="/api/tensorq/health"
            className="text-xs text-graphite-500 hover:text-cyan-neon"
            target="_blank"
            rel="noreferrer"
          >
            engine status &rarr;
          </a>
        </div>
      </header>

      <section className="mx-auto max-w-[1600px] px-6 py-6">
        <CircuitBuilder />
      </section>

      <footer className="border-t border-graphite-700/60 px-6 py-4 text-center text-xs text-graphite-500">
        |&psi;&rang; = U<sub>n</sub> &middot;&middot;&middot; U<sub>1</sub> |0&rang;<sup>&otimes;n</sup>
        &nbsp;&middot;&nbsp; rendered with Next.js 14 &middot; computed with NumPy
      </footer>
    </main>
  );
}
