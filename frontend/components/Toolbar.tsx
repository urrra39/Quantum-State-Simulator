"use client";

interface ToolbarProps {
  readonly numQubits: number;
  readonly onChangeNumQubits: (n: number) => void;
  readonly shots: number;
  readonly onChangeShots: (n: number) => void;
  readonly onRun: () => void;
  readonly onReset: () => void;
  readonly running: boolean;
  readonly error: string | null;
}

export const Toolbar = ({
  numQubits,
  onChangeNumQubits,
  shots,
  onChangeShots,
  onRun,
  onReset,
  running,
  error,
}: ToolbarProps): JSX.Element => {
  return (
    <div className="flex flex-wrap items-end gap-4 rounded-lg border border-graphite-700/70 bg-void-800 p-4">
      <label className="flex flex-col text-[10px] uppercase tracking-widest text-graphite-500">
        Qubits (1&ndash;8)
        <input
          type="number"
          min={1}
          max={8}
          step={1}
          value={numQubits}
          onChange={(e) => {
            const v = Number.parseInt(e.target.value, 10);
            if (Number.isFinite(v)) onChangeNumQubits(Math.max(1, Math.min(8, v)));
          }}
          className="mt-1 w-24 rounded border border-graphite-700 bg-void-900 px-3 py-2 text-sm text-cyan-neon focus:border-cyan-glow focus:outline-none"
        />
      </label>

      <label className="flex flex-col text-[10px] uppercase tracking-widest text-graphite-500">
        Shots (0 = analytic)
        <input
          type="number"
          min={0}
          max={100000}
          step={64}
          value={shots}
          onChange={(e) => {
            const v = Number.parseInt(e.target.value, 10);
            if (Number.isFinite(v)) onChangeShots(Math.max(0, Math.min(100000, v)));
          }}
          className="mt-1 w-32 rounded border border-graphite-700 bg-void-900 px-3 py-2 text-sm text-cyan-neon focus:border-cyan-glow focus:outline-none"
        />
      </label>

      <div className="flex flex-1 items-center justify-end gap-3">
        {error && (
          <span
            role="alert"
            className="max-w-md truncate rounded border border-crimson-err/60 bg-crimson-err/10 px-3 py-2 text-xs text-crimson-err"
            title={error}
          >
            {error}
          </span>
        )}
        <button
          type="button"
          onClick={onReset}
          disabled={running}
          className="rounded border border-graphite-700 bg-void-900 px-4 py-2 text-xs uppercase tracking-widest text-graphite-300 hover:border-graphite-500 disabled:opacity-50"
        >
          Reset
        </button>
        <button
          type="button"
          onClick={onRun}
          disabled={running}
          className="rounded border border-cyan-glow bg-cyan-deep px-5 py-2 text-xs uppercase tracking-widest text-cyan-glow shadow-neon hover:bg-cyan-dim/30 disabled:opacity-50"
        >
          {running ? "Computing\u2026" : "Run Circuit"}
        </button>
      </div>
    </div>
  );
};
