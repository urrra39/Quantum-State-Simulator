"use client";

import { formatComplex } from "@/lib/circuit";
import type { SimulationResponse } from "@/lib/types";

interface StatePanelProps {
  readonly result: SimulationResponse | null;
}

export const StatePanel = ({ result }: StatePanelProps): JSX.Element => {
  if (!result) {
    return (
      <div className="rounded-lg border border-graphite-700/70 bg-void-800 p-4 text-xs text-graphite-500">
        Run a circuit to inspect the state vector |&psi;&rang;.
      </div>
    );
  }

  const epsilon = 1e-9;
  const rows = result.basis_labels
    .map((basis, i) => {
      const amp = result.amplitudes[i] ?? { real: 0, imag: 0 };
      const prob = result.probabilities[i] ?? 0;
      return { basis, amp, prob, idx: i };
    })
    .filter((r) => r.prob > epsilon);

  return (
    <div className="rounded-lg border border-graphite-700/70 bg-void-800 p-4">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-xs uppercase tracking-[0.3em] text-cyan-neon">
          State Vector
        </h2>
        <span className="text-[10px] uppercase tracking-widest text-graphite-500">
          {result.operations_applied} ops applied &middot; non-zero amplitudes
        </span>
      </div>
      <div className="max-h-72 overflow-auto rounded border border-graphite-700/50">
        <table className="w-full table-fixed text-left text-xs">
          <thead className="sticky top-0 bg-void-900 text-graphite-500">
            <tr>
              <th className="w-24 px-3 py-2 font-normal">basis</th>
              <th className="px-3 py-2 font-normal">amplitude &alpha;<sub>i</sub></th>
              <th className="w-20 px-3 py-2 text-right font-normal">|&alpha;|<sup>2</sup></th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td className="px-3 py-3 text-graphite-500" colSpan={3}>
                  All amplitudes are zero (numerically).
                </td>
              </tr>
            ) : (
              rows.map((r) => (
                <tr
                  key={r.idx}
                  className="border-t border-graphite-700/40 hover:bg-cyan-deep/30"
                >
                  <td className="px-3 py-2 text-cyan-neon">
                    |{r.basis}&rang;
                  </td>
                  <td className="px-3 py-2 text-graphite-200">
                    {formatComplex(r.amp.real, r.amp.imag)}
                  </td>
                  <td className="px-3 py-2 text-right text-cyan-glow">
                    {r.prob.toFixed(4)}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
