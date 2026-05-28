"use client";

import { type CircuitOp, type GateName, isControlled } from "@/lib/types";

interface CircuitGridProps {
  readonly numQubits: number;
  readonly ops: ReadonlyArray<CircuitOp>;
  readonly onDropGate: (gate: GateName, qubit: number) => void;
  readonly onCellClick: (qubit: number) => void;
  readonly onToggleControl: (opId: string, qubit: number) => void;
  readonly onRemoveOp: (opId: string) => void;
}

const TARGET_BOX =
  "flex h-10 w-10 items-center justify-center rounded-md border border-cyan-glow bg-cyan-deep text-cyan-glow shadow-neon font-bold cursor-pointer";
const CONTROLLED_TARGET_BOX =
  "flex h-10 w-10 items-center justify-center rounded-md border border-amber-warn bg-graphite-900 text-amber-warn font-bold cursor-pointer";
const CONTROL_DOT_BTN =
  "flex h-10 w-10 items-center justify-center text-amber-warn text-3xl leading-none cursor-pointer";

export const CircuitGrid = ({
  numQubits,
  ops,
  onDropGate,
  onCellClick,
  onToggleControl,
  onRemoveOp,
}: CircuitGridProps): JSX.Element => {
  const handleDragOver = (e: React.DragEvent<HTMLDivElement>): void => {
    if (e.dataTransfer.types.includes("application/x-tensorq-gate")) {
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
    }
  };

  const handleDrop =
    (qubit: number) =>
    (e: React.DragEvent<HTMLDivElement>): void => {
      e.preventDefault();
      const gate = e.dataTransfer.getData("application/x-tensorq-gate");
      if (!gate) return;
      onDropGate(gate as GateName, qubit);
    };

  // Pre-compute a 2D matrix [qubit][column] -> "target" | "control" | "vline" | null
  // so we can render multi-row controlled gates with a connecting vertical wire.
  type CellKind = "target" | "control" | "vline" | null;
  const matrix: CellKind[][] = Array.from({ length: numQubits }, () =>
    Array<CellKind>(ops.length).fill(null),
  );
  ops.forEach((op, col) => {
    const involved = [...op.targets, ...op.controls];
    if (involved.length === 0) return;
    const minRow = Math.min(...involved);
    const maxRow = Math.max(...involved);
    for (let r = minRow; r <= maxRow; r += 1) {
      const row = matrix[r];
      if (!row) continue;
      if (op.targets.includes(r)) row[col] = "target";
      else if (op.controls.includes(r)) row[col] = "control";
      else row[col] = "vline";
    }
  });

  return (
    <div className="rounded-lg border border-graphite-700/70 bg-void-800 p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-xs uppercase tracking-[0.3em] text-cyan-neon">
          Circuit Canvas
        </h2>
        <span className="text-[10px] uppercase tracking-widest text-graphite-500">
          {numQubits} qubits &middot; {ops.length} ops
        </span>
      </div>

      <div
        className="bg-quantum-grid rounded-md p-4 overflow-x-auto"
        role="grid"
        aria-label="Quantum circuit grid"
      >
        <div className="inline-block min-w-full">
          {Array.from({ length: numQubits }).map((_, q) => {
            const row = matrix[q] ?? [];
            return (
              <div
                key={`row-${q}`}
                className="mb-4 flex items-center gap-4 last:mb-0"
              >
                {/* Qubit label */}
                <div className="w-16 shrink-0 text-right text-xs text-graphite-300">
                  <div className="font-bold text-cyan-neon">q[{q}]</div>
                  <div className="text-[10px] text-graphite-500">|0&rang;</div>
                </div>

                {/* Cells: existing op columns + trailing empty column */}
                <div className="relative flex items-center gap-4">
                  {/* horizontal qubit wire */}
                  <div
                    aria-hidden
                    className="absolute left-0 right-0 top-1/2 -translate-y-1/2 border-t border-graphite-700/70"
                  />

                  {ops.map((op, col) => {
                    const kind: CellKind = row[col] ?? null;

                    if (kind === "target") {
                      const label =
                        op.gate === "CNOT" || op.gate === "CCX"
                          ? "\u2295"
                          : op.gate.replace(/^C/, "");
                      return (
                        <div
                          key={`${op.id}-${q}-t`}
                          className="relative z-10"
                          role="gridcell"
                        >
                          <button
                            type="button"
                            onClick={() => onRemoveOp(op.id)}
                            title={`${op.gate} (click to remove)`}
                            className={
                              isControlled(op.gate)
                                ? CONTROLLED_TARGET_BOX
                                : TARGET_BOX
                            }
                          >
                            {label}
                          </button>
                        </div>
                      );
                    }

                    if (kind === "control") {
                      return (
                        <div
                          key={`${op.id}-${q}-c`}
                          className="relative z-10"
                          role="gridcell"
                        >
                          <button
                            type="button"
                            onClick={() => onToggleControl(op.id, q)}
                            title={`Control on q[${q}] (click to remove)`}
                            className={CONTROL_DOT_BTN}
                          >
                            &bull;
                          </button>
                        </div>
                      );
                    }

                    if (kind === "vline") {
                      // Pass-through wire: vertical line joining target and controls
                      return (
                        <div
                          key={`${op.id}-${q}-v`}
                          className="relative z-10 flex h-10 w-10 items-center justify-center"
                          role="gridcell"
                          aria-hidden
                        >
                          <div className="h-10 w-[2px] bg-amber-warn/60" />
                        </div>
                      );
                    }

                    // Empty cell within an existing column -> click adds a control
                    // for controlled gates only; no drop here (drop only on trailing).
                    return (
                      <div
                        key={`${op.id}-${q}-e`}
                        className="relative z-10"
                        role="gridcell"
                      >
                        {isControlled(op.gate) ? (
                          <button
                            type="button"
                            onClick={() => onToggleControl(op.id, q)}
                            title={`Add control on q[${q}]`}
                            className="h-10 w-10 rounded-md border border-dashed border-amber-warn/40 text-amber-warn/60 hover:bg-amber-warn/10"
                          >
                            +
                          </button>
                        ) : (
                          <div
                            className="h-10 w-10 rounded-md border border-dashed border-graphite-700/40"
                            aria-hidden
                          />
                        )}
                      </div>
                    );
                  })}

                  {/* Trailing empty column — drag/drop and click target */}
                  <div
                    className="relative z-10"
                    role="gridcell"
                    onDragOver={handleDragOver}
                    onDrop={handleDrop(q)}
                  >
                    <button
                      type="button"
                      onClick={() => onCellClick(q)}
                      aria-label={`Add gate on q[${q}]`}
                      className="h-10 w-10 rounded-md border border-dashed border-cyan-dim/40 text-cyan-dim/70 hover:bg-cyan-deep/40 hover:text-cyan-glow"
                    >
                      +
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <p className="mt-3 text-[11px] text-graphite-500">
        Drag a gate onto the trailing <span className="text-cyan-glow">+</span>{" "}
        cell of any qubit, or click a palette tile then click a cell. Click an
        existing gate box to remove it. For controlled gates, click any free
        cell in the same column to toggle that qubit as a control.
      </p>
    </div>
  );
};
