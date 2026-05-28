import {
  CONTROLLED_GATES,
  type CircuitOp,
  type GateName,
  type SingleGateName,
  isControlled,
} from "@/lib/types";

// Stable id generator. We avoid using crypto.randomUUID at module top-level so
// that server-rendered output is deterministic. IDs are minted only in event
// handlers, which run after hydration.
let idCounter = 0;
export const newId = (): string => {
  idCounter += 1;
  return `op_${idCounter.toString(36)}_${Date.now().toString(36)}`;
};

export const makeSingleOp = (
  gate: SingleGateName,
  target: number,
): CircuitOp => ({
  id: newId(),
  gate,
  targets: [target],
  controls: [],
});

export const makeControlledOp = (
  gate: GateName,
  target: number,
  controls: ReadonlyArray<number>,
): CircuitOp => ({
  id: newId(),
  gate,
  targets: [target],
  controls: [...controls],
});

/**
 * Returns a default control configuration when the user drops a controlled
 * gate. Picks the first qubits != target. Caller is expected to refine via
 * UI clicks afterwards.
 */
export const defaultControls = (
  gate: GateName,
  target: number,
  numQubits: number,
): number[] => {
  if (!isControlled(gate)) return [];
  const need = gate === "CCX" ? 2 : 1;
  const out: number[] = [];
  for (let q = 0; q < numQubits && out.length < need; q += 1) {
    if (q !== target) out.push(q);
  }
  return out;
};

export const isControlledName = (g: string): g is GateName =>
  (CONTROLLED_GATES as readonly string[]).includes(g);

export const formatComplex = (re: number, im: number): string => {
  const epsilon = 1e-9;
  const r = Math.abs(re) < epsilon ? 0 : re;
  const i = Math.abs(im) < epsilon ? 0 : im;
  if (i === 0) return r.toFixed(4);
  if (r === 0) return `${i.toFixed(4)}i`;
  const sign = i >= 0 ? "+" : "-";
  return `${r.toFixed(4)} ${sign} ${Math.abs(i).toFixed(4)}i`;
};
