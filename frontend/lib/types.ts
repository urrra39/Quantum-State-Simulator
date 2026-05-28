// Shared types between the dashboard and the FastAPI backend.

export const SINGLE_GATES = ["I", "X", "Y", "Z", "H", "S", "T"] as const;
export const CONTROLLED_GATES = ["CNOT", "CZ", "CY", "CH", "CCX"] as const;

export type SingleGateName = (typeof SINGLE_GATES)[number];
export type ControlledGateName = (typeof CONTROLLED_GATES)[number];
export type GateName = SingleGateName | ControlledGateName;

export interface CircuitOp {
  readonly id: string;
  readonly gate: GateName;
  readonly targets: ReadonlyArray<number>;
  readonly controls: ReadonlyArray<number>;
}

export interface Amplitude {
  readonly real: number;
  readonly imag: number;
}

export interface SimulationResponse {
  readonly num_qubits: number;
  readonly operations_applied: number;
  readonly basis_labels: ReadonlyArray<string>;
  readonly amplitudes: ReadonlyArray<Amplitude>;
  readonly probabilities: ReadonlyArray<number>;
  readonly counts: Readonly<Record<string, number>>;
}

export interface SimulationRequest {
  readonly num_qubits: number;
  readonly operations: ReadonlyArray<{
    gate: GateName;
    targets: ReadonlyArray<number>;
    controls: ReadonlyArray<number>;
  }>;
  readonly shots?: number;
  readonly seed?: number;
}

export interface ApiError {
  readonly error: string;
  readonly message: string;
}

export const isControlled = (g: GateName): g is ControlledGateName =>
  (CONTROLLED_GATES as readonly string[]).includes(g);

export const minControls = (g: GateName): number => {
  if (g === "CCX") return 2;
  if (isControlled(g)) return 1;
  return 0;
};
