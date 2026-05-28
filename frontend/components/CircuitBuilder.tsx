"use client";

import { useCallback, useMemo, useState } from "react";

import { CircuitGrid } from "@/components/CircuitGrid";
import { GatePalette } from "@/components/GatePalette";
import { ProbabilityChart } from "@/components/ProbabilityChart";
import { StatePanel } from "@/components/StatePanel";
import { Toolbar } from "@/components/Toolbar";
import { TensorQApiError, runSimulation } from "@/lib/api";
import {
  defaultControls,
  makeControlledOp,
  makeSingleOp,
} from "@/lib/circuit";
import {
  type CircuitOp,
  type GateName,
  type SimulationResponse,
  isControlled,
  minControls,
} from "@/lib/types";

const DEFAULT_QUBITS = 3;

export const CircuitBuilder = (): JSX.Element => {
  const [numQubits, setNumQubits] = useState<number>(DEFAULT_QUBITS);
  const [ops, setOps] = useState<ReadonlyArray<CircuitOp>>([]);
  const [selectedGate, setSelectedGate] = useState<GateName | null>(null);
  const [shots, setShots] = useState<number>(0);
  const [running, setRunning] = useState<boolean>(false);
  const [result, setResult] = useState<SimulationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleNumQubitsChange = useCallback((n: number) => {
    setNumQubits(n);
    // Drop any operations referencing qubits that no longer exist.
    setOps((prev) =>
      prev.filter((op) => {
        const all = [...op.targets, ...op.controls];
        return all.every((q) => q < n);
      }),
    );
    setResult(null);
  }, []);

  const placeGate = useCallback(
    (gate: GateName, qubit: number) => {
      const newOp: CircuitOp = isControlled(gate)
        ? makeControlledOp(gate, qubit, defaultControls(gate, qubit, numQubits))
        : makeSingleOp(gate, qubit);

      if (isControlled(gate) && newOp.controls.length < minControls(gate)) {
        setError(
          `Gate ${gate} needs ${minControls(gate)} control qubit(s); register only has ${numQubits}.`,
        );
        return;
      }
      setError(null);
      setOps((prev) => [...prev, newOp]);
    },
    [numQubits],
  );

  const handleDropGate = useCallback(
    (gate: GateName, qubit: number) => placeGate(gate, qubit),
    [placeGate],
  );

  const handleCellClick = useCallback(
    (qubit: number) => {
      if (!selectedGate) return;
      placeGate(selectedGate, qubit);
    },
    [placeGate, selectedGate],
  );

  const handleToggleControl = useCallback(
    (opId: string, qubit: number) => {
      setOps((prev) =>
        prev.map((op) => {
          if (op.id !== opId) return op;
          if (op.targets.includes(qubit)) return op; // can't make target a control
          const has = op.controls.includes(qubit);
          const nextControls = has
            ? op.controls.filter((c) => c !== qubit)
            : [...op.controls, qubit];
          // Enforce minimum controls
          if (nextControls.length < minControls(op.gate)) {
            setError(
              `${op.gate} requires at least ${minControls(op.gate)} control qubit(s).`,
            );
            return op;
          }
          setError(null);
          return { ...op, controls: nextControls };
        }),
      );
    },
    [],
  );

  const handleRemoveOp = useCallback((opId: string) => {
    setOps((prev) => prev.filter((op) => op.id !== opId));
  }, []);

  const handleReset = useCallback(() => {
    setOps([]);
    setResult(null);
    setError(null);
  }, []);

  const requestPayload = useMemo(
    () => ({
      num_qubits: numQubits,
      operations: ops.map((op) => ({
        gate: op.gate,
        targets: op.targets,
        controls: op.controls,
      })),
      shots,
    }),
    [numQubits, ops, shots],
  );

  const handleRun = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await runSimulation(requestPayload);
      setResult(res);
    } catch (err) {
      if (err instanceof TensorQApiError) {
        setError(`[${err.code}] ${err.message}`);
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Unknown error during simulation.");
      }
      setResult(null);
    } finally {
      setRunning(false);
    }
  }, [requestPayload]);

  return (
    <div className="space-y-4">
      <Toolbar
        numQubits={numQubits}
        onChangeNumQubits={handleNumQubitsChange}
        shots={shots}
        onChangeShots={setShots}
        onRun={handleRun}
        onReset={handleReset}
        running={running}
        error={error}
      />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[260px_minmax(0,1fr)]">
        <GatePalette onSelect={setSelectedGate} selected={selectedGate} />

        <div className="space-y-4">
          <CircuitGrid
            numQubits={numQubits}
            ops={ops}
            onDropGate={handleDropGate}
            onCellClick={handleCellClick}
            onToggleControl={handleToggleControl}
            onRemoveOp={handleRemoveOp}
          />

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <ProbabilityChart result={result} />
            <StatePanel result={result} />
          </div>
        </div>
      </div>
    </div>
  );
};
