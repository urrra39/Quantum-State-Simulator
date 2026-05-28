"""
State-vector simulator.

Evolves the joint state |psi> in C^{2^n} by sequentially applying lifted
unitary operators built from the gate library. The initial state is always
|0>^{otimes n}, i.e. the basis vector with index 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from numpy.typing import NDArray

from app.core.gates import (
    CONTROLLED_GATES,
    SINGLE_QUBIT_GATES,
    get_gate,
    is_unitary,
)
from app.core.tensor import build_controlled_gate, embed_single_qubit_gate
from app.exceptions import (
    CircuitValidationError,
    DimensionMismatchError,
)

ComplexVector = NDArray[np.complex128]


@dataclass(frozen=True)
class GateOp:
    """Parsed circuit operation."""

    name: str
    targets: tuple[int, ...]
    controls: tuple[int, ...]


@dataclass(frozen=True)
class SimulationResult:
    """Container for simulation output."""

    num_qubits: int
    state_vector: ComplexVector
    operations_applied: int


def _initial_state(num_qubits: int) -> ComplexVector:
    """Return |0>^{otimes n} as a 2^n-dimensional complex vector."""
    dim = 1 << num_qubits
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0j
    return state


def _parse_operation(raw: dict, num_qubits: int) -> GateOp:
    if not isinstance(raw, dict):
        raise CircuitValidationError(f"Operation must be an object, got {type(raw).__name__}.")

    name = raw.get("gate")
    if not isinstance(name, str) or not name.strip():
        raise CircuitValidationError("Operation 'gate' must be a non-empty string.")
    key = name.strip().upper()

    targets_raw = raw.get("targets", [])
    controls_raw = raw.get("controls", []) or []

    if not isinstance(targets_raw, list) or not all(isinstance(t, int) for t in targets_raw):
        raise CircuitValidationError(
            f"Gate '{name}': 'targets' must be a list of integer qubit indices."
        )
    if not isinstance(controls_raw, list) or not all(isinstance(c, int) for c in controls_raw):
        raise CircuitValidationError(
            f"Gate '{name}': 'controls' must be a list of integer qubit indices."
        )

    targets = tuple(targets_raw)
    controls = tuple(controls_raw)

    for q in (*targets, *controls):
        if q < 0 or q >= num_qubits:
            raise CircuitValidationError(
                f"Gate '{name}': qubit index {q} out of range "
                f"for {num_qubits}-qubit register."
            )

    if len(set(targets)) != len(targets):
        raise CircuitValidationError(f"Gate '{name}': duplicate target qubits {targets}.")
    if len(set(controls)) != len(controls):
        raise CircuitValidationError(f"Gate '{name}': duplicate control qubits {controls}.")
    if set(targets) & set(controls):
        raise CircuitValidationError(
            f"Gate '{name}': qubit cannot be both control and target."
        )

    if key in SINGLE_QUBIT_GATES:
        if controls:
            raise CircuitValidationError(
                f"Gate '{name}' is single-qubit and cannot have controls."
            )
        if len(targets) != 1:
            raise CircuitValidationError(
                f"Gate '{name}' requires exactly one target qubit, got {len(targets)}."
            )
    elif key in CONTROLLED_GATES:
        if len(targets) != 1:
            raise CircuitValidationError(
                f"Gate '{name}' requires exactly one target qubit, got {len(targets)}."
            )
        # CCX / TOFFOLI need >=2 controls; CNOT/CX/CY/CZ/CH need >=1
        min_ctrl = 2 if key in {"CCX", "TOFFOLI"} else 1
        if len(controls) < min_ctrl:
            raise CircuitValidationError(
                f"Gate '{name}' requires at least {min_ctrl} control qubit(s), "
                f"got {len(controls)}."
            )
    else:
        raise CircuitValidationError(f"Unknown gate '{name}'.")

    return GateOp(name=key, targets=targets, controls=controls)


def _operator_for(op: GateOp, num_qubits: int) -> NDArray[np.complex128]:
    base = get_gate(op.name)
    if op.controls:
        operator = build_controlled_gate(
            target_gate=base,
            controls=op.controls,
            target=op.targets[0],
            num_qubits=num_qubits,
        )
    else:
        operator = embed_single_qubit_gate(base, op.targets[0], num_qubits)

    expected_dim = 1 << num_qubits
    if operator.shape != (expected_dim, expected_dim):
        raise DimensionMismatchError(
            f"Operator for '{op.name}' has shape {operator.shape}, "
            f"expected ({expected_dim}, {expected_dim})."
        )
    return operator


def simulate_circuit(
    num_qubits: int,
    operations: Sequence[dict],
    *,
    verify_unitarity: bool = False,
) -> SimulationResult:
    """Run a circuit and return the final state vector.

    Parameters
    ----------
    num_qubits
        Number of qubits in the register. Must satisfy 1 <= n <= 12 to keep
        the dense 2^n x 2^n matrix representation tractable in memory.
    operations
        Ordered list of dicts of the form
        ``{"gate": str, "targets": [int, ...], "controls": [int, ...]}``.
    verify_unitarity
        When True, every lifted operator is checked against U^dagger U = I.

    Raises
    ------
    CircuitValidationError, DimensionMismatchError
    """
    if not isinstance(num_qubits, int) or num_qubits < 1:
        raise CircuitValidationError("num_qubits must be a positive integer.")
    if num_qubits > 12:
        raise CircuitValidationError(
            f"num_qubits={num_qubits} exceeds the safety cap of 12 "
            f"(2^12 = 4096-dim state vector, 16M-entry operator matrices)."
        )

    parsed: List[GateOp] = [_parse_operation(op, num_qubits) for op in operations]

    state = _initial_state(num_qubits)

    for op in parsed:
        operator = _operator_for(op, num_qubits)
        if verify_unitarity and not is_unitary(operator):
            raise DimensionMismatchError(
                f"Lifted operator for '{op.name}' failed unitarity check."
            )
        if operator.shape[1] != state.shape[0]:
            raise DimensionMismatchError(
                f"Operator shape {operator.shape} incompatible with "
                f"state dimension {state.shape[0]}."
            )
        state = operator @ state

    # Numerical hygiene: renormalize to defend against tiny floating drift.
    norm = float(np.linalg.norm(state))
    if norm == 0.0:
        raise DimensionMismatchError("Final state has zero norm; numerical breakdown.")
    state = state / norm

    return SimulationResult(
        num_qubits=num_qubits,
        state_vector=state,
        operations_applied=len(parsed),
    )
