"""
Fundamental quantum gate library.

All gates are defined from first principles as 2x2 complex unitary matrices,
without any external quantum library. Computational basis states |0> and |1>
are defined as 2D complex column vectors.

Convention
----------
- Column-vector convention: |psi'> = U |psi>
- Computational basis ordering: |0> = (1, 0)^T, |1> = (0, 1)^T
- Multi-qubit ordering: qubit index 0 is the most significant (leftmost) in
  the tensor product. For two qubits |q0 q1>, the basis-state index is
  i = 2*q0 + q1.
"""

from __future__ import annotations

from typing import Dict, Final

import numpy as np
from numpy.typing import NDArray

ComplexMatrix = NDArray[np.complex128]
ComplexVector = NDArray[np.complex128]

# ---------------------------------------------------------------------------
# Computational basis states (|0> and |1>)
# ---------------------------------------------------------------------------

KET_0: Final[ComplexVector] = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
KET_1: Final[ComplexVector] = np.array([0.0 + 0j, 1.0 + 0j], dtype=np.complex128)

# Outer-product projectors used to build controlled operators:
#   P0 = |0><0|,  P1 = |1><1|
PROJ_0: Final[ComplexMatrix] = np.outer(KET_0, KET_0.conj()).astype(np.complex128)
PROJ_1: Final[ComplexMatrix] = np.outer(KET_1, KET_1.conj()).astype(np.complex128)

# 2x2 identity used pervasively as the "do nothing" tensor factor.
I2: Final[ComplexMatrix] = np.eye(2, dtype=np.complex128)

# ---------------------------------------------------------------------------
# Single-qubit unitary gates (defined from scratch)
# ---------------------------------------------------------------------------

# Pauli-X (bit flip):       X = [[0, 1], [1, 0]]
PAULI_X: Final[ComplexMatrix] = np.array(
    [[0.0 + 0j, 1.0 + 0j], [1.0 + 0j, 0.0 + 0j]], dtype=np.complex128
)

# Pauli-Y:                  Y = [[0, -i], [i, 0]]
PAULI_Y: Final[ComplexMatrix] = np.array(
    [[0.0 + 0j, 0.0 - 1j], [0.0 + 1j, 0.0 + 0j]], dtype=np.complex128
)

# Pauli-Z (phase flip):     Z = [[1, 0], [0, -1]]
PAULI_Z: Final[ComplexMatrix] = np.array(
    [[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, -1.0 + 0j]], dtype=np.complex128
)

# Hadamard:                 H = (1/sqrt(2)) * [[1, 1], [1, -1]]
HADAMARD: Final[ComplexMatrix] = (1.0 / np.sqrt(2.0)) * np.array(
    [[1.0 + 0j, 1.0 + 0j], [1.0 + 0j, -1.0 + 0j]], dtype=np.complex128
)

# Phase gate (S):           S = [[1, 0], [0, i]]
PHASE_S: Final[ComplexMatrix] = np.array(
    [[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 0.0 + 1j]], dtype=np.complex128
)

# pi/8 gate (T):            T = [[1, 0], [0, exp(i*pi/4)]]
T_GATE: Final[ComplexMatrix] = np.array(
    [[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, np.exp(1j * np.pi / 4.0)]],
    dtype=np.complex128,
)

# Identity gate (no-op slot in a circuit)
IDENTITY: Final[ComplexMatrix] = I2.copy()

# ---------------------------------------------------------------------------
# Gate registry
# ---------------------------------------------------------------------------

# Single-qubit gates available to circuits. Keys are normalized to upper case.
SINGLE_QUBIT_GATES: Final[Dict[str, ComplexMatrix]] = {
    "I": IDENTITY,
    "X": PAULI_X,
    "Y": PAULI_Y,
    "Z": PAULI_Z,
    "H": HADAMARD,
    "S": PHASE_S,
    "T": T_GATE,
}

# Controlled gates: each entry maps a circuit-level name to the *target*
# single-qubit unitary that is applied when all control qubits are |1>.
# The full 2^n x 2^n matrix is constructed at simulation time via Kronecker
# products and projector decomposition (see core/tensor.py).
CONTROLLED_GATES: Final[Dict[str, ComplexMatrix]] = {
    "CNOT": PAULI_X,  # CX
    "CX": PAULI_X,
    "CY": PAULI_Y,
    "CZ": PAULI_Z,
    "CH": HADAMARD,
    "TOFFOLI": PAULI_X,  # CCX
    "CCX": PAULI_X,
}

GATE_LIBRARY: Final[Dict[str, ComplexMatrix]] = {
    **SINGLE_QUBIT_GATES,
    **CONTROLLED_GATES,
}


def get_gate(name: str) -> ComplexMatrix:
    """Return the canonical 2x2 unitary for ``name`` (single-qubit or target of controlled).

    Raises
    ------
    KeyError
        If the gate name is not registered.
    """
    key = name.strip().upper()
    if key not in GATE_LIBRARY:
        raise KeyError(f"Unknown gate '{name}'. Registered: {sorted(GATE_LIBRARY)}")
    return GATE_LIBRARY[key]


def is_unitary(matrix: ComplexMatrix, atol: float = 1e-10) -> bool:
    """Sanity check: M is unitary iff M^dagger @ M == I."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    identity = np.eye(matrix.shape[0], dtype=np.complex128)
    return bool(np.allclose(matrix.conj().T @ matrix, identity, atol=atol))
