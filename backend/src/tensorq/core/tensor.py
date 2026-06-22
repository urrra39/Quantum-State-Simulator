"""
Tensor-product (Kronecker) algebra for embedding gates into n-qubit Hilbert space.

Every k-qubit gate U_k acting on specific qubit indices in an n-qubit register
must be lifted to a full 2^n x 2^n unitary on the joint Hilbert space. We do
this via Kronecker products and (for controlled gates) projector decomposition.

Mathematical recap
------------------
For a single-qubit gate U applied to qubit ``target`` of an n-qubit register,
the full operator is

    U_full = I_2^{otimes target} (x) U (x) I_2^{otimes (n - target - 1)}

For a controlled gate with controls C = {c_1, ..., c_k} and target t applying U,

    CU = I_{2^n} + (prod_i P1^{(c_i)}) @ (U^{(t)} - I_{2^n})

where P1^{(q)} embeds |1><1| at qubit q (identity elsewhere) and U^{(t)} is U
embedded at qubit t. Projectors at distinct qubits commute, so the product is
well-defined and equals the full multi-controlled operator.
"""

from __future__ import annotations

from functools import reduce
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from tensorq.core.gates import I2, PROJ_1

ComplexMatrix = NDArray[np.complex128]


def kron_n(matrices: Sequence[ComplexMatrix]) -> ComplexMatrix:
    """Compute the Kronecker product of an ordered sequence of matrices.

    Equivalent to ``M_0 (x) M_1 (x) ... (x) M_{n-1}``, evaluated left-to-right.
    """
    if not matrices:
        raise ValueError("kron_n requires at least one matrix.")
    return reduce(lambda a, b: np.kron(a, b), matrices)


def _validate_qubit(qubit: int, num_qubits: int, *, label: str = "qubit") -> None:
    if not isinstance(qubit, (int, np.integer)):
        raise TypeError(f"{label} index must be an integer, got {type(qubit).__name__}.")
    if qubit < 0 or qubit >= num_qubits:
        raise IndexError(
            f"{label} index {qubit} is out of range for an {num_qubits}-qubit register."
        )


def embed_single_qubit_gate(
    gate: ComplexMatrix, target: int, num_qubits: int
) -> ComplexMatrix:
    """Lift a 2x2 unitary into the full 2^n x 2^n operator acting on ``target``.

    Parameters
    ----------
    gate
        A 2x2 complex unitary.
    target
        The qubit index (0-based, MSB-first) to which the gate applies.
    num_qubits
        Total number of qubits in the register.

    Returns
    -------
    A 2^n x 2^n complex matrix.
    """
    if gate.shape != (2, 2):
        raise ValueError(
            f"embed_single_qubit_gate expected a 2x2 matrix, got shape {gate.shape}."
        )
    _validate_qubit(target, num_qubits, label="target")

    factors: list[ComplexMatrix] = []
    for q in range(num_qubits):
        factors.append(gate if q == target else I2)
    return kron_n(factors)


def _embed_projector_at(qubit: int, num_qubits: int) -> ComplexMatrix:
    """Embed |1><1| at ``qubit`` and identity elsewhere."""
    factors = [PROJ_1 if q == qubit else I2 for q in range(num_qubits)]
    return kron_n(factors)


def build_controlled_gate(
    target_gate: ComplexMatrix,
    controls: Iterable[int],
    target: int,
    num_qubits: int,
) -> ComplexMatrix:
    """Build the full 2^n x 2^n unitary for a multi-controlled single-qubit gate.

    The construction uses projector decomposition:

        CU = I + (P1_{c1} ... P1_{ck}) @ (U_t - I)

    which yields an exact unitary on the joint Hilbert space, equivalent to
    "apply U at qubit t iff every control qubit is |1>".

    Raises
    ------
    ValueError
        If controls and target overlap, or the target gate is not 2x2.
    """
    if target_gate.shape != (2, 2):
        raise ValueError(
            f"build_controlled_gate expected a 2x2 target gate, got {target_gate.shape}."
        )

    control_list = list(controls)
    if not control_list:
        raise ValueError("build_controlled_gate requires at least one control qubit.")

    seen: set[int] = set()
    for c in control_list:
        _validate_qubit(c, num_qubits, label="control")
        if c in seen:
            raise ValueError(f"Duplicate control qubit {c}.")
        seen.add(c)
    _validate_qubit(target, num_qubits, label="target")
    if target in seen:
        raise ValueError(f"Target qubit {target} cannot also be a control.")

    dim = 1 << num_qubits  # 2 ** num_qubits
    identity_full = np.eye(dim, dtype=np.complex128)

    # Project all controls to |1> simultaneously. Projectors on distinct qubits
    # commute, so the product is the joint projector |1...1><1...1|_{controls}
    # tensored with identity on the other qubits.
    control_projector = identity_full.copy()
    for c in control_list:
        control_projector = control_projector @ _embed_projector_at(c, num_qubits)

    target_full = embed_single_qubit_gate(target_gate, target, num_qubits)

    return identity_full + control_projector @ (target_full - identity_full)
