"""Tests for tensor-product embedding (``tensorq.core.tensor``).

A k-qubit gate acting on specific wires of an n-qubit register is lifted to the
full 2^n x 2^n operator on the joint Hilbert space. With the MSB-first
convention (qubit 0 is the most significant tensor factor):

    embed(U, target=t, n) = I^{(x)t}  (x)  U  (x)  I^{(x)(n-t-1)}.

Controlled gates use projector decomposition:

    CU = I + (P1_{c1} ... P1_{ck}) (U_t - I)

which is exactly "apply U on the target iff every control is |1>".

Ground-truth matrices asserted here (n=2, MSB-first, index = 2*q0 + q1):

    CNOT(control=0, target=1) = [[1,0,0,0],
                                 [0,1,0,0],
                                 [0,0,0,1],
                                 [0,0,1,0]]   (swaps |10> <-> |11>)

    CZ = diag(1, 1, 1, -1)  (symmetric under control/target swap)

    Toffoli(controls={0,1}, target=2) = identity except |110> <-> |111>.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorq.core.gates import (
    HADAMARD,
    I2,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    is_unitary,
)
from tensorq.core.tensor import (
    build_controlled_gate,
    embed_single_qubit_gate,
    kron_n,
)

ATOL = 1e-12


# ---------------------------------------------------------------------------
# kron_n
# ---------------------------------------------------------------------------

def test_kron_n_single_element() -> None:
    """The Kronecker product of a one-element sequence is the element itself."""
    assert np.allclose(kron_n([PAULI_X]), PAULI_X, atol=ATOL)


def test_kron_n_two_elements_matches_numpy() -> None:
    """kron_n([A, B]) == np.kron(A, B)."""
    assert np.allclose(kron_n([PAULI_X, I2]), np.kron(PAULI_X, I2), atol=ATOL)


def test_kron_n_is_left_associative_chain() -> None:
    """kron_n([A, B, C]) == kron(kron(A, B), C) and equals the 8x8 expected size."""
    result = kron_n([HADAMARD, PAULI_X, I2])
    expected = np.kron(np.kron(HADAMARD, PAULI_X), I2)
    assert result.shape == (8, 8)
    assert np.allclose(result, expected, atol=1e-9)


def test_kron_n_identity_chain_is_identity() -> None:
    """The Kronecker product of n identities is the 2^n identity."""
    assert np.allclose(kron_n([I2, I2, I2]), np.eye(8), atol=ATOL)


def test_kron_n_empty_raises_valueerror() -> None:
    """An empty sequence has no well-defined Kronecker product."""
    with pytest.raises(ValueError):
        kron_n([])


# ---------------------------------------------------------------------------
# embed_single_qubit_gate
# ---------------------------------------------------------------------------

def test_embed_x_on_qubit0_is_kron_x_i() -> None:
    """embed(X, target=0, n=2) == X (x) I (qubit 0 is the leftmost factor)."""
    embedded = embed_single_qubit_gate(PAULI_X, 0, 2)
    assert embedded.shape == (4, 4)
    assert np.allclose(embedded, np.kron(PAULI_X, I2), atol=ATOL)


def test_embed_x_on_qubit1_is_kron_i_x() -> None:
    """embed(X, target=1, n=2) == I (x) X."""
    embedded = embed_single_qubit_gate(PAULI_X, 1, 2)
    assert np.allclose(embedded, np.kron(I2, PAULI_X), atol=ATOL)


def test_embed_middle_qubit_three_register() -> None:
    """embed(H, target=1, n=3) == I (x) H (x) I."""
    embedded = embed_single_qubit_gate(HADAMARD, 1, 3)
    expected = np.kron(np.kron(I2, HADAMARD), I2)
    assert embedded.shape == (8, 8)
    assert np.allclose(embedded, expected, atol=1e-9)


@pytest.mark.parametrize("gate", [PAULI_X, PAULI_Y, PAULI_Z, HADAMARD])
@pytest.mark.parametrize("target", [0, 1, 2])
def test_embed_preserves_unitarity(gate: np.ndarray, target: int) -> None:
    """Lifting a unitary by tensoring with identities preserves unitarity."""
    embedded = embed_single_qubit_gate(gate, target, 3)
    assert embedded.shape == (8, 8)
    assert is_unitary(embedded)


def test_embed_target_out_of_range_raises_indexerror() -> None:
    """A target index >= num_qubits is invalid."""
    with pytest.raises(IndexError):
        embed_single_qubit_gate(PAULI_X, 2, 2)


def test_embed_negative_target_raises_indexerror() -> None:
    """A negative target index is invalid."""
    with pytest.raises(IndexError):
        embed_single_qubit_gate(PAULI_X, -1, 2)


def test_embed_non_2x2_gate_raises_valueerror() -> None:
    """embed_single_qubit_gate requires a 2x2 gate."""
    with pytest.raises(ValueError):
        embed_single_qubit_gate(np.eye(4, dtype=np.complex128), 0, 2)


# ---------------------------------------------------------------------------
# build_controlled_gate -- exact CNOT
# ---------------------------------------------------------------------------

CNOT_C0_T1 = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=np.complex128,
)


def test_cnot_exact_matrix_control0_target1() -> None:
    """CNOT(control=0, target=1) on 2 qubits swaps |10> <-> |11> (indices 2,3)."""
    cnot = build_controlled_gate(PAULI_X, controls=[0], target=1, num_qubits=2)
    assert cnot.shape == (4, 4)
    assert np.allclose(cnot, CNOT_C0_T1, atol=ATOL)


def test_cnot_action_on_basis_states() -> None:
    """CNOT flips the target only when the control |0> is set.

    With control=0: |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10>.
    Basis indices: 0,1 unchanged; 2<->3 swapped.
    """
    cnot = build_controlled_gate(PAULI_X, controls=[0], target=1, num_qubits=2)
    e = np.eye(4, dtype=np.complex128)
    assert np.allclose(cnot @ e[:, 0], e[:, 0])  # |00> -> |00>
    assert np.allclose(cnot @ e[:, 1], e[:, 1])  # |01> -> |01>
    assert np.allclose(cnot @ e[:, 2], e[:, 3])  # |10> -> |11>
    assert np.allclose(cnot @ e[:, 3], e[:, 2])  # |11> -> |10>


def test_cnot_control1_target0_swaps_indices_1_and_3() -> None:
    """CNOT(control=1, target=0): flips qubit 0 iff qubit 1 is |1>.

    Affected basis states are those with q1=1: |01> (idx 1) and |11> (idx 3),
    which map to |11> and |01> respectively.
    """
    cnot = build_controlled_gate(PAULI_X, controls=[1], target=0, num_qubits=2)
    e = np.eye(4, dtype=np.complex128)
    assert np.allclose(cnot @ e[:, 0], e[:, 0])  # |00> -> |00>
    assert np.allclose(cnot @ e[:, 1], e[:, 3])  # |01> -> |11>
    assert np.allclose(cnot @ e[:, 2], e[:, 2])  # |10> -> |10>
    assert np.allclose(cnot @ e[:, 3], e[:, 1])  # |11> -> |01>


def test_cnot_is_unitary() -> None:
    """CNOT is its own inverse and is unitary."""
    cnot = build_controlled_gate(PAULI_X, controls=[0], target=1, num_qubits=2)
    assert is_unitary(cnot)
    assert np.allclose(cnot @ cnot, np.eye(4), atol=ATOL)


# ---------------------------------------------------------------------------
# build_controlled_gate -- CZ symmetry
# ---------------------------------------------------------------------------

def test_cz_is_diagonal_phase() -> None:
    """CZ = diag(1, 1, 1, -1): a (-1) phase only on |11>."""
    cz = build_controlled_gate(PAULI_Z, controls=[0], target=1, num_qubits=2)
    assert np.allclose(cz, np.diag([1, 1, 1, -1]).astype(np.complex128), atol=ATOL)


def test_cz_is_symmetric_under_control_target_swap() -> None:
    """CZ is symmetric: swapping control and target yields the identical matrix."""
    cz_01 = build_controlled_gate(PAULI_Z, controls=[0], target=1, num_qubits=2)
    cz_10 = build_controlled_gate(PAULI_Z, controls=[1], target=0, num_qubits=2)
    assert np.allclose(cz_01, cz_10, atol=ATOL)


def test_cz_is_unitary_and_hermitian() -> None:
    """CZ is both unitary and Hermitian (it is diagonal and real)."""
    cz = build_controlled_gate(PAULI_Z, controls=[0], target=1, num_qubits=2)
    assert is_unitary(cz)
    assert np.allclose(cz, cz.conj().T, atol=ATOL)


# ---------------------------------------------------------------------------
# build_controlled_gate -- Toffoli / CCX
# ---------------------------------------------------------------------------

def test_toffoli_exact_matrix() -> None:
    """Toffoli(controls={0,1}, target=2) is the 8x8 identity except |110> <-> |111>.

    Basis indices 6 (|110>) and 7 (|111>) are swapped; all other rows untouched.
    """
    ccx = build_controlled_gate(PAULI_X, controls=[0, 1], target=2, num_qubits=3)
    expected = np.eye(8, dtype=np.complex128)
    expected[[6, 7]] = expected[[7, 6]]
    assert ccx.shape == (8, 8)
    assert np.allclose(ccx, expected, atol=ATOL)


def test_toffoli_only_swaps_last_two_basis_states() -> None:
    """CCX leaves indices 0..5 fixed and swaps 6<->7."""
    ccx = build_controlled_gate(PAULI_X, controls=[0, 1], target=2, num_qubits=3)
    e = np.eye(8, dtype=np.complex128)
    for i in range(6):
        assert np.allclose(ccx @ e[:, i], e[:, i]), f"|{i:03b}> should be unchanged"
    assert np.allclose(ccx @ e[:, 6], e[:, 7])  # |110> -> |111>
    assert np.allclose(ccx @ e[:, 7], e[:, 6])  # |111> -> |110>


def test_toffoli_is_unitary_and_involution() -> None:
    """Toffoli is unitary and its own inverse."""
    ccx = build_controlled_gate(PAULI_X, controls=[0, 1], target=2, num_qubits=3)
    assert is_unitary(ccx)
    assert np.allclose(ccx @ ccx, np.eye(8), atol=ATOL)


def test_controlled_gate_control_order_irrelevant() -> None:
    """Distinct-qubit control projectors commute, so control order does not matter."""
    a = build_controlled_gate(PAULI_X, controls=[0, 1], target=2, num_qubits=3)
    b = build_controlled_gate(PAULI_X, controls=[1, 0], target=2, num_qubits=3)
    assert np.allclose(a, b, atol=ATOL)


# ---------------------------------------------------------------------------
# build_controlled_gate -- validation errors
# ---------------------------------------------------------------------------

def test_build_controlled_empty_controls_raises_valueerror() -> None:
    """A controlled gate needs at least one control qubit."""
    with pytest.raises(ValueError):
        build_controlled_gate(PAULI_X, controls=[], target=0, num_qubits=2)


def test_build_controlled_duplicate_control_raises_valueerror() -> None:
    """Duplicate control qubits are rejected."""
    with pytest.raises(ValueError):
        build_controlled_gate(PAULI_X, controls=[0, 0], target=1, num_qubits=3)


def test_build_controlled_target_equals_control_raises_valueerror() -> None:
    """The target qubit cannot also be a control qubit."""
    with pytest.raises(ValueError):
        build_controlled_gate(PAULI_X, controls=[0], target=0, num_qubits=2)


def test_build_controlled_non_2x2_target_raises_valueerror() -> None:
    """The target unitary must be 2x2."""
    with pytest.raises(ValueError):
        build_controlled_gate(np.eye(4, dtype=np.complex128), controls=[0], target=1, num_qubits=2)


def test_build_controlled_control_out_of_range_raises_indexerror() -> None:
    """An out-of-range control qubit is rejected."""
    with pytest.raises(IndexError):
        build_controlled_gate(PAULI_X, controls=[5], target=0, num_qubits=2)


def test_build_controlled_target_out_of_range_raises_indexerror() -> None:
    """An out-of-range target qubit is rejected."""
    with pytest.raises(IndexError):
        build_controlled_gate(PAULI_X, controls=[0], target=5, num_qubits=2)
