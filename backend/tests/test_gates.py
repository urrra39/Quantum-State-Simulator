"""Tests for the fundamental gate library (``tensorq.core.gates``).

These tests pin down the *exact* numerical form of every elementary quantum
gate, verify that the whole library is unitary, and check the canonical
single-qubit algebraic identities that any correct gate set must satisfy:

    X^2 = Y^2 = Z^2 = I,  H^2 = I,  S^2 = Z,  T^2 = S,  T^4 = Z,  T^8 = I,
    H Z H = X,  H X H = Z,  i (X Z) = Y.

A gate is unitary iff U^dagger U = I. The Pauli matrices, Hadamard, S, T and
the identity are all unitary by construction; we assert that property holds for
every entry of ``GATE_LIBRARY``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorq.core import gates
from tensorq.core.gates import (
    GATE_LIBRARY,
    HADAMARD,
    I2,
    IDENTITY,
    KET_0,
    KET_1,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    PHASE_S,
    PROJ_0,
    PROJ_1,
    T_GATE,
    get_gate,
    is_unitary,
)

# Tightest tolerance for values that are exact in IEEE-754 (integers, 0, 1);
# a slightly looser one for anything derived through sqrt/exp arithmetic.
ATOL_EXACT = 1e-12
ATOL_DERIVED = 1e-9


# ---------------------------------------------------------------------------
# Exact matrix definitions
# ---------------------------------------------------------------------------

def test_pauli_x_matrix() -> None:
    """X = [[0, 1], [1, 0]] -- the bit-flip / NOT gate."""
    expected = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    assert np.allclose(PAULI_X, expected, atol=ATOL_EXACT)


def test_pauli_y_matrix() -> None:
    """Y = [[0, -i], [i, 0]]."""
    expected = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    assert np.allclose(PAULI_Y, expected, atol=ATOL_EXACT)


def test_pauli_z_matrix() -> None:
    """Z = [[1, 0], [0, -1]] -- the phase-flip gate."""
    expected = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    assert np.allclose(PAULI_Z, expected, atol=ATOL_EXACT)


def test_hadamard_matrix() -> None:
    """H = (1/sqrt(2)) [[1, 1], [1, -1]]."""
    expected = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    assert np.allclose(HADAMARD, expected, atol=ATOL_DERIVED)


def test_phase_s_matrix() -> None:
    """S = [[1, 0], [0, i]]; note S = sqrt(Z)."""
    expected = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    assert np.allclose(PHASE_S, expected, atol=ATOL_EXACT)


def test_t_gate_matrix() -> None:
    """T = [[1, 0], [0, exp(i*pi/4)]]; note T = sqrt(S)."""
    expected = np.array([[1, 0], [0, np.exp(1j * np.pi / 4.0)]], dtype=np.complex128)
    assert np.allclose(T_GATE, expected, atol=ATOL_DERIVED)


def test_identity_matrix() -> None:
    """IDENTITY == I2 == eye(2)."""
    assert np.allclose(IDENTITY, np.eye(2), atol=ATOL_EXACT)
    assert np.allclose(I2, np.eye(2), atol=ATOL_EXACT)


def test_t_gate_phase_value() -> None:
    """The nontrivial T entry is exactly (1+i)/sqrt(2) = e^{i pi/4}."""
    assert np.isclose(T_GATE[1, 1], (1 + 1j) / np.sqrt(2.0), atol=ATOL_DERIVED)


# ---------------------------------------------------------------------------
# Basis states and projectors
# ---------------------------------------------------------------------------

def test_basis_kets() -> None:
    """|0> = (1, 0)^T and |1> = (0, 1)^T, each normalized."""
    assert np.allclose(KET_0, np.array([1, 0], dtype=np.complex128))
    assert np.allclose(KET_1, np.array([0, 1], dtype=np.complex128))
    assert np.isclose(np.linalg.norm(KET_0), 1.0)
    assert np.isclose(np.linalg.norm(KET_1), 1.0)


def test_basis_kets_orthonormal() -> None:
    """<0|0> = <1|1> = 1 and <0|1> = 0."""
    assert np.isclose(np.vdot(KET_0, KET_0), 1.0)
    assert np.isclose(np.vdot(KET_1, KET_1), 1.0)
    assert np.isclose(np.vdot(KET_0, KET_1), 0.0)


def test_projectors_are_outer_products() -> None:
    """P0 = |0><0| and P1 = |1><1|."""
    assert np.allclose(PROJ_0, np.array([[1, 0], [0, 0]], dtype=np.complex128))
    assert np.allclose(PROJ_1, np.array([[0, 0], [0, 1]], dtype=np.complex128))


def test_projectors_idempotent_and_complete() -> None:
    """Projectors satisfy P^2 = P and resolve the identity: P0 + P1 = I."""
    assert np.allclose(PROJ_0 @ PROJ_0, PROJ_0)
    assert np.allclose(PROJ_1 @ PROJ_1, PROJ_1)
    assert np.allclose(PROJ_0 + PROJ_1, np.eye(2))
    # Orthogonality: P0 P1 = 0.
    assert np.allclose(PROJ_0 @ PROJ_1, np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# Unitarity of the whole library
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(GATE_LIBRARY.keys()))
def test_every_gate_is_unitary(name: str) -> None:
    """Every registered gate U satisfies U^dagger U = I (it is unitary)."""
    matrix = GATE_LIBRARY[name]
    assert is_unitary(matrix), f"Gate {name} is not unitary"
    # Explicit cross-check independent of is_unitary's internals.
    assert np.allclose(matrix.conj().T @ matrix, np.eye(2), atol=ATOL_DERIVED)


@pytest.mark.parametrize("name", sorted(GATE_LIBRARY.keys()))
def test_every_gate_is_2x2_complex(name: str) -> None:
    """All library gates are 2x2 complex128 matrices."""
    matrix = GATE_LIBRARY[name]
    assert matrix.shape == (2, 2)
    assert matrix.dtype == np.complex128


# ---------------------------------------------------------------------------
# Single-qubit algebraic identities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "gate, label",
    [(PAULI_X, "X"), (PAULI_Y, "Y"), (PAULI_Z, "Z")],
)
def test_pauli_squares_to_identity(gate: np.ndarray, label: str) -> None:
    """Each Pauli is an involution: P^2 = I."""
    assert np.allclose(gate @ gate, np.eye(2), atol=ATOL_DERIVED), label


def test_hadamard_is_involution() -> None:
    """H^2 = I -- the Hadamard is its own inverse."""
    assert np.allclose(HADAMARD @ HADAMARD, np.eye(2), atol=ATOL_DERIVED)


def test_s_squared_is_z() -> None:
    """S^2 = Z, i.e. S = sqrt(Z)."""
    assert np.allclose(PHASE_S @ PHASE_S, PAULI_Z, atol=ATOL_DERIVED)


def test_t_squared_is_s() -> None:
    """T^2 = S, i.e. T = sqrt(S)."""
    assert np.allclose(T_GATE @ T_GATE, PHASE_S, atol=ATOL_DERIVED)


def test_t_fourth_is_z() -> None:
    """T^4 = Z (since T = Z^{1/4})."""
    t4 = np.linalg.matrix_power(T_GATE, 4)
    assert np.allclose(t4, PAULI_Z, atol=ATOL_DERIVED)


def test_t_eighth_is_identity() -> None:
    """T^8 = I (T applies a phase of 2*pi after eight applications)."""
    t8 = np.linalg.matrix_power(T_GATE, 8)
    assert np.allclose(t8, np.eye(2), atol=ATOL_DERIVED)


def test_hadamard_conjugates_z_to_x() -> None:
    """H Z H = X -- the Hadamard rotates the Z axis onto the X axis."""
    assert np.allclose(HADAMARD @ PAULI_Z @ HADAMARD, PAULI_X, atol=ATOL_DERIVED)


def test_hadamard_conjugates_x_to_z() -> None:
    """H X H = Z -- the inverse rotation of the Bloch sphere."""
    assert np.allclose(HADAMARD @ PAULI_X @ HADAMARD, PAULI_Z, atol=ATOL_DERIVED)


def test_pauli_product_xz_gives_y() -> None:
    """i (X Z) = Y, since X Z = [[0, -1], [1, 0]] and multiplying by i yields Y."""
    xz = PAULI_X @ PAULI_Z
    assert np.allclose(xz, np.array([[0, -1], [1, 0]], dtype=np.complex128))
    assert np.allclose(1j * xz, PAULI_Y, atol=ATOL_DERIVED)


def test_pauli_anticommutation() -> None:
    """The Paulis anticommute: X Y = -Y X (and i X Y = -Z up to sign conventions)."""
    assert np.allclose(PAULI_X @ PAULI_Y, -(PAULI_Y @ PAULI_X), atol=ATOL_DERIVED)
    assert np.allclose(PAULI_Y @ PAULI_Z, -(PAULI_Z @ PAULI_Y), atol=ATOL_DERIVED)
    assert np.allclose(PAULI_Z @ PAULI_X, -(PAULI_X @ PAULI_Z), atol=ATOL_DERIVED)


def test_pauli_commutation_relation_xy_z() -> None:
    """X Y = i Z (cyclic Pauli algebra)."""
    assert np.allclose(PAULI_X @ PAULI_Y, 1j * PAULI_Z, atol=ATOL_DERIVED)


# ---------------------------------------------------------------------------
# get_gate name resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, expected",
    [
        ("x", PAULI_X),
        ("X", PAULI_X),
        ("  h  ", HADAMARD),
        ("Cnot", PAULI_X),  # controlled gate resolves to its target unitary
        ("toffoli", PAULI_X),
        ("cz", PAULI_Z),
    ],
)
def test_get_gate_is_case_and_whitespace_insensitive(name: str, expected: np.ndarray) -> None:
    """get_gate normalizes via strip().upper() before lookup."""
    assert np.allclose(get_gate(name), expected, atol=ATOL_DERIVED)


def test_get_gate_unknown_raises_keyerror() -> None:
    """An unregistered gate name raises KeyError."""
    with pytest.raises(KeyError):
        get_gate("NOT_A_GATE")


def test_get_gate_empty_raises_keyerror() -> None:
    """An empty/whitespace name normalizes to '' which is unregistered -> KeyError."""
    with pytest.raises(KeyError):
        get_gate("   ")


# ---------------------------------------------------------------------------
# is_unitary true/false behavior
# ---------------------------------------------------------------------------

def test_is_unitary_true_for_unitaries() -> None:
    """Known unitaries pass the check."""
    assert is_unitary(PAULI_X)
    assert is_unitary(HADAMARD)
    assert is_unitary(np.eye(4, dtype=np.complex128))


def test_is_unitary_false_for_non_unitary() -> None:
    """A scaled / non-normalized matrix is not unitary."""
    non_unitary = np.array([[1, 2], [0, 1]], dtype=np.complex128)
    assert not is_unitary(non_unitary)
    # Zero matrix is not unitary.
    assert not is_unitary(np.zeros((2, 2), dtype=np.complex128))


def test_is_unitary_false_for_non_square() -> None:
    """Non-square matrices cannot be unitary."""
    rect = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.complex128)
    assert not is_unitary(rect)


def test_is_unitary_false_for_non_2d() -> None:
    """A 1-D array is not a unitary operator."""
    assert not is_unitary(np.array([1, 0], dtype=np.complex128))


def test_library_does_not_mutate_via_get_gate() -> None:
    """get_gate returns the library matrix; sanity-check it equals the named const."""
    assert np.allclose(get_gate("S"), PHASE_S, atol=ATOL_EXACT)
    assert np.allclose(get_gate("T"), T_GATE, atol=ATOL_DERIVED)


def test_module_constant_count() -> None:
    """The library is the union of single-qubit and controlled gates with no loss."""
    assert set(GATE_LIBRARY) == set(gates.SINGLE_QUBIT_GATES) | set(gates.CONTROLLED_GATES)
    assert len(GATE_LIBRARY) == len(gates.SINGLE_QUBIT_GATES) + len(gates.CONTROLLED_GATES)
