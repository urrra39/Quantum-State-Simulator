"""Tests for the state-vector simulator (``tensorq.core.simulator``).

The simulator starts from |0...0> and applies a sequence of lifted unitaries.
We assert the canonical textbook circuits produce the correct amplitudes under
the MSB-first convention (qubit 0 is the most significant bit; for n=2 the
basis index is 2*q0 + q1):

    - X|0> = |1>
    - H|0> = (|0> + |1>)/sqrt(2)
    - Bell:  H(0), CNOT(0->1)  =>  (|00> + |11>)/sqrt(2)
    - GHZ:   H(0), CNOT(0->1), CNOT(1->2)  =>  (|000> + |111>)/sqrt(2)
    - Toffoli truth table on prepared classical inputs.

All final states are renormalized, so the L2 norm is always 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorq.core.measurement import probability_distribution
from tensorq.core.simulator import (
    SimulationResult,
    simulate_circuit,
)
from tensorq.exceptions import CircuitValidationError, DimensionMismatchError

ATOL = 1e-12
INV_SQRT2 = 1.0 / np.sqrt(2.0)


def _state(num_qubits, operations, **kwargs) -> np.ndarray:
    """Helper: run a circuit and return just the state vector."""
    return simulate_circuit(num_qubits, operations, **kwargs).state_vector


# ---------------------------------------------------------------------------
# Initial state and trivial circuits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_initial_state_is_all_zeros(n: int) -> None:
    """With no operations, the simulator returns |0...0> (index 0 amplitude = 1)."""
    state = _state(n, [])
    assert state.shape == (1 << n,)
    assert np.isclose(state[0], 1.0, atol=ATOL)
    assert np.allclose(state[1:], 0.0, atol=ATOL)


def test_result_is_dataclass_with_metadata() -> None:
    """simulate_circuit returns a SimulationResult carrying counts and dimensions."""
    result = simulate_circuit(2, [{"gate": "H", "targets": [0]}])
    assert isinstance(result, SimulationResult)
    assert result.num_qubits == 2
    assert result.operations_applied == 1
    assert result.state_vector.shape == (4,)


def test_identity_gate_is_noop() -> None:
    """Applying the identity gate leaves |0> unchanged."""
    state = _state(1, [{"gate": "I", "targets": [0]}])
    assert np.allclose(state, np.array([1.0, 0.0], dtype=np.complex128), atol=ATOL)


def test_x_then_x_returns_to_zero() -> None:
    """X is an involution: X X |0> = |0>."""
    state = _state(1, [{"gate": "X", "targets": [0]}, {"gate": "X", "targets": [0]}])
    assert np.allclose(state, np.array([1.0, 0.0], dtype=np.complex128), atol=ATOL)


# ---------------------------------------------------------------------------
# Single-qubit canonical states
# ---------------------------------------------------------------------------

def test_x_flip_produces_ket_one() -> None:
    """X|0> = |1>: amplitude[1] = 1."""
    state = _state(1, [{"gate": "X", "targets": [0]}])
    assert np.allclose(state, np.array([0.0, 1.0], dtype=np.complex128), atol=ATOL)


def test_hadamard_superposition() -> None:
    """H|0> = (|0> + |1>)/sqrt(2): both amplitudes are +1/sqrt(2), probs are [0.5, 0.5]."""
    state = _state(1, [{"gate": "H", "targets": [0]}])
    assert np.allclose(state, np.array([INV_SQRT2, INV_SQRT2], dtype=np.complex128), atol=1e-12)
    probs = probability_distribution(state)
    assert np.allclose(probs, [0.5, 0.5], atol=1e-12)


def test_gate_name_case_insensitive() -> None:
    """Operation gate names are normalized to upper case before lookup."""
    state_lower = _state(1, [{"gate": "x", "targets": [0]}])
    state_upper = _state(1, [{"gate": "X", "targets": [0]}])
    assert np.allclose(state_lower, state_upper, atol=ATOL)
    assert np.allclose(state_lower, [0.0, 1.0], atol=ATOL)


def test_phase_gate_adds_phase_to_one() -> None:
    """S applied to |1> gives i|1>: H then... actually prepare |1>, apply S -> i|1>."""
    state = _state(1, [{"gate": "X", "targets": [0]}, {"gate": "S", "targets": [0]}])
    assert np.allclose(state, np.array([0.0, 1j], dtype=np.complex128), atol=ATOL)


# ---------------------------------------------------------------------------
# Bell state
# ---------------------------------------------------------------------------

def test_bell_state_amplitudes() -> None:
    """H(0) then CNOT(control=0, target=1) yields (|00> + |11>)/sqrt(2).

    Amplitudes[0] == amplitudes[3] == 1/sqrt(2) (real), and amplitudes[1,2] == 0.
    """
    state = _state(
        2,
        [
            {"gate": "H", "targets": [0]},
            {"gate": "CNOT", "targets": [1], "controls": [0]},
        ],
    )
    expected = np.array([INV_SQRT2, 0.0, 0.0, INV_SQRT2], dtype=np.complex128)
    assert np.allclose(state, expected, atol=1e-12)


def test_bell_state_probabilities() -> None:
    """The Bell state has probabilities [0.5, 0, 0, 0.5]."""
    state = _state(
        2,
        [
            {"gate": "H", "targets": [0]},
            {"gate": "CX", "targets": [1], "controls": [0]},
        ],
    )
    probs = probability_distribution(state)
    assert np.allclose(probs, [0.5, 0.0, 0.0, 0.5], atol=1e-12)


def test_bell_state_is_entangled() -> None:
    """The Bell state cannot be written as a product state: only |00> and |11> survive."""
    state = _state(
        2,
        [
            {"gate": "H", "targets": [0]},
            {"gate": "CNOT", "targets": [1], "controls": [0]},
        ],
    )
    # Off-diagonal computational components must vanish exactly.
    assert np.isclose(state[1], 0.0, atol=ATOL)
    assert np.isclose(state[2], 0.0, atol=ATOL)


# ---------------------------------------------------------------------------
# GHZ state
# ---------------------------------------------------------------------------

def test_ghz_state_amplitudes() -> None:
    """H(0), CNOT(0->1), CNOT(1->2) builds (|000> + |111>)/sqrt(2).

    Indices 0 (|000>) and 7 (|111>) carry 1/sqrt(2); the rest are zero.
    """
    state = _state(
        3,
        [
            {"gate": "H", "targets": [0]},
            {"gate": "CNOT", "targets": [1], "controls": [0]},
            {"gate": "CNOT", "targets": [2], "controls": [1]},
        ],
    )
    expected = np.zeros(8, dtype=np.complex128)
    expected[0] = INV_SQRT2
    expected[7] = INV_SQRT2
    assert np.allclose(state, expected, atol=1e-12)


def test_ghz_state_probabilities() -> None:
    """GHZ probabilities are 0.5 on |000> and |111>, zero elsewhere."""
    state = _state(
        3,
        [
            {"gate": "H", "targets": [0]},
            {"gate": "CNOT", "targets": [1], "controls": [0]},
            {"gate": "CNOT", "targets": [2], "controls": [1]},
        ],
    )
    probs = probability_distribution(state)
    assert np.isclose(probs[0], 0.5, atol=1e-12)
    assert np.isclose(probs[7], 0.5, atol=1e-12)
    assert np.isclose(probs.sum(), 1.0, atol=1e-12)
    # All other basis states have zero probability.
    for i in (1, 2, 3, 4, 5, 6):
        assert np.isclose(probs[i], 0.0, atol=ATOL)


# ---------------------------------------------------------------------------
# Toffoli truth table
# ---------------------------------------------------------------------------

def test_toffoli_flips_target_when_both_controls_set() -> None:
    """Prepare |110> via X(0), X(1); CCX(controls=[0,1], target=2) -> |111> (index 7)."""
    state = _state(
        3,
        [
            {"gate": "X", "targets": [0]},
            {"gate": "X", "targets": [1]},
            {"gate": "CCX", "targets": [2], "controls": [0, 1]},
        ],
    )
    expected = np.zeros(8, dtype=np.complex128)
    expected[7] = 1.0  # |111>
    assert np.allclose(state, expected, atol=ATOL)
    probs = probability_distribution(state)
    assert np.isclose(probs[7], 1.0, atol=ATOL)


def test_toffoli_leaves_single_control_input_unchanged() -> None:
    """With only one control set (|100>), CCX does nothing -> stays |100> (index 4)."""
    state = _state(
        3,
        [
            {"gate": "X", "targets": [0]},
            {"gate": "TOFFOLI", "targets": [2], "controls": [0, 1]},
        ],
    )
    expected = np.zeros(8, dtype=np.complex128)
    expected[4] = 1.0  # |100>
    assert np.allclose(state, expected, atol=ATOL)


def test_toffoli_leaves_other_single_control_input_unchanged() -> None:
    """With only qubit 1 set (|010>), CCX does nothing -> stays |010> (index 2)."""
    state = _state(
        3,
        [
            {"gate": "X", "targets": [1]},
            {"gate": "CCX", "targets": [2], "controls": [0, 1]},
        ],
    )
    expected = np.zeros(8, dtype=np.complex128)
    expected[2] = 1.0  # |010>
    assert np.allclose(state, expected, atol=ATOL)


# ---------------------------------------------------------------------------
# Normalization and unitarity verification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n, ops",
    [
        (1, [{"gate": "H", "targets": [0]}]),
        (1, [{"gate": "T", "targets": [0]}, {"gate": "H", "targets": [0]}]),
        (2, [{"gate": "H", "targets": [0]}, {"gate": "CNOT", "targets": [1], "controls": [0]}]),
        (
            3,
            [
                {"gate": "H", "targets": [0]},
                {"gate": "H", "targets": [1]},
                {"gate": "H", "targets": [2]},
            ],
        ),
    ],
)
def test_final_state_is_normalized(n, ops) -> None:
    """Every simulated state has unit L2 norm (<psi|psi> = 1)."""
    state = _state(n, ops)
    assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-12)
    assert np.isclose(probability_distribution(state).sum(), 1.0, atol=1e-12)


def test_uniform_superposition_three_qubits() -> None:
    """H on all 3 qubits gives a uniform superposition: every prob == 1/8."""
    state = _state(
        3,
        [
            {"gate": "H", "targets": [0]},
            {"gate": "H", "targets": [1]},
            {"gate": "H", "targets": [2]},
        ],
    )
    probs = probability_distribution(state)
    assert np.allclose(probs, np.full(8, 1.0 / 8.0), atol=1e-12)


def test_verify_unitarity_path_succeeds_for_valid_circuit() -> None:
    """verify_unitarity=True accepts a valid circuit (all lifted operators unitary)."""
    result = simulate_circuit(
        2,
        [
            {"gate": "H", "targets": [0]},
            {"gate": "CNOT", "targets": [1], "controls": [0]},
        ],
        verify_unitarity=True,
    )
    assert np.isclose(np.linalg.norm(result.state_vector), 1.0, atol=1e-12)


def test_operations_applied_counter() -> None:
    """operations_applied equals the number of ops in the circuit."""
    result = simulate_circuit(
        2,
        [
            {"gate": "H", "targets": [0]},
            {"gate": "X", "targets": [1]},
            {"gate": "CNOT", "targets": [1], "controls": [0]},
        ],
    )
    assert result.operations_applied == 3


# ---------------------------------------------------------------------------
# CircuitValidationError cases
# ---------------------------------------------------------------------------

def test_unknown_gate_raises() -> None:
    """An unregistered gate name is a validation error."""
    with pytest.raises(CircuitValidationError):
        simulate_circuit(1, [{"gate": "FOO", "targets": [0]}])


def test_qubit_out_of_range_raises() -> None:
    """A target index >= num_qubits is a validation error."""
    with pytest.raises(CircuitValidationError):
        simulate_circuit(2, [{"gate": "X", "targets": [5]}])


def test_negative_qubit_index_raises() -> None:
    """A negative qubit index is out of range."""
    with pytest.raises(CircuitValidationError):
        simulate_circuit(2, [{"gate": "X", "targets": [-1]}])


def test_single_qubit_gate_with_controls_raises() -> None:
    """A single-qubit gate may not carry control qubits."""
    with pytest.raises(CircuitValidationError):
        simulate_circuit(2, [{"gate": "X", "targets": [1], "controls": [0]}])


def test_controlled_gate_missing_controls_raises() -> None:
    """A CNOT requires at least one control qubit."""
    with pytest.raises(CircuitValidationError):
        simulate_circuit(2, [{"gate": "CNOT", "targets": [1]}])


def test_ccx_with_single_control_raises() -> None:
    """CCX/Toffoli requires at least two control qubits."""
    with pytest.raises(CircuitValidationError):
        simulate_circuit(3, [{"gate": "CCX", "targets": [2], "controls": [0]}])


def test_duplicate_targets_raises() -> None:
    """Duplicate target qubits are rejected."""
    with pytest.raises(CircuitValidationError):
        simulate_circuit(2, [{"gate": "X", "targets": [0, 0]}])


def test_target_control_overlap_raises() -> None:
    """A qubit cannot be both a control and a target."""
    with pytest.raises(CircuitValidationError):
        simulate_circuit(2, [{"gate": "CNOT", "targets": [0], "controls": [0]}])


def test_too_many_targets_for_single_qubit_gate_raises() -> None:
    """A single-qubit gate requires exactly one target."""
    with pytest.raises(CircuitValidationError):
        simulate_circuit(2, [{"gate": "H", "targets": [0, 1]}])


def test_num_qubits_below_one_raises() -> None:
    """num_qubits must be a positive integer (>= 1)."""
    with pytest.raises(CircuitValidationError):
        simulate_circuit(0, [])


def test_num_qubits_above_cap_raises() -> None:
    """num_qubits is capped at 12 for memory tractability."""
    with pytest.raises(CircuitValidationError):
        simulate_circuit(13, [])


def test_empty_gate_name_raises() -> None:
    """An empty gate name is a validation error."""
    with pytest.raises(CircuitValidationError):
        simulate_circuit(1, [{"gate": "  ", "targets": [0]}])


def test_exception_hierarchy() -> None:
    """CircuitValidationError and DimensionMismatchError descend from TensorQError."""
    from tensorq.exceptions import TensorQError

    assert issubclass(CircuitValidationError, TensorQError)
    assert issubclass(DimensionMismatchError, TensorQError)
