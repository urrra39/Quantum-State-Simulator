"""Tests for measurement primitives (``tensorq.core.measurement``).

The Born rule says that measuring a normalized state |psi> = sum_i alpha_i |i>
in the computational basis yields outcome i with probability p_i = |alpha_i|^2.
``probability_distribution`` computes that vector; ``sample_measurements`` draws
shots from it (collapse simulation), returning an MSB-first bitstring histogram.

We assert:
    - probabilities sum to 1 and equal |amp|^2,
    - basis_label produces zero-padded MSB-first bitstrings,
    - sampling is deterministic under a seeded Generator,
    - counts always sum to the requested shots,
    - a Bell state only ever yields '00' or '11',
    - and a seeded large-N run is ~50/50 within a generous statistical band.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorq.core.measurement import (
    basis_label,
    probability_distribution,
    sample_measurements,
)
from tensorq.core.simulator import simulate_circuit

INV_SQRT2 = 1.0 / np.sqrt(2.0)


def _bell_state() -> np.ndarray:
    """(|00> + |11>)/sqrt(2)."""
    return simulate_circuit(
        2,
        [
            {"gate": "H", "targets": [0]},
            {"gate": "CNOT", "targets": [1], "controls": [0]},
        ],
    ).state_vector


# ---------------------------------------------------------------------------
# probability_distribution
# ---------------------------------------------------------------------------

def test_probabilities_sum_to_one() -> None:
    """A normalized state has probabilities summing to 1."""
    state = np.array([INV_SQRT2, INV_SQRT2], dtype=np.complex128)
    probs = probability_distribution(state)
    assert np.isclose(probs.sum(), 1.0, atol=1e-12)


def test_probabilities_equal_amplitude_modulus_squared() -> None:
    """p_i == |alpha_i|^2 even for complex amplitudes with nontrivial phase."""
    # State with a complex amplitude: (|0> + i|1>)/sqrt(2).
    state = np.array([INV_SQRT2, 1j * INV_SQRT2], dtype=np.complex128)
    probs = probability_distribution(state)
    assert np.allclose(probs, np.abs(state) ** 2, atol=1e-12)
    assert np.allclose(probs, [0.5, 0.5], atol=1e-12)


def test_probabilities_basis_state() -> None:
    """A pure basis state |10> has probability 1 on index 2 (MSB-first n=2)."""
    state = np.zeros(4, dtype=np.complex128)
    state[2] = 1.0
    probs = probability_distribution(state)
    assert np.allclose(probs, [0, 0, 1, 0], atol=1e-12)


def test_probabilities_normalize_unnormalized_input() -> None:
    """An unnormalized state is renormalized to a valid distribution."""
    state = np.array([2.0, 0.0], dtype=np.complex128)  # norm 2
    probs = probability_distribution(state)
    assert np.allclose(probs, [1.0, 0.0], atol=1e-12)


def test_probabilities_zero_state_raises() -> None:
    """A zero-norm state has no valid probability distribution."""
    state = np.zeros(4, dtype=np.complex128)
    with pytest.raises(ValueError):
        probability_distribution(state)


def test_probabilities_two_dimensional_input_raises() -> None:
    """probability_distribution requires a 1-D state vector."""
    state = np.eye(2, dtype=np.complex128)
    with pytest.raises(ValueError):
        probability_distribution(state)


# ---------------------------------------------------------------------------
# basis_label
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "index, n, expected",
    [
        (0, 2, "00"),
        (1, 2, "01"),
        (2, 2, "10"),
        (3, 2, "11"),
        (0, 3, "000"),
        (5, 3, "101"),
        (7, 3, "111"),
        (0, 1, "0"),
        (1, 1, "1"),
    ],
)
def test_basis_label_msb_first(index: int, n: int, expected: str) -> None:
    """basis_label zero-pads to n bits in MSB-first order."""
    assert basis_label(index, n) == expected


def test_basis_label_length_matches_qubit_count() -> None:
    """Every label has exactly num_qubits characters."""
    for i in range(16):
        assert len(basis_label(i, 4)) == 4


# ---------------------------------------------------------------------------
# sample_measurements -- determinism and bookkeeping
# ---------------------------------------------------------------------------

def test_sampling_is_deterministic_with_same_seed() -> None:
    """Two seeded Generators with the same seed yield identical histograms."""
    state = _bell_state()
    rng_a = np.random.default_rng(12345)
    rng_b = np.random.default_rng(12345)
    counts_a = sample_measurements(state, 2, 1000, rng=rng_a)
    counts_b = sample_measurements(state, 2, 1000, rng=rng_b)
    assert counts_a == counts_b


def test_sampling_differs_with_different_seed() -> None:
    """Different seeds (very likely) produce different histograms for a fair coin."""
    state = _bell_state()
    counts_a = sample_measurements(state, 2, 1000, rng=np.random.default_rng(1))
    counts_b = sample_measurements(state, 2, 1000, rng=np.random.default_rng(2))
    # Not a hard guarantee mathematically, but overwhelmingly likely for 1000 shots.
    assert counts_a != counts_b


def test_counts_sum_to_shots() -> None:
    """The histogram totals exactly the number of requested shots."""
    state = _bell_state()
    counts = sample_measurements(state, 2, 777, rng=np.random.default_rng(42))
    assert sum(counts.values()) == 777


def test_zero_shots_returns_empty_dict() -> None:
    """shots=0 means no sampling and an empty histogram."""
    state = _bell_state()
    assert sample_measurements(state, 2, 0) == {}


def test_negative_shots_raises() -> None:
    """A negative shot count is invalid."""
    state = _bell_state()
    with pytest.raises(ValueError):
        sample_measurements(state, 2, -5)


def test_sampling_keys_are_valid_bitstrings() -> None:
    """Every histogram key is a 2-bit MSB-first label."""
    state = _bell_state()
    counts = sample_measurements(state, 2, 500, rng=np.random.default_rng(7))
    for key in counts:
        assert len(key) == 2
        assert set(key) <= {"0", "1"}


# ---------------------------------------------------------------------------
# sample_measurements -- physics: support of the distribution
# ---------------------------------------------------------------------------

def test_bell_sampling_only_yields_00_and_11() -> None:
    """A Bell state has zero amplitude on |01> and |10>, so they are never sampled."""
    state = _bell_state()
    counts = sample_measurements(state, 2, 5000, rng=np.random.default_rng(2024))
    assert set(counts.keys()) <= {"00", "11"}
    # Both outcomes should appear for a fair, large sample.
    assert "00" in counts and "11" in counts
    assert "01" not in counts and "10" not in counts


def test_deterministic_state_samples_single_outcome() -> None:
    """A classical state |1> always collapses to '1'."""
    state = simulate_circuit(1, [{"gate": "X", "targets": [0]}]).state_vector
    counts = sample_measurements(state, 1, 100, rng=np.random.default_rng(0))
    assert counts == {"1": 100}


# ---------------------------------------------------------------------------
# sample_measurements -- statistical sanity check
# ---------------------------------------------------------------------------

def test_bell_sampling_is_approximately_balanced() -> None:
    """For a fair Bell state, '00' and '11' each appear ~50% over many shots.

    With N = 200_000 shots, the standard deviation of the count fraction is
    sqrt(p(1-p)/N) ~= 0.0011, so a +/- 0.03 band is hugely generous (>25 sigma)
    yet still catches a genuinely biased sampler.
    """
    state = _bell_state()
    shots = 200_000
    counts = sample_measurements(state, 2, shots, rng=np.random.default_rng(99))
    frac_00 = counts.get("00", 0) / shots
    frac_11 = counts.get("11", 0) / shots
    assert abs(frac_00 - 0.5) < 0.03
    assert abs(frac_11 - 0.5) < 0.03


def test_uniform_three_qubit_sampling_covers_all_outcomes() -> None:
    """A uniform 3-qubit superposition samples all 8 basis strings with shots enough."""
    state = simulate_circuit(
        3,
        [
            {"gate": "H", "targets": [0]},
            {"gate": "H", "targets": [1]},
            {"gate": "H", "targets": [2]},
        ],
    ).state_vector
    counts = sample_measurements(state, 3, 40_000, rng=np.random.default_rng(11))
    assert len(counts) == 8
    expected = 40_000 / 8.0
    for n in counts.values():
        # Each bucket within a generous band of the 5000 expectation.
        assert abs(n - expected) < 0.15 * expected
