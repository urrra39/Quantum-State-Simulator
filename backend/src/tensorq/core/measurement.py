"""
Measurement primitives: Born-rule probability distribution and sampling.

For a normalized state |psi> = sum_i alpha_i |i>, the probability of observing
basis state |i> upon a computational-basis measurement is

    p_i = |alpha_i|^2 = alpha_i * conj(alpha_i).

Sampling N shots simulates wavefunction collapse: each shot independently
draws a basis state from the discrete distribution {p_i}.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from numpy.typing import NDArray

ComplexVector = NDArray[np.complex128]


def probability_distribution(state_vector: ComplexVector) -> NDArray[np.float64]:
    """Return the |alpha_i|^2 distribution for a normalized state vector."""
    if state_vector.ndim != 1:
        raise ValueError(
            f"state_vector must be 1-D, got shape {state_vector.shape}."
        )
    probs = np.abs(state_vector) ** 2
    # Snap negligible negative noise to zero, then renormalize defensively.
    probs = np.clip(probs.real, 0.0, None)
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("State has zero total probability.")
    return probs / total


def basis_label(index: int, num_qubits: int) -> str:
    """Return the bitstring label '|q0 q1 ... q_{n-1}>' for a basis index."""
    return format(index, f"0{num_qubits}b")


def sample_measurements(
    state_vector: ComplexVector,
    num_qubits: int,
    shots: int,
    *,
    rng: np.random.Generator | None = None,
) -> Dict[str, int]:
    """Simulate ``shots`` measurements and return a counts histogram.

    The histogram keys are bitstring labels in MSB-first order, matching the
    qubit-indexing convention used elsewhere in the engine.
    """
    if shots < 0:
        raise ValueError("shots must be non-negative.")
    probs = probability_distribution(state_vector)
    if probs.size != (1 << num_qubits):
        raise ValueError(
            f"Probability vector size {probs.size} does not match 2^{num_qubits}."
        )

    counts: Dict[str, int] = {}
    if shots == 0:
        return counts

    generator = rng if rng is not None else np.random.default_rng()
    samples = generator.choice(probs.size, size=shots, p=probs)

    unique, freq = np.unique(samples, return_counts=True)
    for idx, n in zip(unique.tolist(), freq.tolist()):
        counts[basis_label(int(idx), num_qubits)] = int(n)
    return counts
