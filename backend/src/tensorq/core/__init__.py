"""Core quantum math primitives: gates, tensor products, simulator, measurement."""

from tensorq.core.gates import (
    KET_0,
    KET_1,
    GATE_LIBRARY,
    SINGLE_QUBIT_GATES,
    CONTROLLED_GATES,
    get_gate,
)
from tensorq.core.tensor import (
    embed_single_qubit_gate,
    build_controlled_gate,
    kron_n,
)
from tensorq.core.simulator import simulate_circuit, SimulationResult
from tensorq.core.measurement import probability_distribution, sample_measurements

__all__ = [
    "KET_0",
    "KET_1",
    "GATE_LIBRARY",
    "SINGLE_QUBIT_GATES",
    "CONTROLLED_GATES",
    "get_gate",
    "embed_single_qubit_gate",
    "build_controlled_gate",
    "kron_n",
    "simulate_circuit",
    "SimulationResult",
    "probability_distribution",
    "sample_measurements",
]
