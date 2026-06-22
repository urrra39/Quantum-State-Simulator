"""REST routes for the TensorQ-Engine simulator."""

from __future__ import annotations

from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException, status

from tensorq import __version__
from tensorq.core import (
    CONTROLLED_GATES,
    GATE_LIBRARY,
    SINGLE_QUBIT_GATES,
    probability_distribution,
    sample_measurements,
    simulate_circuit,
)
from tensorq.core.measurement import basis_label
from tensorq.exceptions import CircuitValidationError, DimensionMismatchError
from tensorq.models.schemas import (
    Amplitude,
    GateInfo,
    HealthResponse,
    SimulationRequest,
    SimulationResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@router.get("/gates", response_model=List[GateInfo], tags=["meta"])
def list_gates() -> List[GateInfo]:
    """Return metadata for every gate the engine supports."""
    descriptions = {
        "I": "Identity (no-op).",
        "X": "Pauli-X (bit flip): |0> <-> |1>.",
        "Y": "Pauli-Y: bit + phase flip.",
        "Z": "Pauli-Z (phase flip): |1> -> -|1>.",
        "H": "Hadamard: maps |0> -> (|0>+|1>)/sqrt(2).",
        "S": "Phase gate: |1> -> i|1>.",
        "T": "pi/8 gate: |1> -> e^{i pi/4}|1>.",
        "CNOT": "Controlled-X with one control.",
        "CX": "Controlled-X with one control.",
        "CY": "Controlled-Y with one control.",
        "CZ": "Controlled-Z with one control.",
        "CH": "Controlled-Hadamard with one control.",
        "TOFFOLI": "Doubly-controlled X (CCX).",
        "CCX": "Doubly-controlled X (Toffoli).",
    }

    out: List[GateInfo] = []
    for name, matrix in GATE_LIBRARY.items():
        kind = "single" if name in SINGLE_QUBIT_GATES else "controlled"
        rows: list[list[tuple[float, float]]] = [
            [(float(c.real), float(c.imag)) for c in row] for row in matrix
        ]
        out.append(
            GateInfo(
                name=name,
                kind=kind,
                matrix=rows,
                description=descriptions.get(name, ""),
            )
        )
    return out


@router.post(
    "/simulate",
    response_model=SimulationResponse,
    tags=["simulation"],
    status_code=status.HTTP_200_OK,
)
def simulate(req: SimulationRequest) -> SimulationResponse:
    """Run a circuit and return amplitudes, probabilities, and optional shots."""
    try:
        result = simulate_circuit(
            num_qubits=req.num_qubits,
            operations=[op.model_dump() for op in req.operations],
        )
    except CircuitValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "circuit_validation", "message": str(exc)},
        ) from exc
    except DimensionMismatchError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "dimension_mismatch", "message": str(exc)},
        ) from exc
    except Exception as exc:  # pragma: no cover - safety net
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "internal", "message": str(exc)},
        ) from exc

    state = result.state_vector
    probs = probability_distribution(state)

    amplitudes = [
        Amplitude(real=float(c.real), imag=float(c.imag)) for c in state.tolist()
    ]
    labels = [basis_label(i, result.num_qubits) for i in range(state.size)]

    counts: dict[str, int] = {}
    if req.shots > 0:
        rng = np.random.default_rng(req.seed) if req.seed is not None else None
        counts = sample_measurements(state, result.num_qubits, req.shots, rng=rng)

    return SimulationResponse(
        num_qubits=result.num_qubits,
        operations_applied=result.operations_applied,
        basis_labels=labels,
        amplitudes=amplitudes,
        probabilities=[float(p) for p in probs.tolist()],
        counts=counts,
    )
