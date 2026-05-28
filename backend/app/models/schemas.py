"""Pydantic v2 schemas for the simulator HTTP API."""

from __future__ import annotations

from typing import Dict, List, Tuple

from pydantic import BaseModel, Field, field_validator


class GateOperation(BaseModel):
    """A single instruction in a quantum circuit."""

    gate: str = Field(..., description="Gate name (e.g. 'H', 'X', 'CNOT', 'TOFFOLI').")
    targets: List[int] = Field(
        default_factory=list,
        description="Target qubit indices the gate acts on.",
    )
    controls: List[int] = Field(
        default_factory=list,
        description="Control qubit indices for controlled gates (empty for single-qubit gates).",
    )

    @field_validator("gate")
    @classmethod
    def _strip_gate(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Gate name must be a non-empty string.")
        return v.strip().upper()

    @field_validator("targets", "controls")
    @classmethod
    def _non_negative(cls, v: List[int]) -> List[int]:
        for idx in v:
            if idx < 0:
                raise ValueError("Qubit indices must be non-negative.")
        return v


class SimulationRequest(BaseModel):
    """Full circuit specification."""

    num_qubits: int = Field(..., ge=1, le=12, description="Number of qubits (1-12).")
    operations: List[GateOperation] = Field(
        default_factory=list,
        description="Ordered list of gate operations to apply to |0...0>.",
    )
    shots: int = Field(
        default=0,
        ge=0,
        le=1_000_000,
        description="Optional number of measurement shots; 0 to skip sampling.",
    )
    seed: int | None = Field(
        default=None, description="Optional RNG seed for reproducible sampling."
    )


class Amplitude(BaseModel):
    """Complex amplitude split into real/imag for JSON safety."""

    real: float
    imag: float


class SimulationResponse(BaseModel):
    """Output of a circuit simulation."""

    num_qubits: int
    operations_applied: int
    basis_labels: List[str] = Field(
        ..., description="Bitstring label for each basis index (MSB-first)."
    )
    amplitudes: List[Amplitude] = Field(
        ..., description="Complex amplitude alpha_i for each basis state |i>."
    )
    probabilities: List[float] = Field(
        ..., description="|alpha_i|^2 for each basis state |i>."
    )
    counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Sampled measurement counts (empty when shots=0).",
    )


class GateInfo(BaseModel):
    """Metadata describing a gate exposed by the engine."""

    name: str
    kind: str  # "single" | "controlled"
    matrix: List[List[Tuple[float, float]]]
    description: str


class HealthResponse(BaseModel):
    status: str
    version: str
