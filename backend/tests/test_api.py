"""Tests for the HTTP API (``tensorq.main.app`` via ``tensorq.api.routes``).

These exercise the real FastAPI application end-to-end with FastAPI's
``TestClient``:

    GET  /                 -> service metadata
    GET  /api/v1/health    -> {"status": "ok", "version": ...}
    GET  /api/v1/gates     -> one GateInfo per library gate
    POST /api/v1/simulate  -> amplitudes, probabilities, optional shot counts

Validation boundaries are split by layer:
    - Pydantic schema rejects num_qubits=0 / 13 with HTTP 422.
    - The simulator rejects an out-of-range qubit index with a
      CircuitValidationError, surfaced as HTTP 400 with
      detail.error == "circuit_validation".
"""

from __future__ import annotations

import numpy as np

from tensorq import __version__
from tensorq.core.gates import GATE_LIBRARY, SINGLE_QUBIT_GATES

INV_SQRT2 = 1.0 / np.sqrt(2.0)


# ---------------------------------------------------------------------------
# Meta endpoints
# ---------------------------------------------------------------------------

def test_root_endpoint(client) -> None:
    """GET / returns the service banner with the package version."""
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "TensorQ-Engine"
    assert body["version"] == __version__


def test_health_endpoint(client) -> None:
    """GET /api/v1/health reports status 'ok' and the version string."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"status": "ok", "version": __version__}


# ---------------------------------------------------------------------------
# Gates listing
# ---------------------------------------------------------------------------

def test_gates_endpoint_lists_full_library(client) -> None:
    """GET /api/v1/gates returns one entry per registered gate."""
    resp = client.get("/api/v1/gates")
    assert resp.status_code == 200
    gates = resp.json()
    assert len(gates) == len(GATE_LIBRARY)
    names = {g["name"] for g in gates}
    assert names == set(GATE_LIBRARY.keys())


def test_gates_entries_have_expected_shape(client) -> None:
    """Each gate has name/kind/matrix/description; matrix is 2x2 of [real, imag] pairs."""
    resp = client.get("/api/v1/gates")
    gates = resp.json()
    for g in gates:
        assert set(g.keys()) >= {"name", "kind", "matrix", "description"}
        assert g["kind"] in {"single", "controlled"}
        matrix = g["matrix"]
        assert len(matrix) == 2  # two rows
        for row in matrix:
            assert len(row) == 2  # two columns
            for entry in row:
                # Each complex entry is serialized as a [real, imag] pair.
                assert len(entry) == 2
                assert all(isinstance(x, (int, float)) for x in entry)


def test_gates_kind_matches_registry(client) -> None:
    """A gate's 'kind' agrees with whether it lives in SINGLE_QUBIT_GATES."""
    resp = client.get("/api/v1/gates")
    for g in resp.json():
        expected_kind = "single" if g["name"] in SINGLE_QUBIT_GATES else "controlled"
        assert g["kind"] == expected_kind


def test_gates_x_matrix_serialization(client) -> None:
    """The Pauli-X entry serializes to [[ [0,0],[1,0] ], [ [1,0],[0,0] ]]."""
    resp = client.get("/api/v1/gates")
    x = next(g for g in resp.json() if g["name"] == "X")
    assert x["matrix"] == [[[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]]


# ---------------------------------------------------------------------------
# Simulate -- Bell state
# ---------------------------------------------------------------------------

def _bell_request(shots: int = 0, seed=None) -> dict:
    body = {
        "num_qubits": 2,
        "operations": [
            {"gate": "H", "targets": [0]},
            {"gate": "CNOT", "targets": [1], "controls": [0]},
        ],
    }
    if shots:
        body["shots"] = shots
    if seed is not None:
        body["seed"] = seed
    return body


def test_simulate_bell_probabilities(client) -> None:
    """POST /simulate of a Bell circuit gives probabilities [0.5, 0, 0, 0.5]."""
    resp = client.post("/api/v1/simulate", json=_bell_request())
    assert resp.status_code == 200
    body = resp.json()
    assert body["num_qubits"] == 2
    assert body["operations_applied"] == 2
    assert body["basis_labels"] == ["00", "01", "10", "11"]
    probs = body["probabilities"]
    assert np.allclose(probs, [0.5, 0.0, 0.0, 0.5], atol=1e-9)


def test_simulate_bell_amplitudes(client) -> None:
    """Bell amplitudes: index 0 and 3 are real 1/sqrt(2); 1 and 2 are zero."""
    resp = client.post("/api/v1/simulate", json=_bell_request())
    amps = resp.json()["amplitudes"]
    assert np.isclose(amps[0]["real"], INV_SQRT2, atol=1e-9)
    assert np.isclose(amps[0]["imag"], 0.0, atol=1e-12)
    assert np.isclose(amps[3]["real"], INV_SQRT2, atol=1e-9)
    assert np.isclose(amps[1]["real"], 0.0, atol=1e-12)
    assert np.isclose(amps[2]["real"], 0.0, atol=1e-12)


def test_simulate_bell_counts_with_shots_and_seed(client) -> None:
    """With shots+seed, counts only span '00'/'11' and total to shots."""
    resp = client.post("/api/v1/simulate", json=_bell_request(shots=2000, seed=123))
    assert resp.status_code == 200
    counts = resp.json()["counts"]
    assert set(counts.keys()) <= {"00", "11"}
    assert sum(counts.values()) == 2000


def test_simulate_seed_reproducibility(client) -> None:
    """Two identical requests with the same seed produce identical counts."""
    req = _bell_request(shots=1500, seed=2024)
    counts_a = client.post("/api/v1/simulate", json=req).json()["counts"]
    counts_b = client.post("/api/v1/simulate", json=req).json()["counts"]
    assert counts_a == counts_b


def test_simulate_no_shots_gives_empty_counts(client) -> None:
    """Without shots, the response carries empty counts but full amplitudes."""
    resp = client.post("/api/v1/simulate", json=_bell_request())
    body = resp.json()
    assert body["counts"] == {}
    assert len(body["amplitudes"]) == 4
    assert len(body["probabilities"]) == 4


def test_simulate_ghz_three_qubits(client) -> None:
    """A GHZ request returns 8 probabilities with 0.5 on |000> and |111>."""
    body = {
        "num_qubits": 3,
        "operations": [
            {"gate": "H", "targets": [0]},
            {"gate": "CNOT", "targets": [1], "controls": [0]},
            {"gate": "CNOT", "targets": [2], "controls": [1]},
        ],
    }
    resp = client.post("/api/v1/simulate", json=body)
    assert resp.status_code == 200
    probs = resp.json()["probabilities"]
    assert len(probs) == 8
    assert np.isclose(probs[0], 0.5, atol=1e-9)
    assert np.isclose(probs[7], 0.5, atol=1e-9)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_simulate_out_of_range_qubit_returns_400(client) -> None:
    """An out-of-range qubit index is a circuit_validation error -> HTTP 400."""
    body = {
        "num_qubits": 2,
        "operations": [{"gate": "X", "targets": [5]}],
    }
    resp = client.post("/api/v1/simulate", json=body)
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["error"] == "circuit_validation"
    assert "message" in detail


def test_simulate_unknown_gate_returns_400(client) -> None:
    """An unknown gate name is also a circuit_validation error -> HTTP 400."""
    body = {
        "num_qubits": 1,
        "operations": [{"gate": "BOGUS", "targets": [0]}],
    }
    resp = client.post("/api/v1/simulate", json=body)
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "circuit_validation"


def test_simulate_num_qubits_zero_returns_422(client) -> None:
    """num_qubits=0 violates the pydantic ge=1 constraint -> HTTP 422."""
    resp = client.post("/api/v1/simulate", json={"num_qubits": 0, "operations": []})
    assert resp.status_code == 422


def test_simulate_num_qubits_thirteen_returns_422(client) -> None:
    """num_qubits=13 violates the pydantic le=12 constraint -> HTTP 422."""
    resp = client.post("/api/v1/simulate", json={"num_qubits": 13, "operations": []})
    assert resp.status_code == 422


def test_simulate_negative_qubit_index_returns_422(client) -> None:
    """A negative qubit index is rejected by the schema validator -> HTTP 422."""
    body = {
        "num_qubits": 2,
        "operations": [{"gate": "X", "targets": [-1]}],
    }
    resp = client.post("/api/v1/simulate", json=body)
    assert resp.status_code == 422


def test_simulate_shots_over_cap_returns_422(client) -> None:
    """shots beyond the 1_000_000 cap violates the schema -> HTTP 422."""
    resp = client.post(
        "/api/v1/simulate",
        json={"num_qubits": 1, "operations": [], "shots": 2_000_000},
    )
    assert resp.status_code == 422
