# TensorQ-Engine

> **High-performance quantum circuit simulator** — state-vector evolution and tensor-network contraction for multi-qubit systems, built **without Qiskit, Cirq, or any other quantum library**. All linear algebra is implemented from first principles in NumPy.

```
|ψ⟩ = Uₙ · ... · U₂ · U₁ · |0⟩^⊗n
```

---

## 1. Architecture

```
┌──────────────────────────────┐         ┌────────────────────────────────┐
│  Quantum Dashboard           │  HTTP   │  Quantum Math Engine           │
│  Next.js 14 (App Router)     │ ──────▶ │  FastAPI · NumPy · SciPy       │
│  TypeScript · Tailwind CSS   │  JSON   │  Pure-Python linear algebra    │
│  Recharts visualization      │         │  state-vector simulator        │
└──────────────────────────────┘         └────────────────────────────────┘
        :3000                                        :8000
              \________ docker bridge: tensorq-net ________/
```

Two services, one private bridge network, orchestrated by `docker-compose.yml`.

| Service             | Tech                         | Port  | Image                       |
| ------------------- | ---------------------------- | ----- | --------------------------- |
| `tensorq-engine`    | Python 3.12 · FastAPI · NumPy| 8000  | `tensorq/engine:1.0.0`      |
| `tensorq-dashboard` | Node 20 · Next.js 14 · TS    | 3000  | `tensorq/dashboard:1.0.0`   |

---

## 2. Mathematical Core

The backend implements the linear-algebraic skeleton of gate-model quantum computing from scratch.

### 2.1 Computational basis

Every qubit lives in a 2-D complex Hilbert space spanned by

```
|0⟩ = (1, 0)ᵀ           |1⟩ = (0, 1)ᵀ
```

declared in `backend/app/core/gates.py` as `numpy.complex128` column vectors.

### 2.2 Single-qubit gates

All unitary matrices are written explicitly, with no external dependency:

| Gate | Matrix |
| ---- | ------ |
| **X** (Pauli-X) | `[[0, 1], [1, 0]]` |
| **Y** (Pauli-Y) | `[[0, -i], [i, 0]]` |
| **Z** (Pauli-Z) | `[[1, 0], [0, -1]]` |
| **H** (Hadamard) | `(1/√2) · [[1, 1], [1, -1]]` |
| **S** (Phase) | `[[1, 0], [0, i]]` |
| **T** (π/8) | `[[1, 0], [0, e^{iπ/4}]]` |

### 2.3 Multi-qubit lifting via Kronecker product

A single-qubit gate `U` acting on qubit `t` of an `n`-qubit register is lifted to the joint space `C^{2ⁿ}` via the tensor product:

```
U_full = I₂^⊗t  ⊗  U  ⊗  I₂^⊗(n−t−1)
```

implemented as `embed_single_qubit_gate(U, t, n)` in `backend/app/core/tensor.py`, using `functools.reduce(np.kron, ...)`.

### 2.4 Controlled gates via projector decomposition

For a multi-controlled gate with control set `C = {c₁, …, cₖ}` and target `t`, applying single-qubit unitary `U` only when every control is in state |1⟩:

```
CU = I + (Π_{c ∈ C} P₁⁽ᶜ⁾) · (U⁽ᵗ⁾ − I)
```

where `P₁⁽ᶜ⁾ = embed(|1⟩⟨1|, c, n)` and `U⁽ᵗ⁾ = embed(U, t, n)`. Projectors at distinct qubits commute, so the product is the joint controls-all-one projector. This handles **CNOT, CY, CZ, CH** (single-control) and **Toffoli/CCX** (multi-control) uniformly. See `build_controlled_gate(...)` in `backend/app/core/tensor.py`.

### 2.5 State evolution and measurement

The initial state is `|0⟩^⊗n`, encoded as the basis vector with index 0. Each operation lifts to a `2ⁿ × 2ⁿ` unitary which left-multiplies the state vector. The Born rule yields the measurement distribution:

```
p_i = |α_i|² = α_i · ᾱ_i
```

Sampling `shots` measurements simulates wavefunction collapse via `numpy.random.Generator.choice` over the discrete distribution `{p_i}`.

### 2.6 Convention

- Qubit index `0` is the **most significant** bit in the tensor-product ordering.
- For `n = 2`, basis indices map `0→|00⟩`, `1→|01⟩`, `2→|10⟩`, `3→|11⟩`.
- A bitstring label like `|110⟩` means `q₀=1, q₁=1, q₂=0`.

---

## 3. HTTP API

Base URL: `http://localhost:8000/api/v1` (or via the dashboard proxy at `/api/tensorq/*`).

### `GET /health`
Returns `{"status": "ok", "version": "1.0.0"}`.

### `GET /gates`
Returns metadata + 2×2 matrix for every registered gate.

### `POST /simulate`

**Request**:
```json
{
  "num_qubits": 2,
  "operations": [
    { "gate": "H",    "targets": [0], "controls": []  },
    { "gate": "CNOT", "targets": [1], "controls": [0] }
  ],
  "shots": 1024,
  "seed": 42
}
```

**Response** (Bell state `(|00⟩ + |11⟩)/√2`):
```json
{
  "num_qubits": 2,
  "operations_applied": 2,
  "basis_labels": ["00", "01", "10", "11"],
  "amplitudes": [
    {"real": 0.7071, "imag": 0.0},
    {"real": 0.0,    "imag": 0.0},
    {"real": 0.0,    "imag": 0.0},
    {"real": 0.7071, "imag": 0.0}
  ],
  "probabilities": [0.5, 0.0, 0.0, 0.5],
  "counts": { "00": 512, "11": 512 }
}
```

**Error envelope** (HTTP 400/422):
```json
{ "detail": { "error": "circuit_validation", "message": "Gate 'CNOT': qubit index 5 out of range for 2-qubit register." } }
```

Validation catches: unknown gate names, qubit-index out-of-range, duplicate/overlapping target & control sets, missing controls on a controlled gate, matrix-dimension mismatches during evolution.

---

## 4. Frontend (Quantum Dashboard)

Theme: **Deep Science Dark Mode** — pitch-black `#000000` canvas, neon cyan `#22e2ff` for single-qubit gates, amber `#ffb347` for controlled-gate decorations, mathematical gray for everything else.

### Interactive Circuit Builder

- Drag any gate tile from the **Gate Palette** onto the trailing `+` cell of any qubit wire.
- Or click a tile to **select** it, then click any cell to place it.
- For controlled gates (CNOT, CZ, CY, CH, Toffoli), click any free cell **in the same column** to toggle that qubit as a control. Vertical wires connect controls to the target.
- Click any placed gate box to remove the operation.

### State Visualization

Every run renders:
- **Probability histogram** over computational-basis states (Recharts bar chart).
- **State-vector table** of all non-zero amplitudes `α_i = a + b·i` with `|α_i|²`.

### Strict typing

- `tsconfig.json` enables `strict`, `noUncheckedIndexedAccess`, `noImplicitOverride`, `noFallthroughCasesInSwitch`.
- All API responses are typed via `lib/types.ts`.
- IDs are minted only inside event handlers, never at module load — guaranteeing **no hydration mismatches**.

---

## 5. Local Development

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

OpenAPI docs: <http://localhost:8000/docs>

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard: <http://localhost:3000>

---

## 6. Run with Docker Compose

Single command brings up both services on a private bridge network:

```bash
docker compose up --build
```

| URL | Purpose |
| --- | --- |
| <http://localhost:3000> | Quantum Dashboard |
| <http://localhost:8000/docs> | FastAPI Swagger UI |
| <http://localhost:8000/api/v1/health> | Engine health probe |

Tear down:

```bash
docker compose down
```

The frontend container reaches the backend over the docker network at `http://tensorq-engine:8000`. CORS is locked down to the dashboard origin via `TENSORQ_CORS_ORIGINS`.

---

## 7. Project Layout

```
Quantum-State-Simulator/
├── backend/
│   ├── app/
│   │   ├── main.py                  FastAPI app factory + CORS
│   │   ├── exceptions.py            CircuitValidationError, DimensionMismatchError
│   │   ├── core/
│   │   │   ├── gates.py             |0⟩, |1⟩, X, Y, Z, H, S, T (from scratch)
│   │   │   ├── tensor.py            Kronecker product, embed_single_qubit_gate,
│   │   │   │                        build_controlled_gate
│   │   │   ├── simulator.py         Circuit parsing + state evolution
│   │   │   └── measurement.py       Born rule + shot sampling
│   │   ├── api/routes.py            /health, /gates, /simulate
│   │   └── models/schemas.py        Pydantic v2 request/response models
│   ├── requirements.txt
│   └── Dockerfile                   python:3.12-slim, non-root, healthcheck
├── frontend/
│   ├── app/                         Next.js 14 App Router
│   │   ├── layout.tsx               Root layout (server component)
│   │   ├── page.tsx                 Dashboard shell (server component)
│   │   └── globals.css              Tailwind + quantum grid bg
│   ├── components/                  All "use client"
│   │   ├── CircuitBuilder.tsx       Top-level state machine
│   │   ├── CircuitGrid.tsx          Drag-drop grid, control-target wires
│   │   ├── GatePalette.tsx          Draggable gate tiles
│   │   ├── ProbabilityChart.tsx     Recharts histogram
│   │   ├── StatePanel.tsx           Amplitude table
│   │   └── Toolbar.tsx              Qubit count, shots, run/reset
│   ├── lib/
│   │   ├── api.ts                   fetch wrapper + TensorQApiError
│   │   ├── circuit.ts               op constructors, defaultControls,
│   │   │                            formatComplex
│   │   └── types.ts                 Shared types (mirror backend schemas)
│   ├── next.config.mjs              /api/tensorq/* rewrite to backend
│   ├── tailwind.config.ts           Deep-science dark palette
│   ├── tsconfig.json                strict + noUncheckedIndexedAccess
│   └── Dockerfile                   multi-stage standalone, non-root
├── docker-compose.yml               2 services, 1 bridge net, healthchecks
├── .gitignore
└── README.md
```

---

## 8. Limits and Roadmap

- **Hard cap: 12 qubits.** A 12-qubit register has a 4096-dim state vector and the lifted operators are 16M-entry dense matrices. Beyond that, the dense Kronecker representation becomes memory-bound — the natural next step is sparse operator construction or direct state-vector application that avoids materializing the full `2ⁿ × 2ⁿ` matrix. Tensor-network contraction (MPS / PEPS) for larger registers is a planned follow-up.
- Currently supports computational-basis measurement only. Mid-circuit measurement and partial trace are out of scope for this boilerplate.
- Verified mathematically: Bell state on `H + CNOT` yields `(|00⟩ + |11⟩)/√2`; Toffoli flips `|110⟩ → |111⟩` exactly.

---

## License

MIT — see [`LICENSE`](LICENSE).
