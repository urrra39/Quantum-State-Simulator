"""Shared pytest fixtures for the TensorQ-Engine test suite.

Provides a single FastAPI ``TestClient`` bound to the production ``app``
object so the API tests exercise the real router, schemas, and exception
handlers rather than a bespoke test app.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tensorq.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """A reusable TestClient for the real TensorQ-Engine FastAPI app."""
    return TestClient(app)


@pytest.fixture()
def rng() -> np.random.Generator:
    """A freshly seeded NumPy Generator for deterministic sampling tests."""
    return np.random.default_rng(20240617)
