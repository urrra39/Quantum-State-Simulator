"""Custom exception hierarchy for the quantum simulator."""

from __future__ import annotations


class TensorQError(Exception):
    """Base class for all simulator errors."""


class CircuitValidationError(TensorQError):
    """Raised when a circuit specification is malformed or references unknown gates."""


class DimensionMismatchError(TensorQError):
    """Raised when matrix/vector dimensions are incompatible during evolution."""
