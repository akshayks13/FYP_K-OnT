"""Ontology-Transformer: End-to-end ontology embedding with hyperbolic geometry and role-based rotation."""
from __future__ import annotations

from ont.model import OntologyTransformer

__version__ = "0.1.7"

__all__ = [
    "OntologyTransformer",
    "fit",
]


def __getattr__(name: str):
    if name == "fit":
        from ont.pipeline import fit
        return fit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
