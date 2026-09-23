from __future__ import annotations

from .load import load_local_dataset

__all__ = [
    "prepare_ontology_data",
    "load_local_dataset",
]


def __getattr__(name: str):
    if name == "prepare_ontology_data":
        from .prepare import prepare_ontology_data
        return prepare_ontology_data
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
