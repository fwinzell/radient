"""Foundation model loaders used by :class:`RadiologyProcessor`."""

from .registry import load_foundation_model, available_models

__all__ = ["load_foundation_model", "available_models"]
