"""Uncertainty metrics, including question- and response-level helpers."""

import importlib.util
from pathlib import Path

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "uncertainty.py"
_LEGACY_SPEC = importlib.util.spec_from_file_location(
    "src._question_level_uncertainty", _LEGACY_PATH
)
if _LEGACY_SPEC is None or _LEGACY_SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"could not load question-level uncertainty module: {_LEGACY_PATH}")
_LEGACY_MODULE = importlib.util.module_from_spec(_LEGACY_SPEC)
_LEGACY_SPEC.loader.exec_module(_LEGACY_MODULE)

calculate_uncertainty = _LEGACY_MODULE.calculate_uncertainty
mean_consistency = _LEGACY_MODULE.mean_consistency
standard_deviation = _LEGACY_MODULE.standard_deviation
uncertainty = _LEGACY_MODULE.uncertainty

__all__ = [
    "calculate_uncertainty",
    "mean_consistency",
    "standard_deviation",
    "uncertainty",
]
