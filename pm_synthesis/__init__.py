"""Failure-driven parametric-model synthesis pipeline."""

from .matching import match_panels
from .schema import InputValidationError, load_pm_dataset

__all__ = ["InputValidationError", "load_pm_dataset", "match_panels"]
