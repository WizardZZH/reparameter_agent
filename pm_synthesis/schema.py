from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any


TRAIN_FILENAME = "input_200.json"
TEST_FILENAME = "test_50.json"


class InputValidationError(ValueError):
    """Raised when a PM dataset directory is invalid."""


def _finite_number(value: Any, location: str) -> float:
    if isinstance(value, bool):
        raise InputValidationError(f"{location} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{location} must be numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise InputValidationError(f"{location} must be finite")
    return number


def _convert_parameter(value: Any, declared_type: str, location: str) -> int | float:
    if declared_type == "int":
        number = _finite_number(value, location)
        if not number.is_integer():
            raise InputValidationError(f"{location} must be an integer")
        return int(number)
    if declared_type == "float":
        return _finite_number(value, location)
    raise InputValidationError(f"Unsupported declared parameter type {declared_type!r} at {location}")


def _normalize_params(
    raw: Any,
    expected_names: list[str],
    expected_types: dict[str, str],
    location: str,
) -> dict[str, int | float]:
    if not isinstance(raw, list) or not raw:
        raise InputValidationError(f"{location} must be a non-empty name/value/type list")
    actual_names: list[str] = []
    params: dict[str, Any] = {}
    for index, item in enumerate(raw):
        item_location = f"{location}[{index}]"
        if not isinstance(item, dict) or not {"name", "value", "type"} <= item.keys():
            raise InputValidationError(f"{item_location} must contain name, value, and type")
        name = str(item["name"])
        if name in params:
            raise InputValidationError(f"{location} contains duplicate parameter {name!r}")
        actual_names.append(name)
        actual_type = str(item["type"])
        expected_type = expected_types.get(name)
        if expected_type != actual_type:
            raise InputValidationError(
                f"{item_location}.type is {actual_type!r}, expected {expected_type!r}"
            )
        params[name] = _convert_parameter(item["value"], actual_type, f"{item_location}.value")
    if actual_names != expected_names:
        raise InputValidationError(
            f"{location} parameter order/names {actual_names!r} do not match {expected_names!r}"
        )
    return params


def normalize_panels(raw: Any, location: str) -> list[list[float]]:
    if not isinstance(raw, list) or not raw:
        raise InputValidationError(f"{location} must be a non-empty panel list")
    panels: list[list[float]] = []
    for panel_index, raw_panel in enumerate(raw):
        panel_location = f"{location}[{panel_index}]"
        if not isinstance(raw_panel, (list, tuple)) or len(raw_panel) != 6:
            raise InputValidationError(f"{panel_location} must contain exactly six coordinates")
        panel = [_finite_number(value, f"{panel_location}[{i}]") for i, value in enumerate(raw_panel)]
        if not (panel[0] < panel[3] and panel[1] < panel[4] and panel[2] < panel[5]):
            raise InputValidationError(f"{panel_location} must have strictly positive extent on every axis")
        panels.append(panel)
    return panels


def _load_json_array(path: Path, split_name: str) -> list[Any]:
    if not path.is_file():
        raise InputValidationError(f"Missing required {split_name} file: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except json.JSONDecodeError as exc:
        raise InputValidationError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(value, list) or not value:
        raise InputValidationError(f"{path} must contain a non-empty JSON array")
    return value


def _derive_schema(first_sample: Any, pm_id: str) -> tuple[str, str, list[str], dict[str, str]]:
    if not isinstance(first_sample, dict):
        raise InputValidationError("train[0] must be an object")
    if str(first_sample.get("pm_id", "")) != pm_id:
        raise InputValidationError(f"train[0].pm_id must match directory name {pm_id!r}")
    category = str(first_sample.get("category", "")).strip()
    param_model_id = str(first_sample.get("paramModelId", "")).strip()
    if not category or not param_model_id:
        raise InputValidationError("train[0] must contain category and paramModelId")
    raw_params = first_sample.get("input_pm")
    if not isinstance(raw_params, list) or not raw_params:
        raise InputValidationError("train[0].input_pm must be a non-empty list")
    names: list[str] = []
    types: dict[str, str] = {}
    for index, item in enumerate(raw_params):
        if not isinstance(item, dict) or not {"name", "type"} <= item.keys():
            raise InputValidationError(f"train[0].input_pm[{index}] must contain name and type")
        name = str(item["name"])
        declared_type = str(item["type"])
        if not name or name in types:
            raise InputValidationError(f"train[0].input_pm has invalid/duplicate name {name!r}")
        if declared_type not in {"int", "float"}:
            raise InputValidationError(f"Unsupported parameter type {declared_type!r} for {name!r}")
        names.append(name)
        types[name] = declared_type
    return category, param_model_id, names, types


def _normalize_split(
    raw_samples: list[Any],
    split_name: str,
    pm_id: str,
    category: str,
    param_model_id: str,
    valid_params: list[str],
    param_types: dict[str, str],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, sample in enumerate(raw_samples):
        location = f"{split_name}[{index}]"
        if not isinstance(sample, dict):
            raise InputValidationError(f"{location} must be an object")
        sample_id = str(sample.get("sample_id", "")).strip()
        if not sample_id:
            raise InputValidationError(f"{location}.sample_id is required")
        if sample_id in seen_ids:
            raise InputValidationError(f"Duplicate {split_name} sample_id {sample_id!r}")
        seen_ids.add(sample_id)
        if str(sample.get("pm_id", "")) != pm_id:
            raise InputValidationError(f"{location}.pm_id does not match {pm_id!r}")
        if str(sample.get("category", "")) != category:
            raise InputValidationError(f"{location}.category does not match {category!r}")
        if str(sample.get("paramModelId", "")) != param_model_id:
            raise InputValidationError(f"{location}.paramModelId does not match {param_model_id!r}")
        params = _normalize_params(
            sample.get("input_pm"), valid_params, param_types, f"{location}.input_pm"
        )
        panels = normalize_panels(sample.get("panels"), f"{location}.panels")
        samples.append({"sample_id": sample_id, "params": params, "panels": panels})
    return samples


def load_pm_dataset(dataset_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    pm_id = dataset_dir.resolve().name
    if not dataset_dir.is_dir():
        raise InputValidationError(f"PM dataset directory not found: {dataset_dir}")
    raw_train = _load_json_array(dataset_dir / TRAIN_FILENAME, "training")
    raw_test = _load_json_array(dataset_dir / TEST_FILENAME, "test")
    category, param_model_id, valid_params, param_types = _derive_schema(raw_train[0], pm_id)
    train_samples = _normalize_split(
        raw_train, "train", pm_id, category, param_model_id, valid_params, param_types
    )
    test_samples = _normalize_split(
        raw_test, "test", pm_id, category, param_model_id, valid_params, param_types
    )
    overlap = {sample["sample_id"] for sample in train_samples} & {
        sample["sample_id"] for sample in test_samples
    }
    if overlap:
        raise InputValidationError(f"Train/test sample IDs overlap: {sorted(overlap)}")
    metadata = {
        "pm_id": pm_id,
        "category": category,
        "param_model_id": param_model_id,
        "valid_params": valid_params,
        "valid_params_type": param_types,
    }
    return ({**metadata, "samples": train_samples}, {**metadata, "samples": test_samples})
