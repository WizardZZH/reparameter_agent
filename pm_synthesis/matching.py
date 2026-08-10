from __future__ import annotations

from typing import Any


def _compatible(target: list[float], generated: list[float], tolerance: float) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(target, generated))


def _maximum_matching(adjacency: list[list[int]], generated_count: int) -> dict[int, int]:
    generated_to_target: list[int | None] = [None] * generated_count

    def augment(target_index: int, seen: set[int]) -> bool:
        for generated_index in adjacency[target_index]:
            if generated_index in seen:
                continue
            seen.add(generated_index)
            previous = generated_to_target[generated_index]
            if previous is None or augment(previous, seen):
                generated_to_target[generated_index] = target_index
                return True
        return False

    for target_index in range(len(adjacency)):
        augment(target_index, set())
    return {
        target_index: generated_index
        for generated_index, target_index in enumerate(generated_to_target)
        if target_index is not None
    }


def _nearest_pairs(
    target: list[list[float]],
    generated: list[list[float]],
    unmatched_target: list[int],
    unmatched_generated: list[int],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    available = set(unmatched_generated)
    for target_index in unmatched_target:
        if not available:
            break
        generated_index = min(
            available,
            key=lambda index: max(
                abs(a - b) for a, b in zip(target[target_index], generated[index])
            ),
        )
        available.remove(generated_index)
        delta = [generated_value - target_value for target_value, generated_value in zip(target[target_index], generated[generated_index])]
        pairs.append(
            {
                "target_index": target_index,
                "generated_index": generated_index,
                "delta": delta,
                "max_abs_delta": max(abs(value) for value in delta),
            }
        )
    return pairs


def match_panels(
    target: list[list[float]], generated: list[list[float]], tolerance: float = 0.01
) -> dict[str, Any]:
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    adjacency = [
        [index for index, panel in enumerate(generated) if _compatible(target_panel, panel, tolerance)]
        for target_panel in target
    ]
    matching = _maximum_matching(adjacency, len(generated))
    matched_target = set(matching)
    matched_generated = set(matching.values())
    unmatched_target = [index for index in range(len(target)) if index not in matched_target]
    unmatched_generated = [index for index in range(len(generated)) if index not in matched_generated]
    matched_pairs = [
        {
            "target_index": target_index,
            "generated_index": generated_index,
            "delta": [
                generated_value - target_value
                for target_value, generated_value in zip(target[target_index], generated[generated_index])
            ],
        }
        for target_index, generated_index in sorted(matching.items())
    ]
    success = len(target) == len(generated) and not unmatched_target and not unmatched_generated
    if success:
        failure_type = None
    elif len(target) != len(generated):
        failure_type = "topology-count"
    else:
        failure_type = "geometry-coordinate"
    return {
        "match": success,
        "failure_type": failure_type,
        "tolerance": tolerance,
        "target_count": len(target),
        "generated_count": len(generated),
        "missing_count": len(unmatched_target),
        "extra_count": len(unmatched_generated),
        "matched_pairs": matched_pairs,
        "unmatched_target": [
            {"index": index, "panel": target[index]} for index in unmatched_target
        ],
        "unmatched_generated": [
            {"index": index, "panel": generated[index]} for index in unmatched_generated
        ],
        "nearest_pairs": _nearest_pairs(
            target, generated, unmatched_target, unmatched_generated
        ),
    }
