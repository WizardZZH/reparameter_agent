from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .matching import match_panels
from .schema import InputValidationError, normalize_panels


_ALLOWED_IMPORTS = {"math"}
_FORBIDDEN_NAMES = {
    "__import__", "breakpoint", "compile", "eval", "exec", "globals", "input",
    "locals", "open", "setattr", "delattr", "getattr", "vars",
}
_FORBIDDEN_ATTRIBUTES = {
    "system", "popen", "spawn", "fork", "forkpty", "execv", "execve", "remove",
    "unlink", "rmdir", "rename", "replace", "socket", "connect", "request", "urlopen",
}
_SAFE_BUILTIN_CALLS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int",
    "isinstance", "len", "list", "map", "max", "min", "pow", "range", "reversed",
    "print", "round", "set", "sorted", "str", "sum", "tuple", "zip",
}
_SAFE_METHOD_CALLS = {"append", "clear", "copy", "count", "extend", "index", "insert", "pop", "remove", "reverse", "sort"}


class CandidateValidationError(ValueError):
    """Raised when candidate source violates the PM contract or safety policy."""


def validate_candidate_source(source: str, declared_params: list[str]) -> dict[str, Any]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise CandidateValidationError(f"syntax error: {exc}") from exc

    generate_functions = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("generate_")
    ]
    if len(generate_functions) != 1:
        raise CandidateValidationError("candidate must define exactly one top-level generate_* function")
    generate_function = generate_functions[0]
    if isinstance(generate_function, ast.AsyncFunctionDef):
        raise CandidateValidationError("generate function must be synchronous")

    positional = [argument.arg for argument in generate_function.args.posonlyargs + generate_function.args.args]
    keyword_only = [argument.arg for argument in generate_function.args.kwonlyargs]
    accepted = set(positional + keyword_only)
    if generate_function.args.kwarg is None:
        missing = set(declared_params) - accepted
        if missing:
            raise CandidateValidationError(f"generate function does not accept declared parameters: {sorted(missing)}")

    defined_functions = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    math_module_names: set[str] = set()
    math_function_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "math":
                    math_module_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "math":
            math_function_names.update(alias.asname or alias.name for alias in node.names)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in _ALLOWED_IMPORTS:
                    raise CandidateValidationError(f"import of {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.module is None or node.module.split(".")[0] not in _ALLOWED_IMPORTS:
                raise CandidateValidationError(f"import from {node.module!r} is not allowed")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name in _FORBIDDEN_NAMES or name.startswith("__"):
                    raise CandidateValidationError(f"call to {name!r} is not allowed")
                if name not in _SAFE_BUILTIN_CALLS | defined_functions | math_function_names:
                    raise CandidateValidationError(f"call to unapproved function {name!r} is not allowed")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in _FORBIDDEN_ATTRIBUTES or node.func.attr.startswith("_"):
                    raise CandidateValidationError(f"call to attribute {node.func.attr!r} is not allowed")
                is_math_call = isinstance(node.func.value, ast.Name) and node.func.value.id in math_module_names
                if not is_math_call and node.func.attr not in _SAFE_METHOD_CALLS:
                    raise CandidateValidationError(f"call to unapproved method {node.func.attr!r} is not allowed")
            else:
                raise CandidateValidationError("indirect callable expressions are not allowed")
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            raise CandidateValidationError("dunder names are not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise CandidateValidationError("private or dunder attribute access is not allowed")

    return {
        "function_name": generate_function.name,
        "accepted_parameters": positional + keyword_only,
        "accepts_kwargs": generate_function.args.kwarg is not None,
    }


def _resource_limiter(cpu_seconds: int, memory_mb: int):
    def apply_limits() -> None:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
            memory_bytes = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        except (ImportError, OSError, ValueError):
            pass

    return apply_limits


def execute_candidate(
    candidate_path: Path,
    params: dict[str, Any],
    timeout_seconds: float = 5.0,
    memory_mb: int = 512,
) -> dict[str, Any]:
    worker_path = Path(__file__).with_name("worker.py")
    payload = json.dumps({"candidate_path": str(candidate_path.resolve()), "params": params})
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [sys.executable, "-I", str(worker_path)],
            input=payload,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
            env={"PATH": os.environ.get("PATH", "")},
            preexec_fn=_resource_limiter(max(1, int(timeout_seconds)), memory_mb) if os.name == "posix" else None,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "error": f"execution exceeded {timeout_seconds} seconds",
            "elapsed_seconds": time.monotonic() - started,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }

    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        return {
            "status": "runtime-error",
            "error": f"worker exited with code {completed.returncode}",
            "elapsed_seconds": elapsed,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "status": "runtime-error",
            "error": "worker returned invalid JSON",
            "elapsed_seconds": elapsed,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    result["elapsed_seconds"] = elapsed
    return result


def evaluate_candidate(
    candidate_path: Path,
    normalized_input: dict[str, Any],
    tolerance: float = 0.01,
    timeout_seconds: float = 5.0,
    memory_mb: int = 512,
) -> dict[str, Any]:
    source = candidate_path.read_text(encoding="utf-8")
    try:
        validation = validate_candidate_source(source, normalized_input["valid_params"])
    except CandidateValidationError as exc:
        return {
            "valid": False,
            "validation": {"status": "invalid-candidate", "error": str(exc)},
            "coverage": 0.0,
            "pass_ids": [],
            "fail_ids": [sample["sample_id"] for sample in normalized_input["samples"]],
            "samples": [],
        }

    sample_results: list[dict[str, Any]] = []
    pass_ids: list[str] = []
    for sample in normalized_input["samples"]:
        execution = execute_candidate(candidate_path, sample["params"], timeout_seconds, memory_mb)
        sample_result: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "params": sample["params"],
            "target_panels": sample["panels"],
            "execution": execution,
        }
        if execution.get("status") == "success":
            try:
                generated = normalize_panels(execution.get("panels"), f"generated[{sample['sample_id']}]")
            except InputValidationError as exc:
                sample_result["status"] = "invalid-output"
                sample_result["discrepancy"] = {"match": False, "failure_type": "invalid-output", "error": str(exc)}
            else:
                sample_result["generated_panels"] = generated
                discrepancy = match_panels(sample["panels"], generated, tolerance)
                sample_result["discrepancy"] = discrepancy
                sample_result["status"] = "pass" if discrepancy["match"] else discrepancy["failure_type"]
                if discrepancy["match"]:
                    pass_ids.append(sample["sample_id"])
        else:
            sample_result["status"] = execution.get("status", "runtime-error")
            sample_result["discrepancy"] = {
                "match": False,
                "failure_type": sample_result["status"],
                "error": execution.get("error"),
            }
        sample_results.append(sample_result)

    all_ids = [sample["sample_id"] for sample in normalized_input["samples"]]
    pass_set = set(pass_ids)
    return {
        "valid": True,
        "validation": {"status": "valid", **validation},
        "coverage": len(pass_ids) / len(all_ids),
        "pass_ids": pass_ids,
        "fail_ids": [sample_id for sample_id in all_ids if sample_id not in pass_set],
        "samples": sample_results,
    }
