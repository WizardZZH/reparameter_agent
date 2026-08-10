from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import json
import sys
import traceback
from pathlib import Path
from typing import Any


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    candidate_path = Path(payload["candidate_path"])
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            spec = importlib.util.spec_from_file_location("candidate_pm", candidate_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("unable to load candidate module")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            functions = [
                function
                for name, function in inspect.getmembers(module, inspect.isfunction)
                if name.startswith("generate_") and function.__module__ == module.__name__
            ]
            if len(functions) != 1:
                raise RuntimeError("candidate must expose exactly one generate_* function")
            panels = functions[0](**payload["params"])
        return {
            "status": "success",
            "panels": panels,
            "stdout": captured_stdout.getvalue(),
            "stderr": captured_stderr.getvalue(),
        }
    except BaseException as exc:
        return {
            "status": "runtime-error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "stdout": captured_stdout.getvalue(),
            "stderr": captured_stderr.getvalue(),
        }


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        result = _run(payload)
        json.dump(result, sys.stdout, ensure_ascii=False)
        return 0
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
