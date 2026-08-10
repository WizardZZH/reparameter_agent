from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .coordinator import AgentInvoker, FailureDrivenCoordinator, SubprocessAgentInvoker
from .io_utils import (
    atomic_write_json,
    atomic_write_text,
    create_task_workspace,
    redact_secrets,
    sha256_file,
)
from .sandbox import evaluate_candidate
from .schema import TEST_FILENAME, TRAIN_FILENAME, load_pm_dataset


REPO_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Failure-driven parametric-model synthesis")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="PM directory containing input_200.json and test_50.json",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "agent" / "config.yaml")
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--sample-timeout", type=float, default=5.0)
    parser.add_argument("--agent-timeout", type=float, default=1800.0)
    parser.add_argument("--memory-mb", type=int, default=512)
    return parser


def _load_and_redact_config(config_path: Path) -> Any:
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        return redact_secrets(json.loads(text))
    return redact_secrets(yaml.safe_load(text) or {})


def run(args: argparse.Namespace, agent_invoker: AgentInvoker | None = None) -> tuple[int, Path]:
    dataset_dir = args.dataset_dir.resolve()
    workspace = create_task_workspace(args.output_dir, dataset_dir.name or "invalid-pm")
    input_dir = workspace / "input"
    train_source = dataset_dir / TRAIN_FILENAME
    test_source = dataset_dir / TEST_FILENAME
    for source in (train_source, test_source):
        if source.is_file():
            atomic_write_text(input_dir / source.name, source.read_text(encoding="utf-8"))
    started_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "status": "initializing",
        "started_at": started_at,
        "dataset_dir": str(dataset_dir),
        "train_source": str(train_source),
        "test_source": str(test_source),
        "config_source": str(args.config.resolve()),
        "arguments": {
            "batch_size": args.batch_size,
            "max_rounds": args.max_rounds,
            "seed": args.seed,
            "tolerance": args.tolerance,
            "sample_timeout": args.sample_timeout,
            "agent_timeout": args.agent_timeout,
            "memory_mb": args.memory_mb,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    atomic_write_json(workspace / "manifest.json", manifest)

    try:
        train_input, test_input = load_pm_dataset(dataset_dir)
        atomic_write_json(input_dir / "normalized-train.json", train_input)
        atomic_write_json(input_dir / "normalized-test.json", test_input)
        if not args.config.is_file():
            raise FileNotFoundError(f"agent config not found: {args.config}")
        atomic_write_json(workspace / "config.redacted.json", _load_and_redact_config(args.config))
        manifest["config_sha256"] = sha256_file(args.config)
        manifest["train_sha256"] = sha256_file(train_source)
        manifest["test_sha256"] = sha256_file(test_source)
        manifest["pm_id"] = train_input["pm_id"]
        manifest["category"] = train_input["category"]
        manifest["train_sample_count"] = len(train_input["samples"])
        manifest["test_sample_count"] = len(test_input["samples"])
        manifest["status"] = "running"
        atomic_write_json(workspace / "manifest.json", manifest)

        invoker = agent_invoker or SubprocessAgentInvoker(
            config_path=args.config.resolve(),
            prompts_path=args.prompts.resolve() if args.prompts else None,
            timeout_seconds=args.agent_timeout,
        )
        coordinator = FailureDrivenCoordinator(
            workspace=workspace,
            normalized_input=train_input,
            agent_invoker=invoker,
            batch_size=args.batch_size,
            max_rounds=args.max_rounds,
            seed=args.seed,
            tolerance=args.tolerance,
            sample_timeout_seconds=args.sample_timeout,
            memory_mb=args.memory_mb,
        )
        summary = coordinator.run()

        final_dir = workspace / "final"
        final_model = final_dir / "model.py"
        if final_model.is_file():
            test_evaluation = evaluate_candidate(
                final_model,
                test_input,
                tolerance=args.tolerance,
                timeout_seconds=args.sample_timeout,
                memory_mb=args.memory_mb,
            )
            atomic_write_json(final_dir / "test-evaluation.json", test_evaluation)
            summary.update(
                {
                    "test_sample_count": len(test_input["samples"]),
                    "test_coverage": test_evaluation["coverage"],
                    "test_pass_ids": test_evaluation["pass_ids"],
                    "test_fail_ids": test_evaluation["fail_ids"],
                    "test_evaluation": "final/test-evaluation.json",
                }
            )
        else:
            summary.update(
                {
                    "test_sample_count": len(test_input["samples"]),
                    "test_coverage": None,
                    "test_pass_ids": [],
                    "test_fail_ids": [],
                    "test_evaluation": None,
                    "test_evaluation_skipped": "no final model was committed",
                }
            )
        atomic_write_json(final_dir / "summary.json", summary)
        manifest["status"] = summary["status"]
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["summary"] = "final/summary.json"
        manifest["train_coverage"] = summary["train_coverage"]
        manifest["test_coverage"] = summary["test_coverage"]
        atomic_write_json(workspace / "manifest.json", manifest)
        return int(summary["exit_code"]), workspace
    except BaseException as exc:
        failure = {
            "status": "fatal-error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(workspace / "failure.json", failure)
        manifest["status"] = "fatal-error"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(workspace / "manifest.json", manifest)
        return 1, workspace


def main() -> int:
    code, workspace = run(build_parser().parse_args())
    print(json.dumps({"exit_code": code, "workspace": str(workspace)}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
