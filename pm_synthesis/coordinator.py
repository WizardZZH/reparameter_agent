from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .io_utils import atomic_write_json, atomic_write_text, load_json, sha256_file
from .sandbox import evaluate_candidate


class AgentInvocationError(RuntimeError):
    """Raised when a sub-agent CLI fails."""


class AgentInvoker(Protocol):
    def run(self, role: str, workspace: Path, round_number: int) -> Path: ...


@dataclass
class SubprocessAgentInvoker:
    config_path: Path
    prompts_path: Path | None = None
    timeout_seconds: float = 1800.0

    def run(self, role: str, workspace: Path, round_number: int) -> Path:
        command = [
            sys.executable,
            "-m",
            "pm_synthesis.agent_cli",
            "--role",
            role,
            "--workspace",
            str(workspace),
            "--round",
            str(round_number),
            "--config",
            str(self.config_path),
        ]
        if self.prompts_path is not None:
            command.extend(["--prompts", str(self.prompts_path)])
        repo_root = Path(__file__).resolve().parent.parent
        environment = os.environ.copy()
        python_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(repo_root) + (os.pathsep + python_path if python_path else "")
        try:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            raise AgentInvocationError(f"{role} agent timed out after {self.timeout_seconds}s\n{output}") from exc

        role_dir = workspace / "rounds" / f"round-{round_number:03d}" / "agents" / role
        atomic_write_text(role_dir / "console.log", completed.stdout)
        if completed.returncode != 0:
            raise AgentInvocationError(f"{role} agent exited with code {completed.returncode}")
        result = role_dir / ("candidate.py" if role == "coding" else "result.json")
        if not result.is_file():
            raise AgentInvocationError(f"{role} agent did not produce {result.name}")
        return result


def decide_gate(
    candidate_evaluation: dict[str, Any],
    historical_pass_ids: set[str],
    baseline_pass_ids: set[str],
) -> dict[str, Any]:
    candidate_pass = set(candidate_evaluation.get("pass_ids", []))
    regressions = sorted(historical_pass_ids - candidate_pass)
    valid = bool(candidate_evaluation.get("valid"))
    improved = len(candidate_pass) > len(baseline_pass_ids)
    accepted = valid and not regressions and improved
    if not valid:
        reason = "invalid-candidate"
    elif regressions:
        reason = "historical-regression"
    elif not improved:
        reason = "no-strict-coverage-gain"
    else:
        reason = "strict-monotone-improvement"
    return {
        "accepted": accepted,
        "reason": reason,
        "candidate_pass_count": len(candidate_pass),
        "baseline_pass_count": len(baseline_pass_ids),
        "historical_pass_count": len(historical_pass_ids),
        "regression_ids": regressions,
        "new_pass_ids": sorted(candidate_pass - baseline_pass_ids),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


class FailureDrivenCoordinator:
    def __init__(
        self,
        workspace: Path,
        normalized_input: dict[str, Any],
        agent_invoker: AgentInvoker,
        batch_size: int = 5,
        max_rounds: int = 10,
        seed: int = 42,
        tolerance: float = 0.01,
        sample_timeout_seconds: float = 5.0,
        memory_mb: int = 512,
    ) -> None:
        if batch_size <= 0 or max_rounds <= 0:
            raise ValueError("batch_size and max_rounds must be positive")
        self.workspace = workspace
        self.input = normalized_input
        self.agent_invoker = agent_invoker
        self.batch_size = batch_size
        self.max_rounds = max_rounds
        self.seed = seed
        self.tolerance = tolerance
        self.sample_timeout_seconds = sample_timeout_seconds
        self.memory_mb = memory_mb
        self.sample_by_id = {sample["sample_id"]: sample for sample in normalized_input["samples"]}

    def _evaluate(self, candidate_path: Path) -> dict[str, Any]:
        return evaluate_candidate(
            candidate_path,
            self.input,
            tolerance=self.tolerance,
            timeout_seconds=self.sample_timeout_seconds,
            memory_mb=self.memory_mb,
        )

    def _base_request(
        self,
        round_number: int,
        sampled_ids: list[str],
        baseline_code: str | None,
        previous_verifier: dict[str, Any] | None,
        previous_regressions: list[str],
    ) -> dict[str, Any]:
        return {
            "round": round_number,
            "pm": {
                "pm_id": self.input["pm_id"],
                "product_category": self.input["category"],
                "param_model_id": self.input["param_model_id"],
                "valid_params": self.input["valid_params"],
                "valid_params_type": self.input["valid_params_type"],
            },
            "sampled_examples": [self.sample_by_id[sample_id] for sample_id in sampled_ids],
            "current_baseline_code": baseline_code,
            "previous_verifier_diagnosis": previous_verifier,
            "previous_regression_ids": previous_regressions,
        }

    def _verifier_request(
        self,
        sampled_ids: list[str],
        evaluation: dict[str, Any],
        gate: dict[str, Any],
    ) -> dict[str, Any]:
        results = {result["sample_id"]: result for result in evaluation.get("samples", [])}
        batch_details = []
        for sample_id in sampled_ids:
            result = results.get(sample_id)
            if result is None:
                sample = self.sample_by_id[sample_id]
                batch_details.append(
                    {
                        "sample_id": sample_id,
                        "params": sample["params"],
                        "target_panels": sample["panels"],
                        "generated_panels": None,
                        "status": evaluation.get("validation", {}).get("status", "not-executed"),
                        "discrepancy": evaluation.get("validation", {}),
                    }
                )
                continue
            batch_details.append(
                {
                    "sample_id": sample_id,
                    "params": result["params"],
                    "target_panels": result["target_panels"],
                    "generated_panels": result.get("generated_panels"),
                    "status": result["status"],
                    "discrepancy": result["discrepancy"],
                }
            )
        return {
            "historical_coverage_preserved": not gate["regression_ids"],
            "regression_ids": gate["regression_ids"],
            "gate_decision": gate,
            "current_batch": batch_details,
        }

    def run(self) -> dict[str, Any]:
        rng = random.Random(self.seed)
        all_ids = [sample["sample_id"] for sample in self.input["samples"]]
        failure_ids = list(all_ids)
        baseline_pass_ids: set[str] = set()
        historical_pass_ids: set[str] = set()
        baseline_code: str | None = None
        baseline_evaluation: dict[str, Any] | None = None
        previous_verifier: dict[str, Any] | None = None
        previous_regressions: list[str] = []
        trajectory: list[dict[str, Any]] = []
        accepted_round: int | None = None
        termination_reason = "maximum-rounds-reached"

        state_dir = self.workspace / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        for round_number in range(1, self.max_rounds + 1):
            round_dir = self.workspace / "rounds" / f"round-{round_number:03d}"
            round_dir.mkdir(parents=True, exist_ok=False)
            sampled_ids = rng.sample(failure_ids, min(self.batch_size, len(failure_ids)))
            atomic_write_json(
                round_dir / "sampled_input.json",
                {
                    "sample_ids": sampled_ids,
                    "samples": [self.sample_by_id[sample_id] for sample_id in sampled_ids],
                },
            )
            atomic_write_text(round_dir / "baseline-before.py", baseline_code or "")
            base_request = self._base_request(
                round_number,
                sampled_ids,
                baseline_code,
                previous_verifier,
                previous_regressions,
            )
            domain_dir = round_dir / "agents" / "domain"
            atomic_write_json(domain_dir / "request.json", base_request)

            round_record: dict[str, Any] = {
                "round": round_number,
                "sample_ids": sampled_ids,
                "baseline_pass_count_before": len(baseline_pass_ids),
            }
            try:
                domain_result_path = self.agent_invoker.run("domain", self.workspace, round_number)
                domain_result = load_json(domain_result_path)
                coding_request = {**base_request, "domain_reasoning": domain_result}
                coding_dir = round_dir / "agents" / "coding"
                atomic_write_json(coding_dir / "request.json", coding_request)
                candidate_path = self.agent_invoker.run("coding", self.workspace, round_number)
            except BaseException as exc:
                round_record.update(
                    {
                        "proposal_status": "agent-error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "accepted": False,
                        "committed_train_coverage": len(baseline_pass_ids) / len(all_ids),
                    }
                )
                atomic_write_json(round_dir / "round-summary.json", round_record)
                trajectory.append(round_record)
                atomic_write_json(state_dir / "trajectory.json", trajectory)
                continue

            candidate_copy = round_dir / "candidate.py"
            atomic_write_text(candidate_copy, candidate_path.read_text(encoding="utf-8"))
            evaluation = self._evaluate(candidate_copy)
            atomic_write_json(round_dir / "evaluation.json", evaluation)
            _write_jsonl(round_dir / "sample-traces.jsonl", evaluation.get("samples", []))
            gate = decide_gate(evaluation, historical_pass_ids, baseline_pass_ids)
            atomic_write_json(round_dir / "gate.json", gate)

            if gate["accepted"]:
                baseline_code = candidate_copy.read_text(encoding="utf-8")
                baseline_pass_ids = set(evaluation["pass_ids"])
                historical_pass_ids.update(baseline_pass_ids)
                baseline_evaluation = evaluation
                accepted_round = round_number
                failure_ids = list(evaluation["fail_ids"])
                atomic_write_text(state_dir / "baseline.py", baseline_code)
                atomic_write_json(
                    state_dir / "evidence.json",
                    {
                        "historical_pass_ids": sorted(historical_pass_ids),
                        "failure_ids": failure_ids,
                        "accepted_round": accepted_round,
                    },
                )
            previous_regressions = list(gate["regression_ids"])
            atomic_write_text(round_dir / "baseline-after.py", baseline_code or "")

            verifier_request = self._verifier_request(sampled_ids, evaluation, gate)
            verifier_dir = round_dir / "agents" / "verifier"
            atomic_write_json(verifier_dir / "request.json", verifier_request)
            verifier_error = None
            try:
                verifier_result_path = self.agent_invoker.run("verifier", self.workspace, round_number)
                previous_verifier = load_json(verifier_result_path)
            except BaseException as exc:
                verifier_error = f"{type(exc).__name__}: {exc}"
                previous_verifier = {"error": verifier_error}

            committed_train_coverage = len(baseline_pass_ids) / len(all_ids)
            round_record.update(
                {
                    "proposal_status": "evaluated",
                    "candidate_valid": evaluation["valid"],
                    "candidate_train_coverage": evaluation["coverage"],
                    "accepted": gate["accepted"],
                    "gate_reason": gate["reason"],
                    "regression_ids": gate["regression_ids"],
                    "new_pass_ids": gate["new_pass_ids"],
                    "committed_train_coverage": committed_train_coverage,
                    "verifier_error": verifier_error,
                }
            )
            atomic_write_json(round_dir / "round-summary.json", round_record)
            trajectory.append(round_record)
            atomic_write_json(state_dir / "trajectory.json", trajectory)

            if baseline_pass_ids == set(all_ids):
                termination_reason = "full-coverage"
                break

        final_dir = self.workspace / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        if baseline_code is not None and baseline_evaluation is not None:
            atomic_write_text(final_dir / "model.py", baseline_code)
            atomic_write_json(final_dir / "evaluation.json", baseline_evaluation)
        final_coverage = len(baseline_pass_ids) / len(all_ids)
        if final_coverage == 1.0:
            exit_code = 0
            status = "success"
        elif baseline_code is not None:
            exit_code = 2
            status = "partial"
        else:
            exit_code = 1
            status = "failed"
            termination_reason = "no-valid-improving-candidate"
        summary = {
            "status": status,
            "exit_code": exit_code,
            "termination_reason": termination_reason,
            "pm_id": self.input["pm_id"],
            "train_sample_count": len(all_ids),
            "train_coverage": final_coverage,
            "train_pass_ids": sorted(baseline_pass_ids),
            "train_fail_ids": [
                sample_id for sample_id in all_ids if sample_id not in baseline_pass_ids
            ],
            "accepted_round": accepted_round,
            "rounds_executed": len(trajectory),
            "trajectory": trajectory,
            "final_model": "final/model.py" if baseline_code is not None else None,
            "final_model_sha256": sha256_file(final_dir / "model.py") if baseline_code is not None else None,
        }
        atomic_write_json(final_dir / "summary.json", summary)
        return summary
