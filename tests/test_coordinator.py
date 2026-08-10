import argparse
import json
import tempfile
import unittest
from pathlib import Path

from pm_synthesis.coordinator import FailureDrivenCoordinator, decide_gate
from pm_synthesis.io_utils import atomic_write_json, atomic_write_text
from pm_synthesis.main import run as run_main


ROUND_CODES = {
    1: """def generate_box(W):
    if W == 1:
        return [[0, 0, 0, 1, 1, 1]]
    return [[0, 0, 0, 1, 1, 1]]
""",
    2: """def generate_box(W):
    if W == 2:
        return [[0, 0, 0, 2, 1, 1]]
    return [[0, 0, 0, 2, 1, 1]]
""",
    3: """def generate_box(W):
    return [[0, 0, 0, W, 1, 1]]
""",
}


class FakeAgentInvoker:
    def run(self, role: str, workspace: Path, round_number: int) -> Path:
        role_dir = workspace / "rounds" / f"round-{round_number:03d}" / "agents" / role
        if role == "coding":
            path = role_dir / "candidate.py"
            atomic_write_text(path, ROUND_CODES[round_number])
            return path
        path = role_dir / "result.json"
        atomic_write_json(path, {"role": role, "round": round_number})
        return path


class GateTests(unittest.TestCase):
    def test_rejects_equal_coverage(self):
        result = decide_gate({"valid": True, "pass_ids": ["a"]}, {"a"}, {"a"})
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "no-strict-coverage-gain")

    def test_rejects_regression_even_with_different_pass(self):
        result = decide_gate({"valid": True, "pass_ids": ["b", "c"]}, {"a"}, {"a"})
        self.assertFalse(result["accepted"])
        self.assertEqual(result["regression_ids"], ["a"])


class CoordinatorIntegrationTests(unittest.TestCase):
    def test_rejects_regression_then_commits_full_coverage(self):
        normalized = {
            "pm_id": "demo",
            "category": "box",
            "param_model_id": "model-demo",
            "valid_params": ["W"],
            "valid_params_type": {"W": "int"},
            "samples": [
                {"sample_id": "s1", "params": {"W": 1}, "panels": [[0, 0, 0, 1, 1, 1]]},
                {"sample_id": "s2", "params": {"W": 2}, "panels": [[0, 0, 0, 2, 1, 1]]},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            summary = FailureDrivenCoordinator(
                workspace=workspace,
                normalized_input=normalized,
                agent_invoker=FakeAgentInvoker(),
                batch_size=1,
                max_rounds=3,
                seed=42,
            ).run()
            round_two = (workspace / "rounds" / "round-002" / "gate.json").read_text()
            final_code = (workspace / "final" / "model.py").read_text()
        self.assertEqual(summary["exit_code"], 0)
        self.assertEqual(summary["train_coverage"], 1.0)
        self.assertIn("historical-regression", round_two)
        self.assertIn("return [[0, 0, 0, W, 1, 1]]", final_code)

    def test_main_creates_unique_archived_workspace(self):
        def raw_sample(sample_id, width, target_width):
            return {
                "sample_id": sample_id,
                "pm_id": "demo-main",
                "category": "box",
                "paramModelId": "model-demo-main",
                "input_pm": [{"name": "W", "value": width, "type": "int"}],
                "panels": [[0, 0, 0, target_width, 1, 1]],
            }

        raw_train = [raw_sample("train-1", 1, 1)]
        # The first-round model always returns width 1 for W != 1, so held-out
        # coverage is zero. It must not affect train-based selection/exit status.
        raw_test = [raw_sample("test-1", 2, 2)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "demo-main"
            dataset_dir.mkdir()
            config_path = root / "config.json"
            (dataset_dir / "input_200.json").write_text(json.dumps(raw_train), encoding="utf-8")
            (dataset_dir / "test_50.json").write_text(json.dumps(raw_test), encoding="utf-8")
            config_path.write_text(json.dumps({"api_key": "secret"}), encoding="utf-8")
            args = argparse.Namespace(
                dataset_dir=dataset_dir,
                output_dir=root / "output",
                config=config_path,
                prompts=None,
                batch_size=1,
                max_rounds=3,
                seed=42,
                tolerance=0.01,
                sample_timeout=1.0,
                agent_timeout=1.0,
                memory_mb=512,
            )
            code, workspace = run_main(args, agent_invoker=FakeAgentInvoker())
            manifest = json.loads((workspace / "manifest.json").read_text())
            redacted = json.loads((workspace / "config.redacted.json").read_text())
            summary = json.loads((workspace / "final" / "summary.json").read_text())
            test_evaluation = json.loads(
                (workspace / "final" / "test-evaluation.json").read_text()
            )
            train_archived = (workspace / "input" / "input_200.json").is_file()
            test_archived = (workspace / "input" / "test_50.json").is_file()
            workspace_name = workspace.name
        self.assertEqual(code, 0)
        self.assertEqual(manifest["status"], "success")
        self.assertEqual(redacted["api_key"], "<redacted>")
        self.assertEqual(summary["train_coverage"], 1.0)
        self.assertEqual(summary["test_coverage"], 0.0)
        self.assertEqual(test_evaluation["fail_ids"], ["test-1"])
        self.assertTrue(train_archived)
        self.assertTrue(test_archived)
        self.assertTrue(workspace_name.startswith("demo-main_"))


if __name__ == "__main__":
    unittest.main()
