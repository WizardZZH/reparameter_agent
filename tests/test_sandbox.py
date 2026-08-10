import tempfile
import unittest
from pathlib import Path

from pm_synthesis.sandbox import (
    CandidateValidationError,
    evaluate_candidate,
    validate_candidate_source,
)


class SandboxTests(unittest.TestCase):
    def test_validates_generate_signature(self):
        result = validate_candidate_source(
            "def generate_box(W, **kwargs):\n    return [[0, 0, 0, W, 1, 1]]\n",
            ["W", "N"],
        )
        self.assertEqual(result["function_name"], "generate_box")
        self.assertTrue(result["accepts_kwargs"])

    def test_rejects_file_access(self):
        source = "def generate_box(W):\n    open('/tmp/x', 'w')\n    return []\n"
        with self.assertRaisesRegex(CandidateValidationError, "open"):
            validate_candidate_source(source, ["W"])

    def test_rejects_non_allowlisted_import(self):
        source = "import os\ndef generate_box(W):\n    return [[0, 0, 0, W, 1, 1]]\n"
        with self.assertRaisesRegex(CandidateValidationError, "not allowed"):
            validate_candidate_source(source, ["W"])

    def test_rejects_dynamic_builtin_escape(self):
        source = "def generate_box(W):\n    return getattr(__builtins__, 'open')('/tmp/x', 'w')\n"
        with self.assertRaisesRegex(CandidateValidationError, "getattr|indirect"):
            validate_candidate_source(source, ["W"])

    def test_allows_normal_list_building(self):
        source = "def generate_box(W):\n    panels = []\n    panels.append([0, 0, 0, W, 1, 1])\n    return panels\n"
        validate_candidate_source(source, ["W"])

    def test_evaluates_candidate_in_worker(self):
        normalized = {
            "valid_params": ["W"],
            "samples": [
                {
                    "sample_id": "s1",
                    "params": {"W": 2.0},
                    "panels": [[0.0, 0.0, 0.0, 2.0, 1.0, 1.0]],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.py"
            path.write_text(
                "def generate_box(W):\n    print('captured')\n    return [[0, 0, 0, W, 1, 1]]\n",
                encoding="utf-8",
            )
            evaluation = evaluate_candidate(path, normalized)
        self.assertEqual(evaluation["coverage"], 1.0)
        self.assertEqual(evaluation["samples"][0]["execution"]["stdout"], "captured\n")

    def test_invalid_output_does_not_crash_evaluation(self):
        normalized = {
            "valid_params": ["W"],
            "samples": [
                {"sample_id": "s1", "params": {"W": 2}, "panels": [[0, 0, 0, 2, 1, 1]]}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.py"
            path.write_text("def generate_box(W):\n    return []\n", encoding="utf-8")
            evaluation = evaluate_candidate(path, normalized)
        self.assertEqual(evaluation["coverage"], 0.0)
        self.assertEqual(evaluation["samples"][0]["status"], "invalid-output")

    def test_timeout_is_isolated_per_sample(self):
        normalized = {
            "valid_params": ["W"],
            "samples": [
                {"sample_id": "s1", "params": {"W": 2}, "panels": [[0, 0, 0, 2, 1, 1]]}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.py"
            path.write_text(
                "def generate_box(W):\n    while True:\n        pass\n",
                encoding="utf-8",
            )
            evaluation = evaluate_candidate(path, normalized, timeout_seconds=0.1)
        self.assertEqual(evaluation["samples"][0]["status"], "timeout")


if __name__ == "__main__":
    unittest.main()
