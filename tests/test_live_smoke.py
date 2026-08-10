import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(
    os.environ.get("RUN_LIVE_PM_SYNTHESIS") == "1",
    "set RUN_LIVE_PM_SYNTHESIS=1 to spend model tokens on the live smoke test",
)
class LiveSynthesisSmokeTests(unittest.TestCase):
    def test_one_round_live_pipeline(self):
        dataset_dir = os.environ.get("PM_SYNTHESIS_LIVE_DATASET_DIR")
        if not dataset_dir:
            self.fail("PM_SYNTHESIS_LIVE_DATASET_DIR must point to a PM dataset directory")
        repo_root = Path(__file__).resolve().parent.parent
        config_path = os.environ.get(
            "PM_SYNTHESIS_LIVE_CONFIG", str(repo_root / "agent" / "config.yaml")
        )
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "output"
            completed = subprocess.run(
                [
                    "/opt/conda/envs/agent/bin/python",
                    str(repo_root / "synthesize_pm.py"),
                    "--dataset-dir",
                    dataset_dir,
                    "--output-dir",
                    str(output_root),
                    "--config",
                    config_path,
                    "--batch-size",
                    "1",
                    "--max-rounds",
                    "1",
                ],
                cwd=repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=1800,
                check=False,
            )
            self.assertIn(completed.returncode, {0, 2}, completed.stdout)
            status = json.loads(completed.stdout.strip().splitlines()[-1])
            workspace = Path(status["workspace"])
            self.assertTrue((workspace / "manifest.json").is_file())
            self.assertTrue((workspace / "rounds" / "round-001" / "agents" / "domain" / "conversation.json").is_file())
            self.assertTrue((workspace / "rounds" / "round-001" / "agents" / "coding" / "candidate.py").is_file())
            self.assertTrue((workspace / "rounds" / "round-001" / "agents" / "verifier" / "conversation.json").is_file())
            self.assertTrue((workspace / "final" / "model.py").is_file())


if __name__ == "__main__":
    unittest.main()
