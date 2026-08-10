import tempfile
import unittest
from pathlib import Path

from pm_synthesis.agent_runtime import _load_config, extract_json_object, extract_python


class AgentRuntimeTests(unittest.TestCase):
    def test_extracts_last_python_block(self):
        text = "```python\ndef generate_old(): pass\n```\n```python\ndef generate_new(): pass\n```"
        self.assertIn("generate_new", extract_python(text))

    def test_extracts_nested_json_object(self):
        result = extract_json_object('```json\n{"a": {"b": 1}, "items": [1, 2]}\n```')
        self.assertEqual(result["a"]["b"], 1)

    def test_missing_block_is_error(self):
        with self.assertRaises(ValueError):
            extract_python("plain text")

    def test_flat_config_does_not_become_self_referential(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("agent_args:\n  agent_name: old\n  sys_prompt: old\nmodel_args: {}\n", encoding="utf-8")
            config = _load_config(path, "domain", "system")
        self.assertNotIn("agent_config", config)
        self.assertEqual(config["agent_args"]["agent_name"], "pm_domain_agent")


if __name__ == "__main__":
    unittest.main()
