import json
import tempfile
import unittest
from pathlib import Path

from pm_synthesis.schema import InputValidationError, load_pm_dataset


def sample(sample_id: str, pm_id: str = "123", width: float = 2.0) -> dict:
    return {
        "sample_id": sample_id,
        "pm_id": pm_id,
        "category": "box",
        "paramModelId": "model-123",
        "source_orderDesignId": 1,
        "input_pm": [
            {"name": "W", "value": width, "type": "float"},
            {"name": "N", "value": 1, "type": "int"},
        ],
        "panels": [[0, 0, 0, width, 1, 1]],
    }


class DatasetDirectory:
    def __init__(self, root: Path) -> None:
        self.path = root / "123"
        self.path.mkdir()
        self.train = [sample("train-1")]
        self.test = [sample("test-1", width=3.0)]

    def write(self) -> None:
        (self.path / "input_200.json").write_text(json.dumps(self.train), encoding="utf-8")
        (self.path / "test_50.json").write_text(json.dumps(self.test), encoding="utf-8")


class SchemaTests(unittest.TestCase):
    def test_loads_train_and_test_with_derived_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetDirectory(Path(directory))
            fixture.write()
            train, test = load_pm_dataset(fixture.path)
        self.assertEqual(train["pm_id"], "123")
        self.assertEqual(train["category"], "box")
        self.assertEqual(train["valid_params"], ["W", "N"])
        self.assertEqual(train["valid_params_type"], {"W": "float", "N": "int"})
        self.assertEqual(train["samples"][0]["params"], {"W": 2.0, "N": 1})
        self.assertEqual(test["samples"][0]["sample_id"], "test-1")

    def test_rejects_pm_id_not_matching_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetDirectory(Path(directory))
            fixture.train[0]["pm_id"] = "wrong"
            fixture.write()
            with self.assertRaisesRegex(InputValidationError, "directory"):
                load_pm_dataset(fixture.path)

    def test_rejects_cross_split_parameter_type_change(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetDirectory(Path(directory))
            fixture.test[0]["input_pm"][0]["type"] = "int"
            fixture.write()
            with self.assertRaisesRegex(InputValidationError, "type"):
                load_pm_dataset(fixture.path)

    def test_rejects_cross_split_parameter_order_change(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetDirectory(Path(directory))
            fixture.test[0]["input_pm"].reverse()
            fixture.write()
            with self.assertRaisesRegex(InputValidationError, "order/names"):
                load_pm_dataset(fixture.path)

    def test_rejects_duplicate_id_across_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetDirectory(Path(directory))
            fixture.test[0]["sample_id"] = "train-1"
            fixture.write()
            with self.assertRaisesRegex(InputValidationError, "overlap"):
                load_pm_dataset(fixture.path)

    def test_rejects_zero_volume_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetDirectory(Path(directory))
            fixture.train[0]["panels"] = [[0, 0, 0, 0, 1, 1]]
            fixture.write()
            with self.assertRaisesRegex(InputValidationError, "positive extent"):
                load_pm_dataset(fixture.path)

    def test_rejects_non_integral_int(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = DatasetDirectory(Path(directory))
            fixture.train[0]["input_pm"][1]["value"] = 1.5
            fixture.write()
            with self.assertRaisesRegex(InputValidationError, "integer"):
                load_pm_dataset(fixture.path)


if __name__ == "__main__":
    unittest.main()
