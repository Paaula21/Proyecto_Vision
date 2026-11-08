import importlib.util
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = PROJECT_ROOT / ".artifacts/inference/predictions_sample.json"
GROUND_TRUTH = PROJECT_ROOT / "fruits_part1/train/annotations.json"


class OfficialEvaluationTests(unittest.TestCase):
    def test_real_predictions_are_accepted_by_official_validator(self):
        self.assertTrue(PREDICTIONS.is_file(), "Falta predictions_sample.json")
        spec = importlib.util.spec_from_file_location("official_evaluate", PROJECT_ROOT / "Startert Kit/evaluate.py")
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)

        ground_truth = evaluator.load_gt(GROUND_TRUTH)
        predictions = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
        kept, warnings = evaluator.validate(predictions, ground_truth)

        self.assertGreater(len(kept), 0)
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
