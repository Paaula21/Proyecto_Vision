import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.evaluate_detector import aggregate_fold_metrics, select_error_examples


class EvaluationAnalysisTests(unittest.TestCase):
    def test_select_error_examples_is_deterministic_and_limited(self):
        summary = {
            "by_image": {
                "12": {"false_positives": 1, "false_negatives": 4, "classification_errors": 0, "matched": 2},
                "3": {"false_positives": 3, "false_negatives": 1, "classification_errors": 2, "matched": 5},
                "7": {"false_positives": 3, "false_negatives": 0, "classification_errors": 2, "matched": 4},
            }
        }

        self.assertEqual(
            select_error_examples(summary, limit_per_type=2),
            {
                "false_positives": [3, 7],
                "false_negatives": [12, 3],
                "classification_errors": [3, 7],
                "overlap_errors": [],
                "high_confidence_false_positives": [],
                "small_false_negatives": [],
                "occluded_false_negatives": [],
            },
        )

    def test_aggregate_fold_metrics_reports_mean_and_standard_deviation(self):
        self.assertEqual(
            aggregate_fold_metrics({0: {"AP": 0.2}, 1: {"AP": 0.4}}),
            {"AP": {"mean": 0.30000000000000004, "std": 0.1, "folds": 2}},
        )


if __name__ == "__main__":
    unittest.main()
