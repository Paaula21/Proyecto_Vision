import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.evaluate_detector import (
    aggregate_fold_metrics,
    filter_predictions,
    mask_iou,
    select_error_examples,
)
from main import coco_rle
from analysis.run_issue6_analysis import aggregate


class EvaluationAnalysisTests(unittest.TestCase):
    def test_filter_predictions_keeps_scores_at_or_above_threshold(self):
        predictions = [
            {"image_id": 1, "score": 0.5},
            {"image_id": 1, "score": 0.49},
            {"image_id": 2, "score": 0.9},
        ]

        self.assertEqual(
            filter_predictions(predictions, 0.5),
            [predictions[0], predictions[2]],
        )

    def test_mask_iou_uses_pixels_not_bounding_boxes(self):
        mask = np.array([[True, False, False], [False, True, False], [False, False, False]])
        first = coco_rle(mask)
        second = coco_rle(mask)

        self.assertEqual(mask_iou(first, second), 1.0)

    def test_mask_iou_matches_rle_overlap(self):
        first = coco_rle(np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8))
        second = coco_rle(np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]], dtype=np.uint8))

        self.assertEqual(mask_iou(first, second), 0.5)

    def test_issue6_aggregate_reports_mean_and_population_std(self):
        result = aggregate({
            "0.25": {
                "0": {"metrics": {"AP_agnostic": 0.2, "AP75": 0.1}},
                "1": {"metrics": {"AP_agnostic": 0.4, "AP75": 0.3}},
            }
        })

        self.assertEqual(result["0.25"]["AP_agnostic"]["mean"], 0.30000000000000004)
        self.assertEqual(result["0.25"]["AP_agnostic"]["std"], 0.1)

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
