import json
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import create_model


ARTIFACTS = PROJECT_ROOT / ".artifacts/inference"


class ArtifactContractTests(unittest.TestCase):
    def test_checkpoint_and_sidecar_match_the_shared_model(self):
        weights = ARTIFACTS / "detector.pth"
        sidecar = ARTIFACTS / "config.json"
        self.assertTrue(weights.is_file())
        self.assertTrue(sidecar.is_file())

        configuration = json.loads(sidecar.read_text(encoding="utf-8"))
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)

        self.assertEqual(checkpoint["epoch"], 1)
        self.assertEqual(checkpoint["config"], configuration)
        self.assertEqual(
            configuration["categories"],
            {"0": "background", "1": "orange", "2": "apple", "3": "pear", "4": "lemon"},
        )
        self.assertEqual(configuration["architecture"], "maskrcnn_resnet50_fpn")

        model = create_model(pretrained=False)
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        self.assertTrue(
            all(torch.isfinite(value).all().item() for value in checkpoint["model_state_dict"].values())
        )


if __name__ == "__main__":
    unittest.main()
