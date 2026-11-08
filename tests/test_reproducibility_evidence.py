import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA = PROJECT_ROOT / ".artifacts/inference/run_metadata.json"


class ReproducibilityEvidenceTests(unittest.TestCase):
    def test_inference_evidence_records_inputs_and_runtime(self):
        self.assertTrue(METADATA.is_file(), "Falta run_metadata.json")
        metadata = json.loads(METADATA.read_text(encoding="utf-8"))
        for key in ("command", "weights_sha256", "config_sha256", "python", "torch", "device"):
            self.assertIn(key, metadata)
        self.assertEqual(metadata["device"], "cpu")
        self.assertEqual(metadata["weights"], ".artifacts/inference/detector.pth")
        self.assertEqual(metadata["config"], ".artifacts/inference/config.json")


if __name__ == "__main__":
    unittest.main()
