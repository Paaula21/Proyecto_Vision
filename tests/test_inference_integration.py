import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RealInferenceIntegrationTests(unittest.TestCase):
    def test_real_local_artifacts_produce_evaluable_coco_json(self):
        """Los artefactos reales de #20 deben completar el flujo oficial de detect."""
        weights = PROJECT_ROOT / ".artifacts/inference/detector.pth"
        configuration = PROJECT_ROOT / ".artifacts/inference/config.json"
        self.assertTrue(
            weights.is_file() and configuration.is_file(),
            "Faltan .artifacts/inference/detector.pth y config.json para la prueba de integración",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            sample_dir = Path(temporary_directory) / "input"
            image_dir = sample_dir / "images"
            image_dir.mkdir(parents=True)
            dataset = json.loads(
                (PROJECT_ROOT / "fruits_part1/train/images.json").read_text(encoding="utf-8")
            )
            sample_images = dataset["images"][:1]
            for image in sample_images:
                shutil.copy2(
                    PROJECT_ROOT / "fruits_part1/train/images" / image["file_name"],
                    image_dir / image["file_name"],
                )
            (sample_dir / "images.json").write_text(
                json.dumps({"images": sample_images}), encoding="utf-8"
            )
            output = Path(temporary_directory) / "predictions.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "detect",
                    "--input",
                    str(sample_dir),
                    "--output",
                    str(output),
                    "--device",
                    "cpu",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            predictions = json.loads(output.read_text(encoding="utf-8"))
            self.assertIsInstance(predictions, list)
            self.assertTrue({prediction["image_id"] for prediction in predictions} <= {1})


if __name__ == "__main__":
    unittest.main()
