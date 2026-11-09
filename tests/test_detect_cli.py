"""Contrato de la interfaz pública del comando ``detect``."""

import json
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main


class DetectCommandContractTests(unittest.TestCase):
    def test_detect_requires_local_weights_and_accepts_cpu_or_cuda(self):
        """El comando declara pesos locales obligatorios y los dispositivos oficiales."""
        help_result = subprocess.run(
            [sys.executable, "main.py", "detect", "--help"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )

        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--weights", help_result.stdout)
        self.assertIn("{cpu,cuda}", help_result.stdout)

        missing_weights = subprocess.run(
            [
                sys.executable,
                "main.py",
                "detect",
                "--input",
                "input",
                "--output",
                "predictions.json",
                "--device",
                "cpu",
                "--weights",
                ".artifacts/inference/missing-detector.pth",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )

        self.assertNotEqual(missing_weights.returncode, 0)
        self.assertIn("missing-detector.pth", missing_weights.stderr)

    def test_detect_writes_coco_rle_predictions_for_every_input_image(self):
        """El comando conserva los IDs y escribe máscara, categoría y puntuación COCO."""
        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class Detector:
            def eval(self):
                return self

            def __call__(self, images):
                mask = np.array([[[[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]]])
                return [{"masks": Tensor(mask), "labels": Tensor([2]), "scores": Tensor([0.9])} for _ in images]

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_dir = Path(temporary_directory) / "input"
            image_dir = input_dir / "images"
            image_dir.mkdir(parents=True)
            for file_name in ("first.png", "second.png"):
                Image.new("RGB", (3, 2)).save(image_dir / file_name)
            (input_dir / "images.json").write_text(
                json.dumps({"images": [
                    {"id": 41, "file_name": "first.png", "width": 3, "height": 2},
                    {"id": 99, "file_name": "second.png", "width": 3, "height": 2},
                ]})
            )
            output = Path(temporary_directory) / "predictions.json"

            with (
                patch.object(main, "load_detector", return_value=Detector()),
                patch.object(main, "image_to_tensor", return_value=object()),
            ):
                main.main([
                    "detect", "--input", str(input_dir), "--output", str(output),
                    "--weights", "local-detector.pth", "--device", "cpu",
                ])

            predictions = json.loads(output.read_text())

        self.assertEqual(
            predictions,
            [
                {"image_id": 41, "category_id": 2, "score": 0.9,
                 "segmentation": {"size": [2, 3], "counts": "0120O0"}},
                {"image_id": 99, "category_id": 2, "score": 0.9,
                 "segmentation": {"size": [2, 3], "counts": "0120O0"}},
            ],
        )

    def test_detect_rejects_background_as_prediction_category(self):
        """La salida COCO no puede usar la categoría de fondo como detección."""
        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class Detector:
            def __call__(self, _images):
                return [{
                    "masks": Tensor(np.array([[[[1.0]]]])),
                    "labels": Tensor([0]),
                    "scores": Tensor([0.9]),
                }]

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_dir = Path(temporary_directory) / "input"
            (input_dir / "images").mkdir(parents=True)
            Image.new("RGB", (1, 1)).save(input_dir / "images" / "image.png")
            (input_dir / "images.json").write_text(json.dumps({"images": [
                {"id": 1, "file_name": "image.png", "width": 1, "height": 1},
            ]}))

            with patch.object(main, "image_to_tensor", return_value=object()):
                with self.assertRaisesRegex(ValueError, "category_id"):
                    main.write_predictions(input_dir, Path(temporary_directory) / "predictions.json", Detector())

    def test_detect_rejects_scores_outside_coco_range(self):
        """La salida COCO solo admite puntuaciones entre cero y uno."""
        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class Detector:
            def __call__(self, _images):
                return [{
                    "masks": Tensor(np.array([[[[1.0]]]])),
                    "labels": Tensor([1]),
                    "scores": Tensor([1.1]),
                }]

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_dir = Path(temporary_directory) / "input"
            (input_dir / "images").mkdir(parents=True)
            Image.new("RGB", (1, 1)).save(input_dir / "images" / "image.png")
            (input_dir / "images.json").write_text(json.dumps({"images": [
                {"id": 1, "file_name": "image.png", "width": 1, "height": 1},
            ]}))

            with patch.object(main, "image_to_tensor", return_value=object()):
                with self.assertRaisesRegex(ValueError, "score"):
                    main.write_predictions(input_dir, Path(temporary_directory) / "predictions.json", Detector())

    def test_detect_rejects_masks_with_wrong_image_dimensions(self):
        """La máscara COCO debe tener la altura y anchura declaradas por la imagen."""
        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class Detector:
            def __call__(self, _images):
                return [{
                    "masks": Tensor(np.array([[[[1.0]]]])),
                    "labels": Tensor([1]),
                    "scores": Tensor([0.9]),
                }]

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_dir = Path(temporary_directory) / "input"
            (input_dir / "images").mkdir(parents=True)
            Image.new("RGB", (2, 2)).save(input_dir / "images" / "image.png")
            (input_dir / "images.json").write_text(json.dumps({"images": [
                {"id": 1, "file_name": "image.png", "width": 2, "height": 2},
            ]}))

            with patch.object(main, "image_to_tensor", return_value=object()):
                with self.assertRaisesRegex(ValueError, "segmentation size"):
                    main.write_predictions(input_dir, Path(temporary_directory) / "predictions.json", Detector())

    def test_detect_keeps_only_the_300_highest_scoring_detections_per_image(self):
        """La salida respeta el límite de 300 detecciones del contrato oficial."""
        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class Detector:
            def __call__(self, _images):
                return [{
                    "masks": Tensor(np.ones((301, 1, 1, 1))),
                    "labels": Tensor(np.ones(301, dtype=int)),
                    "scores": Tensor(np.arange(301, dtype=float) / 300.0),
                }]

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_dir = Path(temporary_directory) / "input"
            (input_dir / "images").mkdir(parents=True)
            Image.new("RGB", (1, 1)).save(input_dir / "images" / "image.png")
            (input_dir / "images.json").write_text(json.dumps({"images": [
                {"id": 1, "file_name": "image.png", "width": 1, "height": 1},
            ]}))
            output = Path(temporary_directory) / "predictions.json"

            with patch.object(main, "image_to_tensor", return_value=object()):
                main.write_predictions(input_dir, output, Detector())

            predictions = json.loads(output.read_text())

        self.assertEqual(len(predictions), 300)
        self.assertEqual(predictions[0]["score"], 1.0)
        self.assertEqual(predictions[-1]["score"], 1 / 300)

    def test_load_detector_uses_only_the_local_checkpoint_and_shared_model(self):
        """La inferencia no pide pesos preentrenados ni consulta la red."""
        configuration = {
            "architecture": "maskrcnn_resnet50_fpn",
            "categories": {"0": "background", "1": "orange", "2": "apple", "3": "pear", "4": "lemon"},
        }

        class Detector:
            def load_state_dict(self, state_dict):
                self.state_dict = state_dict

            def to(self, device):
                self.device = device
                return self

            def eval(self):
                self.is_evaluation = True
                return self

        detector = Detector()
        calls = []
        torch = ModuleType("torch")
        torch.load = lambda path, map_location: calls.append((path, map_location)) or {
            "model_state_dict": {"local": "weights"}, "config": configuration,
        }
        model = ModuleType("src.model")
        model.create_model = lambda *, pretrained: calls.append(("create_model", pretrained)) or detector

        with tempfile.TemporaryDirectory() as temporary_directory:
            weights = Path(temporary_directory) / "detector.pth"
            (weights.parent / "config.json").write_text(json.dumps(configuration))
            with patch.dict(sys.modules, {"torch": torch, "src.model": model}):
                loaded = main.load_detector(weights, "cpu")

        self.assertIs(loaded, detector)
        self.assertEqual(calls, [("create_model", False), (weights, "cpu")])
        self.assertEqual(detector.state_dict, {"local": "weights"})
        self.assertEqual(detector.device, "cpu")
        self.assertTrue(detector.is_evaluation)

    def test_load_detector_rejects_an_incompatible_sidecar_configuration(self):
        """Los pesos no se usan si su configuración no define las categorías oficiales."""
        incompatible_configuration = {
            "architecture": "maskrcnn_resnet50_fpn",
            "categories": {"0": "background", "1": "apple", "2": "orange", "3": "pear", "4": "lemon"},
        }
        torch = ModuleType("torch")
        torch.load = lambda *_args, **_kwargs: self.fail("no debe cargar pesos incompatibles")
        model = ModuleType("src.model")
        model.create_model = lambda *, pretrained: self.fail("no debe crear un modelo incompatible")

        with tempfile.TemporaryDirectory() as temporary_directory:
            weights = Path(temporary_directory) / "detector.pth"
            (weights.parent / "config.json").write_text(json.dumps(incompatible_configuration))
            with patch.dict(sys.modules, {"torch": torch, "src.model": model}):
                with self.assertRaisesRegex(ValueError, "categorías"):
                    main.load_detector(weights, "cpu")

    def test_load_detector_rejects_an_incompatible_checkpoint_configuration(self):
        """Los pesos no se usan si su configuración embebida contradice al sidecar."""
        sidecar_configuration = {
            "architecture": "maskrcnn_resnet50_fpn",
            "categories": {"0": "background", "1": "orange", "2": "apple", "3": "pear", "4": "lemon"},
        }
        checkpoint_configuration = {
            **sidecar_configuration,
            "categories": {"0": "background", "1": "apple", "2": "orange", "3": "pear", "4": "lemon"},
        }

        class Detector:
            def load_state_dict(self, _state_dict):
                self.loaded = True

            def to(self, _device):
                return self

            def eval(self):
                return self

        torch = ModuleType("torch")
        torch.load = lambda *_args, **_kwargs: {
            "model_state_dict": {"local": "weights"},
            "config": checkpoint_configuration,
        }
        model = ModuleType("src.model")
        model.create_model = lambda *, pretrained: Detector()

        with tempfile.TemporaryDirectory() as temporary_directory:
            weights = Path(temporary_directory) / "detector.pth"
            (weights.parent / "config.json").write_text(json.dumps(sidecar_configuration))
            with patch.dict(sys.modules, {"torch": torch, "src.model": model}):
                with self.assertRaisesRegex(ValueError, "configuración"):
                    main.load_detector(weights, "cpu")

    def test_detect_rejects_cuda_when_it_is_not_available(self):
        """El comando no cambia silenciosamente de CUDA a CPU."""
        torch = ModuleType("torch")
        torch.cuda = SimpleNamespace(is_available=lambda: False)
        errors = io.StringIO()

        with (
            patch.dict(sys.modules, {"torch": torch}),
            patch.object(main, "load_detector", side_effect=AssertionError("no debe cargar pesos")),
            redirect_stderr(errors),
            self.assertRaises(SystemExit) as stopped,
        ):
            main.main([
                "detect", "--input", "input", "--output", "predictions.json",
                "--weights", "local-detector.pth", "--device", "cuda",
            ])

        self.assertEqual(stopped.exception.code, 2)
        self.assertIn("Se solicitó CUDA", errors.getvalue())

    def test_detect_forwards_cuda_to_the_detector_when_available(self):
        """El comando conserva CUDA cuando el dispositivo está disponible."""
        torch = ModuleType("torch")
        torch.cuda = SimpleNamespace(is_available=lambda: True)
        detector = object()

        with (
            patch.dict(sys.modules, {"torch": torch}),
            patch.object(main, "load_detector", return_value=detector) as load_detector,
            patch.object(main, "write_predictions") as write_predictions,
        ):
            main.main([
                "detect", "--input", "input", "--output", "predictions.json",
                "--weights", "local-detector.pth", "--device", "cuda",
            ])

        load_detector.assert_called_once_with(Path("local-detector.pth"), "cuda")
        write_predictions.assert_called_once_with(
            Path("input"), Path("predictions.json"), detector
        )


if __name__ == "__main__":
    unittest.main()
