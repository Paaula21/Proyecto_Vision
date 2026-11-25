"""Evaluate the detector on a reproducible train/validation split.

The script runs inference once, scores the same predictions on the selected
fold's train and validation image IDs, and writes metrics, error summaries and
visualisation inputs to an output directory.

Example::

    python analysis/evaluate_detector.py --fold 0 \
        --data fruits_part1/train \
        --splits splits \
        --weights .artifacts/inference/detector.pth \
        --output .artifacts/evaluation/fold_0
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official_evaluator():
    """Load the assignment evaluator without copying or modifying it."""
    path = PROJECT_ROOT / "Startert Kit" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("official_evaluate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se puede cargar el evaluador oficial: {path}")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    return evaluator


def score_predictions(predictions_path: Path, ground_truth_path: Path, image_ids: list[int]) -> dict[str, Any]:
    """Validate and score predictions for exactly ``image_ids``."""
    evaluator = load_official_evaluator()
    ground_truth = evaluator.load_gt(ground_truth_path)
    predictions = load_json(predictions_path)
    valid_predictions, warnings = evaluator.validate(predictions, ground_truth)
    with fast_mask_iou(evaluator):
        metrics = evaluator.evaluate(ground_truth, valid_predictions, image_ids)
    return {"metrics": metrics, "warnings": warnings}


def filter_predictions(predictions: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    """Keep predictions whose confidence is at least ``threshold``."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("El umbral debe estar entre 0 y 1")
    return [prediction for prediction in predictions if float(prediction["score"]) >= threshold]


class _FastMaskIoU:
    """Compute exact RLE IoU without decoding every pair to a full image."""

    def __init__(self) -> None:
        self._cache: dict[int, tuple[tuple[int, ...], int, tuple[int, int, int, int]]] = {}

    @staticmethod
    def _runs(rle: dict[str, Any]) -> tuple[int, ...]:
        counts = rle["counts"]
        runs = list(counts) if isinstance(counts, list) else []
        if not isinstance(counts, list):
            value = shift = 0
            for character in counts:
                code = ord(character) - 48
                value |= (code & 0x1F) << shift
                if code & 0x20:
                    shift += 5
                    continue
                if code & 0x10:
                    value |= -1 << shift
                runs.append(value + (runs[-2] if len(runs) > 1 else 0))
                value = shift = 0
        return tuple(int(run) for run in runs)

    def _record(self, rle: dict[str, Any]) -> tuple[tuple[int, ...], int, tuple[int, int, int, int]]:
        key = id(rle)
        record = self._cache.get(key)
        if record is not None:
            return record
        runs = self._runs(rle)
        area = sum(runs[1::2])
        from pycocotools import mask as mask_utils

        x, y, width, height = (int(value) for value in mask_utils.toBbox([rle])[0])
        record = (runs, area, (x, y, x + width, y + height))
        self._cache[key] = record
        return record

    @staticmethod
    def _intersection(first: tuple[int, ...], second: tuple[int, ...]) -> int:
        first_index = second_index = 0
        first_remaining = first[0] if first else 0
        second_remaining = second[0] if second else 0
        intersection = 0
        while first_index < len(first) and second_index < len(second):
            shared = min(first_remaining, second_remaining)
            if first_index % 2 and second_index % 2:
                intersection += shared
            first_remaining -= shared
            second_remaining -= shared
            if first_remaining == 0:
                first_index += 1
                if first_index < len(first):
                    first_remaining = first[first_index]
            if second_remaining == 0:
                second_index += 1
                if second_index < len(second):
                    second_remaining = second[second_index]
        return intersection

    def iou(self, detections: list[dict[str, Any]], ground_truth: list[dict[str, Any]], iscrowd: list[int]) -> np.ndarray:
        result = np.zeros((len(detections), len(ground_truth)), dtype=float)
        gt_records = [self._record(mask) for mask in ground_truth]
        for detection_index, detection in enumerate(detections):
            detection_runs, detection_area, detection_bbox = self._record(detection)
            dx0, dy0, dx1, dy1 = detection_bbox
            for gt_index, (gt_runs, gt_area, (gx0, gy0, gx1, gy1)) in enumerate(gt_records):
                if dx1 <= gx0 or gx1 <= dx0 or dy1 <= gy0 or gy1 <= dy0:
                    continue
                intersection = self._intersection(detection_runs, gt_runs)
                union = detection_area if iscrowd[gt_index] else detection_area + gt_area - intersection
                result[detection_index, gt_index] = intersection / union if union else 0.0
        return result


class fast_mask_iou:
    """Temporarily replace the official evaluator's repeated RLE IoU calls."""

    def __init__(self, evaluator: Any) -> None:
        self.evaluator = evaluator
        self.original = evaluator.mask_utils.iou
        self.calculator = _FastMaskIoU()

    def __enter__(self) -> "fast_mask_iou":
        self.evaluator.mask_utils.iou = self.calculator.iou
        return self

    def __exit__(self, exception_type: Any, exception: Any, traceback: Any) -> None:
        self.evaluator.mask_utils.iou = self.original


def mask_iou(first: dict[str, Any], second: dict[str, Any]) -> float:
    """Return pixel IoU for two COCO RLE masks."""
    return float(_FastMaskIoU().iou([first], [second], [0])[0][0])


def _prediction_groups(predictions: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        grouped[int(prediction["image_id"])].append(prediction)
    return grouped


def classify_errors(
    predictions: list[dict[str, Any]],
    ground_truth_path: Path,
    image_ids: list[int],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Classify prediction errors using deterministic mask-IoU matching."""
    ground_truth = load_official_evaluator().load_gt(ground_truth_path)
    grouped = _prediction_groups(predictions)
    iou_calculator = _FastMaskIoU()
    selected = set(image_ids)
    summary: dict[str, Any] = {
        "iou_threshold": iou_threshold,
        "images": len(image_ids),
        "false_positives": 0,
        "false_negatives": 0,
        "classification_errors": 0,
        "overlap_errors": 0,
        "high_confidence_false_positives": 0,
        "small_false_negatives": 0,
        "occluded_false_negatives": 0,
        "matched": 0,
        "by_image": {},
    }

    for image_id in image_ids:
        annotations = [
            annotation
            for annotation in ground_truth.imgToAnns.get(image_id, [])
            if not annotation.get("iscrowd", 0)
        ]
        # Error inspection focuses on the highest-scoring proposals. Official
        # metrics still score the complete validated prediction file.
        image_predictions = sorted(grouped.get(image_id, []), key=lambda item: -item["score"])[:100]
        gt_masks = [annotation["segmentation"] for annotation in annotations]
        used_gt: set[int] = set()
        image_errors = {
            "false_positives": 0,
            "false_negatives": 0,
            "classification_errors": 0,
            "overlap_errors": 0,
            "high_confidence_false_positives": 0,
            "small_false_negatives": 0,
            "occluded_false_negatives": 0,
            "matched": 0,
        }

        ious = iou_calculator.iou(
            [prediction["segmentation"] for prediction in image_predictions],
            gt_masks,
            [0] * len(gt_masks),
        )
        for prediction_index, prediction in enumerate(image_predictions):
            if not gt_masks:
                image_errors["false_positives"] += 1
                if prediction["score"] >= 0.75:
                    image_errors["high_confidence_false_positives"] += 1
                continue
            candidates = [(float(ious[prediction_index, index]), index) for index in range(len(gt_masks)) if index not in used_gt]
            best_iou, best_gt = max(candidates, default=(0.0, -1))
            if best_iou < iou_threshold or best_gt in used_gt:
                image_errors["false_positives"] += 1
                if prediction["score"] >= 0.75:
                    image_errors["high_confidence_false_positives"] += 1
                continue
            used_gt.add(best_gt)
            image_errors["matched"] += 1
            if prediction["category_id"] != annotations[best_gt]["category_id"]:
                image_errors["classification_errors"] += 1
            if best_iou < 0.75:
                image_errors["overlap_errors"] += 1

        unmatched = set(range(len(annotations))) - used_gt
        image_errors["false_negatives"] = len(unmatched)
        for index in unmatched:
            annotation = annotations[index]
            area = float(annotation.get("area", 0))
            width, height = annotation["bbox"][2:4]
            if area < 32 * 32:
                image_errors["small_false_negatives"] += 1
            if width * height and area / (width * height) < 0.5:
                image_errors["occluded_false_negatives"] += 1
        for key in image_errors:
            summary[key] += image_errors[key]
        if any(image_errors[key] for key in image_errors):
            summary["by_image"][str(image_id)] = image_errors

    summary["images"] = len(selected)
    return summary


def select_error_examples(summary: dict[str, Any], limit_per_type: int = 10) -> dict[str, list[int]]:
    """Select deterministic representative image IDs for visual inspection."""
    names = ("false_positives", "false_negatives", "classification_errors")
    selected: dict[str, list[int]] = {}
    names = (
        "false_positives", "false_negatives", "classification_errors", "overlap_errors",
        "high_confidence_false_positives", "small_false_negatives", "occluded_false_negatives",
    )
    for name in names:
        ranked = sorted(
            (
                (int(image_id), counts[name])
                for image_id, counts in summary["by_image"].items()
                if counts.get(name, 0)
            ),
            key=lambda item: (-item[1], item[0]),
        )
        selected[name] = [image_id for image_id, _ in ranked[:limit_per_type]]
    return selected


def aggregate_fold_metrics(fold_metrics: dict[int, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Return mean and population standard deviation across completed folds."""
    import math

    if not fold_metrics:
        return {}
    names = sorted(set.intersection(*(set(metrics) for metrics in fold_metrics.values())))
    aggregate: dict[str, dict[str, float]] = {}
    for name in names:
        values = [float(fold_metrics[fold][name]) for fold in sorted(fold_metrics)]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        aggregate[name] = {"mean": mean, "std": math.sqrt(variance), "folds": len(values)}
    return aggregate


def write_report(output: Path, fold_info: dict[str, Any], scored: dict[str, Any], errors: dict[str, Any]) -> None:
    train = scored["train"]["metrics"]
    validation = scored["validation"]["metrics"]
    caveat = (
        f"El fold {fold_info['fold']} valida el árbol/especie «{fold_info['val_species']}», "
        "que no aparece en el entrenamiento. Por tanto, su AP por especie no mide "
        "solo detección: la clasificación de esa especie está fuera de distribución. "
        "AP_agnostic es la métrica principal para valorar si se encuentran frutas."
    )
    lines = [
        f"# Evaluación del detector — fold {fold_info['fold']}",
        "",
        caveat,
        "",
        "## Métricas",
        "",
        "| Métrica | Entrenamiento | Validación |",
        "|---|---:|---:|",
    ]
    metric_names = [
        "AP", "AP50", "AP75", "AP_agnostic", "accuracy",
        "AP_small", "AP_medium", "AP_large",
        "AP_orange", "AP_apple", "AP_pear", "AP_lemon",
    ]
    for name in metric_names:
        lines.append(f"| `{name}` | {train.get(name, float('nan')):.4f} | {validation.get(name, float('nan')):.4f} |")
    lines += [
        "",
        f"Imágenes: {validation['images']} de validación y {train['images']} de entrenamiento.",
        f"Detecciones puntuadas: {validation['detections']} en validación y {train['detections']} en entrenamiento.",
        "",
        "## Errores diagnósticos en validación",
        "",
        f"- Coincidencias: {errors['matched']}",
        f"- Falsos positivos: {errors['false_positives']}",
        f"- Falsos negativos: {errors['false_negatives']}",
        f"- Errores de especie: {errors['classification_errors']}",
        f"- Errores de solapamiento (IoU de máscara): {errors['overlap_errors']}",
        f"- Falsos positivos con confianza ≥ 0,75: {errors['high_confidence_false_positives']}",
        f"- Falsos negativos pequeños: {errors['small_false_negatives']}",
        f"- Falsos negativos con indicios de oclusión: {errors['occluded_false_negatives']}",
        "",
        "## Mejoras priorizadas",
        "",
        "1. Revisar los falsos negativos pequeños y las máscaras de bajo AP75: el dataset "
        "está dominado por frutas pequeñas y parcialmente ocultas.",
        "2. Si hay muchos falsos positivos de alta confianza, revisar resolución, "
        "augmentations y calibración de puntuaciones.",
        "3. Tratar por separado la clasificación de la especie excluida del fold: "
        "su AP no es comparable con las especies vistas durante el entrenamiento.",
        "",
        "La comparación entrenamiento/validación es diagnóstica y no modifica los pesos. "
        "Una diferencia grande entre ambas puede indicar sobreajuste.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_examples(
    data_dir: Path,
    ground_truth: Path,
    predictions: Path,
    output: Path,
    examples: dict[str, list[int]],
) -> list[int]:
    """Render prediction and ground-truth views for selected error images."""
    image_ids = sorted({image_id for ids in examples.values() for image_id in ids})
    if not image_ids:
        return []
    ids = [str(image_id) for image_id in image_ids]
    visualizer = PROJECT_ROOT / "Startert Kit" / "visualize.py"
    predictions_dir = output / "visualizations" / "predictions"
    ground_truth_dir = output / "visualizations" / "ground_truth"
    subprocess.run(
        [
            sys.executable, str(visualizer), "--images", str(data_dir / "images"),
            "--pred", str(predictions), "--images-json", str(data_dir / "images.json"),
            "--ids", *ids, "--out", str(predictions_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable, str(visualizer), "--images", str(data_dir / "images"),
            "--coco", str(ground_truth), "--ids", *ids, "--out", str(ground_truth_dir),
        ],
        check=True,
    )
    return image_ids


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("fruits_part1/train"))
    parser.add_argument("--splits", type=Path, default=Path("splits"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--weights", type=Path, default=Path(".artifacts/inference/detector.pth"))
    parser.add_argument("--predictions", type=Path, help="Reuse an existing full-dataset COCO predictions file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-examples-per-type", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from src.dataset import load_split

    args.output.mkdir(parents=True, exist_ok=True)
    train_ids, val_ids, fold_info, annotations_hash = load_split(args.data, args.splits, args.fold)
    predictions_path = args.predictions or (args.output / "predictions.json")
    if args.predictions is None:
        from main import load_detector, write_predictions

        write_predictions(args.data, predictions_path, load_detector(args.weights, args.device))

    ground_truth = args.data / "annotations.json"
    predictions = load_json(predictions_path)
    scored = {
        "train": score_predictions(predictions_path, ground_truth, train_ids),
        "validation": score_predictions(predictions_path, ground_truth, val_ids),
    }
    errors = classify_errors(predictions, ground_truth, val_ids)
    examples = select_error_examples(errors, args.max_examples_per_type)
    rendered_ids = render_examples(args.data, ground_truth, predictions_path, args.output, examples)
    (args.output / "metrics.json").write_text(json.dumps(scored, indent=2) + "\n", encoding="utf-8")
    (args.output / "errors.json").write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
    (args.output / "error_examples.json").write_text(json.dumps(examples, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "fold": args.fold,
        "train_ids": train_ids,
        "validation_ids": val_ids,
        "validation_species": fold_info["val_species"],
        "annotations_sha256": annotations_hash,
        "weights": str(args.weights),
        "weights_sha256": sha256(args.weights),
        "predictions": str(predictions_path),
        "device": args.device,
        "visualized_image_ids": rendered_ids,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_report(args.output, fold_info, scored, errors)
    print(f"Evaluación guardada en: {args.output}")


if __name__ == "__main__":
    main()
