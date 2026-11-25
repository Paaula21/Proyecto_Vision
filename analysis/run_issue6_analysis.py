"""Run the reproducible four-fold analysis for issue #6.

The command can train fold checkpoints with ``--train`` or consume existing
checkpoints. Predictions are generated once per fold and then rescored at the
configured confidence thresholds.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from .evaluate_detector import (
        classify_errors,
        filter_predictions,
        load_json,
        render_examples,
        score_predictions,
        select_error_examples,
        sha256,
    )
except ImportError:  # Direct execution: python analysis/run_issue6_analysis.py
    from evaluate_detector import (
        classify_errors,
        filter_predictions,
        load_json,
        render_examples,
        score_predictions,
        select_error_examples,
        sha256,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (0.05, 0.25, 0.50)


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def train_fold(args: argparse.Namespace, fold: int, output: Path) -> Path:
    command = [
        sys.executable, "train.py",
        "--data", str(args.data), "--splits", str(args.splits),
        "--fold", str(fold), "--output", str(output),
        "--device", args.device, "--epochs", str(args.epochs),
        "--learning-rate", str(args.learning_rate), "--seed", str(args.seed),
    ]
    if args.exclude_iscrowd:
        command.append("--exclude-iscrowd")
    run_command(command)
    return output / "detector.pth"


def _inference_input(args: argparse.Namespace, fold: int, output: Path) -> Path:
    """Return the dataset input, optionally reduced to validation images.

    The reduced mode is deliberately opt-in and is reported as provisional. It
    keeps the official annotations untouched while making a CPU smoke analysis
    possible with an existing checkpoint.
    """
    if args.image_limit is None:
        return args.data

    folds = load_json(args.splits / "folds.json")["folds"]
    val_ids = [int(image_id) for image_id in folds[fold]["val_ids"]][:args.image_limit]
    source_images = {int(item["id"]): item for item in load_json(args.data / "images.json")["images"]}
    subset = output / "inference_input"
    (subset / "images").mkdir(parents=True, exist_ok=True)
    selected = [source_images[image_id] for image_id in val_ids]
    for image in selected:
        source = args.data / "images" / image["file_name"]
        target = subset / "images" / image["file_name"]
        if not target.exists():
            shutil.copy2(source, target)
    (subset / "images.json").write_text(json.dumps({"images": selected}), encoding="utf-8")
    return subset


def infer_fold(args: argparse.Namespace, fold: int, weights: Path, output: Path) -> Path:
    predictions = output / "predictions.json"
    input_dir = _inference_input(args, fold, output)
    run_command([
        sys.executable, "main.py", "detect",
        "--input", str(input_dir), "--output", str(predictions),
        "--weights", str(weights), "--device", args.device,
    ])
    return predictions


def threshold_label(threshold: float) -> str:
    return "all" if threshold == 0.0 else f"threshold_{threshold:.2f}".replace(".", "_")


def aggregate(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_threshold: dict[str, Any] = {}
    for threshold, fold_results in results.items():
        metrics = [item["metrics"] for item in fold_results.values()]
        names = sorted(set.intersection(*(set(item) for item in metrics))) if metrics else []
        aggregate_metrics = {}
        for name in names:
            values = [float(item[name]) for item in metrics]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            aggregate_metrics[name] = {
                "mean": mean,
                "std": variance ** 0.5,
                "folds": len(values),
            }
        by_threshold[threshold] = aggregate_metrics
    return by_threshold


def choose_recommendation(aggregate_metrics: dict[str, Any], errors: dict[str, Any]) -> str:
    def key(label: str) -> tuple[float, float, float]:
        metrics = aggregate_metrics[label]
        error = errors[label]
        matched = max(1, sum(item["errors"]["matched"] for item in error.values()))
        imprecise = sum(item["errors"]["overlap_errors"] for item in error.values()) / matched
        return (metrics.get("AP_agnostic", {}).get("mean", float("-inf")),
                metrics.get("AP75", {}).get("mean", float("-inf")), -imprecise)

    return max(aggregate_metrics, key=key)


def write_report(
    output: Path,
    aggregate_metrics: dict[str, Any],
    recommendation: str,
    errors: dict[str, dict[str, Any]],
    folds: list[int],
    image_limit: int | None,
) -> None:
    provisional = image_limit is not None or len(folds) < 4
    lines = [
        "# Issue #6 — análisis de experimentos y errores",
        "",
        ("**Estado: provisional.** Este resultado usa una muestra limitada y no "
         "debe compararse como resultado LOTO final."
         if provisional else "**Estado: completo.** Se han evaluado los cuatro folds LOTO."),
        "",
        "La selección usa la media de `AP_agnostic` en los folds disponibles. "
        "`AP75` y la proporción de máscaras imprecisas sirven como desempate.",
        "",
        f"Folds evaluados: {', '.join(map(str, folds))}. "
        f"Imágenes por fold: {'máximo ' + str(image_limit) if image_limit else 'todas las de validación'}.",
        "",
        f"## Configuración recomendada: `{recommendation}`",
        "",
        "| Umbral | AP agnóstico medio | AP75 medio | AP pequeño medio | Folds |",
        "|---:|---:|---:|---:|---:|",
    ]
    for label, metrics in aggregate_metrics.items():
        lines.append(
            f"| {label} | {metrics.get('AP_agnostic', {}).get('mean', float('nan')):.4f} "
            f"| {metrics.get('AP75', {}).get('mean', float('nan')):.4f} "
            f"| {metrics.get('AP_small', {}).get('mean', float('nan')):.4f} "
            f"| {metrics.get('AP_agnostic', {}).get('folds', 0)} |"
        )
    lines += ["", "## Errores diagnósticos agregados", "", "| Umbral | FP | FN | especie | máscaras imprecisas | pequeños FN |", "|---:|---:|---:|---:|---:|---:|"]
    for label, fold_errors in errors.items():
        totals = {name: sum(item["errors"].get(name, 0) for item in fold_errors.values()) for name in (
            "false_positives", "false_negatives", "classification_errors", "overlap_errors", "small_false_negatives"
        )}
        lines.append(
            f"| {label} | {totals['false_positives']} | {totals['false_negatives']} | "
            f"{totals['classification_errors']} | {totals['overlap_errors']} | {totals['small_false_negatives']} |"
        )
    lines += [
        "",
        "La especie excluida se conserva como condición de interpretación de cada fold: "
        "manzana, pera, naranja y limón, respectivamente. Las métricas por especie no "
        "se usan para seleccionar el umbral.",
    ]
    report = "\n".join(lines) + "\n"
    (output / "issue6_report.md").write_text(report, encoding="utf-8")
    (PROJECT_ROOT / "analysis" / "issue6_report.md").write_text(report, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("fruits_part1/train"))
    parser.add_argument("--splits", type=Path, default=Path("splits"))
    parser.add_argument("--output", type=Path, default=Path(".artifacts/issue6"))
    parser.add_argument("--weights-root", type=Path)
    parser.add_argument("--weights", type=Path, help="Checkpoint existente para un único fold")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--image-limit", type=int, help="Limita imágenes de validación; el informe queda provisional")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-iscrowd", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--skip-visualizations", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.train and args.weights_root is None and args.weights is None:
        raise SystemExit("Usa --train, --weights-root o --weights con un checkpoint")
    if not args.folds or any(fold not in range(4) for fold in args.folds):
        raise SystemExit("--folds solo admite valores entre 0 y 3")
    if args.image_limit is not None and args.image_limit < 1:
        raise SystemExit("--image-limit debe ser positivo")
    if args.weights is not None and len(args.folds) != 1:
        raise SystemExit("--weights solo puede usarse con un único fold")

    ground_truth = args.data / "annotations.json"
    aggregate_inputs: dict[str, dict[str, Any]] = {str(t): {} for t in THRESHOLDS}
    errors_by_threshold: dict[str, dict[str, Any]] = {str(t): {} for t in THRESHOLDS}

    folds_to_run = list(dict.fromkeys(args.folds))
    for fold in folds_to_run:
        fold_output = args.output / f"fold_{fold}"
        fold_output.mkdir(parents=True, exist_ok=True)
        weights_output = fold_output / "weights"
        if args.train:
            weights = train_fold(args, fold, weights_output)
        elif args.weights is not None:
            weights = args.weights
        else:
            weights = args.weights_root / f"fold_{fold}" / "weights" / "detector.pth"
        predictions_path = fold_output / "predictions.json"
        if not predictions_path.exists():
            infer_fold(args, fold, weights, fold_output)
        predictions = load_json(predictions_path)
        fold_info_path = args.splits / "folds.json"
        folds = load_json(fold_info_path)["folds"]
        val_ids = [int(image_id) for image_id in folds[fold]["val_ids"]]

        for threshold in THRESHOLDS:
            label = str(threshold)
            selected = filter_predictions(predictions, threshold)
            selected_path = fold_output / f"predictions_{threshold_label(threshold)}.json"
            selected_path.write_text(json.dumps(selected), encoding="utf-8")
            errors_path = fold_output / f"errors_{threshold_label(threshold)}.json"
            if errors_path.exists():
                cached = load_json(errors_path)
                errors = cached["errors"]
                examples = cached.get("examples", select_error_examples(errors))
                if "metrics" in cached:
                    scored = {"metrics": cached["metrics"]}
                else:
                    scored = score_predictions(selected_path, ground_truth, val_ids)
                    errors_path.write_text(
                        json.dumps({"metrics": scored["metrics"], "errors": errors, "examples": examples}, indent=2),
                        encoding="utf-8",
                    )
            else:
                scored = score_predictions(selected_path, ground_truth, val_ids)
                errors = classify_errors(selected, ground_truth, val_ids)
                examples = select_error_examples(errors)
                errors_path.write_text(
                    json.dumps({"metrics": scored["metrics"], "errors": errors, "examples": examples}, indent=2),
                    encoding="utf-8",
                )
            aggregate_inputs[label][str(fold)] = {"metrics": scored["metrics"]}
            errors_by_threshold[label][str(fold)] = {"metrics": scored["metrics"], "errors": errors}
            if not args.skip_visualizations and threshold == 0.25:
                render_examples(args.data, ground_truth, selected_path, fold_output / "visualizations", examples)
        (fold_output / "metadata.json").write_text(json.dumps({
            "fold": fold, "weights": str(weights), "weights_sha256": sha256(weights),
            "device": args.device, "seed": args.seed, "thresholds": list(THRESHOLDS),
            "image_limit": args.image_limit,
            "exclude_iscrowd": args.exclude_iscrowd,
        }, indent=2) + "\n", encoding="utf-8")

    aggregate_metrics = aggregate(aggregate_inputs)
    recommendation = choose_recommendation(aggregate_metrics, errors_by_threshold)
    (args.output / "aggregate_metrics.json").write_text(json.dumps(aggregate_metrics, indent=2) + "\n", encoding="utf-8")
    (args.output / "recommendation.json").write_text(json.dumps({"threshold": recommendation}, indent=2) + "\n", encoding="utf-8")
    write_report(args.output, aggregate_metrics, recommendation, errors_by_threshold, folds_to_run, args.image_limit)


if __name__ == "__main__":
    main()
