"""Punto de entrada del detector de frutas."""

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_CATEGORIES = {
    "0": "background",
    "1": "orange",
    "2": "apple",
    "3": "pear",
    "4": "lemon",
}
DEFAULT_WEIGHTS = Path(".artifacts/inference/detector.pth")


def load_detector(weights, device):
    """Carga un checkpoint local en la definición compartida del detector."""
    weights = Path(weights)
    configuration_path = weights.with_name("config.json")
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    if configuration.get("categories") != EXPECTED_CATEGORIES:
        raise ValueError("categorías incompatibles con el modelo oficial")

    import torch

    from src.model import create_model

    model = create_model(pretrained=False)
    checkpoint = torch.load(weights, map_location=device)
    if checkpoint.get("config") != configuration:
        raise ValueError("configuración del checkpoint incompatible con config.json")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def image_to_tensor(image_path):
    """Convierte una imagen RGB en el tensor que espera Mask R-CNN."""
    import torch

    with Image.open(image_path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1).float().div(255.0)


def coco_rle(mask):
    """Codifica una máscara booleana en el RLE comprimido de COCO."""
    values = np.asarray(mask, dtype=np.uint8).ravel(order="F")
    counts = []
    current, length = 0, 0
    for value in values:
        if value != current:
            counts.append(length)
            current, length = value, 0
        length += 1
    counts.append(length)

    encoded = []
    for index, count in enumerate(counts):
        value = count - (counts[index - 2] if index > 2 else 0)
        while True:
            character = value & 0x1F
            value >>= 5
            more = value != (-1 if character & 0x10 else 0)
            if more:
                character |= 0x20
            character += 48
            encoded.append(chr(character + (1 if character >= 92 else 0)))
            if not more:
                break
    return {"size": list(mask.shape), "counts": "".join(encoded)}


def write_predictions(input_dir, output_path, detector):
    """Ejecuta el detector para todas las imágenes del contrato oficial."""
    images = json.loads((input_dir / "images.json").read_text(encoding="utf-8"))["images"]
    try:
        import torch
        context = torch.no_grad()
    except ModuleNotFoundError:
        context = contextlib.nullcontext()

    predictions = []
    with context:
        for image in images:
            outputs = detector([image_to_tensor(input_dir / "images" / image["file_name"])])[0]
            masks = outputs["masks"].detach().cpu().numpy()[:, 0] >= 0.5
            labels = outputs["labels"].detach().cpu().numpy()
            scores = outputs["scores"].detach().cpu().numpy()
            image_predictions = []
            for mask, label, score in zip(masks, labels, scores):
                category_id = int(label)
                if category_id not in (1, 2, 3, 4):
                    raise ValueError(f"category_id inválido para una detección: {category_id}")
                score = float(score)
                if not 0.0 <= score <= 1.0:
                    raise ValueError(f"score inválido para una detección: {score}")
                expected_shape = (int(image["height"]), int(image["width"]))
                if tuple(mask.shape) != expected_shape:
                    raise ValueError(
                        f"segmentation size inválido: {tuple(mask.shape)} != {expected_shape}"
                    )
                image_predictions.append({
                    "image_id": image["id"],
                    "category_id": category_id,
                    "score": score,
                    "segmentation": coco_rle(mask),
                })
            image_predictions.sort(key=lambda prediction: -prediction["score"])
            predictions.extend(image_predictions[:300])
    output_path.write_text(json.dumps(predictions), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="task", required=True)

    detect = subcommands.add_parser("detect", help="Genera predicciones de segmentación")
    detect.add_argument("--input", type=Path, required=True)
    detect.add_argument("--output", type=Path, required=True)
    detect.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    detect.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    args = parser.parse_args(argv)
    if args.task == "detect":
        if args.device == "cuda":
            import torch

            if not torch.cuda.is_available():
                parser.error("Se solicitó CUDA, pero no hay una GPU CUDA disponible")
        write_predictions(args.input, args.output, load_detector(args.weights, args.device))


if __name__ == "__main__":
    main()
