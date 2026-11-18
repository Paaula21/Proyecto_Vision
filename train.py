"""Entrena el detector inicial con un fold generado por splits/make_split.py."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from src.dataset import FruitDataset, collate_fn, load_split
from src.model import create_model


def main():
    """
    Leemos los parámetros del comando
        Rutas
        Fold
        Dispositivo
        Numero de épocas
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Carpeta train con images/ y annotations.json")
    parser.add_argument("--splits", type=Path, default=Path("splits"), help="Carpeta con los folds generados")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="Directorio nuevo para pesos y configuración")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.learning_rate <= 0:
        parser.error("--epochs y --learning-rate deben ser positivos")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("Se solicitó CUDA, pero no hay una GPU CUDA disponible")
    if args.output.exists():
        parser.error("--output debe ser un directorio nuevo para no sobrescribir experimentos")

    "Comprobamos el split"
    train_ids, val_ids, fold_info, annotations_hash = load_split(args.data, args.splits, args.fold)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        
    "Preparamos las imágenes de entrenamiento"
    dataset = FruitDataset(args.data, train_ids)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)
    device = torch.device(args.device)
    # Los pesos preentrenados pueden descargarse durante el entrenamiento.
    model = create_model(pretrained=True).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)

    config = {
        "architecture": "maskrcnn_resnet50_fpn",
        "initial_weights": "MaskRCNN_ResNet50_FPN_Weights.DEFAULT",
        "categories": {"0": "background", "1": "orange", "2": "apple", "3": "pear", "4": "lemon"},
        "fold": args.fold,
        "validation_species": fold_info["val_species"],
        "train_ids": train_ids,
        "val_ids": val_ids,
        "annotations_sha256": annotations_hash,
        "iscrowd_training": "included_as_positive",
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": 1,
        "seed": args.seed,
        "device": args.device,
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
    }
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    "Entrena durante el número de épocas indicado"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for image_number, (images, targets) in enumerate(loader, start=1):
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"La pérdida no es finita: {losses}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if image_number % 10 == 0 or image_number == len(loader):
                print(
                    f"Época {epoch}/{args.epochs} | Imagen {image_number}/{len(loader)} | "
                    f"Pérdida media: {total_loss / image_number:.4f}",
                    flush=True,
                )
        "Guarda los resultados"
        torch.save(
            {"model_state_dict": model.state_dict(), "epoch": epoch, "config": config},
            args.output / "detector.pth",
        )
    print(f"Pesos guardados en: {args.output / 'detector.pth'}")


if __name__ == "__main__":
    main()
